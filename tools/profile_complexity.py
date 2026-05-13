#!/usr/bin/env python
# encoding: utf-8

import argparse
import json
import logging
import sys
import time

import torch
from fvcore.nn import flop_count
from fvcore.nn.jit_handles import elementwise_flop_counter
from torch.profiler import ProfilerActivity, profile

sys.path.append('.')

from fastreid.config import get_cfg
from fastreid.data import CommDataset, build_reid_test_loader
from fastreid.data.datasets import DATASET_REGISTRY
from fastreid.data.transforms import build_transforms
from fastreid.evaluation.evaluator import inference_context
from fastreid.modeling import build_model


def parse_args():
    parser = argparse.ArgumentParser(description="Profile params, FLOPs, and latency")
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--dataset-name", default="")
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--flops-batch-size", type=int, default=1)
    parser.add_argument("--flops-mode", choices=("auto", "fvcore", "profiler"), default="auto")
    parser.add_argument("--warmup-batches", type=int, default=5)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--disable-pretrain", action="store_true")
    parser.add_argument("--dummy-latency-iters", type=int, default=0)
    parser.add_argument("--dummy-latency-warmup", type=int, default=10)
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


def supported_ops():
    return {
        "aten::silu": elementwise_flop_counter(0, 1),
        "aten::gelu": elementwise_flop_counter(0, 1),
        "aten::neg": elementwise_flop_counter(0, 1),
        "aten::exp": elementwise_flop_counter(0, 1),
        "aten::flip": elementwise_flop_counter(0, 1),
        "aten::mul": elementwise_flop_counter(0, 1),
        "aten::div": elementwise_flop_counter(0, 1),
        "aten::softmax": elementwise_flop_counter(0, 2),
        "aten::sigmoid": elementwise_flop_counter(0, 1),
        "aten::add": elementwise_flop_counter(0, 1),
        "aten::add_": elementwise_flop_counter(0, 1),
        "aten::radd": elementwise_flop_counter(0, 1),
        "aten::sub": elementwise_flop_counter(0, 1),
        "aten::sub_": elementwise_flop_counter(0, 1),
        "aten::rsub": elementwise_flop_counter(0, 1),
        "aten::mul_": elementwise_flop_counter(0, 1),
        "aten::rmul": elementwise_flop_counter(0, 1),
        "aten::div_": elementwise_flop_counter(0, 1),
        "aten::rdiv": elementwise_flop_counter(0, 1),
        "aten::cumsum": elementwise_flop_counter(0, 1),
        "aten::ne": elementwise_flop_counter(0, 1),
        "aten::silu_": elementwise_flop_counter(0, 1),
        "aten::dropout_": elementwise_flop_counter(0, 1),
        "aten::log_softmax": elementwise_flop_counter(0, 2),
        "aten::argmax": elementwise_flop_counter(0, 1),
        "aten::one_hot": elementwise_flop_counter(0, 1),
        "aten::flatten": elementwise_flop_counter(0, 0),
        "aten::unflatten": elementwise_flop_counter(0, 0),
        "aten::mean": elementwise_flop_counter(1, 0),
        "aten::sum": elementwise_flop_counter(1, 0),
        "aten::abs": elementwise_flop_counter(0, 1),
        "aten::tanh": elementwise_flop_counter(0, 1),
        "aten::relu": elementwise_flop_counter(0, 1),
        "aten::where": elementwise_flop_counter(0, 1),
        "aten::le": elementwise_flop_counter(0, 1),
        "aten::topk": elementwise_flop_counter(1, 1),
        "aten::sort": elementwise_flop_counter(1, 1),
        "aten::argsort": elementwise_flop_counter(1, 1),
        "aten::scatter": elementwise_flop_counter(1, 1),
        "aten::gather": elementwise_flop_counter(1, 1),
        "aten::adaptive_max_pool2d": elementwise_flop_counter(1, 0),
    }


def build_cfg(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    if args.disable_pretrain:
        cfg.MODEL.BACKBONE.PRETRAIN = False
    if args.batch_size is not None:
        cfg.TEST.IMS_PER_BATCH = args.batch_size
    cfg.TEST.FLIP.ENABLED = False
    if args.num_inference_steps is not None:
        cfg.TEST.SDMODEL.NUM_INFERENCE_STEPS = args.num_inference_steps
    cfg.freeze()
    return cfg


def make_dummy_inputs(cfg, model, batch_size):
    device = next(model.parameters()).device
    images = torch.randint(0, 255, (batch_size, 3, cfg.INPUT.SIZE_TEST[0], cfg.INPUT.SIZE_TEST[1]), device=device)
    images = images.to(torch.float32)
    camids = torch.zeros(batch_size, dtype=torch.int64, device=device)
    viewids = torch.zeros(batch_size, dtype=torch.int64, device=device)
    targets = torch.zeros(batch_size, dtype=torch.int64, device=device)
    return {
        "images": images,
        "camids": camids,
        "viewids": viewids,
        "targets": targets,
    }


def compute_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def compute_gflops_with_profiler(cfg, model, batch_size, error_message=""):
    dummy_inputs = make_dummy_inputs(cfg, model, batch_size)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    with torch.no_grad():
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            with_flops=True,
            record_shapes=False,
            profile_memory=False,
        ) as prof:
            _ = model(dummy_inputs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
    total_flops = sum((event.flops or 0) for event in prof.key_averages())
    total_gflops = total_flops / 1e9 / batch_size
    return total_gflops, {}, "torch.profiler", error_message


def compute_gflops(cfg, model, batch_size, flops_mode):
    dummy_inputs = make_dummy_inputs(cfg, model, batch_size)
    if flops_mode == "profiler":
        return compute_gflops_with_profiler(cfg, model, batch_size)

    try:
        with torch.no_grad():
            gflops, unsupported = flop_count(model=model, inputs=(dummy_inputs,), supported_ops=supported_ops())
        unsupported_ops_dict = {key: int(val) for key, val in unsupported.items()}
        total_gflops = sum(gflops.values()) / batch_size
        return total_gflops, unsupported_ops_dict, "fvcore", ""
    except Exception as error:
        if flops_mode == "fvcore":
            raise
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return compute_gflops_with_profiler(cfg, model, batch_size, str(error))


def measure_latency(cfg, model, dataset_name, warmup_batches, max_batches):
    dataset = DATASET_REGISTRY.get(dataset_name)(root='datasets')
    transforms = build_transforms(cfg, is_train=False)
    test_set = CommDataset(dataset.query + dataset.gallery, transforms, relabel=False)
    data_loader, _ = build_reid_test_loader(
        test_set=test_set,
        test_batch_size=cfg.TEST.IMS_PER_BATCH,
        num_query=len(dataset.query),
        num_workers=cfg.DATALOADER.NUM_WORKERS,
    )

    total_batches = len(data_loader)
    if total_batches <= 0:
        raise RuntimeError("Test loader is empty")

    effective_warmup = min(warmup_batches, max(total_batches - 1, 0))
    measured_images = 0
    measured_batches = 0
    total_compute_time = 0.0

    with inference_context(model), torch.no_grad():
        for idx, inputs in enumerate(data_loader):
            if max_batches and idx >= max_batches:
                break

            if idx == effective_warmup:
                total_compute_time = 0.0
                measured_images = 0
                measured_batches = 0

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(inputs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            if idx >= effective_warmup:
                total_compute_time += elapsed
                measured_batches += 1
                measured_images += int(inputs["images"].shape[0])

    if hasattr(data_loader, "shutdown"):
        data_loader.shutdown()

    if measured_images == 0 or total_compute_time == 0:
        raise RuntimeError("No measured images collected for latency profiling")

    return {
        "num_query": len(dataset.query),
        "num_gallery": len(dataset.gallery),
        "num_images": measured_images,
        "measured_batches": measured_batches,
        "latency_ms_per_image": total_compute_time / measured_images * 1000.0,
        "throughput_img_per_s": measured_images / total_compute_time,
        "seconds_per_batch": total_compute_time / measured_batches,
    }


def measure_dummy_latency(cfg, model, warmup_iters, measure_iters):
    dummy_inputs = make_dummy_inputs(cfg, model, cfg.TEST.IMS_PER_BATCH)
    total_compute_time = 0.0

    with inference_context(model), torch.no_grad():
        for _ in range(warmup_iters):
            _ = model(dummy_inputs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        for _ in range(measure_iters):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(dummy_inputs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start

    measured_images = cfg.TEST.IMS_PER_BATCH * measure_iters
    return {
        "dummy_latency_iters": measure_iters,
        "dummy_latency_warmup": warmup_iters,
        "num_images": measured_images,
        "latency_ms_per_image": total_compute_time / measured_images * 1000.0,
        "throughput_img_per_s": measured_images / total_compute_time,
        "seconds_per_batch": total_compute_time / measure_iters,
    }


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = build_cfg(args)

    torch.backends.cudnn.benchmark = cfg.CUDNN_BENCHMARK

    model = build_model(cfg)
    model.eval()

    total_params, trainable_params = compute_params(model)
    gflops, unsupported_ops_dict, flop_method, flop_error = compute_gflops(
        cfg,
        model,
        args.flops_batch_size,
        args.flops_mode,
    )

    result = {
        "config_file": args.config_file,
        "model_meta_arch": cfg.MODEL.META_ARCHITECTURE,
        "test_batch_size": cfg.TEST.IMS_PER_BATCH,
        "num_inference_steps": int(cfg.TEST.SDMODEL.NUM_INFERENCE_STEPS),
        "total_params": total_params,
        "trainable_params": trainable_params,
        "total_params_m": total_params / 1e6,
        "trainable_params_m": trainable_params / 1e6,
        "gflops_per_image": gflops,
        "flop_method": flop_method,
        "flops_error": flop_error,
        "unsupported_ops": unsupported_ops_dict,
    }

    if args.dataset_name:
        result.update(measure_latency(cfg, model, args.dataset_name, args.warmup_batches, args.max_batches))
    elif args.dummy_latency_iters > 0:
        result.update(measure_dummy_latency(cfg, model, args.dummy_latency_warmup, args.dummy_latency_iters))

    print(json.dumps(result, indent=2, sort_keys=True))

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()