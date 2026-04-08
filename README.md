<div align="center">

# SDReID

### SD-ReID: View-aware Stable Diffusion for Aerial-Ground Person Re-Identification

Yuhao Wang, Xiang Hu, Lixin Wang, Pingping Zhang, Huchuan Lu

[arXiv](https://arxiv.org/abs/2504.09549) | [PDF](https://arxiv.org/pdf/2504.09549) | [GitHub](https://github.com/924973292/SD-ReID)

</div>

## News

- 2026-04: The project repository is online.
- 2026-04: The paper is under review at IEEE Transactions on Image Processing (TIP).
- Code, pretrained models, and reproduction instructions will be released after acceptance.

## Overview

SDReID is a generative framework for aerial-ground person re-identification. This repository is the official project page for the public arXiv preprint and will host the codebase and supplementary resources after the paper is accepted.

## Abstract

Aerial-Ground Person Re-IDentification (AG-ReID) aims to retrieve specific persons across cameras with different viewpoints. Previous works focus on designing discriminative models to maintain the identity consistency despite drastic changes in camera viewpoints. The core idea behind these methods is quite natural, but designing a view-robust model is a very challenging task. Moreover, they overlook the contribution of view-specific features in enhancing the model's ability to represent persons. To address these issues, we propose a novel generative framework named SD-ReID for AG-ReID, which leverages generative models to mimic the feature distribution of different views while extracting robust identity representations. More specifically, we first train a ViT-based model to extract person representations along with controllable conditions, including identity and view conditions. We then fine-tune the Stable Diffusion (SD) model to enhance person representations guided by these controllable conditions. Furthermore, we introduce the View-Refined Decoder (VRD) to bridge the gap between instance-level and global-level features. Finally, both person representations and all-view features are employed to retrieve target persons. Extensive experiments on five AG-ReID benchmarks (i.e., CARGO, AG-ReIDv1, AG-ReIDv2, LAGPeR and G2APS-ReID) demonstrate the effectiveness of our proposed method.

## Paper Information

- Title: SD-ReID: View-aware Stable Diffusion for Aerial-Ground Person Re-Identification
- Authors: Yuhao Wang, Xiang Hu, Lixin Wang, Pingping Zhang, Huchuan Lu
- Venue status: under review at IEEE TIP
- Preprint: arXiv:2504.09549
- Subject: Computer Vision and Pattern Recognition (cs.CV)

## Framework

The framework figure and method illustration will be added with the public release of the code.

## Results

Quantitative benchmark tables, qualitative retrieval examples, and additional comparisons will be added together with the released implementation.

## Planned Release

The following materials will be added after acceptance:

- training and evaluation code
- pretrained checkpoints
- dataset preparation instructions
- experiment configuration files
- usage and reproduction guidelines

## Citation

```bibtex
@article{wang2025sdreid,
  title={SD-ReID: View-aware Stable Diffusion for Aerial-Ground Person Re-Identification},
  author={Wang, Yuhao and Hu, Xiang and Wang, Lixin and Zhang, Pingping and Lu, Huchuan},
  journal={arXiv preprint arXiv:2504.09549},
  year={2025}
}
```

## Acknowledgement

This repository page is currently based on the public arXiv preprint. More project details will be released together with the official implementation.