# Installation

Use the environment setup in [README.md](README.md#environment). In short:

```bash
conda create -n sdreid python=3.10 -y
conda activate sdreid
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

Install the PyTorch build that matches your CUDA driver if CUDA 11.8 is not suitable for your machine.
