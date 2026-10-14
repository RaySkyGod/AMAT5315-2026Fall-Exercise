# Week 6: Energy-Based Models

AMAT5315 (Fall 2026) week 6 exercises: dataset inspection (MNIST/CIFAR-10),
a restricted Boltzmann machine for handwritten digits in JAX, Hopfield
associative memory capacity, and modern Hopfield retrieval as attention.

Environment (uv, CPU baseline):

```bash
uv sync
uv run python scripts/<name>.py
uv run pytest tests/
```

## Dataset provenance

Downloaded with the commands below (URLs appended by `curl --write-out`),
stored under `inputs/` (not in git):

https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz
https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz
https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz
https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz
