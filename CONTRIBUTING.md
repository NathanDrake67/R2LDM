# Contributing

Thank you for helping improve R2LDM.

## Before opening a change

1. Search existing issues and keep each change focused on one problem.
2. Do not commit datasets, checkpoints, generated point clouds, credentials, machine-specific paths, or compiled CUDA artifacts.
3. Preserve copyright and attribution notices in code derived from other projects.
4. Describe the dataset split, checkpoint, hardware, CUDA/PyTorch versions, and random seed for changes that affect numerical results.

## Validation

At minimum, compile all Python files and run the smallest relevant GPU smoke test. For training or inference changes, include the exact command and enough log output to show that the forward pass, loss, checkpoint, or generated point cloud is valid.

## Pull requests

Explain what changed, why it is needed, how it was tested, and whether it changes published behavior or checkpoint compatibility. Scientific changes should include before/after metrics or a clearly labeled ablation.
