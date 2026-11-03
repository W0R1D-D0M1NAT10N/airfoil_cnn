# Learning experiments

These early Python, image-processing, and neural-network exercises are preserved separately from the aerodynamic implementations. Their contents are unchanged.

| File in `learning/` | Purpose |
| --- | --- |
| `simple.py` | Small fully connected network with a manual weight-update experiment |
| `ConvolutionNN.py` | MNIST digit-classification CNN exercise |
| `seeing_pictures_with_nn.py` | Digit prediction using a saved MNIST network and a local image |
| `pillow_test.py`, `pillow test.py` | Image inversion and pixel inspection; original duplicate names preserved |
| `Python_refresher.py` | Incomplete scratch fragment referencing external `model` and `x` variables |

Each file previously lived at the repository root with the same filename. No aerodynamic script imports these files. They use process-working-directory paths or absolute local paths, rather than paths relative to their source file.

To retain the original relative file behavior, invoke them from the repository root, for example `python experiments/learning/simple.py`. MNIST training and inference share `mnist_cnn.pth` in that working directory. These exercises may download data or write model files when executed; they are not project tests and were not executed for the documentation update.

Earlier **aerodynamic** variants (`airfoilNN.py`, `airfoilNN_linux.py`, and `airfoilNN8020.py`) remain at their original root paths. Their role and differences are documented in the [project guide](../docs/PROJECT_GUIDE.md).
