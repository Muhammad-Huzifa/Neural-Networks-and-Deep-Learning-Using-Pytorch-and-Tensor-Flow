# Deep Learning

Neural-network fundamentals, PyTorch and TensorFlow experiments, CNN/image-classification lessons, tuning, and a YOLO starter. Classical ML and Adult Income have moved to the separate [Machine Learning collection](https://github.com/Muhammad-Huzifa/Machine_Learning).

## Start here

| Section | Contents |
| --- | --- |
| [Neural-network fundamentals](notebooks/01_neural_network_fundamentals/README.md) | 8 notebooks |
| [PyTorch](notebooks/02_frameworks/pytorch/README.md) | 7 notebooks |
| [CNNs and image classification](notebooks/03_convolutional_networks/README.md) | 8 notebooks |
| [Hyperparameter experiments](notebooks/04_tuning/README.md) | 2 notebooks |

The active collection contains 25 notebooks. The [YOLO starter](projects/object_detection/yolo/README.md) provides training, evaluation, and prediction commands. [Incomplete original drafts](archive/incomplete/README.md) are reference material and are not active lessons.

## Setup

Use Python 3.11 or 3.12:

```bash
git clone https://github.com/Muhammad-Huzifa/Neural-Networks-and-Deep-Learning-Using-Pytorch-and-Tensor-Flow.git
cd Neural-Networks-and-Deep-Learning-Using-Pytorch-and-Tensor-Flow
python -m venv .venv
```

| Terminal | Activate the environment |
| --- | --- |
| Windows Command Prompt | `.venv\Scripts\activate.bat` |
| Windows PowerShell | `.\.venv\Scripts\Activate.ps1` |
| Windows Git Bash | `source .venv/Scripts/activate` |
| Linux/macOS | `source .venv/bin/activate` |

Install the environment for the selected lesson:

```bash
python -m pip install -r requirements/pytorch.txt
jupyter lab
```

For TensorFlow, use `python -m pip install -r requirements/tensorflow.txt` in a separate environment. Mixed-framework examples need both. YOLO uses `requirements/detection.txt`; its README explains the model and dataset arguments.

## Structure

| Path | Purpose |
| --- | --- |
| `notebooks/` | Neural, framework, CNN, and tuning lessons |
| `projects/object_detection/` | YOLO training/evaluation/prediction starter |
| `requirements/` | Separate framework environments |
| `data/` | Source CSV, image-input guide, and original archive |
| `archive/incomplete/` | Clearly labeled historical drafts |
| `scripts/` | Offline notebook checks |
| `docs/` | Inputs, provenance, and development |

Read [the dataset guide](docs/DATASETS.md) before execution. Custom images and some CSVs are external; named datasets and pretrained weights may download when a lesson is run. For larger models, use an appropriate Kaggle/Colab runtime.

Active notebook syntax and YOLO CLI help were checked. Full framework installation, GPU training, dataset downloads, and real detector inference have not been reproduced in this organization pass. See [development notes](docs/DEVELOPMENT.md).

Muhammad Huzifa — [GitHub](https://github.com/Muhammad-Huzifa)
