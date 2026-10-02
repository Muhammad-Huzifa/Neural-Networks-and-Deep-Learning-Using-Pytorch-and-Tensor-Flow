# Deep Learning

A collection of 33 active notebooks covering neural-network fundamentals, PyTorch/TensorFlow, CNNs, sequence models, attention, Transformers, detection calculations, and autoencoders, with a [YOLO starter](projects/object_detection/yolo/README.md). Classical ML and Adult Income live in the separate [Machine Learning collection](https://github.com/Muhammad-Huzifa/machine-learning).

## Learning path

| Section | Lessons | Topics |
| --- | --- | --- |
| [Neural-network fundamentals](notebooks/01_neural_network_fundamentals/README.md) | 9 | Original ANN lessons and complete NumPy backpropagation |
| [PyTorch](notebooks/02_frameworks/pytorch/README.md) | 8 | Tensors, datasets, modules, training, evaluation, checkpoints |
| [Convolutional networks](notebooks/03_convolutional_networks/README.md) | 9 | Convolution, image classification, transfer examples, CPU CNN |
| [Hyperparameter experiments](notebooks/04_tuning/README.md) | 2 | Optuna and hidden-neuron comparisons |
| [Sequence models](notebooks/05_sequence_models/README.md) | 1 | RNN, LSTM, and GRU comparison |
| [Attention and Transformers](notebooks/06_attention/README.md) | 2 | Attention mechanics, causal masks, a small encoder |
| [Detection calculations](notebooks/07_object_detection/README.md) | 1 | IoU and class-aware NMS |
| [Representation learning](notebooks/08_representation_learning/README.md) | 1 | Autoencoder reconstruction |

For a laptop without a GPU, follow the [eight self-contained CPU lessons](notebooks/README.md). They use built-in digits, generated signals, or small arrays, and include explanations and practice. Original larger experiments retain their dataset and framework requirements. [Two incomplete historical drafts](archive/incomplete/README.md) are separate from the active lessons.

## Setup

Use Python 3.11 or 3.12:

```bash
git clone https://github.com/Muhammad-Huzifa/deep-learning.git
cd deep-learning
python -m venv .venv
```

| Terminal | Activate the environment |
| --- | --- |
| Windows Command Prompt | `.venv\Scripts\activate.bat` |
| Windows PowerShell | `.\.venv\Scripts\Activate.ps1` |
| Windows Git Bash | `source .venv/Scripts/activate` |
| Linux/macOS | `source .venv/bin/activate` |

For the new CPU learning path:

```bash
python -m pip install -r requirements/base.txt
python -m pip install 'torch>=2.4,<3' --index-url https://download.pytorch.org/whl/cpu
jupyter lab
```

The three NumPy-only additions (backpropagation, attention, IoU/NMS) do not need PyTorch. The five new PyTorch lessons explicitly use CPU and need neither torchvision nor TensorFlow. After dependency installation, these eight lessons require no dataset or pretrained-weight download.

Choose the environment for an original experiment:

| Material | Requirements |
| --- | --- |
| Original PyTorch experiments | `requirements/pytorch.txt` |
| TensorFlow experiments | `requirements/tensorflow.txt` in a separate environment |
| Mixed-framework examples | Both frameworks in an appropriate environment |
| YOLO starter | `requirements/detection.txt`; see its dataset/model arguments |

Select the environment's Python kernel, restart it, and run cells in order. Read [the dataset guide](docs/DATASETS.md) before an original experiment; some need external CSVs, class-folder images, or downloads. Larger models may need a Kaggle/Colab runtime.

## Structure

| Path | Purpose |
| --- | --- |
| `notebooks/01_*` through `notebooks/08_*` | Ordered foundations, models, and complete CPU lessons |
| `projects/object_detection/` | YOLO training/evaluation/prediction starter |
| `requirements/` | Base, validation, and separate framework environments |
| `data/` | Original CSV, input guides, and retained archive |
| `artifacts/` | Ignored local reports and checkpoints |
| `archive/incomplete/` | Historical drafts |
| `scripts/`, `docs/` | Notebook checks, execution, and provenance |

## Validation

After the CPU setup above:

```bash
python -m pip install -r requirements/validation.txt
python scripts/check_notebooks.py
python scripts/execute_lessons.py
```

The execution command runs the eight new lessons in fresh kernels without writing outputs into source notebooks. GitHub Actions performs this execution, checks all active notebook syntax, and checks YOLO CLI help. All 40 new lesson code cells passed local CPU execution, including training and checkpoint reload. The original 25 lessons and binary archive are preserved. Larger TensorFlow/GPU experiments and real detector inference were not rerun in this expansion. See [development notes](docs/DEVELOPMENT.md) and [source provenance](docs/SOURCE_MAP.md).

Muhammad Huzifa — [GitHub](https://github.com/Muhammad-Huzifa)
