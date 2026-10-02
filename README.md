# Machine Learning and Deep Learning

A learning collection covering classical machine learning, neural-network fundamentals, PyTorch and TensorFlow experiments, image classification, tuning, and applied projects. The material is organized by topic rather than by its original repository.

## Start here

| Section | Material |
| --- | --- |
| [Machine learning](notebooks/01_machine_learning/README.md) | Linear, polynomial, logistic regression, and SVR |
| [Neural network fundamentals](notebooks/02_neural_network_fundamentals/README.md) | Neurons, forward passes, dense classifiers, and MNIST |
| [PyTorch](notebooks/03_frameworks/pytorch/README.md) | Tensors, autograd, nn.Module, datasets, and DataLoader |
| [Deep learning](notebooks/04_deep_learning/README.md) | CNNs, CIFAR, LeNet, ResNet50, and VGG16 |
| [Tuning](notebooks/05_tuning/README.md) | Optuna and neuron-count experiments |
| [Adult Income project](projects/adult_income/README.md) | Train, evaluate, save, and serve a tabular model |
| [Object detection](projects/object_detection/README.md) | Generic YOLO training and inference starter |

The notebooks are original learning experiments with descriptive names, cleared outputs, and portable dataset paths. Some require external CSVs, custom images, or framework downloads. They are not all independently reproduced benchmark implementations; read [the input guide](docs/DATASETS.md) before running them. Incomplete experiments are kept in [the reference archive](archive/incomplete/README.md).

## Setup

Use Python 3.11 or 3.12. For basic ML:

```bash
git clone https://github.com/Muhammad-Huzifa/Neural-Networks-and-Deep-Learning-Using-Pytorch-and-Tensor-Flow.git
cd Neural-Networks-and-Deep-Learning-Using-Pytorch-and-Tensor-Flow
python -m venv .venv
```

| Terminal | Activation command |
| --- | --- |
| Windows Command Prompt | `.venv\Scripts\activate.bat` |
| Windows PowerShell | `.\.venv\Scripts\Activate.ps1` |
| Windows Git Bash | `source .venv/Scripts/activate` |
| Linux/macOS | `source .venv/bin/activate` |

```bash
python -m pip install -r requirements/base.txt
jupyter lab
```

Use separate environments for framework experiments:

```bash
python -m pip install -r requirements/pytorch.txt
```

or:

```bash
python -m pip install -r requirements/tensorflow.txt
```

Both files include the basic notebook tools. Install both only for mixed-framework examples. Full framework installations and GPU training depend on the platform and have not been verified by this migration.

## Structure and resources

| Path | Purpose |
| --- | --- |
| `notebooks/` | Topic-based lessons and section indexes |
| `projects/` | Independent projects with their own environments and commands |
| `requirements/` | Base, PyTorch, TensorFlow, and detection environments |
| `data/` | Bundled source CSVs, original archive, and external-data instructions |
| `artifacts/` | Ignored local model outputs |
| `docs/` | Dataset guide, development notes, and migration source map |
| `archive/incomplete/` | Explicitly unfinished original drafts |

The collection keeps useful variants and removes identical source-cell duplicates. All original sources are traced in [SOURCE_MAP.md](docs/SOURCE_MAP.md). The suggested final repository name is `Machine-Learning-and-Deep-Learning`; clone commands currently use the existing GitHub name.

## Notebook index

| Notebook |
| --- |
| [Simple linear regression](notebooks/01_machine_learning/01_simple_linear_regression.ipynb) |
| [Multiple linear regression](notebooks/01_machine_learning/02_multiple_linear_regression.ipynb) |
| [Polynomial regression](notebooks/01_machine_learning/03_polynomial_regression.ipynb) |
| [Logistic regression numpy](notebooks/01_machine_learning/04_logistic_regression_numpy.ipynb) |
| [Logistic regression patient records](notebooks/01_machine_learning/05_logistic_regression_patient_records.ipynb) |
| [Logistic regression titanic](notebooks/01_machine_learning/06_logistic_regression_titanic.ipynb) |
| [Support vector regression](notebooks/01_machine_learning/07_support_vector_regression.ipynb) |
| [Neurons and layers tf](notebooks/02_neural_network_fundamentals/01_neurons_and_layers_tf.ipynb) |
| [Numpy tensorflow forward pass](notebooks/02_neural_network_fundamentals/02_numpy_tensorflow_forward_pass.ipynb) |
| [Dense network tf](notebooks/02_neural_network_fundamentals/03_dense_network_tf.ipynb) |
| [Ann classification tf](notebooks/02_neural_network_fundamentals/04_ann_classification_tf.ipynb) |
| [Mnist ann intro tf](notebooks/02_neural_network_fundamentals/05_mnist_ann_intro_tf.ipynb) |
| [Mnist ann tf](notebooks/02_neural_network_fundamentals/06_mnist_ann_tf.ipynb) |
| [Diabetes classification tf](notebooks/02_neural_network_fundamentals/07_diabetes_classification_tf.ipynb) |
| [Diabetes classification variant tf](notebooks/02_neural_network_fundamentals/08_diabetes_classification_variant_tf.ipynb) |
| [Tensor operations](notebooks/03_frameworks/pytorch/00_tensor_operations.ipynb) |
| [Nn module](notebooks/03_frameworks/pytorch/01_nn_module.ipynb) |
| [Forward backward pass](notebooks/03_frameworks/pytorch/02_forward_backward_pass.ipynb) |
| [Dataset and dataloader](notebooks/03_frameworks/pytorch/03_dataset_and_dataloader.ipynb) |
| [Breast cancer dataloader](notebooks/03_frameworks/pytorch/04_breast_cancer_dataloader.ipynb) |
| [Breast cancer ann](notebooks/03_frameworks/pytorch/05_breast_cancer_ann.ipynb) |
| [Nn module tabular](notebooks/03_frameworks/pytorch/06_nn_module_tabular.ipynb) |
| [Convolution on matrices](notebooks/04_deep_learning/00_convolution_on_matrices.ipynb) |
| [Cifar cnn pytorch tensorflow](notebooks/04_deep_learning/01_cifar_cnn_pytorch_tensorflow.ipynb) |
| [Lenet5 tf](notebooks/04_deep_learning/02_lenet5_tf.ipynb) |
| [Fashion mnist pytorch](notebooks/04_deep_learning/03_fashion_mnist_pytorch.ipynb) |
| [Resnet50 classification pytorch](notebooks/04_deep_learning/04_resnet50_classification_pytorch.ipynb) |
| [Penguin turtle basic tf](notebooks/04_deep_learning/05_penguin_turtle_basic_tf.ipynb) |
| [Penguin turtle extended tf](notebooks/04_deep_learning/06_penguin_turtle_extended_tf.ipynb) |
| [Vgg16 feature extraction tf](notebooks/04_deep_learning/07_vgg16_feature_extraction_tf.ipynb) |
| [Optuna pytorch](notebooks/05_tuning/01_optuna_pytorch.ipynb) |
| [Neuron count comparison pytorch](notebooks/05_tuning/02_neuron_count_comparison_pytorch.ipynb) |

## Author

Muhammad Huzifa — [GitHub](https://github.com/Muhammad-Huzifa)
