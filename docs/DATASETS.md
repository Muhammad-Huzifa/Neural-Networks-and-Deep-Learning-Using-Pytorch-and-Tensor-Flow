# Dataset and environment guide

Some notebooks use synthetic arrays or scikit-learn built-in datasets. TensorFlow MNIST and PyTorch/CIFAR examples download their public datasets when run. Downloads, pretrained framework weights, and GPU training require an appropriate local or hosted environment.

| Input | Location | Used by |
| --- | --- | --- |
| Original diabetes CSV | `data/tabular/diabetes.csv` | TensorFlow diabetes classifier |
| Coffee-roasting CSV with the original expected schema | `data/tabular/coffee_roasting.csv` | NumPy/TensorFlow forward-pass example |
| Fashion MNIST training CSV | `data/tabular/fashion-mnist_train.csv` | Fashion MNIST and tuning notebooks |
| Custom class-folder images | `data/images/image_classification/` | ANN/CNN and ResNet50 examples |
| Penguin/turtle class folders | `data/images/penguin_turtle/train/`, `test/` | Penguin/turtle classifiers |
| Dog/cat class folders | `data/images/dogs_vs_cats/train/`, `test/` | TensorFlow VGG16 example |

The original source links in `SOURCE_MAP.md` show expected column usage and custom image conventions. Not all datasets are included, and framework training has not been repeated during restructuring.

MNIST, CIFAR, Fashion MNIST, downloaded breast-cancer data, and custom image data have their own upstream terms and citations. Record the exact source used for a run. Install the environment listed by its notebook section; mixed PyTorch/TensorFlow examples need both sets of dependencies.

For a laptop without a GPU, begin with the tensor and neural-network fundamentals. Use Kaggle or Colab for larger CNN experiments after cloning the repository into the notebook environment. Mounted personal-drive paths and credential-upload cells have been removed from active notebooks.
