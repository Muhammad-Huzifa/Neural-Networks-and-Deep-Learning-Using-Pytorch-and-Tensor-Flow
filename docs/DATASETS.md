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

## New self-contained lessons

These inputs are generated in the notebook or bundled with scikit-learn. No external file, dataset download, credentials, or GPU is required after installing dependencies.

| Lesson | Input |
| --- | --- |
| [Backpropagation from scratch with a gradient check](../notebooks/01_neural_network_fundamentals/09_backpropagation_numpy.ipynb) | Four generated XOR examples |
| [Training, evaluation, and portable checkpoints](../notebooks/02_frameworks/pytorch/07_training_evaluation_and_checkpoints.ipynb) | Built-in scikit-learn 8x8 digits |
| [A small convolutional classifier on CPU](../notebooks/03_convolutional_networks/08_cnn_digits_cpu.ipynb) | Built-in scikit-learn 8x8 digits |
| [RNN, LSTM, and GRU sequence classification](../notebooks/05_sequence_models/01_rnn_lstm_gru_sequence_classification.ipynb) | Generated independent increasing/decreasing signals |
| [Scaled dot-product attention and causal masking](../notebooks/06_attention/01_scaled_dot_product_attention_numpy.ipynb) | Small generated query, key, and value matrices |
| [A small Transformer encoder on CPU](../notebooks/06_attention/02_transformer_encoder_cpu.ipynb) | Generated independent increasing/decreasing signals |
| [Intersection over union and class-aware non-maximum suppression](../notebooks/07_object_detection/01_iou_and_non_max_suppression_numpy.ipynb) | Small handcrafted bounding boxes |
| [A small autoencoder for digit reconstruction](../notebooks/08_representation_learning/01_autoencoder_digits_cpu.ipynb) | Built-in scikit-learn 8x8 digits |

Built-in digits here are 8×8 images, not MNIST. Synthetic/handcrafted examples are teaching data; their scores do not validate a production model. Each supervised lesson keeps test data out of fitting and model/epoch selection. Clustering is explicitly exploratory.
