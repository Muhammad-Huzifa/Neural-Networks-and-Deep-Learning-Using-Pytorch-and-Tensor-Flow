# Deep Learning notebook catalog

Run each notebook from a fresh kernel in the environment it needs. The new self-contained lessons use generated, handcrafted, or built-in data; older lessons retain their original input requirements.

## Self-contained CPU learning path

| Order | Lesson | Environment | Input |
| --- | --- | --- | --- |
| 1 | [Backpropagation from scratch with a gradient check](01_neural_network_fundamentals/09_backpropagation_numpy.ipynb) | Base | Four generated XOR examples |
| 2 | [Training, evaluation, and portable checkpoints](02_frameworks/pytorch/07_training_evaluation_and_checkpoints.ipynb) | CPU PyTorch | Built-in scikit-learn 8x8 digits |
| 3 | [A small convolutional classifier on CPU](03_convolutional_networks/08_cnn_digits_cpu.ipynb) | CPU PyTorch | Built-in scikit-learn 8x8 digits |
| 4 | [RNN, LSTM, and GRU sequence classification](05_sequence_models/01_rnn_lstm_gru_sequence_classification.ipynb) | CPU PyTorch | Generated independent increasing/decreasing signals |
| 5 | [Scaled dot-product attention and causal masking](06_attention/01_scaled_dot_product_attention_numpy.ipynb) | Base | Small generated query, key, and value matrices |
| 6 | [A small Transformer encoder on CPU](06_attention/02_transformer_encoder_cpu.ipynb) | CPU PyTorch | Generated independent increasing/decreasing signals |
| 7 | [Intersection over union and class-aware non-maximum suppression](07_object_detection/01_iou_and_non_max_suppression_numpy.ipynb) | Base | Small handcrafted bounding boxes |
| 8 | [A small autoencoder for digit reconstruction](08_representation_learning/01_autoencoder_digits_cpu.ipynb) | CPU PyTorch | Built-in scikit-learn 8x8 digits |

## Full collection

### Neural-network fundamentals

| Notebook | Environment / input |
| --- | --- |
| [Neurons and layers — TensorFlow](01_neural_network_fundamentals/01_neurons_and_layers_tf.ipynb) | Original lesson; see section environment and dataset guide |
| [Forward pass — NumPy and TensorFlow](01_neural_network_fundamentals/02_numpy_tensorflow_forward_pass.ipynb) | Original lesson; see section environment and dataset guide |
| [Dense networks — TensorFlow](01_neural_network_fundamentals/03_dense_network_tf.ipynb) | Original lesson; see section environment and dataset guide |
| [ANN classification — TensorFlow](01_neural_network_fundamentals/04_ann_classification_tf.ipynb) | Original lesson; see section environment and dataset guide |
| [MNIST ANN introduction — TensorFlow](01_neural_network_fundamentals/05_mnist_ann_intro_tf.ipynb) | Original lesson; see section environment and dataset guide |
| [MNIST ANN — TensorFlow](01_neural_network_fundamentals/06_mnist_ann_tf.ipynb) | Original lesson; see section environment and dataset guide |
| [Diabetes classification example — TensorFlow](01_neural_network_fundamentals/07_diabetes_classification_tf.ipynb) | Original lesson; see section environment and dataset guide |
| [Diabetes classification variant — TensorFlow](01_neural_network_fundamentals/08_diabetes_classification_variant_tf.ipynb) | Original lesson; see section environment and dataset guide |
| [Backpropagation from scratch with a gradient check](01_neural_network_fundamentals/09_backpropagation_numpy.ipynb) | Base; Four generated XOR examples |

### PyTorch

| Notebook | Environment / input |
| --- | --- |
| [Tensor operations](02_frameworks/pytorch/00_tensor_operations.ipynb) | Original lesson; see section environment and dataset guide |
| [The nn.Module interface](02_frameworks/pytorch/01_nn_module.ipynb) | Original lesson; see section environment and dataset guide |
| [Forward and backward passes](02_frameworks/pytorch/02_forward_backward_pass.ipynb) | Original lesson; see section environment and dataset guide |
| [Dataset and DataLoader](02_frameworks/pytorch/03_dataset_and_dataloader.ipynb) | Original lesson; see section environment and dataset guide |
| [Breast-cancer example — DataLoader](02_frameworks/pytorch/04_breast_cancer_dataloader.ipynb) | Original lesson; see section environment and dataset guide |
| [Breast-cancer example — ANN](02_frameworks/pytorch/05_breast_cancer_ann.ipynb) | Original lesson; see section environment and dataset guide |
| [Tabular classification with nn.Module](02_frameworks/pytorch/06_nn_module_tabular.ipynb) | Original lesson; see section environment and dataset guide |
| [Training, evaluation, and portable checkpoints](02_frameworks/pytorch/07_training_evaluation_and_checkpoints.ipynb) | CPU PyTorch; Built-in scikit-learn 8x8 digits |

### Convolutional networks

| Notebook | Environment / input |
| --- | --- |
| [Convolution on matrices](03_convolutional_networks/00_convolution_on_matrices.ipynb) | Original lesson; see section environment and dataset guide |
| [CIFAR CNN — PyTorch and TensorFlow](03_convolutional_networks/01_cifar_cnn_pytorch_tensorflow.ipynb) | Original lesson; see section environment and dataset guide |
| [LeNet-5 — TensorFlow](03_convolutional_networks/02_lenet5_tf.ipynb) | Original lesson; see section environment and dataset guide |
| [Fashion MNIST — PyTorch](03_convolutional_networks/03_fashion_mnist_pytorch.ipynb) | Original lesson; see section environment and dataset guide |
| [ResNet50 image classification — PyTorch](03_convolutional_networks/04_resnet50_classification_pytorch.ipynb) | Original lesson; see section environment and dataset guide |
| [Penguin/turtle classifier — TensorFlow](03_convolutional_networks/05_penguin_turtle_basic_tf.ipynb) | Original lesson; see section environment and dataset guide |
| [Penguin/turtle extended classifier — TensorFlow](03_convolutional_networks/06_penguin_turtle_extended_tf.ipynb) | Original lesson; see section environment and dataset guide |
| [VGG16 feature extraction — TensorFlow](03_convolutional_networks/07_vgg16_feature_extraction_tf.ipynb) | Original lesson; see section environment and dataset guide |
| [A small convolutional classifier on CPU](03_convolutional_networks/08_cnn_digits_cpu.ipynb) | CPU PyTorch; Built-in scikit-learn 8x8 digits |

### Hyperparameter experiments

| Notebook | Environment / input |
| --- | --- |
| [Hyperparameter tuning with Optuna](04_tuning/01_optuna_pytorch.ipynb) | Original lesson; see section environment and dataset guide |
| [Hidden-neuron count comparison](04_tuning/02_neuron_count_comparison_pytorch.ipynb) | Original lesson; see section environment and dataset guide |

### Sequence models

| Notebook | Environment / input |
| --- | --- |
| [RNN, LSTM, and GRU sequence classification](05_sequence_models/01_rnn_lstm_gru_sequence_classification.ipynb) | CPU PyTorch; Generated independent increasing/decreasing signals |

### Attention and Transformers

| Notebook | Environment / input |
| --- | --- |
| [Scaled dot-product attention and causal masking](06_attention/01_scaled_dot_product_attention_numpy.ipynb) | Base; Small generated query, key, and value matrices |
| [A small Transformer encoder on CPU](06_attention/02_transformer_encoder_cpu.ipynb) | CPU PyTorch; Generated independent increasing/decreasing signals |

### Object-detection calculations

| Notebook | Environment / input |
| --- | --- |
| [Intersection over union and class-aware non-maximum suppression](07_object_detection/01_iou_and_non_max_suppression_numpy.ipynb) | Base; Small handcrafted bounding boxes |

### Representation learning

| Notebook | Environment / input |
| --- | --- |
| [A small autoencoder for digit reconstruction](08_representation_learning/01_autoencoder_digits_cpu.ipynb) | CPU PyTorch; Built-in scikit-learn 8x8 digits |

[Dataset guide](../docs/DATASETS.md) · [Development and validation](../docs/DEVELOPMENT.md)
