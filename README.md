# 🧠 Neural Networks and Deep Learning using PyTorch & TensorFlow

A comprehensive collection of **Artificial Neural Network (ANN)** and **Convolutional Neural Network (CNN)** implementations — built from scratch and using frameworks like **PyTorch** and **TensorFlow**.  
This repository demonstrates how deep learning models are applied to **classification, regression, and image recognition** tasks using real-world datasets.

---

## 📘 Overview

This repository is designed for students, researchers, and developers aiming to **master core neural network concepts** and **apply deep learning models** to practical datasets.  
It includes:
- Fully working Jupyter notebooks
- Hands-on projects on ANN, CNN, and Transfer Learning
- Implementations using both **PyTorch** and **TensorFlow**

The collection progresses from **simple neural networks** to **state-of-the-art architectures** like **LeNet-5, ResNet50, and Transfer Learning** on modern datasets.

---

## 🧩 Topics Covered

| Category | Topics / Implementations |
|-----------|--------------------------|
| **Artificial Neural Networks (ANN)** | Regression, Classification, Optuna tuning, Real datasets |
| **Deep Neural Networks (DNN)** | Multilayer perceptrons, Backpropagation, Hyperparameter tuning |
| **Convolutional Neural Networks (CNN)** | CIFAR dataset, Custom datasets, Feature extraction |
| **Transfer Learning** | ResNet50, Feature Extraction, Fine-tuning on custom datasets |
| **PyTorch Experiments** | Dataset, DataLoader, nn.Module, forward/backward propagation |
| **TensorFlow Models** | LeNet-5, MNIST classification |
| **Optimization Techniques** | Adam, SGD, Optuna Hyperparameter Optimization |
| **Real-world Data Applications** | Breast Cancer dataset, Fashion MNIST, Regression datasets |

---

## 🧠 Key Features
- Implementation of **ANNs, CNNs, and DNNs** from scratch and using frameworks.  
- Experiments in **both PyTorch and TensorFlow**.  
- Real dataset projects such as **Breast Cancer**, **Fashion MNIST**, and **Custom Image Datasets**.  
- Demonstrations of **Transfer Learning** using **ResNet50**.  
- **Hyperparameter tuning** using **Optuna**.  
- Educational explanations and step-by-step implementations.

---

## 🏗️ Projects Structure

<details>
<summary>Click to expand</summary>
Neural-Networks-and-Deep-Learning-Using-Pytorch-and-Tensor-Flow/
│
├── ANN with Real DataSets/
│ ├── Artifitial Neural_Network.ipynb
│ ├── Regression_DataSet_Using_ANN.ipynb
│ ├── Simple_Neural_Networks_MINST.ipynb
│ ├── diabetes.csv
│
├── Deep Neural Networks/
│ ├── CNN/
│ │ ├── CIFAR Using CNN.ipynb
│ │ ├── CNN-AlexNet with Real Dataset.ipynb
│ │ ├── Convulutional Operationn on Matrix.ipynb
│ │ ├── Penguin_Vs_Turtle_Classifaction(CNN).ipynb
│ │ ├── Classification of Balls Cars and Cone using Resnet50.ipynb
│ │
│ ├── Deep_Nueral_Networks_SImple_DNNS.ipynb
│ ├── Forward_Backward_prop_Pytorch.ipynb
│ ├── Lenet_5 Using Tensor_flow.ipynb
│
├── Pytorch/
│ ├── (nn)Module in Pytorch.ipynb
│ ├── ANN_Breast_Cancer_DataSet_Using pytorch.ipynb
│ ├── Data loader and dataset.ipynb
│ ├── Dataset and Dataloader using Breast Cancer.ipynb
│ ├── Fasion MNIST dataset using pytorch.ipynb
│ ├── Hyperparameter Tuning using Optuna.ipynb
│ ├── Real DataSet nnmodule().ipynb
│ ├── Transfer Learning on Fashion Dataset using pytorch.ipynb
│ ├── Transfer Learning_feature_Extraction .ipynb
│
├── requirements.txt
└── README.md
</details>

---

## ⚙️ Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/Muhammad-Huzifa/Neural-Networks-and-Deep-Learning-Using-Pytorch-and-Tensor-Flow.git
cd Neural-Networks-and-Deep-Learning-Using-Pytorch-and-Tensor-Flow
python -m venv venv
venv\Scripts\activate   # For Windows
pip install -r requirements.txt
```

🧪 Usage

Open any notebook (.ipynb) in Jupyter Notebook or VS Code.

Follow the notebook instructions for data preprocessing, model creation, and training.

Modify network parameters or architectures to experiment with learning behavior.

Evaluate models and visualize results interactively.

| Project                              | Description                                                                |
| ------------------------------------ | -------------------------------------------------------------------------- |
| **ANN with Real Datasets**           | Implements regression and classification using artificial neural networks. |
| **Penguin vs Turtle Classification** | CNN-based image classification of custom animal dataset.                   |
| **CIFAR CNN**                        | Training a CNN from scratch on CIFAR-10 dataset.                           |
| **ResNet50 Transfer Learning**       | Classification of multiple object categories using ResNet50.               |
| **Breast Cancer Detection**          | Binary classification using ANN in PyTorch.                                |
| **LeNet-5 in TensorFlow**            | Classic CNN architecture trained on MNIST.                                 |
| **Optuna Hyperparameter Tuning**     | Automated search for optimal ANN configurations.                           |


Tools & Libraries

Programming Language: Python 3.8+

Frameworks: PyTorch, TensorFlow, Keras

Libraries: NumPy, Pandas, Matplotlib, Seaborn, OpenCV, Optuna, scikit-learn

Environment: Jupyter Notebook / Google Colab


💡 Learning Outcomes

By exploring this repository, you will:

Understand the mathematical and computational foundations of neural networks.

Learn how to design, train, and evaluate models using PyTorch and TensorFlow.

Explore Transfer Learning and feature extraction techniques.

Gain practical skills in hyperparameter optimization and data handling.


💫 Future Work

Add Transformer-based architectures (ViT, BERT-like for vision)

Implement GANs (Generative Adversarial Networks)

Extend to object detection (YOLO, Faster-RCNN)

Include more real-world datasets for medical and industrial use cases


👨‍💻 Author

Muhammad Huzifa
🔗 GitHub Profile

💬 Passionate about Deep Learning, AI Systems, and Computer Vision.

🙏 Acknowledgements

This repository is a compilation of projects created during the learning and exploration of Deep Learning fundamentals and modern architectures using PyTorch and TensorFlow.


---
requirements.txt

```text
torch
torchvision
tensorflow
numpy
matplotlib
pandas
opencv-python
optuna
scikit-learn
```
