# Computer Vision Foundations

Cleaned portfolio versions of MSc computer vision and deep learning coursework notebooks, covering classical image classification, optimization, neural networks, convolutional networks, and vision transformers.

## Notebooks

| Notebook | Topic | Main techniques | What it demonstrates |
| --- | --- | --- | --- |
| [01_knn_cifar10.ipynb](notebooks/01_knn_cifar10.ipynb) | CIFAR-10 image classification with k-nearest neighbours | L1/L2 distances, vectorization, cross-validation | Classical image classification from raw pixels and nearest-neighbour retrieval analysis |
| [02_svm_cifar10.ipynb](notebooks/02_svm_cifar10.ipynb) | Linear SVM for CIFAR-10 | Multiclass hinge loss, numerical and analytic gradients, stochastic gradient descent | Implementing and validating a linear classifier for images |
| [03_softmax_cifar10.ipynb](notebooks/03_softmax_cifar10.ipynb) | Softmax classifier for CIFAR-10 | Cross-entropy loss, vectorized gradients, stochastic gradient descent | Multiclass probabilistic classification with a linear model |
| [04_numpy_two_layer_net.ipynb](notebooks/04_numpy_two_layer_net.ipynb) | Two-layer neural network from scratch | Affine layers, ReLU, softmax loss, backpropagation, gradient checking | Building neural-network training components directly in NumPy |
| [05_pytorch_two_layer_net.ipynb](notebooks/05_pytorch_two_layer_net.ipynb) | Two-layer neural networks with PyTorch | Autograd, `nn.Module`, `nn.Sequential`, optimizers | PyTorch training pipelines for image classification |
| [06_pytorch_convnets.ipynb](notebooks/06_pytorch_convnets.ipynb) | Convolutional neural networks | Conv layers, pooling, batch normalization, ResNet-18 transfer learning | CNN model construction and training workflows in PyTorch |
| [07_vision_transformer.ipynb](notebooks/07_vision_transformer.ipynb) | Vision Transformer | Patch embeddings, positional embeddings, self-attention, multi-head attention, transformer encoders | Implementing and experimenting with transformer-based vision models |

## Relevance

This local project demonstrates practical foundations in image classification, computer vision, deep learning, PyTorch, CNNs, and transformer-based vision models. The notebooks cover both classical machine-learning baselines and modern neural architectures, moving from raw-pixel classifiers to convolutional and attention-based models.

## Academic Context

These notebooks are cleaned portfolio versions of coursework and implementation practice. They are included to demonstrate technical learning and implementation skills; they should not be presented as peer-reviewed research or production-level software.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

Open the notebooks from the `notebooks/` directory after Jupyter starts.

## Dataset Note

Datasets such as CIFAR-10 should be downloaded through standard library utilities where applicable and should not be stored in this project. Local dataset folders such as `data/` and `datasets/` are ignored by `.gitignore`.

## Execution Notes

The NumPy/classical notebooks expect the original local `engine` helper package from the coursework environment. That helper package is not included here, so those notebooks need that code restored before they can be rerun end to end.
