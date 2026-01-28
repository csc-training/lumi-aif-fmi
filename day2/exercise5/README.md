# Exercise 5: Classification with Bayesian Neural Networks

## Dataset

We are switching from MNIST to [FashionMNIST](https://docs.pytorch.org/vision/0.24/generated/torchvision.datasets.FashionMNIST.html) for this task.
Similarly to MNIST FashionMNIST are low resolution images, but instead of digits they depict products from the fashion industry.

## `torch_blue`

`torch_blue` is a bayesian neural networks framework that sits on top of PyTorch and tries to keep its interface as similar as possible.
It supports BNNs via variational inference.

### `VILinear`

We will return to using `Linear` layers for this exercise.
Replace the ones from `torch.nn` from a deterministic model with `torch_blue.vi.VILinear` ones.
The constructor and forward pass have the same signature, but the output is different. How?

### Kullback-Leibler Loss

Since we are optimizing distributions, we use a `torch_blue.vi.KullbackLeiblerLoss` as our loss function.
Its constructor needs a `predictive_distribution` and the number of samples in the dataset.
Since we are doing classification, a `torch_blue.vi.Categorical` should be suitable.
