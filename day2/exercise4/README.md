# Exercise 4: Shape Graph Classification with Graph Neural Networks

## Data

The ModelNet dataset contains meshes of objects (e.g. chair).
The task is to classify each graph

## pytorch-geometric

PyTorch Geometric (PyG) is a graph neural network library on top of PyTorch.

### Data

A `torch_geometric.Data` instance represents a graph.
It contains multiple tensors, some of which are optional.
In this example you will need the `.pos`, `.edge_index`, and `.batch` attributes, containing the position (i.e. 3D vertex features), the vertices of the vertices contributing to an edge, and the index of the graph in the batch, that vertex belongs to.
All of these are packed into `torch.Tensor`s.

What shapes are these tensors?

### DataLoader

The graph `DataLoader` works slightly differently than the normal torch one: Instead of stacking tensors into a batch of tensors, it takes several graphs (`Data` objects).
Some operations (e.g. global max pooling) require information on which graph a verteix belongs to, which is stored in the `.batch` attribute.

### GCN

The graph operation we are mainly using is the `pytorch_geometric.nn.GCNConv` layer.
It is a simple graph convolution.
Its constructor takes the number of input and output features as first two arguments.
```python
conv = pytorch_geometric.nn.GCNConv(in_features, out_geatures, normalize:bool)
```
Its forward pass takes tensors of node feautures, and edges as input.

```python
conv(data.pos, data.edge_index)
```

### Global Pooling

Similarly to the convolutional models on images, we can contend with a variable size input tensor for a fully feed-forward classification sub-module.

Instead of a flatten operation, we can use a global pooling (`pytorch_geometric.nn.global_max_pool` or `pytorch_geometric.nn.global_average_pool`).
Its forward pass takes vertex features and batch information:
```python
m = pytorch_geometric.nn.global_max_pool(data.pos, data.batch)
```

Note that pooling operations might currently not be GPU accelerated.
You can transfer the tensors back to CPU to perform the operation.
Do not forget to return the result to the GPU, if you perform further operations there.
