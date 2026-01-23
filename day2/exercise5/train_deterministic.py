from torch.utils.data import DataLoader
import torch
from torch import nn
from tqdm import tqdm

from dataset import HousingDataset


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        n1 = nn.BatchNorm1d(8, track_running_stats=False)
        l1 = nn.Linear(8, 16)
        a1 = nn.ReLU()
        l2 = nn.Linear(16, 16)
        a2 = nn.ReLU()
        l3 = nn.Linear(16, 1)
        self.f = nn.Sequential(n1, l1, a1, l2, a2, l3, nn.Flatten())

    def forward(self, x):
        # TODO then add an embedding module to your network (see torch.nn.Embedding)
        # TODO concatenate or add the embedding to the input vector
        # TODO does including the ocean_proximity information improve performance?
        return self.f(x)


def main():
    epochs = 2
    learning_rate = 1e-4
    batch_size = 64
    device = torch.device("cuda:0")
    ds = HousingDataset()
    # TODO split into training and test data
    dl = DataLoader(ds, batch_size=batch_size)
    model = Model().to(device)
    loss_fn = nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        # for step, batch in enumerate(tqdm(dl)):
        for step, batch in enumerate(dl):
            x, y = batch
            x = x.to(device)
            y = y.to(device)

            optim.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optim.step()
            print(loss/(torch.sum(y)/batch_size))
        print(loss/(torch.sum(y)/batch_size))

if __name__ == "__main__":
    main()
