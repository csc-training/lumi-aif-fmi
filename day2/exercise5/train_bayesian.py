import torch
from torch import nn
from torch.utils.data import DataLoader
from torch_blue import vi

from dataset import HousingDataset


class Model(vi.VIModule):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(
            nn.BatchNorm1d(8, track_running_stats=False),
            vi.VILinear(8, 16),
            nn.ReLU(),
            vi.VILinear(16, 16),
            nn.ReLU(),
            vi.VILinear(16, 1),
            nn.Flatten(),
        )
    def forward(self, x):
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
    # NOTE return log probs of weights needed by KL loss
    model.return_log_probs = True
    model.train()
    print(model)

    predictive_distribution = vi.distributions.MeanFieldNormal()
    loss_fn = vi.KullbackLeiblerLoss(predictive_distribution, dataset_size=len(ds))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        # for step, batch in enumerate(tqdm(dl)):
        for step, batch in enumerate(dl):
            x, y = batch
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            print(loss/(torch.sum(y)/batch_size))
        print(loss/(torch.sum(y)/batch_size))

if __name__ == "__main__":
    main()
