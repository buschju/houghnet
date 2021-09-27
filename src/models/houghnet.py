import numpy
import torch
from torch import Tensor
from torch.nn import Parameter, Module
from torch.nn.init import uniform_
from torch.optim import Adam
from torch.utils.data import DataLoader


class HoughNet(Module):
    def __init__(self,
                 num_dims: int,
                 num_clusters: int,
                 lambda_activation: float,
                 initial_data_mean: numpy.ndarray,
                 ):
        super().__init__()

        self.lambda_activation = lambda_activation
        self.initial_data_mean = torch.FloatTensor(initial_data_mean)

        self.weight = Parameter(torch.empty(num_dims, num_clusters, dtype=torch.float32),
                                requires_grad=True,
                                )
        self.bias = Parameter(torch.empty(num_clusters, 1, dtype=torch.float32),
                              requires_grad=True,
                              )

        self.reset_parameters()

    def fit(self,
            x: numpy.ndarray,
            batch_size: int,
            learning_rate: float,
            num_epochs: int,
            ) -> None:
        dataset = torch.utils.data.TensorDataset(torch.FloatTensor(x))

        if batch_size is None:
            batch_size = len(dataset)
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                drop_last=True,
                                shuffle=True,
                                num_workers=0,
                                pin_memory=True,
                                )

        optimizer = Adam(params=self.parameters(),
                         lr=learning_rate,
                         )

        for epoch in range(1, num_epochs + 1):
            for batch in dataloader:
                self.train()

                x_batch = batch[0].to(next(self.parameters()).device)

                activated_distances = self(x_batch)
                loss = self.get_product_loss(activated_distances)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def predict(self,
                x: numpy.ndarray,
                ) -> numpy.ndarray:
        self.eval()

        with torch.no_grad():
            x = torch.FloatTensor(x).to(next(self.parameters()).device)
            projection_distances = self.get_projection_distance(x)
            cluster_assignments = self.get_soft_assignments(projection_distances).argmax(dim=1).detach().cpu().numpy()

        return cluster_assignments

    def fit_predict(self,
                    x: numpy.ndarray,
                    **training_parameters
                    ) -> numpy.ndarray:
        self.fit(x=x,
                 **training_parameters
                 )
        return self.predict(x)

    def forward(self,
                x: Tensor,
                ):
        projection_distances = self.get_projection_distance(x)
        activated_distances = self.get_activated_distance(projection_distances)

        return activated_distances

    def get_projection_distance(self,
                                x: Tensor,
                                ) -> Tensor:
        weights_normalized = self.weight / self.weight.norm(dim=0, keepdim=True).detach()
        projection_distances = x @ weights_normalized - self.bias

        return projection_distances

    def get_soft_assignments(self,
                             distances: Tensor,
                             ) -> Tensor:
        normalizer = torch.std(distances, dim=0, keepdim=True) * self.lambda_activation
        distances = distances / normalizer
        soft_assignments = 1. / (1. + distances ** 2)

        return soft_assignments

    def get_activated_distance(self,
                               distances: Tensor,
                               ) -> Tensor:
        soft_assignments = self.get_soft_assignments(distances)
        activated_distances = 1. - soft_assignments

        return activated_distances

    @staticmethod
    def get_product_loss(activated_distances: Tensor,
                         ) -> Tensor:
        return activated_distances.prod(dim=1).mean()

    def reset_parameters(self):
        uniform_(self.weight,
                 a=0.,
                 b=1.,
                 )
        self.weight.data /= self.weight.data.norm(dim=0, keepdim=True)

        self.bias.data = self.initial_data_mean[None, :].to(self.weight.device) @ self.weight.data
