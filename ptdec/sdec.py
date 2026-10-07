import torch
import torch.nn as nn
import torch
import numpy as np
from sklearn.cluster import KMeans

def compute_initial_centroids(x, num_classes):
  # Initializating centroids using KMeans
  kmeans = KMeans(n_clusters=num_classes, n_init=20, random_state=25)

  kmeans.fit(x.detach().numpy())

  cluster_centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float, requires_grad=True)
  return cluster_centers

class SDEC(nn.Module):
    def __init__(self, encoder, init_centroids, alpha=1.0):
        super().__init__()

        self.encoder = encoder
        self.alpha = alpha

        self.centroids = nn.Parameter(
            init_centroids.clone()
        )

    def soft_assignments(self, z):
        factor = -(self.alpha + 1) / 2

        # Vectorized distance calculation
        dist2 = torch.sum((z.unsqueeze(1) - self.centroids) ** 2, dim=2)

        q = (1 + dist2 / self.alpha).pow(factor)

        q = q / q.sum(dim=1, keepdim=True)

        return q

    def target_distribution(self, q):
        fj = q.sum(axis=0) # shape 10

        numerator = (q ** 2) / fj # shape 70000, 10

        denom = numerator.sum(dim=1, keepdim=True)
        p = numerator / denom
        return p

    def compute_pairwise_constraints(self, Y: torch.Tensor) -> torch.Tensor:
        # Ensure Y is a column vector
        Y = Y.view(-1, 1)  # Shape (n, 1)

        # Compare each pair: same -> 1, different -> -1
        same = (Y == Y.T).float()  # must-link: 1
        different = (Y != Y.T).float()  # cannot-link: 1

        A = same - different  # must-link = 1, cannot-link = -1

        # Set diagonal to 0 (no self constraint)
        A.fill_diagonal_(0)

        return A

    def supervised_loss_all_pairs(self, Z, A, lambd, margin=2.0):
        diff = Z.unsqueeze(0) - Z.unsqueeze(1)
        dist = torch.norm(diff, dim=-1)

        must = (A == 1)
        cannot = (A == -1)

        ml_loss = (dist[must] ** 2).mean() # must link losses

        cl_loss = torch.relu(margin - dist[cannot]).pow(2).mean() # cannot link losses

        return lambd * (ml_loss + cl_loss)

    def supervised_loss(
        self,
        Z: torch.Tensor,
        Y: torch.Tensor,
        lambd: float,
        margin: float = 2.0,
        num_pairs: int = 50000000,
    ):
        N = Z.size(0)
        device = Z.device

        # Sample random pairs
        i = torch.randint(0, N, (num_pairs,), device=device)
        j = torch.randint(0, N, (num_pairs,), device=device)

        # Ignore self-pairs
        mask = i != j
        i = i[mask]
        j = j[mask]

        zi = Z[i]
        zj = Z[j]

        # Euclidean distance
        dist = torch.norm(zi - zj, dim=1)

        # Pair types
        must = (Y[i] == Y[j])
        cannot = ~must

        loss = 0.0

        if must.any():
            ml_loss = (dist[must] ** 2).mean()
            loss += ml_loss

        if cannot.any():
            cl_loss = torch.relu(margin - dist[cannot]).pow(2).mean()
            loss += cl_loss

        return lambd * loss

    def forward(self, x):
        z = self.encoder(x)

        q = self.soft_assignments(z)

        return z, q