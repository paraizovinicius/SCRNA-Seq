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


def cluster_acc(Y_pred, Y):
  from scipy.optimize import linear_sum_assignment as linear_assignment

  assert Y_pred.size == Y.size
  D = max(Y_pred.max(), Y.max())+1
  w = np.zeros((D,D), dtype=np.int64)
  for i in range(Y_pred.size):
    w[Y_pred[i], Y[i]] += 1
  row_ind, col_ind = linear_assignment(w.max() - w)

  return sum(w[i, j] for i, j in zip(row_ind, col_ind)) / Y_pred.size


import torch
import torch.nn as nn

class IDEC(nn.Module):
    def __init__(self, encoder, decoder, init_centroids, alpha=1.0):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder # Now the decoder is added and will be finetuned
        self.alpha = alpha

        self.centroids = nn.Parameter(
            init_centroids.clone()
        )

    def soft_assignments(self, z):
        """
        Student-t distribution.
        z: (B, latent_dim)
        μ: (K, latent_dim)
        """
        factor = -(self.alpha + 1) / 2

        # Vectorized distance calculation
        distancia = torch.sum((z.unsqueeze(1) - self.centroids) ** 2, dim=2) / self.alpha + 1

        # Vectorized power operation
        distancia_quadrado = distancia.pow(factor)

        # Vectorized sum and normalization
        sum_distancia = distancia_quadrado.sum(dim=1, keepdim=True)

        # Vectorized division (broadcasting handles the normalization)
        q = distancia_quadrado / sum_distancia
        return q

    def target_distribution(self, q):
        fj = q.sum(axis=0)

        numerator = (q ** 2) / fj # shape 70000, 10

        denom = numerator.sum(dim=1, keepdim=True)
        p = numerator / denom
        return p

    def forward(self, x):
        z = self.encoder(x)

        q = self.soft_assignments(z)

        return z, q