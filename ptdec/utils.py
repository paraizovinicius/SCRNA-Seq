import numpy as np
import random
import torch
from typing import Optional
from scipy.optimize import linear_sum_assignment


def set_random_seeds(seed: int = 42) -> None:
    """
    Set global random seeds for reproducible runs across numpy, python random and PyTorch.

    This configures CPU and CUDA RNGs and attempts to make cuDNN deterministic.

    :param seed: integer seed to use
    :return: None
    """
    if seed is None:
        return
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    # for multi-gpu
    try:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        # CUDA may not be available; ignore
        pass

    # Make cuDNN deterministic where possible. This may slow down training.
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def cluster_accuracy(y_true, y_predicted, cluster_number: Optional[int] = None):
    """
    Calculate clustering accuracy after using the linear_sum_assignment function in SciPy to
    determine reassignments.

    :param y_true: list of true cluster numbers, an integer array 0-indexed
    :param y_predicted: list  of predicted cluster numbers, an integer array 0-indexed
    :param cluster_number: number of clusters, if None then calculated from input
    :return: reassignment dictionary, clustering accuracy
    """
    if cluster_number is None:
        cluster_number = (
            max(y_predicted.max(), y_true.max()) + 1
        )  # assume labels are 0-indexed
    count_matrix = np.zeros((cluster_number, cluster_number), dtype=np.int64)
    for i in range(y_predicted.size):
        count_matrix[y_predicted[i], y_true[i]] += 1

    row_ind, col_ind = linear_sum_assignment(count_matrix.max() - count_matrix)
    reassignment = dict(zip(row_ind, col_ind))
    accuracy = count_matrix[row_ind, col_ind].sum() / y_predicted.size
    return reassignment, accuracy


def target_distribution(batch: torch.Tensor) -> torch.Tensor:
    """
    Compute the target distribution p_ij, given the batch (q_ij), as in 3.1.3 Equation 3 of
    Xie/Girshick/Farhadi; this is used the KL-divergence loss function.

    :param batch: [batch size, number of clusters] Tensor of dtype float
    :return: [batch size, number of clusters] Tensor of dtype float
    """
    weight = (batch ** 2) / torch.sum(batch, 0)
    return (weight.t() / torch.sum(weight, 1)).t()

def compute_pairwise_constraints(Y: torch.Tensor) -> torch.Tensor:
    """For must-link constraints, when x_i and x_k are assigned to the same cluster, 
    a i_k = 1 . If x_i and x_k satisfy cannot-link constraints, a i_k = −1 . 
    Other entities in this matrix are all zero.

    Args: batch (torch.Tensor)

    Returns: torch.Tensor: constraint matrix 
    """

    # Ensure Y is a column vector
    Y = Y.view(-1, 1)  # Shape (n, 1)
    
    # Compare each pair: same -> 1, different -> -1
    same = (Y == Y.T).float()  # must-link: 1
    different = (Y != Y.T).float()  # cannot-link: 1

    A = same - different  # must-link = 1, cannot-link = -1

    # Set diagonal to 0 (no self constraint)
    A.fill_diagonal_(0)

    return A

def supervised_loss(Z, A, lambd):
    """_summary_

    Args:
        batch (_type_): _description_
        A (_type_): _description_
        lambd (_type_, optional): _description_. Defaults to 1e-5.

    Returns:
        _type_: _description_
    """
    n = Z.shape[0]
    diff = Z.unsqueeze(0) - Z.unsqueeze(1)  # shape: (n, n, latent_dim)
    dists_squared = torch.sum(diff ** 2, dim=-1)  # shape: (n, n)
    res = torch.sum(A * dists_squared)
    return res * lambd / n
