import torch
import numpy as np
from ptdec.utils import set_random_seeds
from ptsdae.sdae import StackedDenoisingAutoEncoder
from torch.utils.data import TensorDataset, DataLoader, RandomSampler


def test_deterministic_initialisation_sdae():
    """Ensure that re-seeding before model construction produces identical initial weights."""
    set_random_seeds(42)
    dims = [16, 8, 4]
    m1 = StackedDenoisingAutoEncoder(dims)
    w1 = m1.encoder[0].linear.weight.detach().cpu().numpy().copy()

    set_random_seeds(42)
    m2 = StackedDenoisingAutoEncoder(dims)
    w2 = m2.encoder[0].linear.weight.detach().cpu().numpy().copy()

    assert np.allclose(w1, w2), "Weights differ between two runs seeded with the same value"


def test_deterministic_dataloader_order():
    """Ensure DataLoader with RandomSampler + seeded Generator yields same ordering across runs."""
    n = 32
    data = torch.arange(n)
    dataset = TensorDataset(data)

    g1 = torch.Generator().manual_seed(1234)
    sampler1 = RandomSampler(dataset, generator=g1)
    loader1 = DataLoader(dataset, batch_size=5, sampler=sampler1, shuffle=False)
    order1 = torch.cat([batch[0].view(-1) for batch in loader1]).tolist()

    g2 = torch.Generator().manual_seed(1234)
    sampler2 = RandomSampler(dataset, generator=g2)
    loader2 = DataLoader(dataset, batch_size=5, sampler=sampler2, shuffle=False)
    order2 = torch.cat([batch[0].view(-1) for batch in loader2]).tolist()

    assert order1 == order2, "DataLoader ordering differs between two runs with the same RandomSampler seed"
