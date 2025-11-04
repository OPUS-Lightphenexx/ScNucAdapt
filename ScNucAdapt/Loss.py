import numpy as np
import copy
import random
import torch.nn.functional as F
from decorator import append
from sklearn.utils._joblib import Parallel, delayed, effective_n_jobs
from sklearn.utils import check_random_state
from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture
import torch
from tqdm import tqdm

def CS(x1, x2, sigma=10, if_use_cdist=False, median_sigma=False):  # conditional cs divergence
    # x1 = torch.tensor(x1)
    # x2 = torch.tensor(x2)

    K1 = GaussianMatrix(x1, x1, sigma, if_use_cdist, median_sigma)
    K2 = GaussianMatrix(x2, x2, sigma, if_use_cdist, median_sigma)

    K12 = GaussianMatrix(x1, x2, sigma, if_use_cdist, median_sigma)

    dim1 = K1.shape[0]
    self_term1 = K1.sum() / (dim1 ** 2)

    dim2 = K2.shape[0]
    self_term2 = K2.sum() / (dim2 ** 2)

    cross_term = K12.sum() / (dim1 * dim2)

    cs = -2 * torch.log(cross_term) + torch.log(self_term1) + torch.log(self_term2)

    return cs
