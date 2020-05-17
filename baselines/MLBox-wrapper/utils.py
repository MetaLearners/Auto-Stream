import numpy as np


def stratified_downsample(y, sample_ratio, seed=1):
    n = len(y)
    pos = np.sum(y)
    neg = n - pos
    index = np.arange(n)
    np.random.seed(seed)
    pos_index = np.random.choice(index[y == 1], size=int(pos * sample_ratio), replace=False)
    np.random.seed(seed)
    neg_index = np.random.choice(index[y == 0], size=int(neg * sample_ratio), replace=False)
    return sorted(np.concatenate([pos_index, neg_index]))


def majority_downsample(y, frac=3.0, seed=1, prob=None):
    n = len(y)
    index = np.arange(n)
    class_0 = index[y == 0]
    class_1 = index[y == 1]
    if len(class_0) < len(class_1):
        major, minor = class_1, class_0
    else:
        major, minor = class_0, class_1
    major_n = len(major)
    minor_n = len(minor)

    np.random.seed(seed)
    select = np.random.choice(major, size=min(major_n, int(minor_n*frac)), replace=False, p=prob)
    return sorted(np.concatenate([select, minor]))