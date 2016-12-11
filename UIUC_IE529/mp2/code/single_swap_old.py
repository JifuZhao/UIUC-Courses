#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
__author__      = "Jifu Zhao"
__email__       = "jzhao59@illinois.edu"
__date__        = "12/05/2016"
"""

import numpy as np
import time
from k_centers import kCenters


def singleSwap(X, K, tau=0.05, random_state=None, verbose=True):
    """ function to implement the single-swap for k-centers algorithm """
    t0 = time.time()

    # calculate the initial centers
    Q, _, pre_cost = kCenters(X, K, random_state=random_state, verbose=False)
    N, d = X.shape

    # compute the distance based on current centers
    distance = np.zeros((N, K))
    for idx in range(K):
        distance[:, idx] = np.sqrt(np.sum((X - Q[idx, :])**2, axis=1))
    cost = np.max(np.min(distance, axis=1))  # calculate cost

    i = 0
    while i < K:
        if i == 0:
            min_dist = np.min(distance[:, 0:], axis=1)
        elif i == (K - 1):
            min_dist = np.min(distance[:, :-1], axis=1)
        else:
            min_dist = np.minimum(np.min(distance[:, :i], axis=1),
                                  np.min(distance[:, (i + 1):], axis=1))
        print('Initial:\t', i, cost)
        min_cost = None
        min_idx = None

        for j in range(N):
            cur_dist = np.sqrt(np.sum((X - X[j, :])**2, axis=1))
            tmp_cost = np.max(np.minimum(min_dist, cur_dist))
            if min_cost is None:
                min_cost = tmp_cost
                min_idx = j
            else:
                if tmp_cost < min_cost:
                    min_cost = tmp_cost
                    min_idx = j

        if min_cost / cost <= (1 - tau):
            Q[i, :] = X[min_idx, :]
            distance[:, i] = np.sqrt(np.sum((X - X[min_idx, :])**2, axis=1))
            i += 1
            print(i, min_idx, cost, min_cost)
            cost = min_cost
        else:
            i += 1
            print(i, cost)

    C = np.argmin(distance, axis=1)

    if verbose is True:
        t = np.round(time.time() - t0, 4)
        print('Single-Swap is finished in ' + str(t) + 's')

    return Q, C, cost


def main():
    """ implement the single-swap algorithm on real-world data """
    print('test case !')


if __name__ == '__main__':
    main()
