# -*- coding: utf-8 -*-
"""Binominal.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1cJ8EHngAvcstFE0ht8Qv89W7UMPeEvrl

# Binominal Option Pricing Model

Implementation of a simple slow and fast binominal pricing model. We treat the binominal tree as a network with nodes (i,j), with i representing time steps and j representing the number of ordered price outcome.

This project focuses on **European Pricing call**.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import wraps
from time import time

"""## Binominal Tree Representation

Stock Tree can be represented using nodes (i,j) and initial stock price $S_0$

$C_{i,j}$ represents contract price at each node (i,j). Where $C_{N_{j}}$ represents final payoff function that we can define.
"""

# Parameters for the tree
S0 = 100 # Initial stock price
K = 100 # Strike price
T = 1 # Time to maturity in years
r = 0.06 # Annual risk-free rate
N = 3 # Number of time steps
u = 1.1 # Up factor in binominal models
d = 1/u # Down factor
opttype = 'C'

"""### Binominal Tree Slow"""

def binominal_tree_slow(S0, K, T, r, N, u, d, opttype='C'):
    # Precompute values
    dt = T / N
    q = (np.exp(r * dt) - d) / (u - d)
    disc = np.exp(-r * dt)

    # Initialize asset prices at maturity
    S = np.zeros(N+1)
    S[0] = S0*d**N
    for i in range(1, N+1):
      S[i] = S[i-1]*u/d

    # Initialize the option values at maturity
    C = np.zeros(N+1)
    for j in range(0, N+1):
      C[j] = max(0, S[j]-K)

    # Step backwards through the tree
    for w in np.arange(N, 0, -1):
      for r in np.arange(0, w):
        C[r] = disc * (q * C[r+1] + (1-q) * C[r])


    return C[0]

binominal_tree_slow(S0, K, T, r, N, u, d, opttype)

"""### Binominal Tree Fast"""

def binominal_tree_fast(S0, K, T, r, N, u, d, opttype='C'):
    dt = T / N
    q = (np.exp(r * dt) - d) / (u - d)
    disc = np.exp(-r * dt)

    # Initialize asset prices at maturity
    C = S0 * d ** (np.arange(N, -1, -1)) * u ** (np.arange(0, N+1, 1))

    # Payoff function
    C = np.maximum(C-K, np.zeros(N+1))

    # Step backwards through the tree
    for w in np.arange(N, 0, -1):
      C = disc * (q * C[1:w+1] + (1-q) * C[0:w])

    return C[0]

binominal_tree_fast(S0, K, T, r, N, u, d, opttype)

"""By using both the fast and slow binominal method we can compute the strike price in two different ways. As we increase the number of nodes, we will see that the slow model will perform much worse which points towards that one should vectorize."""