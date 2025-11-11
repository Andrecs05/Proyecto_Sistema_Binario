import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.special import legendre

def binary_potential_laplace(r, theta, M1, M2, a, G, n_terms):
    """
    Calculate the gravitational potential in a binary system using a multipole expansion.
    """

    M = M1 + M2
    phi = -G * M / r

    for n in range(n_terms):
        Pn = legendre(n)
        Bn = -G * a**n * ( M1 * (M2 / M)**n + (-1)**n * M2 * (M1 / M)**n )
        phi += Bn / r**(n + 1) * Pn(np.cos(theta))
    
    return phi

