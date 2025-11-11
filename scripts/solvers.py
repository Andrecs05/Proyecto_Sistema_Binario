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
    a1 = M2 * a / M  # M1 position (right of COM)
    a2 = M1 * a / M  # M2 position (left of COM)

    for n in range(1, n_terms + 1):
        Pn = legendre(n)
        Bn = -G * ((M1 * a1**n) + (M2 * (-a2)**n))
        phi += Bn / r**(n + 1) * Pn(np.cos(theta))
    
    return phi

def binary_field_laplace(phi, X, Y):
    """
    Calculate the gravitational field in a binary system using a multipole expansion.
    Returns the radial and angular components of the field.
    """

    x = X[0, :]   
    y = Y[:, 0]   

    dphi_dy, dphi_dx = np.gradient(phi, y, x)
    g_x = -dphi_dx
    g_y = -dphi_dy

    return g_x, g_y