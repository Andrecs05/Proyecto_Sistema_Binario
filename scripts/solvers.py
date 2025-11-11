import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.special import legendre
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

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

def binary_potential_poisson_fd(M1, M2, R_star1, R_star2, a, G, L, grid_size):
    """
    Calculate the gravitational potential in a binary system by solving Poisson's equation using finite differences.
    """

    N = grid_size
    h = 2 * L / (N - 1)

    x = np.linspace(-L, L, N)
    y = np.linspace(-L, L, N)
    X, Y = np.meshgrid(x, y)

    rho = np.zeros((N, N))

    a1 = M2 * a / (M1 + M2)  # M1 position (right of COM)
    a2 = M1 * a / (M1 + M2)  # M2 position (left of COM)

    area_1 = np.pi * R_star1**2
    area_2 = np.pi * R_star2**2

    for i in range(N):
        for j in range(N):
            r1 = np.sqrt((X[i, j] - a1)**2 + Y[i, j]**2)
            r2 = np.sqrt((X[i, j] + a2)**2 + Y[i, j]**2)

            if r1 <= R_star1:
                rho[i, j] += M1 / area_1
            if r2 <= R_star2:
                rho[i, j] += M2 / area_2

    phi = np.zeros((N, N))

    N_total = N * N

    main_diag = -4 * np.ones(N_total)
    off_diag1_r = np.ones(N_total - 1)
    off_diag1_l = np.ones(N_total - 1)
    off_diagN = np.ones(N_total - N)

    for i in range(N):
        for j in range(N - 1):
            k = i * N + j
            if j == N - 1:
                off_diag1_r[k] = 0
            if j == 0:
                off_diag1_l[k] = 0

    diagonals = [main_diag, off_diag1_r, off_diag1_l, off_diagN, off_diagN]
    offsets = [0, -1, 1, -N, N]

    A = diags(diagonals, offsets, shape=(N_total, N_total), format='lil')
    b = 4 * np.pi * G * rho.flatten() * h**2

    switching_radius = a + max(R_star1, R_star2)  # Or any radius you choose

    for i in range(N):
        for j in range(N):
            k = i * N + j
            r = np.sqrt(X[i,j]**2 + Y[i,j]**2)
            
            # Apply Laplace solution on AND outside the switching circle
            if r >= switching_radius:
                theta = np.arctan2(Y[i,j], X[i,j])
                phi_boundary = binary_potential_laplace(r, theta, M1, M2, a, G, n_terms=3)
                
                A[k, :] = 0
                A[k, k] = 1
                b[k] = phi_boundary

    A = A.tocsr()

    phi = spsolve(A, b)
    phi = phi.reshape((N, N))

    return phi

def binary_field_poisson_fd(phi, L, grid_size):
    """
    Calculate the gravitational field in a binary system by numerically differentiating the potential obtained from Poisson's equation.
    Returns the x and y components of the field.
    """

    x = np.linspace(-L, L, grid_size)
    y = np.linspace(-L, L, grid_size)

    dphi_dy, dphi_dx = np.gradient(phi, y, x)
    g_x = -dphi_dx
    g_y = -dphi_dy

    return g_x, g_y
