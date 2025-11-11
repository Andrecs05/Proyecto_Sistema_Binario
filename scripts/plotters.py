import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def plot_potential_contours(phi, M1, M2, a, size = 4.0, res = 400, levels = 50):
    """
    Plot contour map of the gravitational potential.
    """

    x = np.linspace(-size, size, res)
    y = np.linspace(-size, size, res)
    X, Y = np.meshgrid(x, y)

    a1 = M2 * a / (M1 + M2)  # M1 position (right of COM)
    a2 = M1 * a / (M1 + M2)  # M2 position (left of COM)

    plt.figure(figsize=(8, 6))
    contour = plt.contourf(X, Y, phi, levels=levels, cmap=cm.inferno.reversed())
    plt.plot(a1, 0, 'ro', markersize=8, label=f'M1 = {M1}')
    plt.plot(-a2, 0, 'bo', markersize=8, label=f'M2 = {M2}')
    plt.colorbar(contour, label='Gravitational Potential')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Gravitational Potential Contours')
    plt.axis('equal')
    plt.legend(loc='best')
    plt.show()

def plot_potential_transverse(phi, x, y, y0=0):
    """
    Plot transverse cut of the gravitational potential at y = y0.
    """

    idx = (np.abs(y - y0)).argmin()
    plt.figure(figsize=(8, 6))
    plt.plot(x, phi[idx, :], label=f'y = {y0}')
    plt.xlabel('x')
    plt.ylabel('Gravitational Potential')
    plt.title('Transverse Cut of Gravitational Potential')
    plt.legend()
    plt.grid()
    plt.show()

def plot_field_log_magnitude(g_x, g_y, X, Y, M1, M2, a):
    """
    Plot logarithmic magnitude of the gravitational field.
    """
    g_magnitude = np.sqrt(g_x**2 + g_y**2)
    log_g_magnitude = np.log10(g_magnitude + 1e-10)  # Avoid log(0)

    a1 = M2 * a / (M1 + M2)  # M1 position (right of COM)
    a2 = M1 * a / (M1 + M2)  # M2 position (left of COM)

    plt.figure(figsize=(8, 6))
    contour = plt.contourf(X, Y, log_g_magnitude, levels=50, cmap=cm.magma)
    plt.plot(a1, 0, 'ro', markersize=8, label=f'M1 = {M1}')
    plt.plot(-a2, 0, 'bo', markersize=8, label=f'M2 = {M2}')
    plt.colorbar(contour, label='Log10 Gravitational Field Magnitude')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Logarithmic Magnitude of Gravitational Field')
    plt.axis('equal')
    plt.legend(loc='best')
    plt.show()

def plot_field_streamlines(g_x, g_y, X, Y, M1, M2, a, density=1.5):
    """
    Plot streamlines of the gravitational field.
    """
    g_magnitude = np.sqrt(g_x**2 + g_y**2)
    a1 = M2 * a / (M1 + M2)  # M1 position (right of COM)
    a2 = M1 * a / (M1 + M2)  # M2 position (left of COM)

    plt.figure(figsize=(8, 6))
    plt.contourf(X, Y, g_magnitude, levels=50, cmap=cm.plasma)
    plt.plot(a1, 0, 'ro', markersize=8, label=f'M1 = {M1}')
    plt.plot(-a2, 0, 'bo', markersize=8, label=f'M2 = {M2}')
    strm = plt.streamplot(X, Y, g_x, g_y, color='w', linewidth=1, density=density, arrowsize=1)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Gravitational Field Streamlines')
    plt.axis('equal')
    plt.legend(loc='best')
    plt.show()