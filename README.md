# Proyecto_Sistema_Binario

Analytical and numerical exploration of the gravitational potential and field of a binary star system.  
Contains multipole (Laplace) analytic solution and a Poisson finite-difference solver with plotting utilities and animations.

## Repository layout

- notebooks/
  - binary_system_laplace.ipynb — multipole / Laplace notebook (interactive plots)
  - binary_system_poisson.ipynb — Poisson finite-difference notebook
- scripts/
  - solvers.py — potential and field calculations (Laplace multipole + Poisson FD)
  - plotters.py — plotting helpers, contour/streamline plots helpers

## Requirements

Recommended Python >= 3.10. Install core libs:

Windows (PowerShell / CMD):
```powershell
pip install numpy matplotlib scipy pillow
pip install ipympl             # enable %matplotlib widget in notebooks (optional)
```


## Quick start

1. Open the notebook in `notebooks/` with JupyterLab / Jupyter Notebook.
2. In the first cell ensure project root is on sys.path (already present in the notebooks):
```python
%matplotlib widget
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))
```
3. Run the cells. If you want interactive toolbar (zoom/pan) make sure `ipympl` is installed and `%matplotlib widget` is enabled.

## Usage examples

From a notebook or Python script:

- Compute analytic multipole potential (Laplace):
```python
from scripts.solvers import binary_potential_laplace
phi = binary_potential_laplace(r, theta, M1, M2, a, G, n_terms=3)
```

- Compute field from a potential grid (Poisson or Laplace grid):
```python
from scripts.solvers import binary_field_laplace, binary_field_poisson_fd
g_x, g_y = binary_field_laplace(Phi, X, Y)          # returns Cartesian components
g_x_poisson, g_y_poisson = binary_field_poisson_fd(phi, L, grid_size)
```

- Plot with helpers:
```python
from scripts.plotters import plot_potential_contours, plot_field_streamlines, create_potential_animation
plot_potential_contours(Phi, M1, M2, a, size=4.0, res=400, levels=50)
plot_field_streamlines(g_x, g_y, X, Y, M1, M2, a)
```

## Notes & troubleshooting

- Sparse matrix construction: the Poisson solver builds a sparse matrix — use `lil` for modifications then convert to `csr` before solving to avoid SparseEfficiencyWarning.
- np.gradient spacing: pass 1‑D x and y arrays (not 2‑D) to `np.gradient` to avoid the ValueError "distances must be either scalars or 1d".
- Interactive notebooks: use `%matplotlib widget` (ipympl) for toolbar (zoom/pan). If unavailable, `%matplotlib notebook` is an alternative.
