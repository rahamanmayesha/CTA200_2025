import numpy as np

def iter_func(xmin, xmax, ymin, ymax, width, height, max_iter):
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    C = x[np.newaxis, :] + 1j * y[:, np.newaxis]  # shape (height, width)
    Z = np.zeros_like(C, dtype=complex)
    diverged = np.zeros(C.shape, dtype=int)
    mask = np.full(C.shape, True, dtype=bool)

    for i in range(max_iter):
        Z[mask] = Z[mask] ** 2 + C[mask]
        mask_new = np.abs(Z) <= 2
        diverged[(mask) & (~mask_new)] = i
        mask &= mask_new

    diverged[diverged == 0] = max_iter  # treat those that never diverged
    return diverged
