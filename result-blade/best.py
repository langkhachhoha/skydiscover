import numpy as np
import time

def circle_packing21() -> np.ndarray:
    '''
    Places 21 non-overlapping circles inside a rectangle of
    perimeter 4, maximizing the sum of their radii.

    Returns:
        circles: numpy array of shape (21, 3), where each row
            stores (x, y, radius).
    '''
    N = 21

    # Search over number of shelves m; for each m, use the most balanced partition of N.
    best = None
    best_data = None
    for m in range(1, N + 1):
        a = N // m
        q = N % m
        if a == 0:
            continue
        counts = [a + 1] * q + [a] * (m - q)
        A = sum(1.0 / n for n in counts)
        R = m / (1.0 + A)
        if best is None or R > best:
            best = R
            best_data = (counts, A)

    counts, A = best_data
    m = len(counts)
    c = 1.0 / (1.0 + A)
    shelf_radii = [c / n for n in counts]

    circles = []
    y_offset = 0.0
    for n, r in zip(counts, shelf_radii):
        y = y_offset + r
        for i in range(n):
            x = r + 2.0 * r * i
            circles.append((x, y, r))
        y_offset += 2.0 * r

    circles = np.array(circles, dtype=float)
    W = 2.0 * c
    H = y_offset

    # Full refinement with adaptive physics-based optimization
    rng = np.random.default_rng(42)
    M_pairs = N * (N - 1) // 2
    pairs_i = []
    pairs_j = []
    for i in range(N - 1):
        for j in range(i + 1, N):
            pairs_i.append(i)
            pairs_j.append(j)
    pairs_i = np.array(pairs_i, dtype=int)
    pairs_j = np.array(pairs_j, dtype=int)

    def compute_constraints(centers, W, H):
        x = centers[:, 0]
        y = centers[:, 1]
        dx = x[pairs_i] - x[pairs_j]
        dy = y[pairs_i] - y[pairs_j]
        d = np.hypot(dx, dy)
        e = np.minimum(np.minimum(x, W - x), np.minimum(y, H - y))
        return d, e

    def simplex_max(A, b, c):
        m, n = A.shape
        T = np.zeros((m + 1, n + m + 1), dtype=float)
        T[:m, :n] = A
        T[:m, n:n + m] = np.eye(m)
        T[:m, -1] = b
        T[m, :n] = -c
        T[m, -1] = 0.0
        basis = np.arange(n, n + m, dtype=int)
        pivots = 0
        while True:
            entering = -1
            for j in range(n + m):
                if T[m, j] < -1e-10:
                    entering = j
                    break
            if entering == -1:
                break
            col = T[:m, entering]
            rhs = T[:m, -1]
            mask = col > 1e-10
            if not np.any(mask):
                break
            ratios = np.where(mask, rhs / col, np.inf)
            candidates = np.where(np.isclose(ratios, np.min(ratios), atol=1e-10))[0]
            leaving = int(candidates[0])
            piv = T[leaving, entering]
            if abs(piv) < 1e-12:
                piv = 1e-12
            T[leaving, :] /= piv
            for i in range(m + 1):
                if i == leaving:
                    continue
                factor = T[i, entering]
                if factor != 0.0:
                    T[i, :] -= factor * T[leaving, :]
            basis[leaving] = entering
            pivots += 1
            if pivots > 20000:
                break
        x = np.zeros(n, dtype=float)
        for i in range(m):
            bi = basis[i]
            if bi < n:
                x[bi] = T[i, -1]
        return np.maximum(x, 0.0), np.dot(c, x)

    # Build constraint matrix A
    A = np.zeros((M_pairs + N, N), dtype=float)
    row = 0
    for k in range(M_pairs):
        i = pairs_i[k]
        j = pairs_j[k]
        A[row, i] = 1.0
        A[row, j] = 1.0
        row += 1
    for i in range(N):
        A[row, i] = 1.0
        row += 1

    centers = circles[:, :2].copy()
    radii = circles[:, 2].copy()
    best_sum = np.sum(radii)
    best_circles = circles.copy()
    step = 0.1
    start_time = time.time()

    while time.time() - start_time < 550:
        d, e = compute_constraints(centers, W, H)
        b = np.concatenate([d, e])
        r, val = simplex_max(A, b, np.ones(N))
        if val <= best_sum + 1e-10:
            step *= 0.95
        else:
            best_sum = val
            best_circles = np.column_stack([centers, r])
            centers += 1e-5 * rng.standard_normal(centers.shape)

        # Force simulation
        F = np.zeros_like(centers)
        s = d - (r[pairs_i] + r[pairs_j])
        near = s < 0.02
        if np.any(near):
            idx_i = pairs_i[near]
            idx_j = pairs_j[near]
            dx = centers[idx_i, 0] - centers[idx_j, 0]
            dy = centers[idx_i, 1] - centers[idx_j, 1]
            dist = np.hypot(dx, dy)
            inv = np.where(dist > 1e-12, 1.0 / dist, 0.0)
            ux = dx * inv
            uy = dy * inv
            w = (0.02 - s[near])
            np.add.at(F[:, 0], idx_i, ux * w)
            np.add.at(F[:, 1], idx_i, uy * w)
            np.add.at(F[:, 0], idx_j, -ux * w)
            np.add.at(F[:, 1], idx_j, -uy * w)
        e = np.minimum(np.minimum(centers[:, 0], W - centers[:, 0]), np.minimum(centers[:, 1], H - centers[:, 1]))
        tight = e - r < 0.02
        if np.any(tight):
            for i in np.where(tight)[0]:
                xi, yi = centers[i]
                dists = np.array([xi, W - xi, yi, H - yi])
                widx = int(np.argmin(dists))
                f = (0.02 - (e[i] - r[i]))
                if widx == 0:
                    F[i, 0] += f
                elif widx == 1:
                    F[i, 0] -= f
                elif widx == 2:
                    F[i, 1] += f
                else:
                    F[i, 1] -= f
        norms = np.linalg.norm(F, axis=1)
        max_norm = np.max(norms)
        if max_norm > 1e-10:
            D = F / max_norm
        else:
            D = 1e-4 * rng.standard_normal(F.shape)
        new_centers = centers + step * D
        new_centers[:, 0] = np.clip(new_centers[:, 0], 0.0, W)
        new_centers[:, 1] = np.clip(new_centers[:, 1], 0.0, H)
        centers = new_centers
        step = np.clip(step * 1.02, 0.001, 0.5)

    # Final scaling to perimeter 4
    min_x = np.min(best_circles[:, 0] - best_circles[:, 2])
    max_x = np.max(best_circles[:, 0] + best_circles[:, 2])
    min_y = np.min(best_circles[:, 1] - best_circles[:, 2])
    max_y = np.max(best_circles[:, 1] + best_circles[:, 2])
    width = max_x - min_x
    height = max_y - min_y
    scale = 2.0 / (width + height)
    best_circles[:, :2] -= [min_x, min_y]
    best_circles *= scale
    best_circles[:, 2] = np.maximum(0.0, best_circles[:, 2] - 1e-12)
    return np.nan_to_num(best_circles, copy=False, nan=0.0, posinf=0.0, neginf=0.0)