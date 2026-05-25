import numpy as np

def circle_packing21() -> np.ndarray:
    rng = np.random.default_rng()
    n = 21
    best_score = -1e18
    best_circles = None

    def build_hex_grid(R=5, C=6, jitter=0.0):
        dx = np.sqrt(3) / 2.0
        dy = 1.0
        xs, ys = [], []
        for row in range(-R, R + 1):
            y = row * dy
            offset = 0.5 if row % 2 else 0.0
            for col in range(-C, C + 1):
                x = (col + offset) * dx
                xs.append(x + rng.normal(0.0, jitter))
                ys.append(y + rng.normal(0.0, jitter))
        return np.array(xs, dtype=float), np.array(ys, dtype=float)

    def farthest_point_selection(points_x, points_y, n_selected):
        M = len(points_x)
        selected = [rng.integers(0, M)]
        selected_mask = np.zeros(M, dtype=bool)
        selected_mask[selected[0]] = True
        pts = np.stack([points_x, points_y], axis=1)
        min_dists = np.full(M, np.inf, dtype=float)

        for _ in range(n_selected - 1):
            d2 = np.sum((pts - pts[selected[-1]]) ** 2, axis=1)
            min_dists = np.minimum(min_dists, d2)
            min_dists[selected_mask] = -np.inf
            next_idx = np.argmax(min_dists)
            selected.append(next_idx)
            selected_mask[next_idx] = True
        return np.array(selected, dtype=int)

    def compute_score(x, y):
        dx = np.subtract.outer(x, x)
        dy = np.subtract.outer(y, y)
        D2 = dx * dx + dy * dy + 1e-18
        np.fill_diagonal(D2, np.inf)
        dmin = np.sqrt(np.min(D2, axis=1))
        r = 0.5 * dmin
        x_lo = x - r
        x_hi = x + r
        y_lo = y - r
        y_hi = y + r
        w = float(x_hi.max() - x_lo.min())
        h = float(y_hi.max() - y_lo.min())
        per = w + h
        if per < 1e-12:
            return -1e18, x, y, r, w, h
        return np.sum(r) / per, x, y, r, w, h

    def refine(x, y, steps=2000, init_step=0.08):
        x, y = x.copy(), y.copy()
        for step in np.linspace(init_step, 5e-5, steps):
            dx = np.subtract.outer(x, x)
            dy = np.subtract.outer(y, y)
            D2 = dx * dx + dy * dy + 1e-18
            np.fill_diagonal(D2, np.inf)
            dmin = np.sqrt(np.min(D2, axis=1))
            r = 0.5 * dmin
            i_max = np.argmax(x + r)
            i_min = np.argmin(x - r)
            i_top = np.argmax(y + r)
            i_bot = np.argmin(y - r)
            grad_x = np.zeros(n)
            grad_y = np.zeros(n)
            jmin = np.argmin(D2, axis=1)
            unit_x = (x - x[jmin]) / np.maximum(dmin, 1e-12)
            unit_y = (y - y[jmin]) / np.maximum(dmin, 1e-12)
            grad_x += unit_x
            grad_y += unit_y
            grad_x[i_max] -= 1.0
            grad_x[i_min] += 1.0
            grad_y[i_top] -= 1.0
            grad_y[i_bot] += 1.0
            norms = np.linalg.norm(np.stack([grad_x, grad_y], axis=1), axis=1)
            valid = norms > 1e-12
            grad_x[valid] /= norms[valid]
            grad_y[valid] /= norms[valid]
            x += step * grad_x
            y += step * grad_y
            if np.random.rand() < 0.05:
                x += rng.normal(0.0, step * 0.15, size=n)
                y += rng.normal(0.0, step * 0.15, size=n)
        return x, y

    for _ in range(16):
        R = 5 if _ % 3 == 0 else 4
        C = 6 if _ % 2 == 0 else 5
        jitter = 0.01 + 0.04 * rng.random()
        base_x, base_y = build_hex_grid(R=R, C=C, jitter=jitter)
        indices = farthest_point_selection(base_x, base_y, n)
        x = base_x[indices]
        y = base_y[indices]
        score, x, y, r, w, h = compute_score(x, y)
        x, y = refine(x, y)
        score, x, y, r, w, h = compute_score(x, y)
        if score > best_score:
            best_score = score
            best_circles = np.column_stack((x, y, r))

    x, y, r = best_circles[:, 0], best_circles[:, 1], best_circles[:, 2]
    x_lo = x - r
    x_hi = x + r
    y_lo = y - r
    y_hi = y + r
    w = float(x_hi.max() - x_lo.min())
    h = float(y_hi.max() - y_lo.min())
    total_perimeter = w + h
    scale = 2.0 / max(total_perimeter, 1e-12)
    x *= scale
    y *= scale
    r *= scale
    cx = 0.5 * (x.min() + x.max())
    cy = 0.5 * (y.min() + y.max())
    x -= cx
    y -= cy
    return np.column_stack((x, y, r))