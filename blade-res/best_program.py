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
    rng = np.random.default_rng(20260705)
    n = 21
    sqrt3 = np.sqrt(3.0)
    
    # Precompute all pairs for fast distance checks
    i_idx, j_idx = np.triu_indices(n, k=1)
    pairs_i, pairs_j = i_idx, j_idx
    
    def compute_distances(x, y, r):
        dx = x[pairs_i] - x[pairs_j]
        dy = y[pairs_i] - y[pairs_j]
        return np.sqrt(dx*dx + dy*dy)
    
    def has_overlap(x, y, r):
        d = compute_distances(x, y, r)
        sr = r[pairs_i] + r[pairs_j]
        return np.any(d < sr - 1e-10)
    
    def normalize_to_perimeter(x, y, r):
        x_ext = x + r
        y_ext = y + r
        xmin, xmax = np.min(x - r), np.max(x_ext)
        ymin, ymax = np.min(y - r), np.max(y_ext)
        W, H = xmax - xmin, ymax - ymin
        scale = 2.0 / (W + H + 1e-12)
        return x * scale, y * scale, r * scale
    
    def hexagonal_layout(counts):
        xs, ys, rs = [], [], []
        r_base = 1.0
        for row, m in enumerate(counts):
            offset = r_base if row % 2 == 0 else 2 * r_base
            y = r_base + row * sqrt3 * r_base
            for i in range(m):
                x = offset + 2 * r_base * i
                xs.append(x)
                ys.append(y)
                rs.append(r_base)
        return np.array(xs, dtype=float), np.array(ys, dtype=float), np.array(rs, dtype=float)
    
    def compute_slacks(x, y, r):
        d = compute_distances(x, y, r)
        sr = r[pairs_i] + r[pairs_j]
        slack = d - sr
        boundary_slack_x = np.minimum(x - r - (x - r).min(), (x + r).max() - r - x)
        boundary_slack_y = np.minimum(y - r - (y - r).min(), (y + r).max() - r - y)
        boundary_slack = np.minimum(boundary_slack_x, boundary_slack_y)
        return np.min(slack), np.min(boundary_slack)
    
    # Try multiple initial layouts
    layouts = [
        [6, 5, 5, 5],
        [5, 5, 5, 6],
        [4, 5, 4, 5, 3],
        [5, 4, 4, 4, 4],
        [4, 4, 5, 4, 4],
        [4, 4, 4, 4, 5]
    ]
    
    best_circles = None
    best_sum = 0.0
    start_time = time.time()
    
    for layout in layouts:
        if np.sum(layout) != n:
            continue
        x0, y0, r0 = hexagonal_layout(layout)
        x0, y0, r0 = normalize_to_perimeter(x0, y0, r0)
        if has_overlap(x0, y0, r0):
            r0 *= 0.99
            x0, y0, r0 = normalize_to_perimeter(x0, y0, r0)
        
        # Local search with adaptive cooling
        x, y, r = x0.copy(), y0.copy(), r0.copy()
        total_r = np.sum(r)
        t_start = time.time()
        T = 0.1 * total_r
        T_min = 1e-8
        max_iter = 50000
        
        for t in range(max_iter):
            if time.time() - t_start > 15.0:
                break
            frac = t / (max_iter - 1) if max_iter > 1 else 0
            T *= T_min / T * (1.0 - frac) if T > T_min else 0
            
            step_size = 0.1 * np.mean(r) * (1.0 - 0.7 * frac)
            aniso_scale = 0.1 * rng.random() * (1.0 - 0.6 * frac)
            
            move = rng.integers(6)
            x2, y2, r2 = x.copy(), y.copy(), r.copy()
            
            if move == 0:
                i = rng.integers(n)
                dx = rng.normal(scale=step_size)
                dy = rng.normal(scale=step_size)
                x2[i] += dx
                y2[i] += dy
            elif move == 1:
                i = rng.integers(n)
                dr = rng.normal(scale=0.05 * np.mean(r))
                if r2[i] + dr < 1e-8:
                    continue
                r2[i] += dr
            elif move == 2:
                k = rng.integers(2, 4)
                idx = rng.choice(n, k, replace=False)
                dx = rng.normal(scale=step_size)
                dy = rng.normal(scale=step_size)
                x2[idx] += dx
                y2[idx] += dy
            elif move == 3:
                i, j = rng.choice(n, 2, replace=False)
                x2[i], y2[i], r2[i], x2[j], y2[j], r2[j] = x2[j], y2[j], r2[j], x2[i], y2[i], r2[i]
            elif move == 4:
                sx = 1.0 + rng.normal(scale=aniso_scale)
                sy = 1.0 + rng.normal(scale=aniso_scale)
                sx, sy = max(0.6, sx), max(0.6, sy)
                x2 *= sx
                y2 *= sy
            else:
                i, j = rng.choice(n, 2, replace=False)
                dr = rng.normal(scale=0.05 * np.mean(r))
                if r2[i] + dr < 1e-8 or r2[j] - dr < 1e-8:
                    continue
                r2[i] += dr
                r2[j] -= dr
            
            x2, y2, r2 = normalize_to_perimeter(x2, y2, r2)
            if has_overlap(x2, y2, r2) or np.any(r2 < 1e-8):
                continue
            
            val2 = np.sum(r2)
            delta = val2 - total_r
            if delta >= 0 or rng.random() < np.exp(delta / max(1e-12, T)):
                x, y, r = x2, y2, r2
                total_r = val2
                if total_r > best_sum:
                    best_sum = total_r
                    best_circles = np.column_stack([x, y, r])
    
    # Fallback layout
    if best_circles is None:
        x0, y0, r0 = hexagonal_layout([4, 4, 4, 4, 5])
        x0, y0, r0 = normalize_to_perimeter(x0, y0, r0)
        x, y, r = x0.copy(), y0.copy(), r0.copy()
        T = 0.1 * np.sum(r)
        T_min = 1e-8
        for t in range(20000):
            step_size = 0.1 * np.mean(r) * (1.0 - 0.7 * t / 19999)
            move = rng.integers(6)
            x2, y2, r2 = x.copy(), y.copy(), r.copy()
            if move == 0:
                i = rng.integers(n)
                dx = rng.normal(scale=step_size)
                dy = rng.normal(scale=step_size)
                x2[i] += dx
                y2[i] += dy
            elif move == 1:
                i = rng.integers(n)
                dr = rng.normal(scale=0.05 * np.mean(r))
                if r2[i] + dr < 1e-8:
                    continue
                r2[i] += dr
            else:
                continue
            x2, y2, r2 = normalize_to_perimeter(x2, y2, r2)
            if has_overlap(x2, y2, r2) or np.any(r2 < 1e-8):
                continue
            val2 = np.sum(r2)
            delta = val2 - np.sum(r)
            if delta >= 0 or rng.random() < np.exp(delta / max(1e-12, T)):
                x, y, r = x2, y2, r2
                if val2 > best_sum:
                    best_sum = val2
                    best_circles = np.column_stack([x, y, r])
    
    # Final normalization and clamp
    x, y, r = best_circles[:, 0], best_circles[:, 1], best_circles[:, 2]
    x, y, r = normalize_to_perimeter(x, y, r)
    circles = np.column_stack([x, y, r])
    circles[:, 2] = np.clip(circles[:, 2], 0.0, 1.0)
    
    assert circles.shape == (21, 3), "Wrong shape"
    assert np.all(np.isfinite(circles)), "Non-finite values"
    assert np.all(circles[:, 2] >= 0), "Negative radius"
    
    return circles