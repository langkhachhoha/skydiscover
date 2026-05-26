import numpy as np
import time
import heapq

def circle_packing21() -> np.ndarray:
    '''
    Places 21 non-overlapping circles inside a rectangle of
    perimeter 4, maximizing the sum of their radii.

    Returns:
        circles: numpy array of shape (21, 3), where each row
            stores (x, y, radius).
    '''
    n = 21
    rng = np.random.default_rng(20260526)
    start_time = time.time()
    time_limit = 580.0

    def aspect_to_WH(a):
        W = 2.0 * a / (1.0 + a)
        H = 2.0 / (1.0 + a)
        return W, H

    def solve_radii_fixed_centers(xs: np.ndarray, ys: np.ndarray, W: float, H: float, tol: float = 1e-12, max_iter: int = 1000):
        u = np.minimum.reduce([xs, ys, W - xs, H - ys])
        u = np.maximum(u, 0.0)
        r = u.copy()
        dx = xs[:, None] - xs[None, :]
        dy = ys[:, None] - ys[None, :]
        d = np.hypot(dx, dy)
        np.fill_diagonal(d, np.inf)

        for _ in range(max_iter):
            cap_mat = d - r[None, :]
            np.fill_diagonal(cap_mat, np.inf)
            minpair = np.min(cap_mat, axis=1)
            r_new = np.minimum(u, minpair)
            np.maximum(r_new, 0.0, out=r_new)
            if np.max(np.abs(r_new - r)) <= tol:
                r = r_new
                break
            r = r_new
        r = np.maximum(r - 1e-12, 0.0)
        for _ in range(3):
            changed = False
            for i in range(n):
                for j in range(i + 1, n):
                    if d[i, j] < 1e-12:
                        continue
                    s = r[i] + r[j] - d[i, j]
                    if s > 1e-12:
                        if r[i] >= r[j]:
                            r[i] -= min(s, r[i])
                        else:
                            r[j] -= min(s, r[j])
                        changed = True
            if not changed:
                break
        r = np.minimum(r, u)
        r = np.maximum(r, 0.0)
        return r

    def repair_to_feasible(xs, ys, rs, W, H, max_passes=6):
        xs = np.clip(xs, 0.0, W)
        ys = np.clip(ys, 0.0, H)
        edge_dist = np.minimum.reduce([xs, ys, W - xs, H - ys])
        rs = np.minimum(rs, edge_dist)
        rs = np.maximum(rs, 0.0)
        
        for _ in range(max_passes):
            dx = xs[:, None] - xs[None, :]
            dy = ys[:, None] - ys[None, :]
            d = np.hypot(dx, dy) + 1e-12
            overlap = (rs[:, None] + rs[None, :]) - d
            np.fill_diagonal(overlap, -1.0)
            max_ov = np.max(overlap)
            if max_ov <= 1e-12:
                break
            overlap_sum = np.sum(overlap > 0, axis=1)
            shrink = np.zeros(n)
            for i in range(n):
                if overlap_sum[i] > 0:
                    total_ov = np.sum(overlap[i, overlap[i] > 0])
                    shrink[i] = total_ov / overlap_sum[i] * 0.5
            rs = np.maximum(rs - shrink, 0.0)
            edge_dist = np.minimum.reduce([xs, ys, W - xs, H - ys])
            rs = np.minimum(rs, edge_dist)
        return xs, ys, rs

    def compute_violations(xs, ys, rs, W, H):
        n = len(xs)
        violations = []
        for i in range(n):
            for j in range(i + 1, n):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                d = np.hypot(dx, dy)
                s = (rs[i] + rs[j]) - d
                if s > 1e-12:
                    priority = s * (rs[i] + rs[j])
                    heapq.heappush(violations, (-priority, i, j))
        return violations

    def resolve_one_violation(xs, ys, rs, W, H, i, j, priority):
        xi, yi, ri = xs[i], ys[i], rs[i]
        xj, yj, rj = xs[j], ys[j], rs[j]
        dx = xi - xj
        dy = yi - yj
        d = np.hypot(dx, dy)
        if d < 1e-12:
            th = rng.uniform(0.0, 2.0 * np.pi)
            ux, uy = np.cos(th), np.sin(th)
        else:
            ux, uy = dx / d, dy / d
        s = (ri + rj) - d
        if s <= 1e-12:
            return False, xs, ys, rs
        alpha = 0.5 * s
        dix, diy = ux * alpha, uy * alpha
        djx, djy = -ux * alpha, -uy * alpha
        ti = 1.0
        if dx > 1e-15:
            ti = min(ti, (W - ri - xi) / dx)
        elif dx < -1e-15:
            ti = min(ti, (ri - xi) / dx)
        if dy > 1e-15:
            ti = min(ti, (H - ri - yi) / dy)
        elif dy < -1e-15:
            ti = min(ti, (ri - yi) / dy)
        tj = 1.0
        if dx > 1e-15:
            tj = min(tj, (W - rj - xj) / dx)
        elif dx < -1e-15:
            tj = min(tj, (rj - xj) / dx)
        if dy > 1e-15:
            tj = min(tj, (H - rj - yj) / dy)
        elif dy < -1e-15:
            tj = min(tj, (rj - yj) / dy)
        xi += ti * dix
        yi += ti * diy
        xj += tj * djx
        yj += tj * djy
        d_new = np.hypot(xi - xj, yi - yj)
        s_new = (ri + rj) - d_new
        if s_new > 1e-12:
            shrink = 0.5 * s_new
            ri = max(0.0, ri - shrink)
            rj = max(0.0, rj - shrink)
        xs[i], ys[i], rs[i] = xi, yi, ri
        xs[j], ys[j], rs[j] = xj, yj, rj
        return True, xs, ys, rs

    def optimize_for_aspect(W, H, max_restarts=5, max_iters=1500):
        best_sum = -1.0
        best_sol = None
        for _ in range(max_restarts):
            g_cols = int(np.ceil(np.sqrt(n)))
            g_rows = int(np.ceil(n / g_cols))
            space_x = 0.7 * W
            space_y = 0.7 * H
            gx = np.linspace(0.15 * W, 0.85 * W, g_cols)
            gy = np.linspace(0.15 * H, 0.85 * H, g_rows)
            pts = np.array([(x, y) for y in gy for x in gx], dtype=float)[:n]
            jitter_x = rng.uniform(-0.04 * W, 0.04 * W, size=n)
            jitter_y = rng.uniform(-0.04 * H, 0.04 * H, size=n)
            X = pts + np.column_stack([jitter_x, jitter_y])
            X[:, 0] = np.clip(X[:, 0], 0.06 * W, 0.94 * W)
            X[:, 1] = np.clip(X[:, 1], 0.06 * H, 0.94 * H)
            R = np.full(n, 0.02 * min(W, H))

            m_x = np.zeros(n); v_x = np.zeros(n)
            m_y = np.zeros(n); v_y = np.zeros(n)
            m_r = np.zeros(n); v_r = np.zeros(n)
            beta1 = 0.9; beta2 = 0.999; eps = 1e-8
            base_lr = 0.02 * min(W, H)
            t_adam = 0

            stagnation = 0
            best_local_sum = -1.0
            for t in range(1, max_iters + 1):
                if time.time() - start_time > time_limit:
                    break
                xs, ys, rs = X[:, 0], X[:, 1], R
                dx = xs[:, None] - xs[None, :]
                dy = ys[:, None] - ys[None, :]
                d = np.hypot(dx, dy) + 1e-12
                overlap = (rs[:, None] + rs[None, :]) - d
                np.fill_diagonal(overlap, -1.0)
                mask = overlap > 0.0
                g_x = np.zeros(n); g_y = np.zeros(n); g_r = -np.ones(n)
                if np.any(mask):
                    total_overlap = np.sum(overlap[mask])
                    lam_o = 10.0 * (1.0 + 10.0 * (t / max_iters)) * (1.0 + total_overlap)
                    contrib = (2.0 * lam_o) * overlap * mask
                    g_x -= np.sum(contrib * (dx / d), axis=1)
                    g_y -= np.sum(contrib * (dy / d), axis=1)
                    g_r += np.sum(contrib, axis=1)
                
                v_left = rs - xs; v_right = rs + xs - W
                v_bot = rs - ys; v_top = rs + ys - H
                lam_w = 50.0
                for v, dx_sign, dy_sign in ((v_left, -1.0, 0.0), (v_right, 1.0, 0.0),
                                           (v_bot, 0.0, -1.0), (v_top, 0.0, 1.0)):
                    m = v > 0.0
                    if np.any(m):
                        g_r[m] += 2.0 * lam_w * v[m]
                        if dx_sign != 0.0:
                            g_x[m] += 2.0 * lam_w * v[m] * dx_sign
                        if dy_sign != 0.0:
                            g_y[m] += 2.0 * lam_w * v[m] * dy_sign

                t_adam += 1
                m_x = beta1 * m_x + (1 - beta1) * g_x
                v_x = beta2 * v_x + (1 - beta2) * (g_x * g_x)
                mhat_x = m_x / (1 - beta1 ** t_adam)
                vhat_x = v_x / (1 - beta2 ** t_adam)
                lr_x = base_lr * (1.0 / (1.0 + 0.001 * t))
                xs_new = xs - lr_x * mhat_x / (np.sqrt(vhat_x) + eps)
                
                m_y = beta1 * m_y + (1 - beta1) * g_y
                v_y = beta2 * v_y + (1 - beta2) * (g_y * g_y)
                mhat_y = m_y / (1 - beta1 ** t_adam)
                vhat_y = v_y / (1 - beta2 ** t_adam)
                lr_y = base_lr * (1.0 / (1.0 + 0.001 * t))
                ys_new = ys - lr_y * mhat_y / (np.sqrt(vhat_y) + eps)
                
                m_r = beta1 * m_r + (1 - beta1) * g_r
                v_r = beta2 * v_r + (1 - beta2) * (g_r * g_r)
                mhat_r = m_r / (1 - beta1 ** t_adam)
                vhat_r = v_r / (1 - beta2 ** t_adam)
                lr_r = base_lr * 0.8 * (1.0 / (1.0 + 0.001 * t))
                rs_new = rs - lr_r * mhat_r / (np.sqrt(vhat_r) + eps)
                
                xs_new = np.clip(xs_new, 0.0, W)
                ys_new = np.clip(ys_new, 0.0, H)
                rs_new = np.maximum(rs_new, 0.0)
                X[:, 0], X[:, 1], R = xs_new, ys_new, rs_new
                
                if t % 200 == 0:
                    # Replace repair_to_feasible with solve_radii_fixed_centers
                    R = solve_radii_fixed_centers(X[:, 0], X[:, 1], W, H)
                
                current_sum = np.sum(R)
                if current_sum > best_local_sum:
                    best_local_sum = current_sum
                    stagnation = 0
                else:
                    stagnation += 1
                if stagnation >= 100 and t < max_iters:
                    break

            # Final feasibility adjustment
            R = solve_radii_fixed_centers(X[:, 0], X[:, 1], W, H)
            total_radius = np.sum(R)
            if total_radius > best_sum:
                best_sum = total_radius
                best_sol = np.column_stack([X[:, 0], X[:, 1], R])
        return best_sol, best_sum

    best_global = None
    best_score = -1.0
    aspect_candidates = np.concatenate([
        np.linspace(0.7, 1.5, 20),
        np.linspace(0.8, 1.2, 20),
        rng.uniform(0.6, 1.6, size=10)
    ])
    rng.shuffle(aspect_candidates)
    
    for a in aspect_candidates:
        W, H = aspect_to_WH(a)
        sol, score = optimize_for_aspect(W, H)
        if sol is not None and score > best_score:
            best_score = score
            best_global = sol
        if time.time() - start_time > time_limit:
            break

    if best_global is None:
        W = H = 1.0
        xs = np.linspace(0.1, 0.9, 7)
        ys = np.linspace(0.1, 0.9, 3)
        pts = []
        for y in ys:
            for x in xs:
                pts.append((x, y))
        pts = np.array(pts)[:n]
        r = np.full(n, 1e-3)
        return np.column_stack([pts[:, 0], pts[:, 1], r])

    circles = best_global.copy()
    min_x = np.min(circles[:, 0] - circles[:, 2])
    min_y = np.min(circles[:, 1] - circles[:, 2])
    if min_x < 0 or min_y < 0:
        circles[:, 0] -= min_x if min_x < 0 else 0.0
        circles[:, 1] -= min_y if min_y < 0 else 0.0
    bbox = np.array([circles[:, 0] - circles[:, 2], circles[:, 0] + circles[:, 2]]).min(axis=0), np.array([circles[:, 0] - circles[:, 2], circles[:, 0] + circles[:, 2]]).max(axis=0)
    bbox_y = np.array([circles[:, 1] - circles[:, 2], circles[:, 1] + circles[:, 2]]).min(axis=0), np.array([circles[:, 1] - circles[:, 2], circles[:, 1] + circles[:, 2]]).max(axis=0)
    width = bbox[1][0] - bbox[0][0]
    height = bbox[1][1] - bbox[0][1]
    perimeter = 2 * (width + height)
    if perimeter > 2.0:
        scale = 2.0 / perimeter
        diameter = circles[:, 2] * 2
        new_diameter = diameter * scale
        circles[:, 2] = new_diameter / 2
    circles[:, 2] = np.maximum(circles[:, 2] - 1e-12, 0.0)
    return circles