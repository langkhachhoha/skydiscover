import numpy as np
import time 

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    '''
    Construct a packing of 26 circles in the unit square.

    Returns:
        centers: numpy array of shape (26, 2)
        radii: numpy array of shape (26,)
        sum_radii: objective value (sum of radii)
    '''
    rng = np.random.default_rng(0)
    n = 26

    def init_centers_radii(n):
        # Construct 5 rows with counts summing to 26: [5,5,5,5,6]
        row_counts = [5, 5, 5, 5, 6]
        assert sum(row_counts) == n
        rows = len(row_counts)
        ys = np.linspace(1.0/(rows+1), rows/(rows+1), rows)
        centers = []
        for ri, m in enumerate(row_counts):
            xs = np.linspace(1.0/(m+1), m/(m+1), m)
            # small stagger to reduce alignment
            if ri % 2 == 1 and m > 1:
                xs += (xs[1]-xs[0]) * 0.3
                xs = np.clip(xs, 1e-3, 1-1e-3)
            for x in xs:
                centers.append([x, ys[ri]])
        centers = np.array(centers, dtype=float)
        # jitter slightly
        centers += rng.normal(scale=1e-3, size=centers.shape)
        centers = np.clip(centers, 1e-3, 1-1e-3)
        # small initial radii strictly feasible
        r = np.full(n, 1e-3, dtype=float)
        # ensure feasibility with tiny margin
        r = np.minimum(r, np.minimum.reduce([centers[:,0], 1-centers[:,0], centers[:,1], 1-centers[:,1]]) - 1e-6)
        r = np.maximum(r, 1e-4)
        return centers, r

    # Precompute pair indices for i<j
    idx_i, idx_j = np.triu_indices(n, k=1)

    def compute_slacks(c, r):
        # Boundary slacks per circle
        s_left  = c[:,0] - r
        s_right = 1.0 - c[:,0] - r
        s_bottom= c[:,1] - r
        s_top   = 1.0 - c[:,1] - r
        # Pairwise slacks for i<j
        diff = c[idx_i] - c[idx_j]
        dist = np.sqrt(np.sum(diff*diff, axis=1))
        s_pair = dist - (r[idx_i] + r[idx_j])
        return (s_left, s_right, s_bottom, s_top), s_pair

    def feasible(c, r, tol=1e-12):
        (s1, s2, s3, s4), s_pair = compute_slacks(c, r)
        smin = min(np.min(s1), np.min(s2), np.min(s3), np.min(s4), np.min(s_pair))
        return bool(smin > tol), float(smin)

    def barrier_objective_and_grad(c, r, lam):
        # Returns objective (sum r + lam*sum log(slacks)), gradients wrt c and r, and min slack
        (s1, s2, s3, s4), s_pair = compute_slacks(c, r)
        # Feasibility required for logs
        smin = min(np.min(s1), np.min(s2), np.min(s3), np.min(s4), np.min(s_pair))
        if not np.isfinite(smin) or smin <= 0:
            return -np.inf, None, None, smin

        # Objective
        phi = np.sum(r) + lam * (np.sum(np.log(s1)) + np.sum(np.log(s2)) + np.sum(np.log(s3)) + np.sum(np.log(s4)) + np.sum(np.log(s_pair)))

        # Gradients initialization
        gc = np.zeros_like(c)
        gr = np.ones_like(r)  # from sum r

        # Boundary contributions
        inv_s1 = lam / s1
        inv_s2 = lam / s2
        inv_s3 = lam / s3
        inv_s4 = lam / s4

        gc[:,0] += inv_s1
        gr      -= inv_s1

        gc[:,0] -= inv_s2
        gr      -= inv_s2

        gc[:,1] += inv_s3
        gr      -= inv_s3

        gc[:,1] -= inv_s4
        gr      -= inv_s4

        # Pairwise contributions
        diff = c[idx_i] - c[idx_j]              # (P,2)
        dist = np.sqrt(np.sum(diff*diff, axis=1))  # (P,)
        # Avoid division by zero in gradient; dist==0 shouldn't happen if feasible with positive s_pair, but guard anyway
        dist = np.maximum(dist, 1e-15)
        inv_sp = lam / s_pair                   # (P,)
        # direction factors
        dir_fac = (inv_sp / dist)[:, None] * diff  # (P,2)

        # scatter-add to centers
        np.add.at(gc, idx_i, dir_fac)
        np.add.at(gc, idx_j, -dir_fac)
        # radii grads
        np.add.at(gr, idx_i, -inv_sp)
        np.add.at(gr, idx_j, -inv_sp)

        return phi, gc, gr, smin

    def adam_ascent(c, r, time_budget=5.0):
        start = time.time()
        # Adam parameters
        lr = 0.02
        beta1 = 0.9
        beta2 = 0.999
        eps_adam = 1e-8

        m = np.zeros((n,3))
        v = np.zeros((n,3))
        t_adam = 0

        lam0 = 0.05  # initial barrier weight
        t_bar = 1.0

        best = (np.copy(c), np.copy(r), -np.inf)

        accept_count = 0
        reject_streak = 0

        max_steps = 4000
        collapse_guard = 0.1
        phi, _, _, smin = barrier_objective_and_grad(c, r, lam0/t_bar)
        if not np.isfinite(phi):
            # Ensure feasibility by shrinking radii
            r = np.minimum(r, np.minimum.reduce([c[:,0], 1-c[:,0], c[:,1], 1-c[:,1]]) * 0.5)
            phi, _, _, smin = barrier_objective_and_grad(c, r, lam0/t_bar)
        for it in range(max_steps):
            if time.time() - start > time_budget:
                break
            lam = lam0 / t_bar
            phi, gc, gr, smin = barrier_objective_and_grad(c, r, lam)
            if not np.isfinite(phi):
                break

            # Flatten gradients into (n,3): [gx, gy, gr]
            g = np.stack([gc[:,0], gc[:,1], gr], axis=1)

            # Gradient clipping
            g_norm = np.linalg.norm(g)
            if g_norm > 100.0:
                g *= (100.0 / (g_norm + 1e-12))

            # Adam update
            t_adam += 1
            m = beta1 * m + (1 - beta1) * g
            v = beta2 * v + (1 - beta2) * (g * g)
            m_hat = m / (1 - beta1**t_adam)
            v_hat = v / (1 - beta2**t_adam)

            step_dir = lr * m_hat / (np.sqrt(v_hat) + eps_adam)

            # Line search
            step_scale = 1.0
            phi_curr = phi
            min_slack_curr = smin
            accepted = False
            for _ in range(25):
                dc = step_scale * step_dir[:, :2]
                dr = step_scale * step_dir[:, 2]

                c_new = c + dc
                r_new = r + dr

                # Keep trivial non-negativity and inside bounds softly by clipping radii lower bound
                r_new = np.maximum(r_new, 1e-12)

                # Feasibility guard: require slacks positive and not collapsing too fast
                feas, smin_new = feasible(c_new, r_new, tol=1e-12)
                if feas and smin_new >= max(1e-12, collapse_guard * min_slack_curr):
                    lam_new = lam0 / (t_bar * 1.0)
                    phi_new, _, _, _ = barrier_objective_and_grad(c_new, r_new, lam_new)
                    if np.isfinite(phi_new) and (phi_new > phi_curr):
                        accepted = True
                        break
                step_scale *= 0.5

            if accepted:
                c, r = c_new, r_new
                accept_count += 1
                reject_streak = 0
                # Slowly reduce barrier influence to approach constraints
                t_bar *= 1.01
                # mild learning-rate growth
                if accept_count % 50 == 0:
                    lr = min(lr * 1.05, 0.05)
                # Track best by true sum of radii (feasible state only)
                sum_r = float(np.sum(r))
                if sum_r > best[2]:
                    best = (np.copy(c), np.copy(r), sum_r)
            else:
                reject_streak += 1
                lr = max(lr * 0.7, 0.002)
                if reject_streak > 50:
                    # Tiny random nudge to centers to escape flatness, keep feasibility by immediate tiny shrink of radii
                    jitter = rng.normal(scale=1e-4, size=c.shape)
                    c = np.clip(c + jitter, 1e-6, 1 - 1e-6)
                    # slight shrink to preserve non-overlap
                    r *= 0.999
                    reject_streak = 0

        return best[0], best[1]

    def maximize_radii_given_centers(c, r_init, sweeps=80, margin=1e-12):
        # Compute boundary caps
        b = np.minimum.reduce([c[:,0], 1.0 - c[:,0], c[:,1], 1.0 - c[:,1]])
        b = np.maximum(b - margin, 0.0)
        # Pairwise distances matrix with inf on diagonal
        diff = c[:, None, :] - c[None, :, :]
        D = np.sqrt(np.sum(diff*diff, axis=2))
        np.fill_diagonal(D, np.inf)

        r = np.clip(r_init, 0.0, b)
        # Gauss-Seidel coordinate ascent on r_i = min(b_i, min_j D_ij - r_j)
        nloc = len(r)
        for _ in range(sweeps):
            changed = False
            for i in range(nloc):
                caps = D[i, :] - r
                caps[i] = np.inf  # ignore self
                cap = min(b[i], float(np.min(caps)))
                cap = max(0.0, cap - margin)
                if abs(cap - r[i]) > 1e-12:
                    r[i] = cap
                    changed = True
            if not changed:
                break
        return r

    # Build initial state
    centers, radii = init_centers_radii(n)

    # Short feasibility repair just in case
    ok, smin0 = feasible(centers, radii)
    if not ok:
        radii *= 0.5

    # Interior-point Adam ascent within a time budget
    c_opt, r_opt = adam_ascent(centers, radii, time_budget=8.0)

    # Final radii-only LP polishing with fixed centers
    r_polish = maximize_radii_given_centers(c_opt, r_opt, sweeps=120, margin=1e-12)

    # Ensure strict feasibility by an ultimate tiny shave if needed
    # Verify constraints, and if any tiny negative slack appears, reduce the larger radius in each violating pair
    def final_repair(c, r):
        (s1, s2, s3, s4), s_pair = compute_slacks(c, r)
        minb = min(np.min(s1), np.min(s2), np.min(s3), np.min(s4))
        if minb <= 0:
            # shave all a bit
            r = np.maximum(r - (1e-12 - minb), 0.0)
        # pairwise repair
        diff = c[idx_i] - c[idx_j]
        dist = np.sqrt(np.sum(diff*diff, axis=1))
        sp = dist - (r[idx_i] + r[idx_j])
        for k, s in enumerate(sp):
            if s <= 0:
                i = idx_i[k]; j = idx_j[k]
                delta = (-s) + 1e-12
                if r[i] >= r[j]:
                    r[i] = max(0.0, r[i] - delta)
                else:
                    r[j] = max(0.0, r[j] - delta)
        return r

    r_final = final_repair(c_opt, r_polish)

    # One more quick polish after repair
    r_final = maximize_radii_given_centers(c_opt, r_final, sweeps=60, margin=1e-12)

    sum_radii = float(np.sum(r_final))
    return c_opt.astype(float), r_final.astype(float), sum_radii

if __name__ == "__main__":
    centers, radii, sum_r = run_packing()
    print("centers shape:", centers.shape)
    print("radii shape:", radii.shape)
    print("sum_radii:", sum_r)