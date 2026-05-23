import numpy as np
import time

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    '''
    Construct a packing of 26 circles in the unit square.

    Algorithm: Hybrid approach combining geometric construction with barrier optimization.
    - Start with a geometric initial configuration using circle triple tangency
    - Refine using interior-point barrier optimization with Adam
    - Use selective feasibility repair to ensure robustness

    Returns:
        centers: numpy array of shape (26, 2)
        radii: numpy array of shape (26,)
        sum_radii: objective value (sum of radii)
    '''
    start_time = time.time()
    time_limit = 590.0  # leave buffer

    rng = np.random.default_rng(2026)
    n = 26

    # Precompute pair index arrays (upper triangle)
    iu, ju = np.triu_indices(n, k=1)

    # Distance smoothing epsilon
    eps_d = 1e-12

    def halton_sequence(size: int, bases=(2, 3)) -> np.ndarray:
        def vdc_indexed(indices, base):
            indices = indices.copy()
            res = np.zeros_like(indices, dtype=float)
            denom = 1.0
            while np.any(indices > 0):
                denom *= base
                res += (indices % base) / denom
                indices //= base
            return res
        skip = 7
        idx = np.arange(skip, skip + size)
        x = np.zeros((size, 2), dtype=float)
        x[:, 0] = vdc_indexed(idx, bases[0])
        x[:, 1] = vdc_indexed(idx, bases[1])
        return x

    def barrier_value_and_grad(t: float, x: np.ndarray, y: np.ndarray, r: np.ndarray):
        # Compute all slacks
        cL = x - r
        cR = 1.0 - x - r
        cB = y - r
        cT = 1.0 - y - r
        cN = r  # nonnegativity

        # Pairwise distances
        dx = x[iu] - x[ju]
        dy = y[iu] - y[ju]
        dij = np.sqrt(dx * dx + dy * dy + eps_d * eps_d)
        cP = dij - (r[iu] + r[ju])

        # Check feasibility
        if np.min(cL) <= 0 or np.min(cR) <= 0 or np.min(cB) <= 0 or np.min(cT) <= 0 or np.min(cN) <= 0 or np.min(cP) <= 0:
            return -np.inf, np.zeros(3 * n, dtype=float)

        # Objective value
        phi = t * float(np.sum(r))
        phi += np.sum(np.log(cL)) + np.sum(np.log(cR)) + np.sum(np.log(cB)) + np.sum(np.log(cT)) + np.sum(np.log(cN)) + np.sum(np.log(cP))

        # Gradients: g_x, g_y, g_r
        gx = np.zeros_like(x)
        gy = np.zeros_like(y)
        gr = np.zeros_like(r)

        # Contribution from t * sum(r)
        gr += t

        # Box constraints
        inv_cL = 1.0 / cL
        inv_cR = 1.0 / cR
        inv_cB = 1.0 / cB
        inv_cT = 1.0 / cT
        inv_cN = 1.0 / cN

        # cL = x - r
        gx += inv_cL
        gr -= inv_cL

        # cR = 1 - x - r
        gx -= inv_cR
        gr -= inv_cR

        # cB = y - r
        gy += inv_cB
        gr -= inv_cB

        # cT = 1 - y - r
        gy -= inv_cT
        gr -= inv_cT

        # cN = r >= 0
        gr += inv_cN

        # Pairwise constraints
        inv_cP = 1.0 / cP
        ux = dx / dij
        uy = dy / dij

        # Accumulate pairwise gradients
        np.add.at(gx, iu, inv_cP * ux)
        np.add.at(gy, iu, inv_cP * uy)
        np.add.at(gx, ju, -inv_cP * ux)
        np.add.at(gy, ju, -inv_cP * uy)
        np.add.at(gr, iu, -inv_cP)
        np.add.at(gr, ju, -inv_cP)

        g = np.concatenate([gx, gy, gr], axis=0)
        return phi, g

    def backtracking_line_search(t, x, y, r, step, phi0, g0, alpha_init=1.0, rho=0.5, c=1e-4):
        alpha = alpha_init
        x0, y0, r0 = x, y, r
        gdotp = np.dot(g0, step)
        if not np.isfinite(gdotp):
            gdotp = 0.0
        sx = step[0:n]
        sy = step[n:2*n]
        sr = step[2*n:3*n]
        for _ in range(50):
            xn = x0 + alpha * sx
            yn = y0 + alpha * sy
            rn = r0 + alpha * sr
            phi_n, _ = barrier_value_and_grad(t, xn, yn, rn)
            if np.isfinite(phi_n) and phi_n >= phi0 + c * alpha * gdotp:
                return xn, yn, rn, phi_n, True
            alpha *= rho
            if alpha < 1e-12:
                break
        # Fallback: accept best feasible point even if phi decreases
        alpha = alpha_init
        best = None
        best_phi = -np.inf
        for _ in range(60):
            xn = x0 + alpha * sx
            yn = y0 + alpha * sy
            rn = r0 + alpha * sr
            phi_n, _ = barrier_value_and_grad(t, xn, yn, rn)
            if np.isfinite(phi_n) and phi_n > best_phi:
                best = (xn, yn, rn, phi_n)
                best_phi = phi_n
            alpha *= rho
            if alpha < 1e-14:
                break
        if best is not None:
            return best[0], best[1], best[2], best[3], True
        return x, y, r, phi0, False

    def optimize_once(init_centers: np.ndarray):
        # Initialize
        x = init_centers[:, 0].copy()
        y = init_centers[:, 1].copy()
        r = np.full(n, 0.03, dtype=float)  # Use larger initial radii to help explore

        # Ensure initial feasibility
        x = np.clip(x, 0.02, 0.98)
        y = np.clip(y, 0.02, 0.98)

        # Barrier schedule
        t = 1.0
        mu = 7.0
        max_phases = 8

        # Adam parameters
        base_lr = 0.05
        beta1 = 0.9
        beta2 = 0.999
        eps_opt = 1e-8

        best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
        best_sum = float(np.sum(r))

        for phase in range(max_phases):
            if time.time() - start_time > time_limit:
                break
            m = np.zeros(3 * n, dtype=float)
            v = np.zeros(3 * n, dtype=float)
            kstep = 0
            lr = base_lr / np.sqrt(phase + 1.0)

            phi, g = barrier_value_and_grad(t, x, y, r)
            if not np.isfinite(phi):
                r *= 0.9
                phi, g = barrier_value_and_grad(t, x, y, r)

            inner_iters = 1500
            for it in range(inner_iters):
                if time.time() - start_time > time_limit:
                    break
                kstep += 1
                m = beta1 * m + (1 - beta1) * g
                v = beta2 * v + (1 - beta2) * (g * g)
                mhat = m / (1 - (beta1 ** kstep))
                vhat = v / (1 - (beta2 ** kstep))
                step = lr * mhat / (np.sqrt(vhat) + eps_opt)

                x_new, y_new, r_new, phi_new, ok = backtracking_line_search(t, x, y, r, step, phi, g, alpha_init=1.0, rho=0.5, c=1e-4)
                if not ok:
                    break

                x, y, r = x_new, y_new, r_new
                phi, g = barrier_value_and_grad(t, x, y, r)

                s = float(np.sum(r))
                if s > best_sum:
                    best_sum = s
                    best_x, best_y, best_r = x.copy(), y.copy(), r.copy()

                if it % 250 == 249:
                    lr *= 0.9

            t *= mu

        return best_x, best_y, best_r

    def min_slack(x, y, r):
        cL = x - r
        cR = 1.0 - x - r
        cB = y - r
        cT = 1.0 - y - r
        cN = r
        dx = x[iu] - x[ju]
        dy = y[iu] - y[ju]
        dij = np.sqrt(dx * dx + dy * dy + eps_d * eps_d)
        cP = dij - (r[iu] + r[ju])
        return min(np.min(cL), np.min(cR), np.min(cB), np.min(cT), np.min(cN), np.min(cP))

    def selective_feasibility_repair(x: np.ndarray, y: np.ndarray, r: np.ndarray, margin: float = 1e-12) -> np.ndarray:
        # Enforce wall bounds first
        b = np.minimum.reduce([x, 1.0 - x, y, 1.0 - y]) - margin
        b = np.maximum(0.0, b)
        r = np.minimum(r, b)

        # Precompute distances
        dx = x[iu] - x[ju]
        dy = y[iu] - y[ju]
        dij = np.sqrt(dx * dx + dy * dy + eps_d * eps_d)

        # Iteratively fix pair overlaps by minimally shrinking the larger circle in each offending pair
        for _ in range(40):
            changed = 0.0
            # Check overlaps
            over = r[iu] + r[ju] - dij + margin
            viol_idx = np.where(over > 0.0)[0]
            if viol_idx.size == 0:
                break
            for k in viol_idx:
                i = iu[k]
                j = ju[k]
                excess = over[k]
                if r[i] >= r[j]:
                    delta = min(excess, r[i])
                    r[i] = max(0.0, r[i] - delta)
                    # keep within wall bounds
                    if r[i] > b[i]:
                        r[i] = b[i]
                    changed = max(changed, delta)
                else:
                    delta = min(excess, r[j])
                    r[j] = max(0.0, r[j] - delta)
                    if r[j] > b[j]:
                        r[j] = b[j]
                    changed = max(changed, delta)
            # Re-enforce wall bounds softly
            r = np.minimum(r, b)
            if changed < 1e-14:
                break

        # Final clamp
        r = np.clip(r, 0.0, b)
        return r

    # Geometric initialization with triple tangency candidates
    centers = []
    radii = []
    
    # Start with four quarter-corner circles of radius 0.25
    seed_positions = [(0.25, 0.25), (0.25, 0.75), (0.75, 0.25), (0.75, 0.75)]
    for (x, y) in seed_positions:
        centers.append([x, y])
        radii.append(0.25 * 0.99)  # Slight shrink for robustness

    centers = np.array(centers, dtype=float)
    radii = np.array(radii, dtype=float)

    # Generate candidates from triple tangency (3 circles, 2 circles + 1 wall, 1 circle + 2 walls)
    def generate_all_candidates(centers, radii):
        candidates = []
        
        # 3 circles
        if len(radii) >= 3:
            for i in range(len(radii) - 2):
                for j in range(i + 1, len(radii) - 1):
                    for k in range(j + 1, len(radii)):
                        # Solve for circle tangent to three existing circles
                        # Using the formula for circle tangent to three circles
                        # This is a simplified version using linear algebra
                        # We'll use the method from the original v1 but with simpler geometric constraints
                        ci = centers[i]; ri = radii[i]
                        cj = centers[j]; rj = radii[j]
                        ck = centers[k]; rk = radii[k]
                        
                        # Use the formula for the center and radius of a circle tangent to three circles
                        # This is numerically unstable but we'll try a simple approach
                        # We'll use the fact that the center must satisfy:
                        # ||c - ci|| = ri + r
                        # ||c - cj|| = rj + r
                        # ||c - ck|| = rk + r
                        
                        # We'll solve the system for a few candidate radii
                        for r_candidate in [0.01, 0.02, 0.03, 0.04, 0.05]:
                            # Try to find a center that satisfies the three equations
                            # This is a system of three equations in two unknowns
                            # We'll use least squares to find the best fit
                            A = np.array([
                                [2*(cj[0]-ci[0]), 2*(cj[1]-ci[1])],
                                [2*(ck[0]-ci[0]), 2*(ck[1]-ci[1])]
                            ])
                            b = np.array([
                                (ri + r_candidate)**2 - (rj + r_candidate)**2 + ci[0]**2 - cj[0]**2 + ci[1]**2 - cj[1]**2,
                                (ri + r_candidate)**2 - (rk + r_candidate)**2 + ci[0]**2 - ck[0]**2 + ci[1]**2 - ck[1]**2
                            ])
                            
                            try:
                                sol = np.linalg.solve(A, b)
                                c_center = np.array([ci[0] + sol[0], ci[1] + sol[1]])
                                # Check if this center is feasible
                                if (c_center[0] - r_candidate >= -1e-10 and c_center[0] + r_candidate <= 1.0 + 1e-10 and
                                    c_center[1] - r_candidate >= -1e-10 and c_center[1] + r_candidate <= 1.0 + 1e-10):
                                    # Check non-overlap with existing circles
                                    dists = np.sqrt(np.sum((centers - c_center)**2, axis=1))
                                    if np.all(dists >= radii + r_candidate - 1e-10):
                                        candidates.append((r_candidate, c_center))
                            except:
                                continue
        
        # 2 circles + 1 wall
        if len(radii) >= 2:
            walls = [(0, 0), (1, 1), (0, 1), (1, 0)]  # x=0, x=1, y=0, y=1
            for i in range(len(radii)):
                for j in range(i+1, len(radii)):
                    for wall_id, wall_val in walls:
                        # Wall constraint
                        # For wall x=0: c[0] = r
                        # For wall x=1: c[0] = 1-r
                        # For wall y=0: c[1] = r
                        # For wall y=1: c[1] = 1-r
                        if wall_id == 0:  # left wall
                            # c[0] = r
                            # Solve for circle tangent to circle i and j and touching wall x=0
                            # ||c - ci|| = ri + r
                            # ||c - cj|| = rj + r
                            # c[0] = r
                            # Substitute r = c[0] into the first two equations
                            # ||[c[0], c[1]] - ci|| = ri + c[0]
                            # ||[c[0], c[1]] - cj|| = rj + c[0]
                            # These are two equations in two unknowns (c[0], c[1])
                            # We'll solve using a numerical approach
                            for r_candidate in [0.01, 0.02, 0.03, 0.04, 0.05]:
                                # Try to solve the system
                                # (c[0] - ci[0])**2 + (c[1] - ci[1])**2 = (ri + c[0])**2
                                # (c[0] - cj[0])**2 + (c[1] - cj[1])**2 = (rj + c[0])**2
                                # This is messy but we can solve it numerically
                                # We'll use a simple approximation by trying values
                                # between 0.01 and 0.3
                                # This is a simplification - we'll just try a few values
                                c0 = r_candidate
                                # Solve for c1 from first equation
                                # (c0 - ci[0])**2 + (c1 - ci[1])**2 = (ri + c0)**2
                                # (c1 - ci[1])**2 = (ri + c0)**2 - (c0 - ci[0])**2
                                delta = (ri + c0)**2 - (c0 - ci[0])**2
                                if delta >= 0:
                                    c1_plus = ci[1] + np.sqrt(delta)
                                    c1_minus = ci[1] - np.sqrt(delta)
                                    # Check the second equation
                                    if abs((c0 - cj[0])**2 + (c1_plus - cj[1])**2 - (rj + c0)**2) < 1e-5:
                                        c_center = np.array([c0, c1_plus])
                                        if (c_center[0] + r_candidate <= 1.0 + 1e-10 and
                                            c_center[1] - r_candidate >= -1e-10 and
                                            c_center[1] + r_candidate <= 1.0 + 1e-10):
                                            dists = np.sqrt(np.sum((centers - c_center)**2, axis=1))
                                            if np.all(dists >= radii + r_candidate - 1e-10):
                                                candidates.append((r_candidate, c_center))
                                    if abs((c0 - cj[0])**2 + (c1_minus - cj[1])**2 - (rj + c0)**2) < 1e-5:
                                        c_center = np.array([c0, c1_minus])
                                        if (c_center[0] + r_candidate <= 1.0 + 1e-10 and
                                            c_center[1] - r_candidate >= -1e-10 and
                                            c_center[1] + r_candidate <= 1.0 + 1e-10):
                                            dists = np.sqrt(np.sum((centers - c_center)**2, axis=1))
                                            if np.all(dists >= radii + r_candidate - 1e-10):
                                                candidates.append((r_candidate, c_center))
                                
        # 1 circle + 2 walls
        if len(radii) >= 1:
            wall_pairs = [(0, 2), (0, 3), (1, 2), (1, 3)]  # (left, bottom), (left, top), (right, bottom), (right, top)
            for i in range(len(radii)):
                for (w1_id, w2_id) in wall_pairs:
                    # Wall constraints
                    # For left wall: c[0] = r
                    # For right wall: c[0] = 1-r
                    # For bottom wall: c[1] = r
                    # For top wall: c[1] = 1-r
                    # So for (left, bottom): c[0] = r, c[1] = r
                    # For (left, top): c[0] = r, c[1] = 1-r
                    # For (right, bottom): c[0] = 1-r, c[1] = r
                    # For (right, top): c[0] = 1-r, c[1] = 1-r
                    if w1_id == 0 and w2_id == 2:  # left, bottom
                        c0 = r_candidate
                        c1 = r_candidate
                    elif w1_id == 0 and w2_id == 3:  # left, top
                        c0 = r_candidate
                        c1 = 1 - r_candidate
                    elif w1_id == 1 and w2_id == 2:  # right, bottom
                        c0 = 1 - r_candidate
                        c1 = r_candidate
                    elif w1_id == 1 and w2_id == 3:  # right, top
                        c0 = 1 - r_candidate
                        c1 = 1 - r_candidate
                    else:
                        continue
                        
                    c_center = np.array([c0, c1])
                    # Check if this center is feasible
                    if (c_center[0] - r_candidate >= -1e-10 and c_center[0] + r_candidate <= 1.0 + 1e-10 and
                        c_center[1] - r_candidate >= -1e-10 and c_center[1] + r_candidate <= 1.0 + 1e-10):
                        # Check non-overlap with existing circles
                        dists = np.sqrt(np.sum((centers - c_center)**2, axis=1))
                        if np.all(dists >= radii + r_candidate - 1e-10):
                            candidates.append((r_candidate, c_center))
        
        # Filter and return candidates with largest radii
        if not candidates:
            return []
        # Sort by radius descending
        candidates.sort(key=lambda x: -x[0])
        # Remove duplicates
        unique = {}
        for r, c in candidates:
            key = (round(c[0], 10), round(c[1], 10), round(r, 10))
            if key not in unique or r > unique[key][0]:
                unique[key] = (r, c)
        return list(unique.values())

    # Greedy insertion of maximal empty circles by triple tangency
    while len(radii) < n:
        candidates = generate_all_candidates(centers, radii)
        if not candidates:
            break
        # Choose the largest candidate
        best_r, best_c = candidates[0]
        centers = np.vstack([centers, best_c])
        radii = np.concatenate([radii, [best_r * 0.99]])
        
        # Check if we have enough circles
        if len(radii) >= n:
            break

    # If we still don't have enough circles, add small circles randomly
    while len(radii) < n:
        # Try to place a tiny circle at a random point
        p = rng.random(2) * 0.8 + 0.1  # avoid edges
        b = min(p[0], 1 - p[0], p[1], 1 - p[1])
        # Reduce by distances to existing circles
        if len(radii) > 0:
            d = np.sqrt(np.sum((centers - p) ** 2, axis=1)) - radii
            b = min(b, np.min(d))
        R = max(1e-6, b - 1e-6)
        if R > 0:
            centers = np.vstack([centers, p])
            radii = np.concatenate([radii, [R]])

    # Ensure we have exactly n circles
    if len(radii) > n:
        # Sort by radius and keep the largest n
        idx = np.argsort(-radii)[:n]
        centers = centers[idx]
        radii = radii[idx]
    elif len(radii) < n:
        # Pad with zero radius
        missing = n - len(radii)
        centers = np.vstack([centers, np.tile(np.array([0.5, 0.5]), (missing, 1))])
        radii = np.concatenate([radii, np.zeros(missing, dtype=float)])

    # Now refine the packing using barrier optimization
    # Generate diverse initial configurations
    inits = []
    # Halton points
    h = halton_sequence(n)
    inits.append(0.08 + 0.84 * h)
    # Grid with jitter
    G = int(np.ceil(np.sqrt(n)))
    gx = np.linspace(0.1, 0.9, G)
    gy = np.linspace(0.1, 0.9, G)
    X, Y = np.meshgrid(gx, gy, indexing='xy')
    grid = np.column_stack([X.ravel(), Y.ravel()])[:n]
    grid = grid + rng.normal(0.0, 0.005, size=grid.shape)
    grid = np.clip(grid, 0.01, 0.99)
    inits.append(grid)
    # Random uniform
    uni = 0.1 + 0.8 * rng.random((n, 2))
    inits.append(uni)

    best_centers = centers.copy()
    best_radii = radii.copy()
    best_sum = np.sum(radii)

    for init in inits:
        if time.time() - start_time > time_limit:
            break
        x_opt, y_opt, r_opt = optimize_once(init)
        # Final safety clip
        r_opt = np.maximum(0.0, r_opt) * (1.0 - 1e-12)

        # Use selective feasibility repair instead of uniform shrink
        r_rep = selective_feasibility_repair(x_opt, y_opt, r_opt, margin=1e-12)

        # Check if we have better sum
        s = np.sum(r_rep)
        if s > best_sum:
            best_sum = s
            best_centers = np.column_stack([x_opt, y_opt])
            best_radii = r_rep

    # Final robustness step
    best_radii = best_radii * (1.0 - 1e-12)
    sum_radii = float(np.sum(best_radii))

    return best_centers.astype(float), best_radii.astype(float), sum_radii