from typing import Callable, Optional

import gudhi as gd
import numpy as np
from joblib import Parallel, delayed
from sklearn.neighbors import NearestNeighbors
from tqdm.auto import tqdm


def ela_difficulty(
    sample_once: Callable[[], list[float]],
    loss_value: Callable[[np.ndarray], float],
    N_max: int = 1024,
    max_pairs: int = 1024,
    compute_hessian: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    n_curvature_points: int = 128,
    curvature_dims: Optional[int] = None,
    topology_k: int = 64,
    bounds: Optional[tuple[np.typing.ArrayLike, np.typing.ArrayLike]] = None,
    seed: Optional[int] = None,
    verbose: bool = True,
    return_features: bool = True,
    n_jobs: int = -1,
):
    """Compute ELA difficulty scores based on:
    - convexity tests
    - finite-difference curvature tests
    This function performs global sampling only once and reuses:
    - sampled points
    - sampled loss values
    - y_scale
    - parameter scale
    - random generator

    Args:
        sample_once: Function returning one parameter vector theta.
        loss_value: Function returning scalar loss at theta.
        N: Number of global samples.
        max_pairs: Maximum number of pairs for convexity test.
        n_curvature_points: Number of points used for curvature estimation.
        curvature_dims: Number of dimensions used for curvature.
            If ``None``, all dimensions are used.
        epsilon: Relative finite-difference step.
            For angles, h = epsilon * angle_period.
        bounds: Optional bounds (lower, upper).
            If ``None``, bounds are inferred from global samples.
        seed: Random seed.
        verbose: Print progress and diagnostics.
        return_features: If True, returns detailed features.

    Returns:
        scores, (and features If return_features is ``True``)
    """

    rng = np.random.default_rng(seed)

    # ============================================================
    # Helpers
    # ============================================================

    # def safe_eval(theta: np.ndarray):
    #     try:
    #         y = loss_value(theta)
    #         if np.isfinite(y):
    #             return float(y)
    #         return np.nan
    #     except Exception:
    #         return np.nan

    # # ============================================================
    # # 0) Global sampling, shared by convexity and curvature
    # # ============================================================

    N_min = 512
    batch_size = 256
    rel_tol = 0.02
    patience = 2

    thetas = []
    ys = []

    previous_y_scale = None
    stable_count = 0

    pbar = tqdm(total=N_max, desc="Global sampling", disable=not verbose, leave=False)

    while len(ys) < N_max:
        # sample one batch
        # for _ in range(batch_size):
        #     if len(ys) >= N_max:
        #         break

        #     theta = np.asarray(sample_once(), dtype=float)
        #     y = safe_eval(theta)

        #     if np.isfinite(y) and np.all(np.isfinite(theta)):
        #         thetas.append(theta)
        #         ys.append(y)

        #     pbar.update(1)

        remaining = N_max - len(ys)
        current_batch_size = min(batch_size, remaining)

        theta_batch = [
            np.asarray(sample_once(), dtype=float) for _ in range(current_batch_size)
        ]

        y_batch = Parallel(n_jobs=n_jobs)(
            delayed(loss_value)(theta) for theta in theta_batch
        )

        for theta, y in zip(theta_batch, y_batch):
            if len(ys) >= N_max:
                break

            assert y is not None
            if np.isfinite(y) and np.all(np.isfinite(theta)):
                thetas.append(theta)
                ys.append(y)

        pbar.update(current_batch_size)

        # for res in results:
        #     if len(ys) >= N_max:
        #         break

        #     if res is not None:
        #         theta, y = res
        #         thetas.append(theta)
        #         ys.append(y)

        # Need enough samples before checking stability
        if len(ys) < N_min:
            continue

        ys_tmp = np.asarray(ys, dtype=float)

        q10, q90 = np.percentile(ys_tmp, [10, 90])
        current_y_scale = max(q90 - q10, 1e-12)

        if previous_y_scale is not None:
            relative_change = abs(current_y_scale - previous_y_scale) / max(
                previous_y_scale, 1e-12
            )

            if verbose:
                tqdm.write(
                    f"[ELA] N={len(ys):d}, y_scale={current_y_scale:.3e}, "
                    f"relative_change={relative_change:.3e}"
                )

            if relative_change < rel_tol:
                stable_count += 1
            else:
                stable_count = 0

            if stable_count >= patience:
                if verbose:
                    tqdm.write(
                        f"[ELA] Stopping global sampling early at N={len(ys)} "
                        f"because y_scale stabilized."
                    )
                break

        previous_y_scale = current_y_scale

    pbar.close()

    thetas = np.asarray(thetas, dtype=float)
    ys = np.asarray(ys, dtype=float)

    if len(ys) < 5:
        raise RuntimeError(
            f"Not enough finite samples to estimate y_scale. "
            f"Only {len(ys)} finite values."
        )

    dim = thetas.shape[1]

    # ============================================================
    # 1) Shared robust output scale
    # ============================================================

    q10, q90 = np.percentile(ys, [10, 90])
    iqr = q90 - q10

    y_scale = max(iqr, 1e-12)
    # y_norm = (ys - np.min(ys)) / y_scale

    if verbose:
        print(f"[ELA] y_scale = {y_scale:.3e}")

    # ============================================================
    # 2) Shared bounds / parameter scale
    # ============================================================

    if bounds is not None:
        lower, upper = bounds
        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)

        if lower.shape != (dim,) or upper.shape != (dim,):
            raise ValueError("bounds must be a tuple (lower, upper) of shape (dim,)")

        span = upper - lower

        if np.any(span <= 0):
            raise ValueError("All bounds must satisfy upper > lower")

    else:
        lower = np.min(thetas, axis=0)
        upper = np.max(thetas, axis=0)
        span = upper - lower
        span = np.maximum(span, 1e-12)

    typical_param_scale = float(np.mean(span))

    if verbose:
        print(f"[ELA] typical_param_scale = {typical_param_scale:.3e}")

    # ============================================================
    # 3) Convexity difficulty
    # ============================================================

    n_pairs = min(max_pairs, len(thetas) * (len(thetas) - 1) // 2)

    if n_pairs <= 0:
        raise RuntimeError("Not enough samples to compute convexity pairs")

    # convex_gaps = []
    # indices = np.arange(len(thetas))

    # for _ in tqdm(range(n_pairs), desc="Convexity test", disable=not verbose):
    #     i, j = rng.choice(indices, size=2, replace=False)

    #     a = thetas[i]
    #     b = thetas[j]

    #     ya = ys[i]
    #     yb = ys[j]

    #     alpha = rng.uniform(0, 1)

    #     # Linear interpolation in parameter space.
    #     # For angles this is a simple interpolation, not geodesic on the circle.
    #     m = alpha * a + (1.0 - alpha) * b

    #     ym = safe_eval(m)

    #     if np.isfinite(ym):
    #         gap = ym - (alpha * ya + (1.0 - alpha) * yb)
    #         convex_gaps.append(gap)

    convex_gaps = []
    indices = np.arange(len(thetas))

    pairs = []
    alphas = []
    ms = []
    linear_values = []

    for _ in tqdm(range(n_pairs), desc="Convexity sampling", disable=not verbose):
        i, j = rng.choice(indices, size=2, replace=False)

        a = thetas[i]
        b = thetas[j]

        ya = ys[i]
        yb = ys[j]

        alpha = rng.uniform(0, 1)

        # Linear interpolation in parameter space.
        # For angles this is a simple interpolation, not geodesic on the circle.
        m = alpha * a + (1.0 - alpha) * b

        pairs.append((i, j))
        alphas.append(alpha)
        ms.append(m)
        linear_values.append(alpha * ya + (1.0 - alpha) * yb)

    ym_values = Parallel(n_jobs=n_jobs)(
        delayed(loss_value)(m)
        for m in tqdm(ms, desc="Convexity eval", disable=not verbose)
    )

    for ym, linear_value in zip(ym_values, linear_values):
        assert ym is not None
        if np.isfinite(ym):
            convex_gaps.append(ym - linear_value)

    convex_gaps = np.asarray(convex_gaps, dtype=float)

    if len(convex_gaps) == 0:
        raise RuntimeError("No valid convexity evaluations")

    normalized_gaps = convex_gaps / y_scale

    eps = 1e-4  # on tolère une violation jusqu’à 0.01% de l’échelle typique de la loss

    convex_satisfied_fraction = float(np.mean(normalized_gaps <= eps))
    convex_violation_fraction = float(np.mean(normalized_gaps > eps))

    mean_gap = float(np.mean(convex_gaps))
    mean_gap_norm = float(np.mean(normalized_gaps))
    median_gap = float(np.median(convex_gaps))
    median_gap_norm = float(np.median(normalized_gaps))
    q90_gap_norm = float(np.quantile(normalized_gaps, 0.9))

    convexity_features = {
        "n_convexity_pairs": int(len(convex_gaps)),
        "convex_satisfied_fraction": convex_satisfied_fraction,
        "convex_violation_fraction": convex_violation_fraction,
        "mean_gap": mean_gap,
        "mean_gap_norm": mean_gap_norm,
        "median_gap": median_gap,
        "median_gap_norm": median_gap_norm,
        "q90_gap_norm": q90_gap_norm,
    }

    # ============================================================
    # 4) Curvature difficulty via Hessian eigenvalues
    # ============================================================

    if compute_hessian is None:
        raise ValueError(
            "compute_hessian must be provided for Hessian-based curvature metrics."
        )

    if curvature_dims is None:
        curvature_dim_indices = np.arange(dim)
    else:
        curvature_dims = min(int(curvature_dims), dim)
        curvature_dim_indices = rng.choice(dim, size=curvature_dims, replace=False)

    curvature_dim_indices = np.asarray(curvature_dim_indices, dtype=int)

    curvature_points = thetas[
        rng.choice(len(thetas), size=n_curvature_points, replace=True)
    ]

    # Natural Hessian scale: loss scale / parameter scale^2
    curvature_scale = y_scale / max(typical_param_scale**2, 1e-12)

    hessian_condition_numbers = []
    normalized_spectral_radii = []
    negative_eigenvalue_fractions = []

    eps = 1e-12
    condition_number_cap = 1e12
    curvature_norm = max(curvature_scale, eps)

    # for theta in tqdm(
    #     curvature_points,
    #     desc="Hessian curvature test",
    #     disable=not verbose,
    # ):
    #     theta = np.asarray(theta, dtype=float)

    #     try:
    #         H_sub = compute_hessian(theta, curvature_dim_indices)
    #         eigvals = np.linalg.eigvalsh(H_sub)
    #     except Exception:
    #         continue

    #     eigvals = np.asarray(eigvals, dtype=float)

    H_list = Parallel(
        n_jobs=n_jobs,
        batch_size=1,  # pyright: ignore[reportArgumentType]
        pre_dispatch="1*n_jobs",
    )(
        delayed(compute_hessian)(np.asarray(theta, dtype=float), curvature_dim_indices)
        for theta in tqdm(
            curvature_points,
            desc="Hessian curvature test",
            disable=not verbose,
        )
    )

    eigvals_list = []

    for H_sub in H_list:
        try:
            assert H_sub is not None
            eigvals = np.linalg.eigvalsh(H_sub)
        except Exception:
            continue

        eigvals = np.asarray(eigvals, dtype=float)

        if not np.all(np.isfinite(eigvals)):
            continue

        eigvals_list.append(eigvals)

    for eigvals in eigvals_list:

        abs_eigvals = np.abs(eigvals)

        max_abs_lambda = float(np.max(abs_eigvals))
        min_abs_lambda = float(np.min(abs_eigvals))

        eig_tol = 1e-6 * max(max_abs_lambda, 1.0)

        # ------------------------------------------------------------
        # Metric 1: Hessian condition number
        # ------------------------------------------------------------
        if min_abs_lambda > eig_tol:
            condition_number = max_abs_lambda / min_abs_lambda
        else:
            condition_number = condition_number_cap

        hessian_condition_numbers.append(min(condition_number, condition_number_cap))

        # ------------------------------------------------------------
        # Metric 2: normalized spectral radius
        # ------------------------------------------------------------
        normalized_spectral_radii.append(max_abs_lambda / curvature_norm)

        # ------------------------------------------------------------
        # Metric 3: negative eigenvalue fraction
        # ------------------------------------------------------------
        negative_eigenvalue_fractions.append(float(np.mean(eigvals < -eig_tol)))

    hessian_condition_numbers = np.asarray(hessian_condition_numbers, dtype=float)

    normalized_spectral_radii = np.asarray(normalized_spectral_radii, dtype=float)

    negative_eigenvalue_fractions = np.asarray(
        negative_eigenvalue_fractions, dtype=float
    )

    def summary_stats(x: np.ndarray, prefix: str):
        q = np.quantile(x, [0.5, 1.0])

        return {
            f"{prefix}_median": float(q[0]),
            f"{prefix}_max": float(q[1]),
        }

    curvature_features = {}

    for values, name in [
        (hessian_condition_numbers, "hessian_condition_number"),
        (normalized_spectral_radii, "normalized_hessian_spectral_radius"),
        (negative_eigenvalue_fractions, "negative_eigenvalue_fraction"),
    ]:
        curvature_features.update(summary_stats(values, name))

    # ============================================================
    # 5) Topological data analysis
    # ============================================================

    # 1) Build kNN graph

    if verbose:
        print("Computing topology features...")

    n = len(ys)
    topology_k = min(int(topology_k), n - 1)

    if topology_k < 1:
        raise RuntimeError("topology_k must be at least 1")

    nn = NearestNeighbors(
        n_neighbors=topology_k + 1,
        algorithm="auto",
        metric="euclidean",
        n_jobs=n_jobs,
    )

    X = thetas
    nn.fit(X)

    neighbors_raw = nn.kneighbors(X, return_distance=False)

    # On retire le point lui-même de sa liste de voisins
    neighbors = np.empty((n, topology_k), dtype=int)

    for i in range(n):
        row = neighbors_raw[i]
        row = row[row != i]
        neighbors[i] = row[:topology_k]

    # 2) Persistent homology with GUDHI

    st = gd.SimplexTree()

    # Vertices: each sample appears at filtration value y_norm[i]
    for i in range(n):
        st.insert([int(i)], filtration=float(ys[i]))

    # Edges: kNN graph edges appear when both endpoints are active,
    # so filtration = max(y_norm[i], y_norm[j])
    for i in range(n):
        for j in neighbors[i]:
            j = int(j)

            if i == j:
                continue

            filt = float(max(ys[i], ys[j]))
            st.insert([int(i), int(j)], filtration=filt)

    st.make_filtration_non_decreasing()
    st.persistence()

    intervals = st.persistence_intervals_in_dimension(0)

    # 3) Component lifetimes

    ymax = float(np.max(ys))

    births = []
    deaths = []
    lifetimes = []

    for birth, death in intervals:
        birth = float(birth)

        if np.isfinite(death):
            death_clipped = float(death)
        else:
            death_clipped = ymax

        births.append(birth)
        deaths.append(death_clipped)
        lifetimes.append(death_clipped - birth)

    births = np.asarray(births, dtype=float)
    deaths = np.asarray(deaths, dtype=float)
    lifetimes = np.asarray(lifetimes, dtype=float)

    # normalized_lifetimes = lifetimes / y_range

    # 4) Approximate component counts over filtration

    components_alive_over_time = []

    for t in np.sort(ys):
        alive = np.sum((births <= t) & (deaths > t))
        components_alive_over_time.append(int(alive))

    components_alive_over_time = np.asarray(components_alive_over_time, dtype=int)

    n_components_created = int(len(intervals))
    n_merges = int(np.sum(np.isfinite(intervals[:, 1]))) if len(intervals) else 0

    max_components_alive = (
        int(np.max(components_alive_over_time))
        if len(components_alive_over_time)
        else 0
    )

    mean_components_alive = (
        float(np.mean(components_alive_over_time))
        if len(components_alive_over_time)
        else np.nan
    )

    median_components_alive = (
        float(np.median(components_alive_over_time))
        if len(components_alive_over_time)
        else np.nan
    )

    # 5) Features

    topology_features = {
        "topology_k": int(topology_k),
        "n_components_created": int(n_components_created),
        "n_merges": int(n_merges),
        "max_components_alive": int(max_components_alive),
        "mean_components_alive": float(mean_components_alive),
        "median_components_alive": float(median_components_alive),
        "mean_component_lifetime": (
            float(np.mean(lifetimes)) if len(lifetimes) else np.nan
        ),
        "median_component_lifetime": (
            float(np.median(lifetimes)) if len(lifetimes) else np.nan
        ),
        "max_component_lifetime": (
            float(np.max(lifetimes)) if len(lifetimes) else np.nan
        ),
    }

    # ============================================================
    # 6) Information content
    # ============================================================

    def finite_differences_from_walk(thetas, ys, walk_indices=None, eps_norm=1e-15):
        """
        Compute Delta C_i normalized along a walk.
        Delta C_i = (y_{i+1} - y_i) / ||theta_{i+1} - theta_i||
        """

        thetas = np.asarray(thetas, dtype=float)
        ys = np.asarray(ys, dtype=float)

        if walk_indices is None:
            walk_indices = np.arange(len(ys))
        else:
            walk_indices = np.asarray(walk_indices)

        theta_walk = thetas[walk_indices]
        y_walk = ys[walk_indices]

        dtheta = theta_walk[1:] - theta_walk[:-1]
        dy = y_walk[1:] - y_walk[:-1]

        norms = np.linalg.norm(dtheta, axis=1)

        valid = norms > eps_norm

        deltas = dy[valid] / norms[valid]

        return deltas

    def symbolize_deltas(deltas, epsilon):

        deltas = np.asarray(deltas)

        symbols = np.zeros(len(deltas), dtype=int)
        symbols[deltas < -epsilon] = -1
        symbols[deltas > epsilon] = +1

        return symbols

    def information_content_from_symbols(symbols):

        from collections import Counter

        symbols = np.asarray(symbols)

        if len(symbols) < 2:
            return 0.0, {}

        pairs = list(zip(symbols[:-1], symbols[1:]))

        # On garde seulement les transitions a != b
        diff_pairs = [(a, b) for a, b in pairs if a != b]

        total_pairs = len(pairs)

        if total_pairs == 0:
            return 0.0, {}

        counts = Counter(diff_pairs)

        H = 0.0
        probs = {}

        for pair, count in counts.items():
            p = count / total_pairs
            probs[pair] = p

            if p > 0:
                H += -p * (np.log(p) / np.log(6))

        return H, probs

    def compute_H_curve(
        thetas,
        ys,
        n_eps=200,
        eps_min=None,
        eps_max=None,
        walk_indices=None,
        random_walk=True,
        seed=None,
    ):

        thetas = np.asarray(thetas, dtype=float)
        ys = np.asarray(ys, dtype=float)

        N = len(ys)

        if walk_indices is None:
            if random_walk:
                rng = np.random.default_rng(seed)
                walk_indices = rng.permutation(N)
            else:
                walk_indices = np.arange(N)

        deltas = finite_differences_from_walk(thetas, ys, walk_indices=walk_indices)

        abs_deltas = np.abs(deltas)
        abs_deltas = abs_deltas[np.isfinite(abs_deltas)]

        if len(abs_deltas) == 0:
            raise ValueError("No Delta C valid")

        if eps_max is None:
            eps_max = np.max(abs_deltas)

        if eps_min is None:
            positive = abs_deltas[abs_deltas > 0]
            if len(positive) > 0:
                eps_min = max(np.min(positive) * 0.1, 1e-15)
            else:
                eps_min = 1e-15

        if eps_max <= eps_min:
            eps_max = eps_min * 10

        epsilons = np.logspace(np.log10(eps_min), np.log10(eps_max), n_eps)

        H_values = []

        for eps in epsilons:
            symbols = symbolize_deltas(deltas, eps)
            H, _ = information_content_from_symbols(symbols)
            H_values.append(H)

        return epsilons, np.asarray(H_values), deltas

    epsilons, H_values, deltas = compute_H_curve(
        thetas,
        ys,
        n_eps=n_eps,
        random_walk=True,
        seed=123,
    )

    idx_max = np.argmax(H_values)
    epsilon_max = epsilons[idx_max]
    H_max = H_values[idx_max]

    infocontent_features = {"epsilon_max": epsilon_max, "H_max": H_max}

    # ============================================================
    # 5) Combined features
    # ============================================================

    features = {
        "global": {
            "n_valid_samples": int(len(ys)),
            "dim": int(dim),
            "y_scale": float(y_scale),
            "typical_param_scale": float(typical_param_scale),
            "bounds_lower": lower,
            "bounds_upper": upper,
        },
        "convexity": convexity_features,
        "curvature": curvature_features,
        "topology": topology_features,
        "infocontent_features": infocontent_features,
    }

    # ============================================================
    # Print results
    # ============================================================

    if verbose:
        print("")
        print("=" * 60)
        print("[ELA] Summary")
        print("-" * 60)
        print("[ELA] Convexity")
        print("-" * 60)
        print(f"[ELA] convex_violation_fraction = {convex_violation_fraction:.3f}")
        print("      Fraction of sampled segments that violate convexity.")
        print(f"[ELA] mean_gap_norm = {mean_gap_norm:.3f}")
        print("      Mean of normalized gap values.")
        print(f"[ELA] mean_gap = {mean_gap:.3f}")
        print("      Mean of gap values.")
        print("")
        print(f"[ELA] median_gap_norm = {median_gap_norm:.3f}")
        print("      Median of normalized gap values.")
        print(f"[ELA] median_gap = {median_gap:.3f}")
        print("      Median of gap values.")
        print("")
        print("-" * 60)
        print("[ELA] Curvature")
        print("-" * 60)

        print("[ELA] Hessian condition number")
        print(
            f"      median = {curvature_features['hessian_condition_number_median']:.3e}"
        )
        print(
            f"      max    = {curvature_features['hessian_condition_number_max']:.3e}"
        )
        print(
            "      Ratio lambda_max / lambda_min for locally positive definite Hessians."
        )
        print(
            "      Higher values indicate ill-conditioning and narrow curved valleys."
        )
        print("")

        print("[ELA] Normalized Hessian spectral radius")
        print(
            f"      median = {curvature_features['normalized_hessian_spectral_radius_median']:.3e}"
        )
        print(
            f"      max    = {curvature_features['normalized_hessian_spectral_radius_max']:.3e}"
        )
        print(
            "      Largest absolute Hessian eigenvalue, normalized by the global curvature scale."
        )
        print("      Higher values indicate strong local curvature.")
        print("")

        print("[ELA] Negative eigenvalue fraction")
        print(
            f"      median = {curvature_features['negative_eigenvalue_fraction_median']:.3f}"
        )
        print(
            f"      max    = {curvature_features['negative_eigenvalue_fraction_max']:.3f}"
        )
        print("      Fraction of Hessian eigenvalues that are significantly negative.")
        print("      Higher values indicate stronger local non-convexity.")
        print("")

        print("=" * 60)
        print("[ELA] Topological data analysis")
        print("-" * 60)

        print("[ELA] Components alive (connected components)")
        print(f"      mean   = {topology_features['mean_components_alive']:.3f}")
        print(f"      median = {topology_features['median_components_alive']:.3f}")
        print("      Number of connected components alive across filtration levels.")
        print(
            "      Higher values indicate more fragmented landscapes with many local minima."
        )
        print("")

        print("[ELA] Component lifetimes")
        print(f"      mean   = {topology_features['mean_component_lifetime']:.3e}")
        print(f"      median = {topology_features['median_component_lifetime']:.3e}")
        print("      Persistence of connected components.")
        print("      Larger values indicate more pronounced and stable basins.")
        print("")

        print("=" * 60)
        print("[ELA] Information content")
        print("-" * 60)
        print(f"      epsilon_max   = {infocontent_features['epsilon_max']:.3f}")
        print(f"      H_max = {infocontent_features['H_max']:.3f}")
        print("      Desc incoming")

        print("")

    if return_features:
        return features

    return None
