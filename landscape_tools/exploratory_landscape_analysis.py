from typing import Callable, Optional

import numpy as np
from joblib import Parallel, delayed
from tqdm.auto import tqdm


@dataclass
class SamplingConfig:
    N_min: int = 256
    batch_size: int = 256
    N_max: int = 2048
    rel_tol = 0.02
    patience = 2


def ela_difficulty(
    sample_once: Callable[[], list[float]],
    loss_value: Callable[[np.ndarray], float],
    sampling: SamplingConfig | None = None,
    max_pairs: int = 1024,
    compute_hessian: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    n_curvature_points: int = 128,
    curvature_dims: Optional[int] = None,
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

    def safe_eval(theta: np.ndarray):
        try:
            y = loss_value(theta)
            if np.isfinite(y):
                return float(y)
            return np.nan
        except Exception:
            return np.nan

    # # ============================================================
    # # 0) Global sampling, shared by convexity and curvature
    # # ============================================================

    if sampling is None:
        sampling = SamplingConfig()

    N_min = sampling.N_min
    N_max = sampling.N_max
    batch_size = sampling.batch_size
    rel_tol = sampling.rel_tol
    patience = sampling.patience

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
            delayed(safe_eval)(theta) for theta in theta_batch
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
        delayed(safe_eval)(m)
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

    def five_number_summary(x: np.ndarray, prefix: str):
        q = np.quantile(x, [0.0, 0.25, 0.5, 0.75, 1.0])

        return {
            f"{prefix}_median": float(q[2]),
            f"{prefix}_max": float(q[4]),
        }

    curvature_features = {}

    for values, name in [
        (hessian_condition_numbers, "hessian_condition_number"),
        (normalized_spectral_radii, "normalized_hessian_spectral_radius"),
        (negative_eigenvalue_fractions, "negative_eigenvalue_fraction"),
    ]:
        curvature_features.update(five_number_summary(values, name))

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

    if return_features:
        return features

    return None
