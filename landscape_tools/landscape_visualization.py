import re
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    NotRequired,
    Optional,
    TypeAlias,
    TypedDict,
)

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm, Normalize
from matplotlib.figure import Figure
from numpy.typing import NDArray
from qiskit import QuantumCircuit
from qiskit.circuit import Clbit, Instruction, Qubit
from sklearn.decomposition import PCA
from tqdm.auto import tqdm

ArrayLike: TypeAlias = NDArray[np.float64]
ParameterVector: TypeAlias = NDArray[np.float64]
LossFunction: TypeAlias = Callable[[ParameterVector], float]


class _QiskitCircuitInstructionProtocol:
    """This is needed because Qiskit dynamic type generation make the access to
    relevant types a hellscape."""

    operation: Instruction
    qubits: tuple[Qubit, ...]
    clbits: tuple[Clbit, ...]


def loss_scan_1d(
    params: ParameterVector,
    direction: ArrayLike,
    loss_function: LossFunction,
    n_steps: int = 200,
    end_points: tuple[float, float] | None = None,
    n_jobs: int = -1,
) -> Figure:
    """Evaluate and plot the loss function along a one-dimensional direction in
    parameter space.

    Parameters:
        params: Reference parameter vector.
        direction: Direction in parameter space used for the scan. It is
            normalized by its maximum absolute value.
        loss_function: Callable returning the loss for a given parameter vector.
        n_steps: Number of scan points.
        end_points: Bounds of the scan parameter. Default is (-π, π).
        n_jobs: Number of parallel jobs used to evaluate the loss.

    Side effects:
        Saves the figure to ``figures/landscape1d.pdf`` and displays it.
    """

    # -------------------- Initialization --------------------

    # Default scan interval
    if end_points is None:
        end_points = (-np.pi, np.pi)

    param0 = params.copy()

    # Copy direction vector
    d = np.asarray(direction).copy()

    # -------------------- Direction normalization --------------------

    max_abs = np.max(np.abs(d))
    if max_abs > 0:
        d = d / max_abs

    # -------------------- Scan grid setup --------------------

    t_vals = np.linspace(*end_points, n_steps)
    l = np.zeros_like(t_vals)

    # -------------------- Loss evaluation loop --------------------

    # for k, t in enumerate(tqdm(t_vals, desc="1D scan progression")):

    #     param = param0.copy()
    #     param = param0 + t * d

    #     L[k] = loss_function(param)

    def evaluate_loss(t: float):
        param = param0 + t * d
        return loss_function(param)

    l = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_loss)(t) for t in tqdm(t_vals, desc="1D scan progression")
    )

    l = np.asarray(l)

    # -------------------- Visualization --------------------

    # plt.figure(figsize=(9, 6))
    # plt.plot(t_vals, l, "-o", ms=3, color="lightgreen")
    # plt.yscale("log")
    # plt.xlabel("t")
    # plt.ylabel("Loss")
    # plt.title("1D Loss Landscape")
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig("figures/landscape1d.pdf")
    # plt.show()

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(t_vals, l, "-o", ms=3, color="lightgreen")
    ax.set_yscale("log")
    ax.set_xlabel("t")
    ax.set_ylabel("Loss")
    ax.set_title("1D Loss Landscape")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig("figures/landscape1d.pdf")

    return fig


def loss_scan_2d_3d(
    params: ParameterVector,
    direction1: ArrayLike,
    direction2: ArrayLike,
    loss_function: LossFunction,
    n_steps: int = 100,
    end_points_x: tuple[float, float] | None = None,
    end_points_y: tuple[float, float] | None = None,
    plot3D: bool = True,
    n_jobs: int = -1,
) -> None:
    """Evaluate and plot the loss function on a two-dimensional parameter plane.

    Parameters:
        params: Reference parameter vector.
        direction1: First scan direction in parameter space. It is normalized by
            its maximum absolute value.
        direction2: Second scan direction in parameter space. It is normalized
            by its maximum absolute value.
        loss_function: Callable returning the loss for a given parameter vector.
        n_steps: Number of points per scan direction. Default is 100.
        end_points_x: Bounds of the scan parameter along the first direction.
            Default is (-π, π).
        end_points_y: Bounds of the scan parameter along the second direction.
            Default is (-π, π).
        plot3D: Whether to generate a 3D visualization of the loss surface.
        n_jobs: Number of parallel jobs used to evaluate the loss.

    Side effects:
        Saves the 2D figure to ``figures/landscape2d.pdf`` and, if requested,
        the 3D figure to ``figures/landscape3d.pdf``. Displays the generated
        figures.
    """

    # -------------------- Initialization --------------------

    if end_points_x is None:
        end_points_x = (-np.pi, np.pi)

    if end_points_y is None:
        end_points_y = (-np.pi, np.pi)

    param0 = params.copy()

    d1 = np.asarray(direction1).copy()
    d2 = np.asarray(direction2).copy()

    # -------------------- Direction normalization --------------------

    max1 = np.max(np.abs(d1))
    if max1 > 0:
        d1 /= max1

    max2 = np.max(np.abs(d2))
    if max2 > 0:
        d2 /= max2

    # -------------------- Scan grid setup --------------------

    t1_vals = np.linspace(*end_points_x, n_steps)
    t2_vals = np.linspace(*end_points_y, n_steps)

    T1, T2 = np.meshgrid(t1_vals, t2_vals, indexing="ij")
    l = np.zeros_like(T1)

    # -------------------- Loss evaluation loop --------------------

    # for i in tqdm(range(n_steps), desc="2D scan progression"):
    #     for j in tqdm(range(n_steps), leave=False):

    #         param = param0.copy()

    #         param = param0 + T1[i, j] * d1 + T2[i, j] * d2

    #         L[i, j] = loss_function(param)

    # -------------------- Parallel loss evaluation --------------------

    def evaluate_row(i: int) -> np.ndarray:
        row = np.empty(n_steps)

        for j in range(n_steps):
            param = param0 + T1[i, j] * d1 + T2[i, j] * d2
            row[j] = loss_function(param)

        return row

    rows = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_row)(i)
        for i in tqdm(range(n_steps), desc="2D scan progression")
    )

    l = np.vstack(rows)  # pyright: ignore[reportCallIssue, reportArgumentType]

    # -------------------- 2D visualization --------------------

    fig, ax = plt.subplots(figsize=(9, 6))

    cp1 = ax.contourf(T1, T2, l, levels=50, cmap="viridis")

    ax.set_title("2D Loss Landscape")
    ax.set_xlabel("t1")
    ax.set_ylabel("t2")
    fig.colorbar(cp1, ax=ax, label="Loss value")

    plt.tight_layout()
    plt.savefig("figures/landscape2d.pdf")
    plt.show()

    # -------------------- 3D visualization --------------------

    if plot3D:

        fig = plt.figure(figsize=(9, 6))
        norm = colors.LogNorm(
            vmin=np.percentile(l[l > 0], 1), vmax=np.percentile(l, 99)
        )

        ax1 = fig.add_subplot(111, projection="3d")
        ax1.set_box_aspect([1, 1, 1])

        _ = ax1.plot_surface(
            T1,
            T2,
            l,
            facecolors=plt.cm.viridis(  # pyright: ignore[reportAttributeAccessIssue]
                norm(l)
            ),
            cmap="viridis",
            linewidth=0,
            antialiased=True,
            shade=False,
            alpha=0.8,
        )

        ax1.set_title("3D Loss Landscape")
        ax1.set_xlabel("t1")
        ax1.set_ylabel("t2")
        ax1.set_zlabel("Loss value")

        mappable = plt.cm.ScalarMappable(norm=norm, cmap="viridis")
        mappable.set_array(l)
        fig.colorbar(mappable, ax=ax1, shrink=0.5, aspect=10)

        plt.tight_layout()
        plt.savefig("figures/landscape3d.pdf")
        plt.show()


class PCAResult(TypedDict):
    X: np.ndarray
    """PCA grid X coordinates"""
    Y: np.ndarray
    """PCA grid Y coordinates"""
    Z: np.ndarray
    """Loss values on the PCA grid"""
    traj_xy: np.ndarray
    """Trajectory projected onto the PCA plane"""
    traj_z: Optional[np.ndarray]
    """Loss values along the trajectory if compute_traj_loss is True, otherwise None"""
    param_history: np.ndarray
    """Parameter trajectory used to fit the PCA"""
    param0: float
    """Reference parameter vector used as the center of the scan"""
    pca: PCA
    """Fitted PCA object"""
    explained_variance_ratio: np.ndarray
    """Variance explained by each PCA component"""
    components: np.ndarray
    """PCA component vectors"""
    x_range: tuple[float, float]
    """Scan bounds along the PCA X axis"""
    y_range: tuple[float, float]
    """Scan bounds along the PCA Y axis"""


def pca_loss_scan(
    params_history: np.ndarray,
    loss_function: LossFunction,
    n_steps: int = 200,
    offset: float | tuple[float, float] = 0.5,
    compute_traj_loss: bool = True,
    n_jobs: int = -1,
    backend: str = "loky",
) -> PCAResult:
    """Evaluate the loss function in the PCA subspace of an optimization trajectory.

    Parameters:
        params_history: Array containing the successive parameter vectors of the
            optimization trajectory.
        loss_function: Callable returning the loss for a given parameter vector.
        n_steps: Number of grid points per PCA axis.
        offset: Margin added to the PCA scan bounds. If a scalar is provided, it
            is converted to ``(-abs(offset), abs(offset))``.
        compute_traj_loss: Whether to evaluate the loss along the original trajectory.
        n_jobs: Number of parallel jobs used to evaluate the loss.
        backend: Joblib backend used for parallel loss evaluations.

    Returns:
        The PCA result
    """

    # -------------------- Load optimization records --------------------

    param_history = np.asarray(params_history)

    # -------------------- PCA computation --------------------

    # Fit PCA on the optimization trajectory to extract main directions
    # These directions capture the dominant variations during training
    pca = PCA(n_components=2)
    pca.fit(param_history)

    pc1 = pca.components_[0]
    pc2 = pca.components_[1]

    # -------------------- Reference point and projection --------------------

    # Use the final parameter vector as the center of the scan
    flat_center = param_history[-1].copy()
    param0: float = flat_center.copy()

    # Project the full trajectory onto the PCA plane
    centered_history = param_history - flat_center[None, :]
    traj_x = centered_history @ pc1
    traj_y = centered_history @ pc2
    traj_xy = np.column_stack([traj_x, traj_y])

    # -------------------- Scan range definition --------------------

    if not isinstance(offset, tuple):
        offset = (-abs(offset), abs(offset))

    # Define scan boundaries by extending trajectory range
    x_min = float(np.min(traj_x) + offset[0])
    x_max = float(np.max(traj_x) + offset[1])
    y_min = float(np.min(traj_y) + offset[0])
    y_max = float(np.max(traj_y) + offset[1])

    # -------------------- Grid construction --------------------

    xs = np.linspace(x_min, x_max, n_steps)
    ys = np.linspace(y_min, y_max, n_steps)
    X, Y = np.meshgrid(xs, ys, indexing="xy")

    # Z = np.empty_like(X, dtype=float)

    # -------------------- Loss evaluation loop on PCA plane --------------------

    # for iy in tqdm(range(n_steps), desc="Scan in PCA directions"):
    #     for ix in tqdm(range(n_steps), leave=False):
    #         a = X[iy, ix]
    #         b = Y[iy, ix]

    #         probe_flat = flat_center + a * pc1 + b * pc2
    #         probe = probe_flat.copy()

    #         Z[iy, ix] = float(loss_function(probe))

    # -------------------- Parallel loss evaluation on PCA plane --------------------

    def evaluate_pca_row(iy: int) -> np.ndarray:
        row = np.empty(n_steps, dtype=float)

        for ix in range(n_steps):
            a = X[iy, ix]
            b = Y[iy, ix]

            probe = flat_center + a * pc1 + b * pc2
            row[ix] = float(loss_function(probe))

        return row

    rows = Parallel(n_jobs=n_jobs, backend=backend)(
        delayed(evaluate_pca_row)(iy)
        for iy in tqdm(range(n_steps), desc="Scan in PCA directions")
    )

    Z = np.vstack(rows)  # pyright: ignore[reportArgumentType, reportCallIssue]

    # -------------------- Trajectory loss evaluation --------------------

    # traj_z = None
    # if compute_traj_loss:
    #     traj_z = np.array([float(loss_function(p)) for p in param_history])

    traj_z = None
    if compute_traj_loss:

        traj_z = Parallel(n_jobs=n_jobs, backend=backend)(
            delayed(loss_function)(p)
            for p in tqdm(param_history, desc="Trajectory loss")
        )

        traj_z = np.asarray(traj_z, dtype=float)

    # -------------------- Return results --------------------

    return {
        "X": X,
        "Y": Y,
        "Z": Z,
        "traj_xy": traj_xy,
        "traj_z": traj_z,
        "param_history": param_history,
        "param0": param0,
        "pca": pca,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "components": pca.components_,
        "x_range": (x_min, x_max),
        "y_range": (y_min, y_max),
    }


def plot_pca_loss_scan_2d(
    scan_result: PCAResult,
    ax: Axes | None = None,
    cmap: str = "viridis",
    contour: bool = False,
    contour_levels: int = 20,
    show_colorbar: bool = True,
    trajectory_kwargs: dict[str, Any] | None = None,
):
    """
    Plot a 2D PCA loss landscape with the optimization trajectory.

    Parameters:
        scan_result: Mapping returned by ``pca_loss_scan``.
        ax: Axis on which to draw the plot.
        cmap: Colormap used for the loss surface.
        contour: Whether to overlay contour lines. Default is False.
        contour_levels: Number of contour levels. Default is 20.
        show_colorbar: Whether to display a colorbar. Default is True.
        trajectory_kwargs: Optional keyword arguments for trajectory plotting.

    Notes:
        - The figure is saved to
          ``figures/pcaLandscape2d.pdf``.
        - If ``LogNorm()`` is used for normalization,
          the values in ``Z`` must be strictly positive.

    Returns:
        fig: Figure containing the plot.
        ax: Axis on which the plot is drawn.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 6))
    else:
        fig = ax.figure

    X = scan_result["X"]
    Y = scan_result["Y"]
    Z = scan_result["Z"]
    traj_xy = scan_result["traj_xy"]
    evr = scan_result["explained_variance_ratio"]

    pcm = ax.pcolormesh(X, Y, Z, shading="auto", cmap=cmap, norm=LogNorm())

    if contour:
        ax.contour(
            X, Y, Z, levels=contour_levels, colors="k", linewidths=0.6, alpha=0.45
        )

    if show_colorbar:
        fig.colorbar(pcm, ax=ax, label="Loss value")

    if trajectory_kwargs is None:
        trajectory_kwargs = {}

    default_traj_kwargs = dict(color="white", lw=2.0, marker="o", ms=3, alpha=0.95)
    default_traj_kwargs.update(trajectory_kwargs)

    ax.plot(
        traj_xy[:, 0],
        traj_xy[:, 1],
        label="Trajectory",
        **default_traj_kwargs,  # pyright: ignore[reportArgumentType]
    )
    ax.scatter(
        traj_xy[0, 0],
        traj_xy[0, 1],
        c="cyan",
        s=50,
        label="Start",
        edgecolors="k",
        zorder=5,
    )
    ax.scatter(
        traj_xy[-1, 0],
        traj_xy[-1, 1],
        c="red",
        s=60,
        label="End",
        edgecolors="k",
        zorder=5,
    )

    ax.set_xlabel(f"PC1 ({100 * evr[0]:.2f}% variance)")
    ax.set_ylabel(f"PC2 ({100 * evr[1]:.2f}% variance)")
    ax.set_title(f"PCA Loss Landscape")
    ax.legend()
    fig.tight_layout()  # pyright: ignore[reportAttributeAccessIssue]
    plt.savefig("figures/pcaLandscape2d.pdf")

    return fig, ax


def plot_pca_loss_scan_3d(
    scan_result: PCAResult,
    cmap: str = "viridis",
    elev: float = 30,
    azim: float = -60,
    alpha_surface: float = 0.85,
    trajectory_kwargs: dict[str, Any] | None = None,
):
    """Plot a 3D PCA loss landscape.

    Args:
        scan_result: Mapping returned by ``pca_loss_scan``.
        cmap: Colormap used for the surface.
        elev: Elevation angle of the view.
        azim: Azimuth angle of the view.
        alpha_surface: Surface transparency.
        trajectory_kwargs: Optional keyword arguments for trajectory plotting.

    Notes:
        - The figure is saved to ``figures/pcaLandscape3d.pdf``.
        - If ``LogNorm()`` is used for normalization, the values in ``Z`` must
            be strictly positive.

    Returns:
        Figure containing the plot and Tuple ``(ax1, ax2)`` containing the main
        PCA landscape axis and the log-scale surface axis.
    """

    from scipy.interpolate import RegularGridInterpolator

    fig = plt.figure(figsize=(16, 6))
    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122, projection="3d")

    X = scan_result["X"]
    Y = scan_result["Y"]
    Z = scan_result["Z"]
    traj_xy = scan_result["traj_xy"]
    evr = scan_result["explained_variance_ratio"]

    # --- Project trajectory onto the PCA loss surface ---
    interp = RegularGridInterpolator(
        (Y[:, 0], X[0, :]),  # meshgrid ordering: Y first, then X
        Z,
        bounds_error=False,
    )

    traj_surface_z = interp(np.column_stack([traj_xy[:, 1], traj_xy[:, 0]]))

    surf = ax1.plot_surface(
        X,
        Y,
        Z,
        cmap=cmap,
        norm=LogNorm(),
        linewidth=0,
        antialiased=True,
        alpha=alpha_surface,
        shade=False,
    )

    fig.colorbar(surf, ax=ax1, shrink=0.7, pad=0.1, label="Loss value")

    if trajectory_kwargs is None:
        trajectory_kwargs = {}

    default_traj_kwargs = dict(color="k", lw=2.5, marker="o", ms=4, alpha=1.0)
    default_traj_kwargs.update(trajectory_kwargs)

    ax1.plot(
        traj_xy[:, 0],
        traj_xy[:, 1],
        traj_surface_z,
        **default_traj_kwargs,  # pyright: ignore[reportArgumentType]
        label="Trajectory",
    )
    ax1.scatter(
        traj_xy[0, 0],
        traj_xy[0, 1],
        traj_surface_z[0],
        c="cyan",
        s=55,
        edgecolors="k",
        label="Start",
    )
    ax1.scatter(
        traj_xy[-1, 0],
        traj_xy[-1, 1],
        traj_surface_z[-1],
        c="red",
        s=65,
        edgecolors="k",
        label="End",
    )

    ax1.set_xlabel(f"PC1 ({100 * evr[0]:.2f}% variance)")
    ax1.set_ylabel(f"PC2 ({100 * evr[1]:.2f}% variance)")
    ax1.set_title(f"PCA Loss Landscape")
    ax1.view_init(elev=elev, azim=azim)
    ax1.legend()

    Z_eps = np.maximum(Z, 1e-14)
    Z_log = np.log(Z_eps)

    _ = ax2.plot_surface(
        X,
        Y,
        Z_log,
        cmap=cmap,
        norm=None,
        linewidth=0,
        antialiased=True,
        alpha=alpha_surface,
        shade=False,
    )

    norm = LogNorm(vmin=Z_eps.min(), vmax=Z_eps.max())

    mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    mappable.set_array(Z_eps)

    fig.colorbar(
        mappable,
        ax=ax2,
        shrink=0.6,
        pad=0.08,
    )

    traj_surface_z_log = np.log(np.maximum(traj_surface_z, 1e-12))

    ax2.plot(
        traj_xy[:, 0],
        traj_xy[:, 1],
        traj_surface_z_log,
        **default_traj_kwargs,  # pyright: ignore[reportArgumentType]
    )

    ax2.scatter(
        traj_xy[0, 0],
        traj_xy[0, 1],
        traj_surface_z_log[0],
        c="cyan",
        s=55,
        edgecolors="k",
    )

    ax2.scatter(
        traj_xy[-1, 0],
        traj_xy[-1, 1],
        traj_surface_z_log[-1],
        c="red",
        s=65,
        edgecolors="k",
    )
    ax2.set_title("Log-scale surface")
    ax2.set_xlabel(f"PC1 ({100 * evr[0]:.2f}%)")
    ax2.set_ylabel(f"PC2 ({100 * evr[1]:.2f}%)")
    ax2.set_zlabel("log(Loss value)")

    ax1.view_init(elev=elev, azim=azim)
    ax2.view_init(elev=elev, azim=azim)

    fig.tight_layout()
    plt.savefig("figures/pcaLandscape3d.pdf")

    return fig, (ax1, ax2)


class ParamEntry(TypedDict):
    index: int
    field: None
    local_index: int
    label: str
    mean: float
    std: float
    range: float
    path_length: float
    global_abs_influence: float
    global_sq_influence: float
    global_path_influence: float
    loading: NotRequired[float]
    abs_loading: NotRequired[float]
    sq_loading: NotRequired[float]
    path_contribution_in_component: NotRequired[float]
    correlation_with_component_score: NotRequired[Optional[float]]


class PCASummary(TypedDict):
    n_steps: int
    n_parameters: int
    n_components_analyzed: int
    explained_variance_ratio: np.ndarray
    cumulative_explained_variance: np.ndarray
    effective_rank_90pct: int
    effective_rank_95pct: int
    dominant_component: int
    most_influential_parameters: list[ParamEntry]
    field_ranking: None


class InfluenceMapInstance(TypedDict):
    global_abs_influence: np.ndarray
    global_sq_influence: np.ndarray
    global_path_influence: np.ndarray
    parameter_std: np.ndarray
    parameter_range: np.ndarray
    path_length_per_param: np.ndarray


class InfluenceMap(TypedDict):
    flat: InfluenceMapInstance


class ComplementReport(TypedDict):
    component_id: int
    explained_variance_ratio: float
    score_mean: float
    score_std: float
    score_min: float
    score_max: float
    top_abs_parameters: list[ParamEntry]
    top_positive_parameters: list[ParamEntry]
    top_negative_parameters: list[ParamEntry]
    top_training_path_parameters: list[ParamEntry]
    field_component_scores: None
    component_vector: np.ndarray
    component_map: np.ndarray
    abs_component_map: np.ndarray
    path_contribution_map: np.ndarray


class RawResult(TypedDict):
    components: np.ndarray
    abs_loadings: np.ndarray
    sq_loadings: np.ndarray
    weights: np.ndarray
    path_contrib_per_component: np.ndarray
    global_abs_influence: np.ndarray
    global_sq_influence: np.ndarray
    global_path_influence: np.ndarray
    param_mean: np.ndarray
    param_std: np.ndarray
    param_range: np.ndarray
    path_length_per_param: np.ndarray
    flat_parameter_labels: list[str]
    parameter_labels: list[str]
    block_slices: None
    index_map: None


class AnalysisResult(TypedDict):
    summary: PCASummary
    """Global PCA summary"""
    scores: np.ndarray
    """PCA scores along the trajectory"""
    component_reports: list[ComplementReport]
    """Per-component analyses"""
    global_ranking: list[ParamEntry]
    """Global parameter importance"""
    field_ranking: None
    """Always None in the flat-parameter analysis mode"""
    correlations: np.ndarray
    """Parameter–score correlations"""
    influence_maps: InfluenceMap
    """Parameter influence arrays"""
    raw: RawResult
    """Raw numerical quantities"""


def analyze_pca(
    scan_result: PCAResult,
    n_components: int | None = None,
    top_k: int = 10,
    weight_mode: str = "variance",
    eps: float = 1e-12,
) -> AnalysisResult:
    """Analyze a PCA applied to an optimization parameter trajectory.

    Parameters:
        scan_result: Mapping containing PCA results and the parameter trajectory.
        n_components: Number of PCA components to analyze.
        top_k: Number of parameters retained in importance rankings.
        weight_mode: Weighting scheme used to aggregate component influence
            across PCA components. Supported values are:

            - ``"variance"``: weights components according
              to their explained variance ratio.
            - ``"uniform"``: assigns equal weight to all
              analyzed components.
        eps: Small constant used for numerical stability.

    Returns:
        PCA interpretability results.
    """

    param_history = np.asarray(scan_result["param_history"])
    pca = scan_result["pca"]
    explained_variance_ratio = np.asarray(scan_result["explained_variance_ratio"])

    if param_history.ndim < 2:
        raise ValueError(
            "scan_result['param_history'] must contain a parameter trajectory."
        )

    if param_history.ndim == 2:
        flat_history = param_history
    else:
        flat_history = param_history.reshape(param_history.shape[0], -1)

    n_steps, n_params = flat_history.shape

    if pca.components_.ndim != 2:
        raise ValueError("scan_result['pca'].components_ unexpected format.")

    max_components: int = pca.components_.shape[0]
    if n_components is None:
        n_components = max_components
    n_components = min(n_components, max_components)

    components = pca.components_[:n_components]  # shape (K, D)
    evr = explained_variance_ratio[:n_components]  # shape (K,)
    scores = pca.transform(flat_history)[:, :n_components]  # shape (T, K)

    flat_parameter_labels = [rf"$\theta_{{{j}}}$" for j in range(n_params)]

    deltas = np.diff(flat_history, axis=0)

    param_mean = flat_history.mean(axis=0)
    param_std: np.ndarray = flat_history.std(axis=0, ddof=0)
    param_range: np.ndarray = flat_history.max(axis=0) - flat_history.min(axis=0)
    path_length_per_param: np.ndarray = np.sum(np.abs(deltas), axis=0)

    if weight_mode == "variance":
        weights = evr.copy()
        weights = weights / max(weights.sum(), eps)
    elif weight_mode == "uniform":
        weights = np.ones_like(evr) / len(evr)
    else:
        raise ValueError("weight_mode must be 'variance' or 'uniform'.")

    abs_loadings = np.abs(components)  # (K, D)
    sq_loadings = components**2  # (K, D)

    global_abs_influence: np.ndarray = weights @ abs_loadings
    global_sq_influence: np.ndarray = weights @ sq_loadings

    path_contrib_per_component = []
    for k in range(n_components):
        contrib_k = np.abs(deltas * components[k][None, :]).sum(axis=0)
        path_contrib_per_component.append(contrib_k)
    path_contrib_per_component = np.asarray(path_contrib_per_component)  # (K, D)

    global_path_influence: np.ndarray = weights @ path_contrib_per_component

    correlations = np.full((n_params, n_components), np.nan)

    for j in range(n_params):
        x = flat_history[:, j]
        x_std = x.std(ddof=0)
        if x_std < eps:
            continue

        for k in range(n_components):
            y = scores[:, k]
            y_std = y.std(ddof=0)
            if y_std < eps:
                continue
            correlations[j, k] = np.corrcoef(x, y)[0, 1]

    def _make_param_entry(idx: int, comp_id: Optional[int] = None) -> ParamEntry:
        entry: ParamEntry = {
            "index": int(idx),
            "field": None,
            "local_index": int(idx),
            "label": flat_parameter_labels[idx],
            "mean": float(param_mean[idx]),
            "std": float(param_std[idx]),
            "range": float(param_range[idx]),
            "path_length": float(path_length_per_param[idx]),
            "global_abs_influence": float(global_abs_influence[idx]),
            "global_sq_influence": float(global_sq_influence[idx]),
            "global_path_influence": float(global_path_influence[idx]),
        }

        if comp_id is not None:
            entry.update(
                {
                    "loading": float(components[comp_id, idx]),
                    "abs_loading": float(abs_loadings[comp_id, idx]),
                    "sq_loading": float(sq_loadings[comp_id, idx]),
                    "path_contribution_in_component": float(
                        path_contrib_per_component[comp_id, idx]
                    ),
                    "correlation_with_component_score": (
                        None
                        if np.isnan(correlations[idx, comp_id])
                        else float(correlations[idx, comp_id])
                    ),
                }
            )

        return entry

    component_reports = []

    for k in range(n_components):
        order_abs = np.argsort(abs_loadings[k])[::-1]
        order_pos = np.argsort(components[k])[::-1]
        order_neg = np.argsort(components[k])

        top_abs = [_make_param_entry(i, comp_id=k) for i in order_abs[:top_k]]
        top_pos = [_make_param_entry(i, comp_id=k) for i in order_pos[:top_k]]
        top_neg = [_make_param_entry(i, comp_id=k) for i in order_neg[:top_k]]

        path_order = np.argsort(path_contrib_per_component[k])[::-1]
        top_path = [_make_param_entry(i, comp_id=k) for i in path_order[:top_k]]

        component_reports.append(
            {
                "component_id": int(k),
                "explained_variance_ratio": float(evr[k]),
                "score_mean": float(scores[:, k].mean()),
                "score_std": float(scores[:, k].std(ddof=0)),
                "score_min": float(scores[:, k].min()),
                "score_max": float(scores[:, k].max()),
                "top_abs_parameters": top_abs,
                "top_positive_parameters": top_pos,
                "top_negative_parameters": top_neg,
                "top_training_path_parameters": top_path,
                "field_component_scores": None,
                "component_vector": components[k].copy(),
                "component_map": components[k].copy(),
                "abs_component_map": abs_loadings[k].copy(),
                "path_contribution_map": path_contrib_per_component[k].copy(),
            }
        )

    global_order = np.argsort(global_sq_influence)[::-1]
    global_ranking = [_make_param_entry(i) for i in global_order[:top_k]]

    cumulative_evr = np.cumsum(evr)

    effective_rank_90 = int(np.searchsorted(cumulative_evr, 0.90) + 1)
    effective_rank_95 = int(np.searchsorted(cumulative_evr, 0.95) + 1)

    effective_rank_90 = min(effective_rank_90, n_components)
    effective_rank_95 = min(effective_rank_95, n_components)

    summary: PCASummary = {
        "n_steps": int(n_steps),
        "n_parameters": int(n_params),
        "n_components_analyzed": int(n_components),
        "explained_variance_ratio": evr.copy(),
        "cumulative_explained_variance": cumulative_evr.copy(),
        "effective_rank_90pct": effective_rank_90,
        "effective_rank_95pct": effective_rank_95,
        "dominant_component": int(np.argmax(evr)),
        "most_influential_parameters": global_ranking,
        "field_ranking": None,
    }

    influence_maps: InfluenceMap = {
        "flat": {
            "global_abs_influence": global_abs_influence.copy(),
            "global_sq_influence": global_sq_influence.copy(),
            "global_path_influence": global_path_influence.copy(),
            "parameter_std": param_std.copy(),
            "parameter_range": param_range.copy(),
            "path_length_per_param": path_length_per_param.copy(),
        }
    }

    return {
        "summary": summary,
        "scores": scores,
        "component_reports": component_reports,
        "global_ranking": global_ranking,
        "field_ranking": None,
        "correlations": correlations,
        "influence_maps": influence_maps,
        "raw": {
            "components": components,
            "abs_loadings": abs_loadings,
            "sq_loadings": sq_loadings,
            "weights": weights,
            "path_contrib_per_component": path_contrib_per_component,
            "global_abs_influence": global_abs_influence,
            "global_sq_influence": global_sq_influence,
            "global_path_influence": global_path_influence,
            "param_mean": param_mean,
            "param_std": param_std,
            "param_range": param_range,
            "path_length_per_param": path_length_per_param,
            "flat_parameter_labels": flat_parameter_labels,
            "parameter_labels": flat_parameter_labels,
            "block_slices": None,
            "index_map": None,
        },
    }


class PCAPlotResult(TypedDict):
    summary: Figure
    components: Figure


def plot_pca_analysis(
    analysis: AnalysisResult,
    n_top: int = 10,
    decimals: int = 4,
    summary_figsize: Optional[tuple[int, int]] = None,
    component_figsize_per_row: Optional[tuple[int, int]] = None,
    max_label_length: int = 28,
) -> PCAPlotResult:
    """Plot summaries of a PCA interpretability analysis.

    Args:
        analysis: Mapping returned by ``analyze_pca``.
        n_top: Number of top entries shown in each plot.
        decimals: Number of decimals used for axis formatting.
        summary_figsize: Figure size for the global summary. Default is (14, 10).
        component_figsize_per_row: Figure size per row for component plots.
            Default is (18, 4).
        max_label_length: Maximum length of labels before truncation.

    Returns:
        Mapping containing the generated figures:
            - summary: Global PCA summary figure
            - components: Component-level analysis figure
    """
    if summary_figsize is None:
        summary_figsize = (14, 10)
    if component_figsize_per_row is None:
        component_figsize_per_row = (18, 4)

    from matplotlib.ticker import FuncFormatter

    def shorten(label: str, max_len: int | None = None):
        if max_len is None:
            max_len = max_label_length
        label = str(label)
        if len(label) <= max_len:
            return label
        return label[: max_len - 3] + "..."

    def plain_formatter(x: float, pos: Any):
        return f"{x:.{decimals}f}"

    def apply_plain_format(ax: Axes, axis: str = "x"):
        formatter = FuncFormatter(plain_formatter)
        if axis == "x":
            ax.xaxis.set_major_formatter(formatter)
        elif axis == "y":
            ax.yaxis.set_major_formatter(formatter)
        elif axis == "both":
            ax.xaxis.set_major_formatter(formatter)
            ax.yaxis.set_major_formatter(formatter)

    def safe_log_scale(ax: Axes, values: np.ndarray | list[float], axis: str = "x"):
        """
        Apply log scale only if all displayed values are strictly positive.
        This avoids matplotlib warnings/errors when values contain zeros.
        """
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]

        if values.size > 0 and np.all(values > 0):
            if axis == "x":
                ax.set_xscale("log")
            elif axis == "y":
                ax.set_yscale("log")

    # ------------------------------------------------------------------
    # 1) Data
    # ------------------------------------------------------------------
    summary = analysis["summary"]
    global_ranking = analysis["global_ranking"][:n_top]
    component_reports = analysis["component_reports"]

    evr = np.asarray(summary["explained_variance_ratio"])
    cum_evr = np.asarray(summary["cumulative_explained_variance"])
    comp_ids = np.arange(len(evr))

    # ------------------------------------------------------------------
    # 2) Summary plot
    # ------------------------------------------------------------------
    fig_summary, axes = plt.subplots(2, 2, figsize=summary_figsize)
    ax_evr, ax_sq, ax_path, ax_std = axes.ravel()

    # --- A. Explained variance
    ax_evr.bar(
        comp_ids,
        evr,
        color="tab:blue",
        alpha=0.8,
        label="Explained variance",
    )
    ax_evr.plot(
        comp_ids,
        cum_evr,
        color="tab:red",
        marker="o",
        lw=2,
        label="Cumulative",
    )

    ax_evr.axhline(0.90, color="gray", ls="--", lw=1, alpha=0.8)
    ax_evr.axhline(0.95, color="gray", ls=":", lw=1, alpha=0.8)

    ax_evr.set_xticks(comp_ids)
    ax_evr.set_xticklabels([f"PC{i}" for i in comp_ids])
    ax_evr.set_ylabel("Variance ratio")
    ax_evr.set_title(
        f"PCA summary\n"
        f"trajectory_size={summary['n_steps']}, "
        f"n_parameters={summary['n_parameters']}"
    )
    ax_evr.legend()
    apply_plain_format(ax_evr, "y")

    # Global data
    labels = [shorten(item["label"]) for item in global_ranking][::-1]
    sq_vals = [item["global_sq_influence"] for item in global_ranking][::-1]
    path_vals = [item["global_path_influence"] for item in global_ranking][::-1]
    std_vals = [item["std"] for item in global_ranking][::-1]

    # --- B. Top global parameters by squared influence
    ax_sq.barh(labels, sq_vals, color="tab:blue", alpha=0.85)
    safe_log_scale(ax_sq, sq_vals, axis="x")
    ax_sq.set_title(f"Top {n_top} parameters — global sq influence")
    ax_sq.set_xlabel("sq influence")
    apply_plain_format(ax_sq, "x")

    # --- C. Top global parameters by path influence
    ax_path.barh(labels, path_vals, color="tab:orange", alpha=0.85)
    safe_log_scale(ax_path, path_vals, axis="x")
    ax_path.set_title(f"Top {n_top} parameters — global path influence")
    ax_path.set_xlabel("path influence")
    apply_plain_format(ax_path, "x")

    # --- D. Top global parameters by std
    ax_std.barh(labels, std_vals, color="tab:green", alpha=0.85)
    safe_log_scale(ax_std, std_vals, axis="x")
    ax_std.set_title(f"Top {n_top} parameters — standard deviation")
    ax_std.set_xlabel("std")
    apply_plain_format(ax_std, "x")

    fig_summary.tight_layout()

    # ------------------------------------------------------------------
    # 3) Component-level plots
    # ------------------------------------------------------------------
    n_components = len(component_reports)
    fig_w, fig_h_per_row = component_figsize_per_row

    # No more F_i / field column.
    n_cols = 2

    fig_components, axes_comp = plt.subplots(
        n_components,
        n_cols,
        figsize=(fig_w, fig_h_per_row * n_components),
        squeeze=False,
    )

    for row, comp in enumerate(component_reports):
        ax_load = axes_comp[row, 0]
        ax_path_comp = axes_comp[row, 1]

        top_items = comp["top_abs_parameters"][:n_top]

        labels_c = [shorten(item["label"]) for item in top_items][::-1]

        loadings = np.array(
            [item["loading"] if "loading" in item else np.nan for item in top_items][
                ::-1
            ],
            dtype=float,
        )

        path_c = np.array(
            [
                (
                    item["path_contribution_in_component"]
                    if "path_contribution_in_component" in item
                    else np.nan
                )
                for item in top_items
            ][::-1],
            dtype=float,
        )

        colors = ["tab:blue" if v >= 0 else "tab:red" for v in loadings]

        # --- A. Signed loadings
        ax_load.barh(labels_c, loadings, color=colors, alpha=0.85)
        ax_load.axvline(0.0, color="black", lw=1)
        ax_load.set_title(f"PC{comp['component_id']} — signed loadings")
        ax_load.set_xlabel("loading")
        apply_plain_format(ax_load, "x")

        # --- B. Path contributions
        ax_path_comp.barh(labels_c, path_c, color="tab:purple", alpha=0.85)
        safe_log_scale(ax_path_comp, path_c, axis="x")
        ax_path_comp.set_title(
            f"PC{comp['component_id']} — path contribution top {n_top}"
        )
        ax_path_comp.set_xlabel("path contribution")
        apply_plain_format(ax_path_comp, "x")

    fig_components.tight_layout()

    plt.show()

    return {
        "summary": fig_summary,
        "components": fig_components,
    }


def plot_pca_circuit_schematic_real_circuit(
    qc: QuantumCircuit,
    analysis: AnalysisResult,
    score_key: str = "global_sq_influence",
    cmap: str = "viridis",
    box_width: float = 0.55,
    box_height: float = 0.5,
    show_values: bool = False,
    title: str | None = None,
    show_gamma: bool = True,
    gamma_label: str | None = None,
    label_mode: str = "label",
    show_entanglers: bool = True,
    entangler_linewidth: float = 1.6,
):
    """Plot a circuit schematic annotated with PCA-based parameter scores.

    Parameters:
        qc: Qiskit quantum circuit to visualize, or any compatible object
            exposing ``qc.data``, ``qc.parameters``, ``qc.num_qubits``,
            and ``qc.find_bit``.
        analysis: Mapping returned by ``analyze_pca``.
        score_key: Influence metric used for coloring.
        cmap: Colormap used to map scores to colors.
        box_width: Width of parameter boxes.
        box_height: Height of parameter boxes.
        show_values: Whether to display numerical scores.
        title: Optional figure title.
        show_gamma: Whether to display the global scaling
            parameter. The gamma parameter is shown only if
            the PCA score vector contains one additional score
            compared to the number of Qiskit circuit parameters.
        gamma_label: Optional label for the scaling parameter.
        label_mode: Labeling scheme used for parameter boxes. Supported values
            are:

            - ``"index"``
            - ``"theta"``
            - ``"full"``
            - ``"gate+index"``
            - ``"gate"``
            - ``"label"``
            - ``"gate+label"``
        show_entanglers: Whether to render entangling gates.
        entangler_linewidth: Line width for entangling
            connections.

    Returns:
        matplotlib.figure.Figure

    Saved Files:
        - ``figures/circuitpcainfluence.pdf``
    """
    if title is None:
        title = ""

    import matplotlib.patheffects as pe
    from matplotlib.cm import get_cmap
    from matplotlib.patches import Rectangle

    # ------------------------------------------------------------------
    # 1) Get flat PCA scores
    # ------------------------------------------------------------------
    if "influence_maps" not in analysis or "flat" not in analysis["influence_maps"]:
        raise KeyError(
            "analysis must contain analysis['influence_maps']['flat']. "
            "Check that analyze_pca(...) uses the updated flat format."
        )

    flat_maps = analysis["influence_maps"]["flat"]

    if score_key not in flat_maps:
        raise KeyError(
            f"score_key='{score_key}' not found in analysis['influence_maps']['flat']."
        )

    local_scores = np.asarray(flat_maps[score_key]).copy()

    # ------------------------------------------------------------------
    # 2) Get flat labels
    # ------------------------------------------------------------------
    flat_labels = None

    if "raw" in analysis:
        if "flat_parameter_labels" in analysis["raw"]:
            flat_labels = list(analysis["raw"]["flat_parameter_labels"])
        elif "parameter_labels" in analysis["raw"]:
            parameter_labels = analysis["raw"]["parameter_labels"]
            if not isinstance(
                parameter_labels, dict
            ):  # pyright: ignore[reportUnnecessaryIsInstance]
                flat_labels = list(parameter_labels)

    # ------------------------------------------------------------------
    # 3) Match PCA scores with Qiskit parameters
    # ------------------------------------------------------------------
    qiskit_params = list(qc.parameters)
    qiskit_param_names = [str(p) for p in qiskit_params]

    gamma_score = None

    # Case: scores = [Gamma, theta_1, ..., theta_n]
    if len(local_scores) == len(qiskit_param_names) + 1:
        gamma_score = float(local_scores[0])
        param_scores = local_scores[1:]

        if flat_labels is not None:
            if len(flat_labels) != len(local_scores):
                raise ValueError(
                    f"flat_parameter_labels size {len(flat_labels)} "
                    f"but needed {len(local_scores)}."
                )

            gamma_label_local = flat_labels[0]
            param_labels_local = flat_labels[1:]
        else:
            gamma_label_local = gamma_label if gamma_label is not None else r"$\Gamma$"
            param_labels_local = None

    # Case: scores = [theta_0, ..., theta_n]
    elif len(local_scores) == len(qiskit_param_names):
        param_scores = local_scores

        if flat_labels is not None:
            if len(flat_labels) != len(local_scores):
                raise ValueError(
                    f"flat_parameter_labels size {len(flat_labels)} "
                    f"but needed {len(local_scores)}."
                )

            gamma_label_local = gamma_label if gamma_label is not None else r"$\Gamma$"
            param_labels_local = flat_labels
        else:
            gamma_label_local = gamma_label if gamma_label is not None else r"$\Gamma$"
            param_labels_local = None

    else:
        raise ValueError(
            f"Incompatible number of PCA scores: "
            f"{len(local_scores)} scores for {len(qiskit_param_names)} "
            f"Qiskit parameters in the circuit. Expected either "
            f"{len(qiskit_param_names)} or {len(qiskit_param_names) + 1}."
        )

    if gamma_label is None:
        gamma_label = gamma_label_local

    score_by_param = dict(zip(qiskit_param_names, param_scores))

    label_by_param = {}
    index_by_param = {}

    for i, pname in enumerate(qiskit_param_names):
        index_by_param[pname] = i

        if param_labels_local is not None:
            label_by_param[pname] = param_labels_local[i]

    # ------------------------------------------------------------------
    # 4) Helpers
    # ------------------------------------------------------------------
    def unpack_instruction(
        inst: (
            _QiskitCircuitInstructionProtocol
            | tuple[Instruction, tuple[Qubit, ...], tuple[Clbit, ...]]
        ),
    ):
        if hasattr(inst, "operation"):
            if TYPE_CHECKING:
                assert isinstance(inst, _QiskitCircuitInstructionProtocol)
            op = inst.operation
            qargs = inst.qubits
            cargs = inst.clbits
        else:
            if TYPE_CHECKING:
                assert isinstance(inst, tuple)
            op, qargs, cargs = inst

        return op, qargs, cargs

    def extract_param_names(op: Instruction):
        names = []

        for p in getattr(op, "params", []):
            if hasattr(p, "parameters") and len(p.parameters) > 0:
                if hasattr(p, "name"):
                    names.append(str(p))
                else:
                    sub = sorted([str(x) for x in p.parameters])
                    names.extend(sub)

        seen = set()
        out = []

        for n in names:
            if n not in seen:
                out.append(n)
                seen.add(n)

        return out

    def pretty_gate_name(gname: str):
        mapping = {
            "rx": "Rx",
            "ry": "Ry",
            "rz": "Rz",
            "u": "U",
            "u1": "U1",
            "u2": "U2",
            "u3": "U3",
            "p": "P",
            "x": "X",
            "y": "Y",
            "z": "Z",
            "h": "H",
            "s": "S",
            "sdg": "Sdg",
            "t": "T",
            "tdg": "Tdg",
            "cx": "CNOT",
            "cz": "CZ",
            "swap": "SWAP",
            "ecr": "ECR",
        }

        return mapping.get(str(gname).lower(), str(gname).upper())

    def fallback_theta_from_name(name: str):
        m = re.search(r"\[(\d+)\]$", name)

        if m:
            idx = m.group(1)
            return idx, rf"$\theta_{{{idx}}}$"

        return name, name

    def label_from_item(param_name: str, gate_name: Optional[str] = None):
        gate_txt = pretty_gate_name(gate_name) if gate_name is not None else ""
        idx_local = index_by_param.get(param_name, None)

        if param_name in label_by_param:
            local_label = label_by_param[param_name]
        else:
            _, local_label = fallback_theta_from_name(param_name)

        if idx_local is None:
            idx_txt = param_name
        else:
            idx_txt = str(idx_local)

        if label_mode == "index":
            return idx_txt
        elif label_mode == "theta":
            return local_label
        elif label_mode == "full":
            return param_name
        elif label_mode == "gate+index":
            return f"{gate_txt} {idx_txt}".strip()
        elif label_mode == "gate":
            return gate_txt
        elif label_mode == "label":
            return local_label
        elif label_mode == "gate+label":
            return f"{gate_txt} {local_label}".strip()
        else:
            return local_label

    # ------------------------------------------------------------------
    # 5) Build visual columns from circuit data
    # ------------------------------------------------------------------

    columns = []
    param_block = []

    def flush_param_block():
        nonlocal param_block, columns

        if not param_block:
            return

        # Group parametrized one-qubit gates by qubit.
        items_by_qubit = {}

        for item in param_block:
            q = item["qubit"]
            items_by_qubit.setdefault(q, []).append(item)

        max_depth = max(len(items) for items in items_by_qubit.values())

        # Create aligned columns:
        # depth d contains the d-th parametrized gate of every qubit.
        for d in range(max_depth):
            col_items = []

            for q in sorted(items_by_qubit.keys()):
                if d < len(items_by_qubit[q]):
                    col_items.append(items_by_qubit[q][d])

            if col_items:
                columns.append(
                    {
                        "type": "param",
                        "items": col_items,
                    }
                )

        param_block = []

    for inst in qc.data:
        op, qargs, _ = unpack_instruction(inst)

        if str(op.name).lower() == "barrier":
            flush_param_block()
            continue

        param_names = extract_param_names(op)
        qubit_indices = [qc.find_bit(q).index for q in qargs]

        # Parametrized single-qubit gate
        if len(param_names) > 0 and len(qubit_indices) == 1:
            q = qubit_indices[0]

            for pname in param_names:
                if pname not in score_by_param:
                    continue

                param_block.append(
                    {
                        "qubit": q,
                        "param_name": pname,
                        "gate_name": op.name,
                        "score": float(score_by_param[pname]),
                    }
                )

        # Entangling / multi-qubit gate
        elif show_entanglers and len(qubit_indices) >= 2:
            flush_param_block()

            columns.append(
                {
                    "type": "entangler",
                    "gate_name": op.name,
                    "qubits": qubit_indices,
                }
            )

        # Non-parametrized single-qubit gates are ignored visually
        else:
            continue

    flush_param_block()

    n_param_columns = sum(1 for c in columns if c["type"] == "param")
    n_total_columns = len(columns)
    n_qubits = qc.num_qubits

    if n_param_columns == 0:
        raise ValueError("No parametrized single-qubit gate detected in the circuit.")

    # ------------------------------------------------------------------
    # 6) Layout
    # ------------------------------------------------------------------
    x_positions = []
    x = 0.0

    # Espacements robustes contre les chevauchements
    param_param_spacing = box_width + 0.05  # entre Ry et Rz
    param_entangler_spacing = box_width / 2 + 0.25  # entre Rz et CNOT
    entangler_param_spacing = box_width / 2 + 0.25  # entre CNOT et Ry
    entangler_entangler_spacing = 0.25  # entre deux CNOT successifs

    for i, c in enumerate(columns):
        x_positions.append((x, c))

        if i == len(columns) - 1:
            break

        next_c = columns[i + 1]

        if c["type"] == "param" and next_c["type"] == "param":
            x += param_param_spacing

        elif c["type"] == "param" and next_c["type"] == "entangler":
            x += param_entangler_spacing

        elif c["type"] == "entangler" and next_c["type"] == "param":
            x += entangler_param_spacing

        elif c["type"] == "entangler" and next_c["type"] == "entangler":
            x += entangler_entangler_spacing

        else:
            x += 1.0

    max_x = x_positions[-1][0] if len(x_positions) > 0 else 0.0

    all_param_scores = np.array(
        [float(v) for v in score_by_param.values()],
        dtype=float,
    )

    vmin = np.nanmin(all_param_scores)
    vmax = np.nanmax(all_param_scores)

    if gamma_score is not None and show_gamma:
        vmin = min(vmin, gamma_score)
        vmax = max(vmax, gamma_score)

    if np.isclose(vmin, vmax):
        vmin -= 1e-12
        vmax += 1e-12

    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = get_cmap(cmap)

    left_margin = -1.6 if (gamma_score is not None and show_gamma) else -0.8

    fig_w = max(
        7,
        1.15 * max(1, n_total_columns)
        + (1.8 if show_gamma and gamma_score is not None else 0)
        + 2.5,
    )
    fig_h = max(4, 0.9 * n_qubits + 2.0)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # ------------------------------------------------------------------
    # 7) Draw qubit lines
    # ------------------------------------------------------------------
    line_x0 = left_margin + 0.25
    line_x1 = max_x + 0.65

    for q in range(n_qubits):
        y = n_qubits - 1 - q

        ax.plot(
            [line_x0, line_x1],
            [y, y],
            color="black",
            lw=1.1,
            zorder=1,
        )

        ax.text(
            left_margin - 0.05,
            y,
            f"q{q}",
            va="center",
            ha="right",
            fontsize=10,
        )

    # ------------------------------------------------------------------
    # 8) Draw gamma if present
    # ------------------------------------------------------------------
    if gamma_score is not None and show_gamma:
        gamma_x = -1.05
        gamma_y = n_qubits - 0.15
        gamma_color = cmap_obj(norm(gamma_score))

        rect = Rectangle(
            (gamma_x - box_width / 2, gamma_y - box_height / 2),
            box_width,
            box_height,
            facecolor=gamma_color,
            edgecolor="black",
            linewidth=1.0,
            zorder=3,
        )

        ax.add_patch(rect)

        gamma_norm_value = norm(np.clip(gamma_score, vmin, vmax))
        gamma_text_color = "white" if gamma_norm_value > 0.5 else "black"

        ax.text(
            gamma_x,
            gamma_y,
            gamma_label,
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            color=gamma_text_color,
            zorder=4,
            path_effects=[pe.withStroke(linewidth=1.0, foreground="black")],
        )

        if show_values:
            ax.text(
                gamma_x,
                gamma_y - 0.42,
                f"{gamma_score:.3f}",
                ha="center",
                va="top",
                fontsize=7,
                zorder=4,
            )

    # ------------------------------------------------------------------
    # 9) Draw circuit columns
    # ------------------------------------------------------------------
    for xcol, col in x_positions:
        if col["type"] == "param":
            for item in col["items"]:
                q = item["qubit"]
                score = item["score"]
                pname = item["param_name"]
                gname = item["gate_name"]

                y = n_qubits - 1 - q
                color = cmap_obj(norm(score))

                rect = Rectangle(
                    (xcol - box_width / 2, y - box_height / 2),
                    box_width,
                    box_height,
                    facecolor=color,
                    edgecolor="black",
                    linewidth=1.0,
                    zorder=3,
                )

                ax.add_patch(rect)

                txt = label_from_item(pname, gate_name=gname)

                ax.text(
                    xcol,
                    y,
                    txt,
                    ha="center",
                    va="center",
                    fontsize=14,
                    fontweight="bold",
                    color="white",
                    zorder=4,
                    path_effects=[pe.withStroke(linewidth=1.0, foreground="black")],
                )

                if show_values:
                    ax.text(
                        xcol,
                        y - 0.40,
                        f"{score:.3f}",
                        ha="center",
                        va="top",
                        fontsize=7,
                        zorder=4,
                    )

        elif col["type"] == "entangler":
            gname = str(col["gate_name"]).lower()
            qs = col["qubits"]

            if len(qs) >= 2:
                q0, q1 = qs[0], qs[1]
                y0 = n_qubits - 1 - q0
                y1 = n_qubits - 1 - q1

                ax.plot(
                    [xcol, xcol],
                    [y0, y1],
                    color="black",
                    lw=entangler_linewidth,
                    zorder=2,
                )

                if gname == "cx":
                    ax.plot(xcol, y0, "ko", ms=7, zorder=4)

                    ax.plot(
                        xcol,
                        y1,
                        marker="o",
                        ms=12,
                        mec="black",
                        mfc="white",
                        zorder=4,
                    )

                    ax.plot(
                        [xcol, xcol],
                        [y1 - 0.14, y1 + 0.14],
                        color="black",
                        lw=1.4,
                        zorder=5,
                    )

                    ax.plot(
                        [xcol - 0.14, xcol + 0.14],
                        [y1, y1],
                        color="black",
                        lw=1.4,
                        zorder=5,
                    )

                    ax.text(
                        xcol,
                        max(y0, y1) + 0.32,
                        "CNOT",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        fontweight="bold",
                    )

                elif gname == "cz":
                    ax.plot(xcol, y0, "ko", ms=7, zorder=4)
                    ax.plot(xcol, y1, "ko", ms=7, zorder=4)

                    ax.text(
                        xcol,
                        max(y0, y1) + 0.32,
                        "CZ",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        fontweight="bold",
                    )

                elif gname == "swap":
                    for yy in [y0, y1]:
                        ax.plot(
                            [xcol - 0.09, xcol + 0.09],
                            [yy - 0.09, yy + 0.09],
                            color="black",
                            lw=1.4,
                            zorder=4,
                        )

                        ax.plot(
                            [xcol - 0.09, xcol + 0.09],
                            [yy + 0.09, yy - 0.09],
                            color="black",
                            lw=1.4,
                            zorder=4,
                        )

                    ax.text(
                        xcol,
                        max(y0, y1) + 0.32,
                        "SWAP",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        fontweight="bold",
                    )

                else:
                    ax.plot(xcol, y0, "ko", ms=6, zorder=4)
                    ax.plot(xcol, y1, "ko", ms=6, zorder=4)

                    ax.text(
                        xcol,
                        max(y0, y1) + 0.32,
                        pretty_gate_name(gname),
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        fontweight="bold",
                    )

            if len(qs) > 2:
                ys = [n_qubits - 1 - q for q in qs]

                for yy in ys[2:]:
                    ax.plot(xcol, yy, "ko", ms=6, zorder=4)

    # ------------------------------------------------------------------
    # 10) Axes, title, colorbar
    # ------------------------------------------------------------------
    ax.set_xlim(left_margin - 0.1, max_x + 1.0)

    ax.set_ylim(
        -1.0,
        n_qubits + (0.8 if (gamma_score is not None and show_gamma) else 0.2),
    )

    ax.set_yticks([])

    xticks = [x for x, _ in x_positions]
    xticklabels = []

    for _, col in x_positions:
        if col["type"] == "param":
            gate_names = [pretty_gate_name(item["gate_name"]) for item in col["items"]]

            uniq = []

            for g in gate_names:
                if g not in uniq:
                    uniq.append(g)

            xticklabels.append("/".join(uniq))

        elif col["type"] == "entangler":
            xticklabels.append(pretty_gate_name(col["gate_name"]))

        else:
            xticklabels.append("")

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation=45, ha="right")

    ax.set_frame_on(False)

    # if title is None:
    #     title = f"Real circuit influence schematic ({score_key})"

    ax.set_title(title)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
    sm.set_array([])

    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.01)
    cbar.set_label(score_key)

    fig.tight_layout()
    plt.savefig("figures/circuitpcainfluence.pdf")
    plt.show()

    return fig


def perform_pca_and_analysis(
    params_history: np.ndarray,
    loss_function: LossFunction,
    n_steps: int,
    offset: float | tuple[float, float],
    n_top: int,
    circuit: QuantumCircuit,
    n_jobs: int = -1,
):
    """
    Run the full PCA-based loss landscape and interpretability analysis pipeline.

    Parameters:
        params_history: Parameter trajectory used to fit the PCA.
        loss_function: Callable returning the loss for a given parameter vector.
        n_steps: Number of grid points per PCA axis.
        offset: Margin added to the PCA scan bounds.
        n_top: Number of top entries shown in analysis plots.
        circuit: Quantum circuit used for the PCA influence schematic.
        n_jobs: Number of parallel jobs used during loss evaluations. Default is -1.

    Returns:
        Mapping returned by ``analyze_pca``.

    Side effects:
        Generates and displays several plots, and saves some figures under ``figures/``.
    """

    print("\n[PCA] Starting PCA-based loss landscape scan...")
    scan = pca_loss_scan(params_history, loss_function, n_steps, offset, n_jobs=n_jobs)

    print("[PCA] Plotting 2D PCA loss landscape")
    plot_pca_loss_scan_2d(
        scan,
        contour=True,
        contour_levels=25,
    )

    print("[PCA] Plotting 3D PCA loss surface")
    plot_pca_loss_scan_3d(
        scan,
        elev=35,
        azim=-50,
    )

    plt.show()

    print("\n[PCA] Computing interpretable PCA analysis...")
    analysis = analyze_pca(
        scan,
        n_components=None,
        top_k=n_top,
        weight_mode="variance",
        eps=1e-12,
    )

    print("[PCA] Visualizing parameter influence on real circuits")
    plot_pca_circuit_schematic_real_circuit(
        qc=circuit,
        analysis=analysis,
        score_key="global_sq_influence",
        show_entanglers=True,
    )

    print("[PCA] Generating global PCA analysis figures")
    plot_pca_analysis(analysis, n_top=n_top)

    print("[PCA] Analysis complete")

    return analysis


# -----------------------------------------------------------------------------
# Useful for tests
# -----------------------------------------------------------------------------


def random_mixed_directions(n: int):
    """Generate two random directions in parameter space.

    Parameters:
        n: Total parameter dimension.

    Returns:
        Two random direction vectors, with the second orthogonalized
        with respect to the first.
    """
    d1 = np.random.randn(n)
    d2 = np.random.randn(n)

    # Global orthogonalization
    d2 -= np.dot(d2, d1) * d1

    return d1, d2
