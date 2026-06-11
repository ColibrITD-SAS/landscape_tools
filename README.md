# Landscape Characterization

## Installation

```bash
pip install --upgrade git+https://github.com/ColibrITD-SAS/landscape_visualisation
```

```python
from landscape_characterization import landscape_visualization as lv
from landscape_characterization import barren_plateaus as bp
```

# Landscape Visualization

This module provides utilities to analyze and visualize the optimization
landscape of variational quantum algorithms (VQAs).

It is designed to work **independently of QUICK / HDES internals** and can be
used with any backend, as long as a suitable cost function is provided.

The main purpose of this module is to:

- perform 1D and 2D loss landscape scans,
- analyze loss landscapes in PCA subspaces,
- study gradient magnitudes and barren plateau effects,
- visualize trajectories and parameter influence.

## Design philosophy

Most optimization algorithms implemented here only depend on a **generic cost
function interface**, not on circuits, observables, or backends.

All landscape analysis functions expect a cost function of the form:

```python
f(theta: np.ndarray) -> float
```

## Available functions

### Loss landscape scans

#### `loss_scan_1d`

Evaluate the loss function along a **one-dimensional direction** in parameter
space.

Typically used to probe local flatness or curvature around a reference point.

```python
lv.loss_scan_1d(
    params,
    direction,
    loss_function,
    n_steps=200,
    end_points=None,
)
```

---

#### `loss_scan_2d_3d`

Evaluate the loss function on a **two-dimensional plane** spanned by two
directions in parameter space.

Optionally produces a 3D surface visualization.

```python
lv.loss_scan_2d_3d(
    params,
    direction1,
    direction2,
    loss_function,
    n_steps=100,
    end_points_x=None,
    end_points_y=None,
    plot3D=True,
)
```

### PCA-based landscape analysis

#### `perform_pca_and_analysis`

Run a **full PCA-based loss landscape analysis pipeline**, including:

- PCA construction from an optimization trajectory
- loss scans in PCA space
- interpretability analysis

```python
lv.perform_pca_and_analysis(
    params_history,
    loss_function,
    n_steps,
    offset,
    n_top,
    isa_circuits,
)
```

## Example: wrapping a cost function

In practice, you must wrap the original cost function so that it only
takes the parameter vector as input and returns a scalar value.

```python
from landscape_characterization import landscape_visualization as lv

def create_cost_value_function(
    use_state_vector: bool,
    isa_circuits: Dict,
    isa_obs_trans: Dict[str, List[Pauli]],
    new_coeff_matrices: Dict,
    x_list: List[List[tuple]] | List[List[float]],
    eq_func: Callable,
    run_mode: str,
    shots_list: List[int],
    original_params: Dict[str, npt.NDArray[np.float64]],
    BC_mode: str,
    backend,
) -> Callable:

    def f2(current_params):
        value, _, _, _, _, _ = cost_function(
            use_state_vector,
            current_params,
            isa_circuits,
            isa_obs_trans,
            new_coeff_matrices,
            x_list,
            eq_func,
            run_mode,
            shots_list,
            original_params,
            BC_mode,
            backend,
        )
        return value

    return f2


loss_value = create_cost_value_function(
    use_state_vector,
    isa_circuits,
    isa_obs_trans,
    new_coeff_matrices,
    x_list,
    eq_func,
    run_mode,
    shots_list,
    original_params,
    BC_mode,
    backend,
)

lv.perform_pca_and_analysis(
    params_history,
    loss_value,
    n_steps=50,
    offset=0.5,
    n_top=12,
    isa_circuits=isa_circuits,
)
```

# Barren Plateau
