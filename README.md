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

Most algorithms implemented here only depend on a **generic cost
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

This utility scans a line in parameter space starting from a reference parameter
vector and evaluates the loss function at regularly spaced points along the
specified direction. It is commonly used to inspect the local geometry of the
loss landscape, such as flatness, curvature, or the presence of nearby minima.

```python
lv.loss_scan_1d(
    params,
    direction,
    loss_function,
    n_steps=200,
    end_points=None,
    n_jobs=-1,
)
```

##### Parameters

* **params** (`ParameterVector`)  
  Reference parameter vector used as the center of the scan.

* **direction** (`ArrayLike`)  
  Direction in parameter space along which the scan is performed.

* **loss\_function** (`Callable[[ParameterVector], float]`)  
  Function returning the scalar loss value associated with a parameter vector.

* **n\_steps** (`int`, optional)  
  Number of evaluation points along the scan direction.  
  Default is `200`.

* **end\_points** (`tuple[float, float] | None`, optional)  
  Bounds of the scan parameter. If `None`, the default interval
  `(-π, π)` is used.

* **n\_jobs** (`int`, optional)  
  Number of parallel jobs used during the scan evaluation.  
  Default is `-1` (use all available CPUs).

````markdown
#### `loss_scan_2d_3d`

Evaluate the loss function on a **two-dimensional parameter plane**.

This utility explores the loss landscape over a plane defined by two directions
in parameter space. The loss function is evaluated on a regular two-dimensional
grid, allowing visualization of local minima, valleys, saddle points, and
overall surface structure. Optionally, a 3D representation of the loss surface
can be generated.

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
    n_jobs=-1,
)
````

##### Parameters

* **params** (`ParameterVector`)  
  Reference parameter vector used as the center of the scan.

* **direction1** (`ArrayLike`)  
  First direction in parameter space defining the scan plane.

* **direction2** (`ArrayLike`)  
  Second direction in parameter space defining the scan plane.

* **loss\_function** (`Callable[[ParameterVector], float]`)  
  Function returning the scalar loss value associated with a parameter vector.

* **n\_steps** (`int`, optional)  
  Number of evaluation points along each scan direction.  
  Default is `100`.

* **end\_points\_x** (`tuple[float, float] | None`, optional)  
  Bounds of the scan parameter along the first direction.  
  If `None`, the default interval `(-π, π)` is used.

* **end\_points\_y** (`tuple[float, float] | None`, optional)  
  Bounds of the scan parameter along the second direction.  
  If `None`, the default interval `(-π, π)` is used.

* **plot3D** (`bool`, optional)  
  Whether to generate a 3D visualization of the loss surface.  
  Default is `True`.

* **n\_jobs** (`int`, optional)  
  Number of parallel jobs used during the scan evaluation.  
  Default is `-1` (use all available CPUs).

```
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
