# Landscape Characterization

## Installation

```bash
pip install --upgrade git+https://github.com/ColibrITD-SAS/landscape_tools
```

```python
from landscape_tools import landscape_visualization as lv
from landscape_tools import barren_plateaus as bp
```

---

# Landscape Tools

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

In practice, you must wrap the original cost function so that it only
takes the parameter vector as input and returns a scalar value.

```python
def your_cost_function(params, circuit, hamiltonian):
    state = circuit.run(params)
    return expectation_value(state, hamiltonian)

wrapped_loss = lambda p: cost_function(
    p,
    circuit=my_circuit,
    hamiltonian=my_hamiltonian,
)
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

- **params** (`ParameterVector`)  
  Reference parameter vector used as the center of the scan.

- **direction** (`ArrayLike`)  
  Direction in parameter space along which the scan is performed.

- **loss_function** (`Callable[[ParameterVector], float]`)  
  Function returning the scalar loss value associated with a parameter vector.

- **n_steps** (`int`, optional)  
  Number of evaluation points along the scan direction.  
  Default is `200`.

- **end_points** (`tuple[float, float] | None`, optional)  
  Bounds of the scan parameter. If `None`, the default interval
  `(-π, π)` is used.

- **n_jobs** (`int`, optional)  
  Number of parallel jobs used during the scan evaluation.  
  Default is `-1` (use all available CPUs).

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
```

##### Parameters

- **params** (`ParameterVector`)  
  Reference parameter vector used as the center of the scan.

- **direction1** (`ArrayLike`)  
  First direction in parameter space defining the scan plane.

- **direction2** (`ArrayLike`)  
  Second direction in parameter space defining the scan plane.

- **loss_function** (`Callable[[ParameterVector], float]`)  
  Function returning the scalar loss value associated with a parameter vector.

- **n_steps** (`int`, optional)  
  Number of evaluation points along each scan direction.  
  Default is `100`.

- **end_points_x** (`tuple[float, float] | None`, optional)  
  Bounds of the scan parameter along the first direction.  
  If `None`, the default interval `(-π, π)` is used.

- **end_points_y** (`tuple[float, float] | None`, optional)  
  Bounds of the scan parameter along the second direction.  
  If `None`, the default interval `(-π, π)` is used.

- **plot3D** (`bool`, optional)  
  Whether to generate a 3D visualization of the loss surface.  
  Default is `True`.

- **n_jobs** (`int`, optional)  
  Number of parallel jobs used during the scan evaluation.  
  Default is `-1` (use all available CPUs).

### PCA-based landscape analysis

#### `perform_pca_and_analysis`

Run a **complete PCA-based loss landscape analysis pipeline**.

This utility builds a low-dimensional representation of an optimization
trajectory using Principal Component Analysis (PCA), performs loss scans in the
resulting PCA subspace, and generates interpretability analyses of the dominant
optimization directions.

The function is designed to help visualize and understand the geometry of the
optimization process and the most significant parameter variations.

```python
lv.perform_pca_and_analysis(
    params_history,
    loss_function,
    n_steps,
    offset,
    n_top,
    circuit,
    n_jobs=-1,
)
```

##### Parameters

- **params_history** (`np.ndarray`)  
  Array containing the parameter vectors recorded during optimization.

- **loss_function** (`Callable[[ParameterVector], float]`)  
  Function returning the scalar loss value associated with a parameter vector.

- **n_steps** (`int`)  
  Number of evaluation points along each PCA direction.

- **offset** (`float | tuple[float, float]`)  
  Margin added to the PCA scan bounds to enlarge the explored region.

- **n_top** (`int`)  
  Number of top-ranked entries displayed in interpretability analyses.

- **circuit** (`Any`)  
  Quantum circuit associated with the analyzed parameter space.

- **n_jobs** (`int`, optional)  
  Number of parallel jobs used during the scan evaluation.  
  Default is `-1` (use all available CPUs).

##### Returns

- Mapping returned by `analyze_pca`.

---

# Barren Plateau

This module provides utilities to study **barren plateau phenomena** in
variational quantum algorithms (VQAs).

It enables systematic scaling analyses of loss variances as a function of:

- the number of qubits,
- the circuit depth,
- the observable structure,
- and padding strategies for growing operators.

The implementation is designed to remain backend-agnostic and relies on generic
user-defined circuit generation, parameter initialization, and cost function
construction routines.

### Main analysis function

#### `barren_plateaus_analysis`

Run a **complete barren plateau scaling analysis pipeline**.

This utility automates the generation of variational circuits, parameter
sampling, variance estimation, bootstrap diagnostics, and visualization of the
loss variance scaling behavior across different system sizes and circuit depths.

It can be used to investigate whether the variance of the cost function decays
exponentially with problem size, a characteristic signature of barren plateaus.

```python
bp.barren_plateaus_analysis(
    experiment,
    cost_function_builder,
    generate_params,
    generate_circuits,
    sampling=None,
    execution=None,
    variance_normalization=None,
    **cost_kwargs,
)
```

##### Parameters

- **experiment** (`ExperimentConfig`)  
  Configuration object defining the barren plateau experiment settings.

- **cost_function_builder** (`Callable`)  
  Function used to construct the cost function evaluated during the analysis.

- **generate_params** (`Callable`)  
  Function generating random parameter vectors for the variational circuits.

- **generate_circuits** (`Callable`)  
  Function generating the quantum circuits associated with the experiment.

- **sampling** (`SamplingConfig | None`, optional)  
  Configuration controlling adaptive sampling and bootstrap variance
  estimation.  
  If `None`, default sampling settings are used.

- **execution** (`ExecutionConfig | None`, optional)  
  Configuration controlling execution parameters such as parallelization and
  verbosity.  
  If `None`, default execution settings are used.

- **variance_normalization** (`Callable | None`, optional)  
  Optional normalization applied to the estimated variances before analysis.

- **\*\*cost_kwargs**  
  Additional keyword arguments forwarded to the cost function builder.

##### Returns

- **dict**  
  Dictionary containing the computed variance statistics, diagnostics, and
  analysis results.

### Available analysis modes

The behavior of the analysis is controlled through the
`experiment.analysis_type` field.

Three main scaling studies are supported.

#### 1. Qubit scaling analysis

Study how the variance evolves as the number of qubits increases while keeping
the circuit depth fixed.

This is the standard barren plateau analysis used to detect exponential decay
with system size.

```python
experiment = bp.ExperimentConfig(
    analysis_type="qubits",
    N_qubits=[4, 6, 8, 10],
    N_layers=[2, 4],
    Ansatz="HardwareEfficient",
)
```

#### 2. Layer scaling analysis

Study how the variance evolves as the circuit depth increases for fixed system
sizes.

This analysis is useful for understanding how expressibility and depth impact
trainability.

```python
experiment = bp.ExperimentConfig(
    analysis_type="layers",
    N_qubits=[6, 8],
    N_layers=[1, 2, 4, 8],
    Ansatz="HardwareEfficient",
)
```

#### 3. Padding scaling analysis

Study how different Pauli-string growth strategies affect barren plateau
behavior when increasing the system size.

This mode compares several operator padding schemes during qubit scaling.

```python
experiment = bp.ExperimentConfig(
    analysis_type="padding",
    N_qubits=[4, 6, 8, 10],
    N_layers=[2, 4],
    Ansatz="HardwareEfficient",
    initial_Pauli_string="ZZ",
    padding_types=[
        "linear_half",
        "linear_full",
        "logarithmic",
    ],
)
```

### Example

```python
from landscape_characterization import barren_plateaus as bp

experiment = bp.ExperimentConfig(
    analysis_type="qubits",
    N_qubits=[4, 6, 8],
    N_layers=[2, 4],
    Ansatz="HardwareEfficient",
)

results = bp.barren_plateaus_analysis(
    experiment=experiment,
    cost_function_builder=build_cost_function,
    generate_params=generate_params,
    generate_circuits=generate_circuits,
)
```
