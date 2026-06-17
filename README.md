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

update the documentation with

```sh
sphinx-build -b html docs build
```

but of course, in order to do that, you need the lastest dev dependancies,
install them with

```sh
pip install -r dependencies-dev.txt
```

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
