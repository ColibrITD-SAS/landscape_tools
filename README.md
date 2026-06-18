<p align="center">
  <a href="https://github.com/ColibrITD-SAS/landscape_tools/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/ColibrITD-SAS/landscape_tools?label=license&style=flat-square" alt="License">
  </a>
  <a href="https://github.com/ColibrITD-SAS/landscape_tools/actions/workflows/doc.yml">
    <img src="https://github.com/ColibrITD-SAS/landscape_tools/actions/workflows/doc.yml/badge.svg" alt="Documentation">
  </a>
  <a href="https://github.com/ColibrITD-SAS/landscape_tools/actions/workflows/pypi.yml">
    <img src="https://github.com/ColibrITD-SAS/landscape_tools/actions/workflows/pypi.yml/badge.svg" alt="PyPI workflow">
  </a>
  <a href="https://pypi.org/project/landscape-tools/">
    <img src="https://img.shields.io/pypi/v/landscape-tools?label=release&style=flat-square" alt="PyPI version">
  </a>
  <a href="https://github.com/ColibrITD-SAS/landscape_tools/stargazers">
    <img src="https://img.shields.io/github/stars/ColibrITD-SAS/landscape_tools?style=flat-square&logo=github" alt="GitHub stars">
  </a>
  <a href="https://pypi.org/project/landscape-tools/">
    <img src="https://img.shields.io/pypi/pyversions/landscape-tools?label=python&style=flat-square&logo=python" alt="Python versions">
  </a>
</p>

![alt text](Designer.png)

# About the library

**Landscape Tools** is a Python library designed to visualize, characterize, and analyze the loss landscapes of variational quantum algorithms. It provides utilities for one-dimensional, two-dimensional, and three-dimensional loss scans, PCA-based landscape projections, optimization trajectory visualization, and interpretable parameter-sensitivity analysis directly on quantum circuits. The library also includes tools for studying barren plateaus through variance-based scaling analyses over circuit depth, number of qubits, observables, and Pauli-string padding strategies. Built with quantum machine learning experiments in mind, Landscape Tools helps researchers diagnose trainability issues, inspect optimization dynamics, and better understand how variational quantum ansätze behave across parameter space.

<p align="center">
  <img src="ezgif.com-animated-gif-maker.gif" alt="Landscape Tools demo" width="800">
</p>

## Installation

```bash
pip install landscape_tools
```

```python
from landscape_tools import landscape_visualization, barren_plateaus
```

## Landscape Characterization

This module provides utilities to analyze and visualize the optimization
landscape of variational quantum algorithms (VQAs).

It is designed to work **independently of QUICK / HDES internals** and can be
used with any backend, as long as a suitable cost function is provided.

The main purpose of this module is to:

- perform 1D and 2D loss landscape scans,
- analyze loss landscapes in PCA subspaces,
- study gradient magnitudes and barren plateau effects,
- visualize trajectories and parameter influence.

Most algorithms implemented here only depend on a
**generic cost function interface**, not on circuits, observables, or backends.

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
