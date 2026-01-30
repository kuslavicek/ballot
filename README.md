# Ballot: Balanced k-means clustering with optimal transport

![](https://img.shields.io/badge/SciPy-654FF0?logo=SciPy&logoColor=white)
[![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=fff)](#)
[![NumPy](https://img.shields.io/badge/NumPy-4DABCF?logo=numpy&logoColor=fff)](#)
[![Pytest](https://img.shields.io/badge/Pytest-fff?logo=pytest&logoColor=000)](#)
[![PyPI](https://img.shields.io/badge/PyPI-3775A9?logo=pypi&logoColor=fff)](#)
![License](https://img.shields.io/github/license/kuslavicek/ballot)
![Version](https://img.shields.io/github/v/release/kuslavicek/ballot)
![Maintained](https://img.shields.io/badge/Maintained%3F-yes-green.svg)

**Ballot** (Balanced Lloyd with Optimal Transport) is a high-performance Python package for balanced clustering. It solves the problem of creating equal-sized clusters (or clusters with specific capacity constraints) by leveraging Optimal Transport theory and Entropic Regularization (Sinkhorn algorithm).

## Features

- **Speed**: Uses Sinkhorn iterations (E-BalLOT) for near-linear time complexity $O(n \log n)$, making it usable for large datasets ($n > 100,000$).
- **Simplicity**: precise, math-driven implementation without complex C++ dependencies.
- **Scikit-learn Compatible**: Designed to fit seamlessly into existing ML pipelines.

## Installation

Install via pip:

```bash
pip install ballot
```

## Usage

```python
import numpy as np
from ballot.core import solve_entropic_kantorovich

# Example usage (API subject to change in v0.1)
# Create random data and centroids...
# Run balanced clustering...
```

## Development

To install in editable mode for development:

```bash
git clone https://github.com/username/ballot.git
cd ballot
pip install -e .
```

Run tests:

```bash
pytest
```

## References

This project incorporates research from the following paper:

- **BalLOT: Balanced k-means clustering with optimal transport**
  Wenyan Luo, Dustin G. Mixon
  *arXiv:2512.05926*
