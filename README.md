# PyLOESS

This is a vectorized implementation of LOESS that supports polynomial models.
It is fast enough for bootstrap resampling for computing prediction intervals.

## Installation

```bash
pip install pyloess
```

## Usage

```python
from pyloess import loess

# Generate some data
import numpy as np
np.random.seed(0)

x = np.random.uniform(0, 10, 100)
y = np.sin(x) + np.random.normal(0, 0.1, 100)
x_new = np.linspace(0, 10, 1000)

# Evaluate the loess model
y_new = loess(x, y, eval_x=x_new, span=0.33, degree=2)
```
