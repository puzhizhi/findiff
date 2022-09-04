# <img src="docs/frontpage/findiff_logo.png" width="100px"> findiff
[![PyPI version](https://badge.fury.io/py/findiff.svg)](https://badge.fury.io/py/findiff)
![Build status](https://img.shields.io/github/workflow/status/maroba/findiff/Checks)
![Coverage](https://img.shields.io/codecov/c/github/maroba/findiff/master.svg)
[![Doc Status](https://readthedocs.org/projects/findiff/badge/?version=latest)](https://findiff.readthedocs.io/en/latest/index.html)
[![PyPI downloads](https://img.shields.io/pypi/dm/findiff.svg)]()
[![Downloads](https://static.pepy.tech/personalized-badge/findiff?period=total&units=international_system&left_color=black&right_color=blue&left_text=Downloads)](https://pepy.tech/project/findiff)


A Python package for finite difference numerical derivatives and partial differential equations in
any number of dimensions. 

## Main Features

* Differentiate arrays of any number of dimensions along any axis with any desired accuracy order
* Accurate treatment of grid boundary
* Can handle arbitrary linear combinations of derivatives with constant and variable coefficients
* Generate stencils of any shape
* Fully vectorized for speed
* Matrix representations of arbitrary linear differential operators


## Installation

```
pip install --upgrade findiff
```

## Documentation and Examples

You can find the documentation of the code including examples of application at https://findiff.readthedocs.io/en/latest/.


## Taking Derivatives

*findiff* allows to easily define derivative operators that you can apply to *numpy* arrays of any dimension.
The syntax for a simple derivative operator is simply 

```python
Diff(axis, degree)
```

where `axis` is the axis along which to take the partial derivative and `degree` (which defaults to 1) is the
degree of the derivative along that axis.

Once defined, you can apply it to any *numpy* array of adequate shape. For instance, consider the 1D case
with a first derivative <img src="docs/frontpage/d_dx.png" height="24"> along the only axis (0):

```
import numpy as np
from findiff import Diff

x = np.linspace(0, 1, 101)
dx = x[1] - x[0]
f = np.sin(x)  # as an example

# Define the derivative:
d_dx = Diff(0, 1)

# Apply it:
df_dx = d_dx(f, spacing=dx) 
```

Similary, you can define partial derivative operators along different axes or of higher degree, for example 
(```f``` is a *numpy* array of suitable shape):

| Math                                                  | *findiff* define                                                                 | *findiff* apply                    |
|-------------------------------------------------------|----------------------------------------------------------------------------------|------------------------------------|
| <img src="docs/frontpage/d_dy.png" height="50px">     | ```d = Diff(1, 1)``` <br> or simply <br> ```Diff(1)```                           | ```d(f, spacing=dy)```             |
| <img src="docs/frontpage/d4_dy4.png" height="50px">   | ```d = Diff(1, 4)```  <br> any degree is possible                                | ```d(f, spacing=dy)```             |
| <img src="docs/frontpage/d3_dx2dz.png" height="50px"> | ```d = Diff(0, 2) * Diff(2, 1)``` <br> or directly <br> ```Diff({0: 2, 2: 1})``` | ```d(f, spacing={0: dx, 2: dz})``` |
| <img src="docs/frontpage/d_dx_10.png" height="50px">  | ```d = Diff(10)```      <br>number of axes not limited                           | ```d(f, spacing={10: dx_10})```    |

We can also take linear combinations of derivative operators, for example:

<img src="docs/frontpage/var_coef.png" alt="variableCoefficients" height="40"/>

is

```python
Coef(2*X) * Diff({0: 2, 2: 1}) + Coef(3*sin(Y)*Z**2) * Diff({0: 1, 1: 2})
```

where `X, Y, Z` are *numpy* arrays with meshed grid points.

Chaining differential operators is also possible, e.g.

<img src="docs/frontpage/chaining.png" alt="chaining" height="40"/>

can be written as

```python
(Diff(0) - Diff(1)) * (Diff(0) + Diff(1))
```

and

```python
Diff(0, 2) - Diff(1, 2)
```

Of course, `Diff` obeys the product rule, if you define some operator with variable coefficients.

Standard operators from vector calculus like gradient, divergence and curl are also available
as shortcuts.


### Accuracy Control

When applying an instance of `Diff`, you can request the desired accuracy
order by setting the keyword argument `acc`. For example:

```
d2_dx2 = Diff(0, 2)
d2f_dx2 = d2_dx2(f, acc=4, spacing=dx)
```

If not specified, second order accuracy will be taken by default.


## Finite Difference Coefficients

Sometimes you may want to have the finite difference coefficients directly.
These can be obtained for __any__ derivative and accuracy order
using `findiff.coefficients(deriv, acc)`. For instance,

```python
import findiff

coefs = findiff.coefficients(deriv=3, acc=4)
```

gives

```
{'backward': {'coefficients': [15/8, -13, 307/8, -62, 461/8, -29, 49/8],
              'offsets': [-6, -5, -4, -3, -2, -1, 0]},
 'center': {'coefficients': [1/8, -1, 13/8, 0, -13/8, 1, -1/8],
            'offsets': [-3, -2, -1, 0, 1, 2, 3]},
 'forward': {'coefficients': [-49/8, 29, -461/8, 62, -307/8, 13, -15/8],
             'offsets': [0, 1, 2, 3, 4, 5, 6]}}
```

If you want to specify the detailed offsets instead of the
accuracy order, you can do this by setting the offset keyword
argument:

```python
import findiff

coefs = findiff.coefficients(deriv=2, offsets=[-2, 1, 0, 2, 3, 4, 7])
```

The resulting accuracy order is computed and part of the output:

```
{'coefficients': [187/1620, -122/27, 9/7, 103/20, -13/5, 31/54, -19/2835], 
 'offsets': [-2, 1, 0, 2, 3, 4, 7], 
 'accuracy': 5}
```

## Matrix Representations

For a given _FinDiff_ differential operator, you can get the matrix representation 
using the `matrix(shape)` method, e.g. for a small 1D grid of 10 points:

```python
from findiff import Diff, matrix_repr
x = np.linspace(...)
d2_dx2 = Diff(0, 2)

mat = matrix_repr(d2_dx2, x.shape)  # this function returns a scipy sparse matrix
print(mat.toarray())
``` 

has the output

```
[[ 2. -5.  4. -1.  0.  0.  0.]
 [ 1. -2.  1.  0.  0.  0.  0.]
 [ 0.  1. -2.  1.  0.  0.  0.]
 [ 0.  0.  1. -2.  1.  0.  0.]
 [ 0.  0.  0.  1. -2.  1.  0.]
 [ 0.  0.  0.  0.  1. -2.  1.]
 [ 0.  0.  0. -1.  4. -5.  2.]]
```

The same works for more general differential operators. Just pass it to `matrix_repr` and it
will return its matrix representation as sparse matrix. 

## Stencils

*findiff* uses standard stencils (patterns of grid points) to evaluate the derivative.
However, you can design your own stencil. A picture says more than a thousand words, so
look at the following example for a standard second order accurate stencil for the 
2D Laplacian <img src="docs/frontpage/laplacian2d.png" height="30">:

<img src="docs/frontpage/laplace2d.png" width="400">

This can be reproduced by *findiff* writing

```
offsets = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
stencil = Stencil(offsets, partials={(2, 0): 1, (0, 2): 1}, spacings=(1, 1))
```

The attribute `stencil.values` contains the coefficients

```
{(0, 0): -4.0, (1, 0): 1.0, (-1, 0): 1.0, (0, 1): 1.0, (0, -1): 1.0}
```

Now for a some more exotic stencil. Consider this one:

<img src="docs/frontpage/laplace2d-x.png" width="400">

With *findiff* you can get it easily:

```
offsets = [(0, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]
stencil = Stencil(offsets, partials={(2, 0): 1, (0, 2): 1}, spacings=(1, 1))
stencil.values
```
which returns

```
{(0, 0): -2.0, (1, 1): 0.5, (-1, -1): 0.5, (1, -1): 0.5, (-1, 1): 0.5}
```

## What about the old API?

The release versions of *findiff* before version 1.0.0 had a different API than the one presented here. 
However, the old API can still be used. It is available in the `findiff.legacy` subpackage now. For example:

```
from findiff.legacy import FinDiff
...
```

## Citations

You have used *findiff* in a publication? Here is how you can cite it:

> M. Baer. *findiff* software package. URL: https://github.com/maroba/findiff. 2018

BibTeX entry:

```
@misc{findiff,
  title = {{findiff} Software Package},
  author = {M. Baer},
  url = {https://github.com/maroba/findiff},
  key = {findiff},
  note = {\url{https://github.com/maroba/findiff}},
  year = {2018}
}
```

## Development

### Set up development environment

- Fork the repository
- Clone your fork to your machine
- Install in development mode:

```
python setup.py develop
```

### Running tests

Install test dependencies.

```
python -m pip install pytest
```

Then run the test from the console (assuming you are in the project root directory):

```
python -m pytest test
```

### Build the Docs

First install the documentation dependencies:

```
pip install -r docs/requirements.txt 
```

Then go to the `./docs` directory and trigger the build:

```
make html
```

The documentation website is then locally available with `docs/_build/html/index.html` as
home page.

### Contributing

Please open an issue for any questions or problems that may arise using *findiff*. 
If you make changes to the code or the documentation, work on your own fork and send me a pull request. 
Before doing so, please make sure that all tests are running and your changes are covered by additional tests.



