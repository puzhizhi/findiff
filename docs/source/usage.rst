==========
User Guide
==========

Installation
::::::::::::

.. code-block:: ipython

    pip install --upgrade findiff


Derivatives
:::::::::::

Defining Derivatives
--------------------

Basic Usage
...........

The :code:`Diff` object in *findiff* allows to define partial derivatives
along any axis. There are different way how you can do that. To define
the *n*-th partial derivative along axis 'k',

.. math::
    \frac{\partial^n}{\partial x_k^n}

you can write

.. code-block:: ipython

    from findiff import Diff

    dk_dxk = Diff(k, n)

The second argument, the degree of the derivative has a default of 1. So
a first derivative along axis *k* can simply be written as :code:`Diff(k)`.

Mixed Partial Derivatives
.........................

The most general way to define a :code:`Diff` object is by using a :code:`dict`.
This can always be used as an alternative, but is the only if you want to specify
a mixed partial derivative, like :math:`\frac{\partial^{2}}{\partial x_0 \partial x_2}`.
For instance, the triple mixed partial derivative

.. math::

    \frac{\partial^{n+m+l}}{\partial x_r^n \partial x_s^m \partial x_t^l}

this would read

.. code-block:: ipython

    Diff({r: n, s: m, t: l}

Keys denote the axes, values the corresponding degrees.


More General Differential Operators
...................................

You can create more general differential operators by forming algebraic expressions
of :code:`Diff` objects. For instance, the 3D Laplace operator

.. math::

    \nabla^2 = \frac{\partial^2}{\partial x_0^2} + \frac{\partial^2}{\partial x_1^2} + \frac{\partial^2}{\partial x_2^2}

can be defined by writing

.. code-block:: ipython

    laplace = Diff(0, 2) + Diff(1, 2) + Diff(2, 2)

Coefficients like in

.. math::

    3 x_1^2 \frac{\partial^2}{\partial x_0^2} + 2 x_0 \frac{\partial^2}{\partial x_1^2}

must be wrapped in `Coef` objects:

.. code-block:: ipython

    from findiff import Diff, Coef

    # (X_0, X_1 are some values or arrays)
    diff_op = Coef(3 * X_1**2) * Diff(0, 2) + Coef(2 * X_0) * Diff(1, 2)

Chaining differential operators is also possible, like

.. math::

    \left(\frac{\partial}{\partial x} - \frac{\partial}{\partial y}\right) \cdot
    \left(\frac{\partial}{\partial x} + \frac{\partial}{\partial y}\right)
    = \frac{\partial^2}{\partial x^2} - \frac{\partial^2}{\partial y^2}

can be written as

.. code-block:: ipython

    (Diff(0) - Diff(1)) * (Diff(0) + Diff(1))

and is equivalent to

.. code-block:: ipython

    D(0, 2) - D(1, 2)


Applying Derivatives to Arrays
------------------------------

Once you have defined a derivative operator, you can apply it to any *numpy* array of the right size.
Right size means, if you have defined a derivative along axis `k`, the array must have at least `k` dimensions.
Applying the derivative is easy, you just call the operator. For instance, let's apply the mixed partial
derivative :math:`\frac{\partial^3}{\partial x \partial y}` to a 2D array :code:`arr` and specify the grid
spacing:

.. code-block:: ipython

    # Define:
    d2_dxdy = Diff({0: 1, 1: 1})

    # Apply:
    result = d2_dxdy(arr, spacing=dx)

Specifying Grid Spacings
........................

The array at which you apply the derivative is just a bunch of numbers that stand for function values
at discrete grid points. The array itself does not know, how far away the grid points are. So you must
specify this information when applying a derivative. There are different ways how you can do that.
The :code:`spacing` argument accepts different formats. When you give just a number, like in

.. code-block:: ipython

    spacing=0.01

*findiff* interprets this as a uniform grid with the same spacing between all points along all axes.
If the spacing is different along the different axes, you can specify that by giving a :code:`dict`
instead:

.. code-block:: ipython

    spacing={0: dx, 1: dy, 2: dz}

You only have to define spacings along axes where you take a derivative. Along all other axes, the
information is not needed.

Finally, as a third option, you can specify the spacing by giving an instance of the `Spacing` class
which is used internally by *findiff*:

.. code-block:: ipython

    spacing=Spacing({0: dx, 1: dy})

The :code:`Spacing` constructor also accepts a :code:`dict` or a single number.


Accuracy Control
................

By default, *findiff* uses finite difference schemes, which have approximation error terms of second order.
That is, as the grid spacing becomes smaller by a factor of 2, the approximation error goes down by a
factor of 4. You can set higher accuracy by giving the :code:`acc` argument when applying a derivative.
For instance, to switch to fourth order:

.. code-block:: ipython

    >>> d_dx = Diff(0)
    >>> result = d_dx(arr, spacing=dx, acc=4)

:code:`acc` can be any positive even integer.


Stencils and Coefficients
:::::::::::::::::::::::::

*findiff* uses finite difference stencils as approximations for differential
operators. When you define an operator, *findiff* creates a standard set of stencils,
which you can inspect. But you can also define your own stencils.

Stencil Sets
............

Suppose you define a 2D Laplacian

.. code-block:: ipython

    >>> diff = Diff(0, 2) + Diff(1, 2)

You can get get access to the stencils of this differential operator by
calling the :code:`stencils_repr` function:

.. code-block:: ipython

    >>> from findiff import stencils_repr
    >>> stencil_set = stencils_repr(diff)

The function returns an instance of the :code:`StencilSet` class. In our example,
*findiff* assumes that the grid spacing is 1 along all axes and that
the dimension of space is implicitly given by the highest axis in the differential
operator (in this case highest axes is 0, so inferred number of dimensions is 1).
This behavior can be modified as described later. But first, let's have a look
at the stencils. The easiest way is to use the :code:`as_dict()` method of the
:code:`StencilSet` object:

.. code-block:: ipython

    >>> stencil_set.as_dict()

    {('L', 'L'): {(0, 0): 4.0, (0, 1): -5.0, (0, 2): 4.0, (0, 3): -1.0, (1, 0): -5.0, (2, 0): 4.0, (3, 0): -1.0},
     ('L', 'C'): {(0, -1): 1.0, (0, 1): 1.0, (1, 0): -5.0, (2, 0): 4.0, (3, 0): -1.0},
     ('L', 'H'): {(0, -3): -1.0, (0, -2): 4.0, (0, -1): -5.0, (0, 0): 4.0, (1, 0): -5.0, (2, 0): 4.0, (3, 0): -1.0},
     ('C', 'L'): {(-1, 0): 1.0, (0, 1): -5.0, (0, 2): 4.0, (0, 3): -1.0, (1, 0): 1.0},
     ('C', 'C'): {(-1, 0): 1.0, (0, -1): 1.0, (0, 0): -4.0, (0, 1): 1.0, (1, 0): 1.0},
     ('C', 'H'): {(-1, 0): 1.0, (0, -3): -1.0, (0, -2): 4.0, (0, -1): -5.0, (0, 0): 0.0, (1, 0): 1.0},
     ('H', 'L'): {(-3, 0): -1.0, (-2, 0): 4.0, (-1, 0): -5.0, (0, 0): 4.0, (0, 1): -5.0, (0, 2): 4.0, (0, 3): -1.0},
     ('H', 'C'): {(-3, 0): -1.0, (-2, 0): 4.0, (-1, 0): -5.0, (0, -1): 1.0, (0, 0): 0.0, (0, 1): 1.0},
     ('H', 'H'): {(-3, 0): -1.0, (-2, 0): 4.0, (-1, 0): -5.0, (0, -3): -1.0, (0, -2): 4.0, (0, -1): -5.0, (0, 0): 4.0}}

In the interior of the grid (the :code:`('C', 'C') case), the stencil looks
like this:

.. image:: images/laplace2d.png
    :width: 400
    :align: center

The blue points denote the grid points used by the stencil, the tu  ple
below denotes the offset from the current grid point and the value
inside the blue dot represents the finite different coefficient for
grid point. So, this stencil evaluates the Laplacian at the center of
the "cross" of blue points. Obviously, this does not work near the boundaries
of the grid because that stencil would require points "outside" of the
grid. So near the boundary, *findiff* switches to asymmetric stencils
(of the same accuracy order), for example

.. image:: images/stencil_laplace2d_corner.png
    :width: 400
    :align: center

for a corner :code:`('L', 'L')`, or

.. image:: images/stencil_laplace2d_border.png
    :width: 400
    :align: center

for the left edge :code:`('L', 'C')`.

The :code:`stencils_repr` method works for grids of all dimensions and not just two. But of course,
it is not easy to visualize for higher dimensions.

Each stencil is basically a collection of key-value pairs. For each offset, we have a coefficient.

If your grid spacing is not 1, or if the number of dimensions is higher than what can
be inferred from your differential operator, you must be more explicit by writing

.. code-block:: ipython

    >>> stencil_set = stencils_repr(diff, spacing={0: dx, 1: dy}, ndims=3)
    >>> stencil_set.as_dict()

which would define a stencil set for a 2D Laplacian in a 3D space.

Stencil sets can directly be applied to arrays. In fact, when you apply a derivative
in *findiff*, it just delegates work to its :code:`StencilSet` instance.

To get access to a specific stencil, you can access it with bracket notation. In
the present case:

.. code-block:: ipython

    >>> inner_stencil = stencil_set['C', 'C']

A stencil can be applied to arrays, but not to all elements, because for nonzero
offsets at some elements, the offset element would be outside of the grid. But
you can apply a stencil to single elements or to masked parts of a grid. For instance,
we can apply the inner stencil defined above to the inner part of the array :code:`arr`:

.. code-block:: ipython

    # Define a mask where to apply the stencil:
    mask = np.zeros_like(arr, dtype=bool)
    mask[1:-1, 1:-1] = True

    # Apply the stencil there:
    result = inner_stencil(arr, on=mask)

which returns an array with the same shape as :code:`arr`, but only values where
:code:`mask` is true are evaluated.

Alternatively, you can apply the stencil only at a single point. To do that, use the
:code:`at` argument:

.. code-block:: ipython

    inner_stencil(arr, at=(3, 4))   # applies to array at index tuple (3, 4)


Create Custom Stencil
.....................

There are situations when you want to create your own stencils and do not
want to use the stencils automatically created by :code:`Diff` objects.
This is mainly for exploratory work. For example, you may wonder, how the
coefficients for the 2D Laplacian look like if you don't use the cross-shaped
stencil from the previous section but rather an x-shaped one:

.. image:: images/laplace2d-x.png
    :width: 400
    :align: center

This can easily be determined with *findiff* by creating a custom stencil from
the :code:`StencilFactory`:

.. code-block:: ipython

    from findiff import StencilFactory

    laplacian = Diff(0, 2) + Diff(1, 2)


    offsets = [(0, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)] # x-shaped offsets
    factory = StencilFactory()
    factory.create(offsets, laplacian, spacing=0.1, symbolic=True)

#    stencil = Stencil(offsets, {(2, 0): 1, (0, 2): 1})

returns

.. code-block:: ipython

    {(0, 0): -2.0, (1, 1): 0.5, (-1, -1): 0.5, (1, -1): 0.5, (-1, 1): 0.5}

The second argument of the :code:`Stencil` constructor defines the derivative operator:

.. code-block::

    {(2, 0): 1, (0, 2): 1}

corresponds to

.. math::
    1 \cdot \frac{\partial^2}{\partial x_0} + 1 \cdot \frac{\partial^2}{\partial x_1}.


StencilFactory
Symbolics


Matrix Representations
::::::::::::::::::::::

For a given FinDiff differential operator, you can get the matrix
representation using the matrix(shape) method, e.g.

.. code-block:: ipython

    x = [np.linspace(0, 6, 7)]
    d2_dx2 = Diff(0, 2)
    u = x**2

    mat = matrix_repr(d2_dx2, u.shape)  # this function returns a scipy sparse matrix
    print(mat.toarray())

yields

.. code-block:: ipython

    [[ 2. -5.  4. -1.  0.  0.  0.]
     [ 1. -2.  1.  0.  0.  0.  0.]
     [ 0.  1. -2.  1.  0.  0.  0.]
     [ 0.  0.  1. -2.  1.  0.  0.]
     [ 0.  0.  0.  1. -2.  1.  0.]
     [ 0.  0.  0.  0.  1. -2.  1.]
     [ 0.  0.  0. -1.  4. -5.  2.]]

Of course this also works for general differential operators.


Stencils
::::::::

Automatic Stencils
------------------


Stencils From Scratch
---------------------

