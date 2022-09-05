import numpy as np
import scipy.sparse

from findiff.stencils import Stencil
from ..algebraic import Algebraic, Numberlike, Add, Mul, Operation
from ..deriv import PartialDerivative, matrix_repr, EquidistantGrid
from ..stencils import StencilSet, SymmetricStencil1D, ForwardStencil1D, BackwardStencil1D
from .pde import BoundaryConditions

__all__ = [
    'FinDiff', 'Coef', 'Identity', 'coefficients', 'Gradient', 'Divergence', 'Curl', 'Laplacian'
]


class FinDiff(Algebraic):
    """ A representation of a general linear differential operator expressed in finite differences.

        FinDiff objects can be added with other FinDiff objects. They can be multiplied by
        objects of type Coefficient.

        FinDiff is callable, i.e. to apply the derivative, just call the object on the array to
        differentiate.

        :param args: variable number of tuples. Defines what derivative to take.
            If only one tuple is given, you can leave away the tuple parentheses.

        Each tuple has the form

               `(axis, spacing, count)`

             `axis` is the dimension along which to take derivative.

             `spacing` is the grid spacing of the uniform grid along that axis.

             `count` is the order of the derivative, which is optional an defaults to 1.


        :param kwargs:  variable number of keyword arguments

            Allowed keywords:

            `acc`:    even integer
                  The desired accuracy order. Default is acc=2.

        This class is actually deprecated and will be replaced by the Diff class in the future.

        **Example**:


       For this example, we want to operate on some 3D array f:

       >>> import numpy as np
       >>> x, y, z = [np.linspace(-1, 1, 100) for _ in range(3)]
       >>> X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
       >>> f = X**2 + Y**2 + Z**2

       To create :math:`\\frac{\\partial f}{\\partial x}` on a uniform grid with spacing dx, dy
       along the 0th axis or 1st axis, respectively, instantiate a FinDiff object and call it:

       >>> d_dx = FinDiff(0, dx)
       >>> d_dy = FinDiff(1, dx)
       >>> result = d_dx(f)

       For :math:`\\frac{\\partial^2 f}{\\partial x^2}` or :math:`\\frac{\\partial^2 f}{\\partial y^2}`:

       >>> d2_dx2 = FinDiff(0, dx, 2)
       >>> d2_dy2 = FinDiff(1, dy, 2)
       >>> result_2 = d2_dx2(f)
       >>> result_3 = d2_dy2(f)

       For :math:`\\frac{\\partial^4 f}{\partial x \\partial^2 y \\partial z}`, do:

       >>> op = FinDiff((0, dx), (1, dy, 2), (2, dz))
       >>> result_4 = op(f)


        """

    legacy = True

    def __init__(self, *args, **kwargs):

        super(FinDiff, self).__init__()
        self.add_handler = DirtyAdd
        self.mul_handler = DirtyMul
        self.acc = 2

        if 'acc' in kwargs:
            self.acc = kwargs['acc']

        degrees, spacings = self._parse_args(args)
        self.partial = PartialDerivative(degrees)

        # The old FinDiff API does not fully specify the grid.
        # So use a dummy-grid for all non-specified values:
        self.grid = EquidistantGrid.from_spacings(max(degrees.keys()) + 1, spacings)
        self._user_specified_spacings = spacings

    def __call__(self, f, acc=None):
        self.acc = acc or self.acc
        return self.apply(f)

    def apply(self, f):
        return self.partial.apply(f, self.grid, self.acc)

    def matrix(self, shape, acc=None):
        acc = acc or self.acc
        if shape != self.grid.shape:
            # The old FinDiff API does not fully specify the grid.
            # The constructor tentatively constructed a dummy grid. Now
            # update this information. In particular, we now know the exact
            # number of space dimensions:
            self.grid = EquidistantGrid.from_shape_and_spacings(shape, self._user_specified_spacings)
        return self.partial.matrix_repr(self.grid, acc)

    def stencil(self, shape):
        return StencilSet(self, shape)

    def _parse_args(self, args):
        assert len(args) > 0
        canonic_args = []

        def parse_tuple(tpl):
            if len(tpl) == 2:
                canonic_args.append(tpl + (1,))
            elif len(tpl) == 3:
                canonic_args.append(tpl)
            else:
                raise ValueError('Invalid input format for FinDiff.')

        if not hasattr(args[0], '__len__'):  # we expect a pure derivative
            parse_tuple(args)
        else:  # we have a mixed partial derivative
            for arg in args:
                parse_tuple(arg)

        degrees = {}
        spacings = {}
        for axis, spacing, degree in canonic_args:
            if axis in degrees:
                raise ValueError('FinDiff: Same axis specified twice.')
            degrees[axis] = degree
            spacings[axis] = spacing

        return degrees, spacings


class DirtyMixin:
    """A class for mixing in behavior which has nothing to do with the
       semantics of the target object.

       For instance, a Mul object has no notion of a matrix.

       The need for this class only arises from the unfortunate API of
       versions 0.* where one could obtain the matrix representation by
       calling a matrix method on the FinDiff objects and consequently
       also on Mul and Add operators.
    """

    legacy = True

    def __init__(self):
        self.add_handler = DirtyAdd
        self.mul_handler = DirtyMul

    def matrix(self, shape):
        if isinstance(self, Operation):
            left = self.left.matrix(shape)
            right = self.right.matrix(shape)
            return self.operation(left, right)
        elif not isinstance(self, FinDiff):
            grid = EquidistantGrid.from_shape_and_spacings(shape, {})
            return matrix_repr(self, 2, grid)
        return self.matrix(shape)

    def stencil(self, shape):
        return StencilSet(self, shape)


class DirtyNumberlike(DirtyMixin, Numberlike):

    def __init__(self, value):
        Numberlike.__init__(self, value)
        DirtyMixin.__init__(self)


class Coef(DirtyNumberlike):
    def __init__(self, value):
        super(Coef, self).__init__(value)


class Identity(DirtyNumberlike):

    legacy = True

    def __init__(self):
        super(Identity, self).__init__(1)

    def __call__(self, f, *args, **kwargs):
        return f

    def matrix(self, shape):
        n = np.prod(shape)
        return scipy.sparse.diags(np.ones(n))


class DirtyAdd(DirtyMixin, Add):
    wrapper_class = DirtyNumberlike

    def __init__(self, *args, **kwargs):
        Add.__init__(self, *args, **kwargs)
        DirtyMixin.__init__(self)


class DirtyMul(DirtyMixin, Mul):
    wrapper_class = DirtyNumberlike

    def __init__(self, *args, **kwargs):
        Mul.__init__(self, *args, **kwargs)
        DirtyMixin.__init__(self)


class PDE:
    """
        Representation of a partial differential equation.
        """

    def __init__(self, lhs, rhs, bcs):
        """
            Initializes the PDE.

            You need to specify the left hand side (lhs) in terms of derivatives
            as well as the right hand side in terms of an array.

            Parameters
            ----------
            lhs: FinDiff object or combination of FinDiff objects
                the left hand side of the PDE
            rhs: numpy.ndarray
                the right hand side of the PDE
            bcs: BoundaryConditions
                the boundary conditions for the PDE

        """
        from findiff.pde import PDE as NewPDE
        self.pde = NewPDE(lhs, rhs, bcs, lhs.grid)

    def solve(self):
        return self.pde.solve()


def coefficients(deriv, acc=None, offsets=None, symbolic=False):
    """
    Calculates the finite difference coefficients for given derivative order and accuracy order.

    If acc is given, the coefficients are calculated for central, forward and backward
    schemes resulting in the specified accuracy order.

    If offsets are given, the coefficients are calculated for the offsets as specified
    and the resulting accuracy order is computed.

    *Either* acc *or* offsets must be given.

    Assumes that the underlying grid is uniform. This function is available at the `findiff`
    package level.

    :param deriv: The derivative order.
    :type deriv: int > 0

    :param acc: The accuracy order.
    :type acc:  even int > 0:

    :param offsets: The offsets for which to calculate the coefficients.
    :type offsets: list of ints

    :raises ValueError: if invalid arguments are given

    :return: dict with the finite difference coefficients and corresponding offsets
    """

    _validate_deriv(deriv)

    if acc and offsets:
        raise ValueError('acc and offsets cannot both be given')

    if offsets:
        return calc_coefs(deriv, offsets, symbolic)

    _validate_acc(acc)
    return {
        scheme: calc_coefs_standard(deriv, acc, scheme, symbolic)
        for scheme in ('center', 'forward', 'backward')
    }


def calc_coefs(deriv, offsets, symbolic=False):
    stencil = Stencil(offsets, {(deriv,): 1}, symbolic=symbolic)
    return {
        "coefficients": [stencil.coefficient(o) for o in offsets],
        "offsets": stencil.offsets,
        "accuracy": stencil.accuracy
    }


def calc_coefs_standard(deriv, acc, scheme, symbolic=False):
    if scheme == 'center':
        stencil = SymmetricStencil1D(deriv, 1, acc, symbolic)
    elif scheme == 'forward':
        stencil = ForwardStencil1D(deriv, 1, acc, symbolic)
    elif scheme == 'backward':
        stencil = BackwardStencil1D(deriv, 1, acc, symbolic)
    return {
        "coefficients": stencil.coefficients,
        "offsets": stencil.offsets,
        "accuracy": acc
    }


def _validate_acc(acc):
    if acc % 2 == 1 or acc <= 0:
        raise ValueError('Accuracy order acc must be positive EVEN integer')


def _validate_deriv(deriv):
    if deriv < 0:
        raise ValueError('Derive degree must be positive integer')


class VectorOperator:
    """Base class for all vector differential operators.
       Shall not be instantiated directly, but through the child classes.
    """

    legacy = True

    def __init__(self, **kwargs):
        """Constructor for the VectorOperator base class.

            kwargs:
            -------

            h       list with the grid spacings of an N-dimensional uniform grid

            coords  list of 1D arrays with the coordinate values along the N axes.
                    This is used for non-uniform grids.

            Either specify "h" or "coords", not both.

        """

        if "acc" in kwargs:
            self.acc = kwargs.pop("acc")
        else:
            self.acc = 2

        if "spac" in kwargs or "h" in kwargs:  # necessary for backward compatibility 0.5.2 => 0.6
            if "spac" in kwargs:
                kw = "spac"
            else:
                kw = "h"
            self.h = kwargs.pop(kw)
            self.ndims = len(self.h)
            self.components = [FinDiff(k, self.h[k], 1) for k in range(self.ndims)]

        if "coords" in kwargs:
            coords = kwargs.pop("coords")
            self.ndims = self.__get_dimension(coords)
            self.components = [FinDiff((k, coords[k], 1), **kwargs) for k in range(self.ndims)]

    def __get_dimension(self, coords):
        return len(coords)


class Gradient(VectorOperator):
    r"""
    The N-dimensional gradient.

    .. math::
        \nabla = \left(\frac{\partial}{\partial x_0}, \frac{\partial}{\partial x_1}, ... , \frac{\partial}{\partial x_{N-1}}\right)

    :param kwargs:  exactly one of *h* and *coords* must be specified

             *h*
                     list with the grid spacings of an N-dimensional uniform grid
             *coords*
                     list of 1D arrays with the coordinate values along the N axes.
                     This is used for non-uniform grids.

             *acc*
                     accuracy order, must be positive integer, default is 2
    """

    legacy = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, f):
        """
        Applies the N-dimensional gradient to the array f.

        :param f:  ``numpy.ndarray``

                Array to apply the gradient to. It represents a scalar function,
                so it must have N axes for the N independent variables.

        :returns: ``numpy.ndarray``

                The gradient of f, which has N+1 axes, i.e. it is
                an array of N arrays of N axes each.

        """

        if not isinstance(f, np.ndarray):
            raise TypeError("Function to differentiate must be numpy.ndarray")

        if len(f.shape) != self.ndims:
            raise ValueError("Gradients can only be applied to scalar functions")

        result = []
        for k in range(self.ndims):
            d_dxk = self.components[k]
            result.append(d_dxk(f, acc=self.acc))

        return np.array(result)


class Divergence(VectorOperator):
    r"""
    The N-dimensional divergence.

    .. math::

       {\rm \bf div} = \nabla \cdot

    :param kwargs:  exactly one of *h* and *coords* must be specified

         *h*
                 list with the grid spacings of an N-dimensional uniform grid
         *coords*
                 list of 1D arrays with the coordinate values along the N axes.
                 This is used for non-uniform grids.

         *acc*
                 accuracy order, must be positive integer, default is 2

    """

    legacy = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, f):
        """
        Applies the divergence to the array f.

        :param f: ``numpy.ndarray``

               a vector function of N variables, so its array has N+1 axes.

        :returns: ``numpy.ndarray``

               the divergence, which is a scalar function of N variables, so it's array dimension has N axes

        """
        if not isinstance(f, np.ndarray) and not isinstance(f, list):
            raise TypeError("Function to differentiate must be numpy.ndarray or list of numpy.ndarrays")

        if len(f.shape) != self.ndims + 1 and f.shape[0] != self.ndims:
            raise ValueError("Divergence can only be applied to vector functions of the same dimension")

        result = np.zeros(f.shape[1:])

        for k in range(self.ndims):
            result += self.components[k](f[k], acc=self.acc)

        return result


class Curl(VectorOperator):
    r"""
    The curl operator.

    .. math::

        {\rm \bf rot} = \nabla \times

    Is only defined for 3D.

    :param kwargs:  exactly one of *h* and *coords* must be specified

     *h*
             list with the grid spacings of a 3-dimensional uniform grid
     *coords*
             list of 1D arrays with the coordinate values along the 3 axes.
             This is used for non-uniform grids.

     *acc*
             accuracy order, must be positive integer, default is 2


    """

    legacy = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if self.ndims != 3:
            raise ValueError("Curl operation is only defined in 3 dimensions. {} were given.".format(self.ndims))

    def __call__(self, f):
        """
        Applies the curl to the array f.

        :param f: ``numpy.ndarray``

               a vector function of N variables, so its array has N+1 axes.

        :returns: ``numpy.ndarray``

               the curl, which is a vector function of N variables, so it's array dimension has N+1 axes

        """

        if not isinstance(f, np.ndarray) and not isinstance(f, list):
            raise TypeError("Function to differentiate must be numpy.ndarray or list of numpy.ndarrays")

        if len(f.shape) != self.ndims + 1 and f.shape[0] != self.ndims:
            raise ValueError("Curl can only be applied to vector functions of the three dimensions")

        result = np.zeros(f.shape)

        result[0] += self.components[1](f[2], acc=self.acc) - self.components[2](f[1], acc=self.acc)
        result[1] += self.components[2](f[0], acc=self.acc) - self.components[0](f[2], acc=self.acc)
        result[2] += self.components[0](f[1], acc=self.acc) - self.components[1](f[0], acc=self.acc)

        return result


class Laplacian(object):
    r"""
        The N-dimensional Laplace operator.

        .. math::

           {\rm \bf \nabla^2} = \sum_{k=0}^{N-1} \frac{\partial^2}{\partial x_k^2}

        :param kwargs:  exactly one of *h* and *coords* must be specified

             *h*
                     list with the grid spacings of an N-dimensional uniform grid
             *coords*
                     list of 1D arrays with the coordinate values along the N axes.
                     This is used for non-uniform grids.

             *acc*
                     accuracy order, must be positive integer, default is 2

        """

    """A representation of the Laplace operator in arbitrary dimensions using finite difference schemes"""

    legacy = True

    def __init__(self, h=[1.], acc=2):
        h = wrap_in_ndarray(h)

        self._parts = [FinDiff((k, h[k], 2), acc=acc) for k in range(len(h))]

    def __call__(self, f):
        """
        Applies the Laplacian to the array f.

        :param f: ``numpy.ndarray``

               a scalar function of N variables, so its array has N axes.

        :returns: ``numpy.ndarray``

               the Laplacian of f, which is a scalar function of N variables, so it's array dimension has N axes

        """
        laplace_f = np.zeros_like(f)

        for part in self._parts:
            laplace_f += part(f)

        return laplace_f


def wrap_in_ndarray(value):
    """Wraps the argument in a numpy.ndarray.

       If value is a scalar, it is converted in a list first.
       If value is array-like, the shape is conserved.

    """

    if hasattr(value, "__len__"):
        return np.array(value)
    else:
        return np.array([value])
