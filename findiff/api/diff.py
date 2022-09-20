import numbers

from findiff.core import InvalidArraySize, EquidistantGrid, InvalidGrid, DEFAULT_ACCURACY
from findiff.core import PartialDerivative
from findiff.utils import parse_spacing


class Diff(PartialDerivative):
    """Defines a (possibly mixed) partial derivative operator.

        Note the difference between defining the derivative operator and applying it.
        For applying the derivative operator, call it, once it is defined.
    """
    def __init__(self, *args):
        """
        Parameters
        ----------
        args:

            If exactly one integer argument is given, it means 'axis', where 'axis' is
            a positive integer, denoting the axis along which to take the (first, degree=1)
            derivative.

            If exactly one dictionary argument is given, it specifies a general, possibly
            mixed partial derivative. Each key denotes an axis along which to take a partial
            derivative, and the corresponding value denotes the degree of the derivative.

            If two integer arguments are given, the first denotes the axis along which
            to take the derivative, the second denotes the degree of the derivative.
        """
        if len(args) == 1 and type(args[0]) == dict:
            degrees = args[0]
        elif len(args) == 2:
            axis, degree = args
            degrees = {axis: degree}
        elif len(args) == 1:
            axis, degree = args[0], 1
            degrees = {axis: degree}
        else:
            raise ValueError('Diff constructor has received invalid argument(s): ' + str(args))

        self._validate_degrees_dict(degrees)

        super(Diff, self).__init__(degrees)

    def __call__(self, f, **kwargs):
        """Applies the partial derivative operator to an array.

        The function delegates to method *self.apply*.
        """
        return self.apply(f, **kwargs)

    def apply(self, f, **kwargs):
        """Applies the partial derivative operator to an array.

        Parameters
        ----------
        f : numpy.ndarray
            The array on which to apply the derivative operator

        kwargs : required keyword arguments

            Keywords:

                spacing : dict
                    Dictionary specifying the grid spacing (key=axis, value=spacing).

                acc :  even int > 0, optional, default: 2
                    The desired accuracy order.

        Returns
        -------
        out : numpy.ndarray
            The array with the evaluated derivative. Same shape as f.

        Examples
        --------
        >> x = y = np.linspace(0, 1, 100)
        >> dx = dy = x[1] - x[0]
        >> X, Y = np.meshgrid(x, y, indexing='ij')
        >> f = X**2 * Y**2
        >> d2_dxdy = Diff({0: 1, 1: 1})    # or: Diff(0) * Diff(1)
        >> d2f_dxdy = d2_dxdy(f, spacing={0: dx, 1: dy})
        """

        # make sure the array shape is big enough
        max_axis = max(self.axes)

        if max_axis >= f.ndim:
            raise InvalidArraySize('Array has not enough dimensions for given derivative operator.'
                                   'Has %d but needs at least %d' % (f.ndim, max_axis))

        if 'spacings' in kwargs:
            raise ValueError('Unknown argument "spacings". Did you mean "spacing"?')

        if 'spacing' in kwargs:
            spacing = parse_spacing(kwargs['spacing'])
            # Assert that spacings along all axes are defined, where derivatives need:
            for axis in self.axes:
                assert spacing.for_axis(axis)
        else:
            raise InvalidGrid('No spacing defined when applying Diff.')

        acc = kwargs.get('acc', DEFAULT_ACCURACY)

        return super().apply(f, spacing=spacing, acc=acc)

    def _validate_degrees_dict(self, degrees):
        assert isinstance(degrees, dict)
        for axis, degree in degrees.items():
            if not isinstance(axis, numbers.Integral) or axis < 0:
                raise ValueError('Axis must be positive integer')
            if not isinstance(degree, numbers.Integral) or degree <= 0:
                raise ValueError('Degree must be positive integer')
