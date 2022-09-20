""" This module contains the classes for algebraic operations like
addition and multiplication between objects like differential operators.
"""


class Algebraic:
    """The base class of all algebraic entities in findiff.

    Algebraic entites are objects that can be composed to calculation graphs.
    """

    def __init__(self):
        # We decided to use variables for handling the Add and
        # Mul operations because this allows other classes
        # to override this behavior. (see the oldapi.py module)
        self.add_handler = Add
        self.mul_handler = Mul

    def __add__(self, other):
        """Returns self + other"""
        return self.add_handler(self, other)

    def __radd__(self, other):
        """Returns other + self"""
        return self.add_handler(other, self)

    def __mul__(self, other):
        """Returns self * other."""
        return self.mul_handler(self, other)

    def __rmul__(self, other):
        """Returns other * self."""
        return self.mul_handler(other, self)

    def __sub__(self, other):
        """Returns self - other."""
        return self.add_handler(self, self.mul_handler(-1, other))

    def __rsub__(self, other):
        """Returns other - self."""
        return self.add_handler(other, self.mul_handler(-1, self))

    def __neg__(self):
        """Converts -self to (-1) * self."""
        return self.mul_handler(-1, self)


class Numberlike(Algebraic):
    """Wrapper class for all numberlike objects (numbers, arrays) that shall
       be used as arithmetic entities.
    """

    def __init__(self, value):
        """Constructor

        Parameters
        ----------
        value : scalar or array-like
            The numeric value to wrap.
        """
        super(Numberlike, self).__init__()
        self.value = value

    def apply(self, target):
        return self.value * target

    def __repr__(self):
        return '%s(%s)' % (self.__class__.__name__, str(self.value))

    def __str__(self):
        return str(self.value)

    def __eq__(self, other):
        if not isinstance(other, Numberlike):
            return False
        return self.value == other.value


class Operation(Algebraic):
    """Base class for all binary operations between arithmetic entitites.

       This class is never instantiated by itself.
    """

    operation = None
    wrapper_class = Numberlike

    def __init__(self, left, right):
        super(Operation, self).__init__()
        if self._needs_wrapping(left):
            self.left = self.wrapper_class(left)
        else:
            self.left = left
        if self._needs_wrapping(right):
            self.right = self.wrapper_class(right)
        else:
            self.right = right

    def __repr__(self):
        return '%s(%s, %s)' % (self.__class__.__name__, self.left, self.right)

    def __str__(self):
        return self.__repr__()

    def _needs_wrapping(self, arg):
        return not arg.__class__.__module__.startswith('findiff')

    def __call__(self, target, *args, **kwargs):
        return self.apply(target, *args, **kwargs)


class Mul(Operation):
    """The multiplication operation."""

    def operation(self, a, b):
        return a * b

    def apply(self, target, *args, **kwargs):
        """Perform the multiplication, broadcasting the command to the left and right side."""

        for side in [self.right, self.left]:
            if isinstance(side, Numberlike): # simplgy multiply
                res = side.apply(target)
            else: # may be specific operation, like partial derivative merging
                res = side.apply(target, *args, **kwargs)
            target = res
        return res


class Add(Operation):
    """The addition operation."""

    def operation(self, a, b):
        return a + b

    def apply(self, target, *args, **kwargs):
        """Perform the addition, broadcasting the command to the left and right side."""

        if type(self.right) != Numberlike:
            right_result = self.right.apply(target, *args, **kwargs)
        else:
            right_result = self.right.apply(target)

        if type(self.left) != Numberlike:
            left_result = self.left.apply(target, *args, **kwargs)
        else:
            left_result = self.left.apply(target)

        return left_result + right_result


class Coordinate(Algebraic):

    def __init__(self, axis):
        assert axis >= 0 and axis == int(axis)
        super(Coordinate, self).__init__()
        self.name = 'x_{%d}' % axis
        self.axis = axis

    def __eq__(self, other):
        return self.axis == other.axis

    def apply(self, f, grid, *args, **kwargs):
        return grid.meshed_coords[self.axis] * f
