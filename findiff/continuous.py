"""This module is for the v1.* API"""


class PartialDerivative:

    def __init__(self, degrees):
        """ Representation of a (possibly mixed) partial derivative.

        Instances of this class are meant to be immutable.

        Parameters
        ----------
        degrees:    dict
            Dictionary describing the partial derivative. key: value <==> axis: degree
        """
        self._validate_degrees(degrees)
        self.degrees = degrees

    def degree(self, axis):
        return self.degrees.get(axis, 0)

    @property
    def axes(self):
        return sorted(self.degrees.keys())

    def __mul__(self, other):
        if not isinstance(other, PartialDerivative):
            raise TypeError('PartialDerivative can only multiply other PartialDerivative.')
        new_degrees = dict(self.degrees)
        for axis, degree in other.degrees.items():
            if axis in new_degrees:
                new_degrees[axis] += degree
            else:
                new_degrees[axis] = degree
        return PartialDerivative(new_degrees)

    def __pow__(self, power):
        assert int(power) == power and power > 0
        return PartialDerivative({axis: degree * power for axis, degree in self.degrees.items()})

    def __eq__(self, other):
        return self.degrees == other.degrees

    def __hash__(self):
        return hash(
            tuple(
                [(k, self.degrees[k]) for k in sorted(self.degrees.keys())]
            )
        )

    def _validate_degrees(self, degrees):
        assert isinstance(degrees, dict)
        for axis, degree in degrees.items():
            if not isinstance(axis, int) or axis < 0:
                raise ValueError('Axis must be positive integer')
            if not isinstance(degree, int) or degree <= 0:
                raise ValueError('Degree must be positive integer')


class DifferentialOperator:
    """Represents a linear combination of partial derivatives."""

    def __init__(self, *args):
        partials_dict = {}
        for arg in args:
            coef, degrees = arg
            pd = PartialDerivative(degrees)
            if pd not in partials_dict:
                partials_dict[pd] = coef
            else:
                partials_dict[pd] += coef

        assert isinstance(partials_dict, dict)
        self._the_sum = partials_dict

    def terms(self):
        return [(c, pd) for pd, c in self._the_sum.items()]

    def coefficient(self, partial):
        if isinstance(partial, PartialDerivative):
            return self._the_sum.get(partial, 0)
        return self._the_sum.get(PartialDerivative(partial), 0)

    def __add__(self, other):
        # Also consider case when other is a number (const * Identity)
        assert isinstance(other, DifferentialOperator)
        new_partials = dict(self._the_sum)
        for pd, c in other._the_sum.items():
            if pd in new_partials:
                new_partials[pd] += c
            else:
                new_partials[pd] = c
        return DifferentialOperator(
            *tuple((c, pd.degrees) for pd, c in new_partials.items())
        )

    def __mul__(self, other):
        if type(other) != PartialDerivative:
            raise TypeError('Multiplication of DifferentialOperator only with PartialDerivative!')
        new_partials = dict()
        for pd, c in self._the_sum.items():
            new_partials[pd * PartialDerivative(other.degrees)] = c
        return DifferentialOperator(
            *tuple((c, pd.degrees) for pd, c in new_partials.items())
        )


class Coordinate:

    def __init__(self, axis):
        assert axis >= 0 and axis == int(axis)
        self.name = 'x_{%d}' % axis
        self.axis = axis

    def __eq__(self, other):
        return self.axis == other.axis
