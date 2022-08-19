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
        assert isinstance(other, PartialDerivative)
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

    def __init__(self, partials_dict):
        assert isinstance(partials_dict, dict)
        self._the_sum = partials_dict

    def coefficient(self, partial):
        return self._the_sum[partial]

    def items(self):
        return {c: pd for pd, c in self._the_sum.items()}


class Coordinate:

    def __init__(self, axis):
        assert axis >= 0 and axis == int(axis)
        self.name = 'x_{%d}' % axis
        self.axis = axis

    def __eq__(self, other):
        return self.axis == other.axis
