"""
The stencils module contains all classes related to stencils, which are
representations of differential operators in finite difference approximations.
"""

import math
import numbers
from itertools import product

import numpy as np
import sympy
from sympy import Matrix


class Stencil:
    """Stencil representation of a differential operator.

    Knows how to apply itself to arrays and how to combine itself with other stencils.

    Stencils are usually created by one of the stencil factories.
    """

    def __init__(self, *args):
        """Constructor.

        Stencil instances are usually not created manually, but by one of
        the stencil factories in this module.

        Stencil coefficients may be floating point numbers or sympy symbols.

        Parameters
        ----------
        offsets : list of tuples
            The offsets for the stencil.
        coefs : list of floats
            The coefficients for the stencil.
        """

        if len(args) == 1 and isinstance(args[0], dict):
            offsets, coefs = list(args[0].keys()), list(args[0].values())
        else:
            offsets, coefs = args

        self.ndims = len(offsets[0])

        self._data = {off: coef for off, coef in zip(offsets, coefs)}
        self._acc = None

    def __getitem__(self, offset):
        """Returns the coefficient for a given offset.

        Parameters
        ----------
        offset : tuple of ints, or int
            The offset for which to return the coefficient.

        Returns
        -------
        out : float or Symbol
            The coefficient.
        """
        if not hasattr(offset, '__len__'):
            offset = offset,
        return self._data.get(offset, 0)

    def __str__(self):
        return 'Length: ' + str(len(self._data)) + ' ' + str(self.as_dict())

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return len(self.as_dict())

    def __eq__(self, other):
        return self.as_dict() == other.as_dict()

    def __call__(self, arr, at=None, on=None):
        """Apply the stencil to an array at a given point (`at`) or on a mask (`on`).

        Parameters
        ----------
        arr : numpy.ndarray
            The array to which to apply the stencil.
        at : index tuple, default: None
            If given, evaluate stencil at this index tuple.
        on : numpy.ndarray of dtype bool
            If given, evaluate stencil on the mask given by `on`

        Returns
        -------
        out : numpy.ndarray or float
            The result of the stencil application.
        """
        if at is not None and on is not None:
            raise ValueError('Either specify *at* or *on*, not both.')
        if at is None and on is None:
            raise ValueError('Specify either *at* or *on*.')
        if on is not None:
            if isinstance(on, np.ndarray):  # assuming at is a mask
                return self.apply(arr, on)
            else:
                raise ValueError('When giving *on*, pass a numpy mask.')
        mask = np.zeros_like(arr, dtype=bool)
        mask[at] = True
        return self.apply(arr, mask)[at]

    def apply(self, arr, mask):
        result = np.zeros_like(arr)
        for offset, coeff in self.as_dict().items():
            offset_mask = self._make_offset_mask(mask, offset)
            result[mask] += coeff * arr[offset_mask]
        return result

    def __add__(self, other):
        data = dict(self.as_dict())
        for off, coef in other.as_dict().items():
            if off in data:
                data[off] += coef
            else:
                data[off] = coef
        return Stencil(data)

    def __mul__(self, other):
        assert isinstance(other, Stencil)
        all_combinations = product(self.offsets, other.offsets)
        offsets = []
        coeffs = []
        for off1, off2 in all_combinations:
            off = tuple(np.array(off1) + np.array(off2))
            offsets.append(off)
            coeffs.append(self[off1] * other[off2])
        return Stencil(offsets, coeffs)

    @property
    def offsets(self):
        return list(self.as_dict().keys())

    @property
    def coefficients(self):
        return list(self.as_dict().values())

    def coefficient(self, offset):
        return self[offset]

    @property
    def accuracy(self):
        return self._acc

    def as_dict(self):
        return self._data

    def _make_offset_mask(self, mask, offset):
        offset_mask = np.full_like(mask, fill_value=False, dtype=bool)
        mslice_off = []
        mslice_base = []
        for off_ in offset:
            if off_ == 0:
                sl_off = slice(None, None)
                sl_base = slice(None, None)
            elif off_ > 0:
                sl_off = slice(off_, None)
                sl_base = slice(None, -off_)
            else:
                sl_off = slice(None, off_)
                sl_base = slice(-off_, None)
            mslice_off.append(sl_off)
            mslice_base.append(sl_base)
        offset_mask[tuple(mslice_off)] = mask[tuple(mslice_base)]
        return offset_mask


class SymmetricStencil(Stencil):
    ...


class ForwardStencil(Stencil):
    ...


class BackwardStencil(Stencil):
    ...


class StandardStencilFactory:
    """The StandardStencilFactory can create the 1D standard stencils used
    to evaluate partial derivatives.
    """

    def create(self, stencil_type, deriv: int, spacing: float, acc=2, symbolic=False):
        """Factory method for creating a stencil of given type.

        Parameters
        ----------
        stencil_type : type
            Which kind of stencil to create. May be SymmetricStencil, ForwardStencil or
            BackwardStencil.
        deriv : int > 0
            The degree of derivative for the stencil.
        spacing : float
            The grid spacing.
        acc : int > 0, even
            The accuracy order.
        symbolic : bool
            Use symbolic computation or not.

        Returns
        -------
        out :
            Returns stencil of requested type.
        """

        if not isinstance(spacing, numbers.Real):
            raise TypeError(
                '"spacing" must be a float in factory method, not %s.' %
                (type(spacing).__name__)
            )

        offsets = self._calc_offsets(stencil_type, deriv, acc)
        coefs = self._calc_coefficients(deriv, offsets, spacing, symbolic)
        return stencil_type(self._convert_to_tuples(offsets), coefs)

    def _calc_coefficients(self, deriv, offsets, dx, symbolic):
        A = self._build_matrix(offsets)
        rhs = self._build_rhs(deriv, offsets)
        if not symbolic:
            coefs = np.linalg.solve(A, rhs) * (dx ** (-deriv))
        else:
            sol = sympy.linsolve((Matrix(A), Matrix(rhs)))
            coefs = list(sol)[0]
        return coefs

    def _calc_offsets(self, stencil_type, deriv, acc):
        if stencil_type == SymmetricStencil:
            num_central = 2 * math.floor((deriv + 1) / 2) - 1 + acc
            p = num_central // 2
            offsets = np.array(list(range(-p, p + 1)))
        elif stencil_type == ForwardStencil:
            num_coefs = 2 * math.floor((deriv + 1) / 2) - 1 + acc
            if deriv % 2 == 0:
                num_coefs = num_coefs + 1
            offsets = np.array(list(range(0, num_coefs)))
        elif stencil_type == BackwardStencil:
            num_coefs = 2 * math.floor((deriv + 1) / 2) - 1 + acc
            if deriv % 2 == 0:
                num_coefs = num_coefs + 1
            offsets = -np.array(list(range(0, num_coefs)))
        else:
            raise TypeError('Cannot calculate offsets for this stencil type: %s' % str(stencil_type))

        return offsets

    def _convert_to_tuples(self, offsets):
        return [(off,) for off in offsets]

    def _build_matrix(self, offsets):
        """Constructs the equation system matrix for the finite difference coefficients"""
        A = [([1 for _ in offsets])]
        for i in range(1, len(offsets)):
            A.append([j ** i for j in offsets])
        return np.array(A)

    def _build_rhs(self, deriv, offsets):
        """The right hand side of the equation system matrix"""
        b = [0 for _ in offsets]
        b[deriv] = math.factorial(deriv)
        return np.array(b)


class StencilFactory:
    r"""Create a stencil based on given offsets for a given differential operator of the form

        .. math::
            \sum_i c_i \prod_{j=0}^N\partial^{n_j}_{j}

        based on a given list of index offsets, where :math:`\partial_i^{k}` is the :math:`k`-th
        partial derivative with respect to axis :math:`i`, :math:`N` is the dimension
        of space and :math:`c_i` are real constants and :math:`n_j` are non-negative
        integers.
        """

    def __init__(self):
        self.max_order_taylor = 1000

    def create(self, offsets, partials, spacing=None, symbolic=False):
        """Factory method.

        Parameters
        ----------
        offsets :   list of ints (1D) or list of int-tuples (>1D)
            The offsets from which to compute the stencil.
        partials :  dict or sympy expression with DerivativeSymbol
            The differential operator for which to compute the stencil.
        spacings : list of float, or list of str, or list of Symbols, default = [1]*ndims
            The grid spacing along each axis. Can be a string or a sympy Symbol
            in case of symbolic calculation.
        symbolic : bool
            Flag to trigger symbolic calculation instead of numerical.
        """

        offsets = self._validate_and_convert_offsets(offsets)

        ndims = len(offsets[0])
        sys_matrix, taylor_terms = self._system_matrix(offsets, ndims)
        rhs = [0] * len(offsets)

        for i, term in enumerate(taylor_terms):
            if term in partials:
                weight = partials[term]
                multiplicity = np.prod([math.factorial(a) for a in term])
                if symbolic:
                    vol = sympy.Mul(*[spacing[j] ** term[j] for j in range(ndims)])
                else:
                    vol = np.prod([spacing[j] ** term[j] for j in range(ndims)])
                rhs[i] = weight * multiplicity / vol

        if symbolic:
            sol = sympy.linsolve((Matrix(sys_matrix), Matrix(rhs)))
            sol = list(sol)[0]
        else:
            sol = np.linalg.solve(sys_matrix, rhs)
        assert len(sol) == len(offsets)

        stencil = Stencil({off: coef for off, coef in zip(offsets, sol) if coef != 0})
        stencil._acc = self._calc_accuracy(partials, sol, offsets, ndims, symbolic)

        return stencil

    def _validate_and_convert_offsets(self, offsets):
        """The factory want offsets to be tuples, even in the 1D case."""
        if not hasattr(offsets[0], '__len__'):
            return [(off,) for off in offsets]
        return offsets

    def _system_matrix(self, offsets, ndims):
        rows = []
        used_taylor_terms = []
        for deriv_order in range(self.max_order_taylor):
            taylor_terms_for_deriv_order = self._multinomial_powers(deriv_order, ndims)
            for term in taylor_terms_for_deriv_order:
                rows.append(self._system_matrix_row(term, offsets))
                used_taylor_terms.append(term)
                if not self._rows_are_linearly_independent(rows):
                    rows.pop()
                    used_taylor_terms.pop()
                if len(rows) == len(offsets):
                    return np.array(rows), used_taylor_terms
        raise Exception('Not enough terms. Try to increase max_order.')

    def _system_matrix_row(self, powers, offsets):
        row = []
        for a in offsets:
            value = 1
            for i, power in enumerate(powers):
                value *= a[i] ** power
            row.append(value)
        return row

    def _multinomial_powers(self, the_sum, ndims):
        """Returns all tuples of a given dimension that add up to the_sum."""
        all_combs = list(product(range(the_sum + 1), repeat=ndims))
        return list(filter(lambda tpl: sum(tpl) == the_sum, all_combs))

    def _rows_are_linearly_independent(self, matrix):
        """Checks the linear independence of the rows of a matrix."""
        matrix = np.array(matrix).astype(float)
        return np.linalg.matrix_rank(matrix) == len(matrix)

    def _calc_accuracy(self, partials, sol, offsets, ndims, symbolic):
        tol = 1.E-6
        deriv_order = 0
        for pows in partials.keys():
            order = sum(pows)
            if order > deriv_order:
                deriv_order = order
        for order in range(deriv_order, deriv_order + 10):
            terms = self._multinomial_powers(order, ndims)
            for term in terms:
                row = self._system_matrix_row(term, offsets)
                if not symbolic:
                    resid = np.sum(np.array(sol) * np.array(row))
                    if abs(resid) > tol and term not in partials:
                        return order - deriv_order
                else:
                    resid = Matrix(sol).dot(Matrix(row).transpose())
                    if term not in partials and sympy.simplify(resid) != 0:
                        return order - deriv_order


class StencilSet:

    def __init__(self, stencils):
        self._stencils = stencils
        self.ndims = len(list(stencils.keys())[0])

    def __add__(self, other):
        assert isinstance(other, StencilSet)
        stencils = {}
        for char_pt, self_stencil in self.as_dict().items():
            stencils[char_pt] = self_stencil + other[char_pt]
        return StencilSet(stencils)

    def __mul__(self, other):
        assert isinstance(other, StencilSet)
        stencils = {}
        for char_pt, self_stencil in self.as_dict().items():
            stencils[char_pt] = self_stencil * other[char_pt]
        return StencilSet(stencils)

    def as_dict(self):
        return self._stencils

    def __getitem__(self, char_pt):
        return self._stencils[char_pt]

    def __getitem__(self, char_pt):
        return self.as_dict()[tuple(char_pt)]

    def __eq__(self, other):
        for key, stl in self.as_dict().items():
            if not (stl == other[key]):
                return False
        return True

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return str(self.as_dict())


class StandardStencilSet(StencilSet):
    """
    Has the complete set of standard stencils for a given grid for a given partial derivative.

    Can combine (add) with other StencilSet objects.
    Can apply itself to arrays. The StencilSet knows where to apply each of its stencils.

    When a PartialDerivative is applied to an array, it creates a StencilSet object for itself,
    and then applies that to the array.
    """

    def __init__(self, partial, spacing, ndims, acc):
        """

        Parameters
        ----------
        partial : PartialDerivative
        spacing : findiff.core.grids.Spacing
        ndims : int
        acc : int
        """
        self._stencils = self._create_stencils(acc, ndims, partial, spacing)
        self.ndims = ndims
        self.inner_mask = None

    def apply(self, arr):
        if self.inner_mask is None or not self.inner_mask.shape == arr.shape:
            self.inner_mask = self._determine_inner_mask(arr.shape)
        result = np.zeros_like(arr)
        masks_applied = np.zeros_like(arr, dtype=int)
        for char_pt, stencil in self.as_dict().items():  # e.g. char_ot == ('L', 'C', 'C')
            mask = self._determine_mask_where_to_apply_stencil(char_pt)
            if np.any(masks_applied[mask] > 0):
                continue
            partial_result = stencil.apply(arr, mask)
            result[mask] = partial_result[mask]
            masks_applied[mask] += 1

        # Make sure masks do not overlap and leave no gaps:
        assert np.max(masks_applied) == 1
        assert np.min(masks_applied) == 1

        return result

    def as_dict(self):
        return self._stencils

    def _determine_mask_where_to_apply_stencil(self, char_pt):
        inner = self.inner_mask
        mslice = []
        bdry_sizes = self._get_boundary_sizes()

        for axis, pos in enumerate(char_pt):
            size = bdry_sizes[axis]
            sl = slice(None, None)
            if pos == 'C':
                if size != 0:
                    sl = slice(size, -size)
            elif pos == 'L':
                if size != 0:
                    sl = slice(0, size)
            elif pos == 'H':
                if size != 0:
                    sl = slice(-size, None)
            else:
                raise ValueError('Invalid position code: %s.' % str(pos))
            mslice.append(sl)
        mask = np.zeros_like(inner, dtype=bool)
        mask[mslice] = True
        return mask

    def _determine_inner_mask(self, shape):
        sizes = self._get_boundary_sizes()
        mslice = []
        for axis in range(self.ndims):
            start = sizes[axis]
            stop = -start
            mslice.append(slice(start, stop))
        mask = np.zeros(shape, dtype=bool)
        mask[tuple(mslice)] = True
        return mask

    def _get_boundary_sizes(self):
        inner_stencil = self._get_inner_stencil()
        offsets = np.array(inner_stencil.offsets)
        sizes = []
        for axis in range(self.ndims):
            size = np.max(np.abs(offsets[:, axis]))
            sizes.append(size)
        return sizes

    def _get_inner_stencil(self):
        idx = tuple(['C'] * self.ndims)
        inner_stencil = self.as_dict()[idx]
        return inner_stencil

    def _create_stencils(self, acc, ndims, partial, spacing):
        char_pts = self._get_characteristic_points(ndims, partial)

        stencils = {}
        for char_pt in char_pts:  # e.g. ('L', 'C', 'C')

            stencil = None

            for axis, pos in enumerate(char_pt):
                degree = partial.degree(axis)
                if not degree:
                    continue
                stl = self._get_1D_stencil(pos, degree, acc, spacing.for_axis(axis))
                stl_ndim = self._expand_dims(stl, ndims, axis)

                if not stencil:
                    stencil = stl_ndim
                else:
                    stencil = stencil * stl_ndim

            stencils[char_pt] = stencil
        return stencils

    def _get_characteristic_points(self, ndims, partial):
        candidates = list(product(('L', 'C', 'H'), repeat=ndims))
        return candidates
        # TODO: filter characteristic points because depending some can be redundant for certain diff operators

    #        all_axes = list(range(ndims))
    #        for axis in all_axes:
    #            if axis in partial.axes:
    #                continue
    #            candidates = list(filter(lambda item: item[axis] != 'C', candidates))
    #        return list(candidates)

    def _get_1D_stencil(self, pos, degree, acc, spacing, symbolic=False):
        if pos == 'C':
            return StencilStore.get_stencil(SymmetricStencil, degree, spacing, acc, symbolic)
        elif pos == 'L':
            return StencilStore.get_stencil(ForwardStencil, degree, spacing, acc, symbolic)
        elif pos == 'H':
            return StencilStore.get_stencil(BackwardStencil, degree, spacing, acc, symbolic)
        else:
            raise ValueError('Invalid value for "pos": %s' % str(pos))

    def _expand_dims(self, stencil, ndims, axis):
        offsets = stencil.offsets
        new_offsets = []
        for off in offsets:
            m_off = [0] * ndims
            m_off[axis] = off[0]
            new_offsets.append(tuple(m_off))
        return Stencil(new_offsets, stencil.coefficients)


class TrivialStencilSet(StencilSet):

    def __init__(self, value, ndims):
        self.ndims = ndims
        self.value = value
        self.char_pts = list(product(('L', 'C', 'H'), repeat=ndims))
        only_offset = tuple([0] * self.ndims)
        self._stencil = Stencil([only_offset], [self.value])

    def __getitem__(self, char_pt):

        # Always return the same stencil, independent of characteristic point:
        return self._stencil

    def as_dict(self):
        return {char_pt: self._stencil for char_pt in self.char_pts}

    def apply(self, arr):
        return self.value * arr


class StencilStore:
    """
    Clients obtain stencils from the store. If a requested stencil is not in the store,
    the store asks the factory to create one, which is then saved in the store.
    """

    registered_stencils = {}

    @classmethod
    def register(cls, key, stencil):
        cls.registered_stencils[key] = stencil

    @classmethod
    def get_stencil(cls, stencil_type, *args):
        key = (stencil_type, args)
        if key in cls.registered_stencils:
            return cls.registered_stencils[key]

        if stencil_type == Stencil:
            factory = StencilFactory()
        else:
            factory = StandardStencilFactory()

        stencil = factory.create(stencil_type, *args)
        cls.register(key, stencil)
        return stencil
