import math
import numbers
from functools import wraps
from itertools import product

import numpy as np
import sympy
from sympy import Symbol, Matrix, IndexedBase, Add, Mul, Expr

from findiff.symbolics.deriv import DerivativeSymbol


class StencilSet:
    """
    Represents the finite difference stencils for a given differential operator.
    """

    def __init__(self, partial, grid, acc):
        """
        Constructor for StencilSet objects.
        """

        self.char_pts = self._det_characteristic_points(grid.shape)
        self.data = self._create_stencil(partial, grid, acc)

    def as_dict(self):
        return {cp: self.data[cp].as_dict() for cp in self.char_pts}

    def __str__(self):
        return str(self.data)

    def __repr__(self):
        return str(self.data)

    def __getitem__(self, key):
        return self.data.get(key)

    def apply(self, u, idx0):
        """ Applies the stencil to a point in an equidistant grid.

        :param u: ndarray
            An array with the function to differentiate.

        :param idx0: int or tuple of ints
            The index of the grid point where to differentiate the function.

        :return:
            The derivative at the given point.
        """

        if not hasattr(idx0, '__len__'):
            idx0 = (idx0,)

        typ = []
        for axis in range(len(self.shape)):
            if idx0[axis] == 0:
                typ.append('L')
            elif idx0[axis] == self.shape[axis] - 1:
                typ.append('H')
            else:
                typ.append('C')
        typ = tuple(typ)

        stl = self.data[typ]

        idx0 = np.array(idx0)
        du = 0.
        for o, c in stl.items():
            idx = idx0 + o
            du += c * u[tuple(idx)]

        return du

    def apply_all(self, u):
        """ Applies the stencil to all grid points.

        :param u: ndarray
            An array with the function to differentiate.

        :return:
            An array with the derivative.
        """

        assert self.shape == u.shape

        ndims = len(u.shape)
        if ndims == 1:
            indices = list(range(len(u)))
        else:
            axes_indices = []
            for axis in range(ndims):
                axes_indices.append(list(range(u.shape[axis])))

            axes_indices = tuple(axes_indices)
            indices = list(product(*axes_indices))

        du = np.zeros_like(u)

        for idx in indices:
            du[idx] = self.apply(u, idx)

        return du

    def _create_stencil(self, partial, grid, acc):

        stencils = {}

        for char_pt in self.char_pts:  # e.g. ('L', 'C', 'C')

            stencil = None

            for axis, pos in enumerate(char_pt):
                degree = partial.degree(axis)
                if not degree:
                    continue
                stl = self._get_1D_stencil(pos, degree, acc, grid.spacing(axis))
                stl_ndim = stl.expand_dims(grid.ndims, axis)

                if not stencil:
                    stencil = stl_ndim
                else:
                    stencil = stencil + stl_ndim

            stencils[char_pt] = stencil

        return stencils

    def _get_1D_stencil(self, pos, degree, acc, spacing):
        if pos == 'C':
            return StencilStore.get_stencil(SymmetricStencil1D, deriv=degree, acc=acc, spacing=spacing)
        elif pos == 'L':
            return StencilStore.get_stencil(ForwardStencil1D, deriv=degree, acc=acc, spacing=spacing)
        elif pos == 'H':
            return StencilStore.get_stencil(BackwardStencil1D, deriv=degree, acc=acc, spacing=spacing)
        else:
            raise ValueError('Invalid value for "pos": %s' % str(pos))

    def _det_characteristic_points(self, shape):
        ndim = len(shape)
        typ = [("L", "C", "H")] * ndim
        return list(product(*typ))


def not_symbolic(func):
    @wraps(func)
    def inner(obj, *args, **kwargs):
        if obj.symbolic:
            raise NotImplementedError('%s cannot be used in symbolic mode.' % func.__name__)
        return func(obj, *args, **kwargs)

    return inner


class StencilStore:
    """The StencilStore saves already calculated stencils for later reuse.

    Kind of a singleton class.
    """
    stencils = {}

    def __init__(self):
        raise NotImplementedError('StencilStore cannot be instantiated. Use its class methods!')

    @classmethod
    def add(cls, stencil, deriv, acc, spacing):
        key = (type(stencil), deriv, acc, spacing)
        cls.stencils[key] = stencil

    @classmethod
    def get_stencil(cls, stencil_type, **kwargs):
        deriv, acc, dx = kwargs['deriv'], kwargs['acc'], kwargs['spacing']
        key = (stencil_type, deriv, acc, dx)
        if key not in cls.stencils:
            if (stencil_type == SymmetricStencil1D
                    or stencil_type == ForwardStencil1D
                    or stencil_type == BackwardStencil1D):
                stencil = stencil_type(deriv, dx, acc)
                cls.add(stencil, deriv, acc, dx)
            else:
                raise NotImplementedError()
        return cls.stencils[key]


class GenericStencil:

    def __init__(self, data):
        self._data = data

    def __len__(self):
        return len(self.as_dict())

    def __str__(self):
        return str(self.as_dict())

    def __repr__(self):
        return str(self.as_dict())

    def __getitem__(self, offset):
        """ Return the coefficient of the stencil at a given offset."""
        if not hasattr(offset, '__len__'):
            offset = offset,
        return self.as_dict().get(offset)

    def __add__(self, other):
        assert isinstance(other, GenericStencil)
        data = dict(self.as_dict())
        for off, coef in other.as_dict().items():
            if off in data:
                data[off] += coef
            else:
                data[off] = coef
        return GenericStencil(data)

    def __mul__(self, other):
        assert isinstance(other, numbers.Real)
        return GenericStencil(
            {off: other * coef for off, coef in self.as_dict().items()}
        )

    def __rmul__(self, other):
        return self.__mul__(other)

    def keys(self):
        return self.as_dict().keys()

    @property
    def coefficients(self):
        return np.array(list(self.as_dict().values()))

    @property
    def offsets(self):
        return np.array(list(self._data.keys()))

    @property
    def coefficients(self):
        return list(self.as_dict().values())

    def coefficient(self, offset):
        """Returns the coefficient for a given offset or offset tuple.

        Parameters
        ----------
        offset : int or tuple of ints
            The offset for which to return the coefficient.

        Returns
        -------
        out : float
            The coefficient corresponding to the offset.
        """
        if not isinstance(offset, tuple):
            offset = offset,
        return self.as_dict().get(offset, 0)

    def as_dict(self):
        """Returns a dictionary representation of the stencil."""
        return self._data

    def __call__(self, f, at=None, on=None):
        if at is not None and on is None:
            return self._apply_at_single_point(f, at)
        if at is None and on is not None:
            if isinstance(on[0], slice):
                return self._apply_on_multi_slice(f, on)
            else:
                return self._apply_on_mask(f, on)
        raise Exception('Cannot specify both *at* and *on* parameters.')

    def _apply_on_mask(self, f, mask):
        result = np.zeros_like(f)
        for offset, coeff in self.as_dict().items():
            offset_mask = self._make_offset_mask(mask, offset)
            result[mask] += coeff * f[offset_mask]
        return result

    def _apply_on_multi_slice(self, f, on):
        result = np.zeros_like(f)
        base_mslice = [self._canonic_slice(sl, f.shape[axis]) for axis, sl in enumerate(on)]
        for off, coeff in self.as_dict().items():
            off_mslice = list(base_mslice)
            for axis, off_ in enumerate(off):
                start = base_mslice[axis].start + off_
                stop = base_mslice[axis].stop + off_
                off_mslice[axis] = slice(start, stop)
            result[tuple(base_mslice)] += coeff * f[tuple(off_mslice)]
        return result

    def _apply_at_single_point(self, f, at):
        result = 0.
        at = np.array(at)
        for off, coeff in self.as_dict().items():
            off = np.array(off)
            eval_at = at + off
            if np.any(eval_at < 0) or not np.all(eval_at < f.shape):
                raise Exception('Cannot evaluate outside of grid.')
            result += coeff * f[tuple(eval_at)]
        return result

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
        offset_mask[tuple(mslice_base)] = mask[tuple(mslice_off)]
        return offset_mask

    def _canonic_slice(self, sl, length):
        start = sl.start
        if start is None:
            start = 0
        if start < 0:
            start = length - start
        stop = sl.stop
        if stop is None:
            stop = 0
        if stop < 0:
            stop = length - start
        return slice(start, stop)


class Stencil(GenericStencil):
    r"""Create a stencil based on given offsets for a given differential operator of the
        form

        .. math::
            \sum_i c_i \prod_{j=0}^N\partial^{n_j}_{j}

        based on a given list of index offsets, where :math:`\partial_i^{k}` is the :math:`k`-th
        partial derivative with respect to axis :math:`i`, :math:`N` is the dimension
        of space and :math:`c_i` are real constants and :math:`n_j` are non-negative
        integers.
    """

    def __init__(self, offsets, partials, spacings=None, symbolic=False):
        """

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

        self.symbolic = symbolic
        self.max_order_taylor = 1000

        if not hasattr(offsets[0], "__len__"):
            ndims = 1
            self._offsets = [(off,) for off in offsets]
        else:
            ndims = len(offsets[0])
            self._offsets = offsets
        self.ndims = ndims

        if isinstance(partials, set):
            raise TypeError('partials should be a dict, not a set.')

        if not isinstance(partials, dict):
            partials = self._convert_partials_to_dict(partials)

        self.partials = partials

        if spacings is None:
            spacings = [1] * ndims
        elif not hasattr(spacings, "__len__"):
            if isinstance(spacings, str):
                spacings = [Symbol(spacings)] * ndims
            else:
                spacings = [spacings] * ndims
        elif hasattr(spacings, "__len__") and isinstance(spacings[0], str):
            assert symbolic
            spacings = [Symbol(s) for s in spacings]

        assert len(spacings) == ndims
        self.spacings = spacings

        self._sol, sol_as_dict = self._make_stencil()
        super(Stencil, self).__init__(sol_as_dict)

    def __call__(self, *args, **kwargs):
        if self.symbolic:
            raise NotImplementedError('__call__ cannot be used in symbolic mode.')
        return super(Stencil, self).__call__(*args, **kwargs)

    @property
    def accuracy(self):
        """Returns the accuracy (error order) of a given stencil."""
        return self._calc_accuracy()

    def as_expression(self, func_symbol='u', index_symbols=None):
        """Convert stencil to sympy expression.

        Parameters
        ----------
        func_symbol : str
            The name of the function used in the returned expression.
        index_symbols : list of str, default = ['i_0', 'i_1', 'i_2', ... ]
            The name of the indices, one per axis, to be used in the expression.

        Returns
        -------
        expr, symbols :  sympy.Expr, dict
            The sympy expression and a dictionary with the used sympy symbols.
        """
        if isinstance(index_symbols, str):
            index_symbols = [Symbol(c) for c in index_symbols]
        if not index_symbols:
            index_symbols = [Symbol('i_%d' % axis) for axis in range(self.ndims)]
        assert len(index_symbols) == self.ndims
        if isinstance(index_symbols[0], str):
            index_symbols = [Symbol(s) for s in index_symbols]
        u = IndexedBase(func_symbol)
        expr = 0
        for off, coef in self.as_dict().items():
            off_inds = [index_symbols[axis] + off[axis] for axis in range(self.ndims)]
            expr += coef * u[off_inds]
        symbols = {'indices': index_symbols, 'function': u}

        if isinstance(self.spacings[0], Symbol):
            symbols['spacings'] = self.spacings

        return expr, symbols

    def _calc_accuracy(self):
        tol = 1.E-6
        deriv_order = 0
        for pows in self.partials.keys():
            order = sum(pows)
            if order > deriv_order:
                deriv_order = order
        for order in range(deriv_order, deriv_order + 10):
            terms = self._multinomial_powers(order)
            for term in terms:
                row = self._system_matrix_row(term)
                resid = np.sum(np.array(self._sol) * np.array(row))
                if abs(resid) > tol and term not in self.partials:
                    return order - deriv_order

    def _make_stencil(self):
        sys_matrix, taylor_terms = self._system_matrix()
        rhs = [0] * len(self._offsets)

        for i, term in enumerate(taylor_terms):
            if term in self.partials:
                weight = self.partials[term]
                multiplicity = np.prod([math.factorial(a) for a in term])
                if self.symbolic:
                    vol = sympy.Mul(*[self.spacings[j] ** term[j] for j in range(self.ndims)])
                else:
                    vol = np.prod([self.spacings[j] ** term[j] for j in range(self.ndims)])
                rhs[i] = weight * multiplicity / vol

        if self.symbolic:
            sol = sympy.linsolve((Matrix(sys_matrix), Matrix(rhs)))
            sol = list(sol)[0]
        else:
            sol = np.linalg.solve(sys_matrix, rhs)
        assert len(sol) == len(self._offsets)
        return sol, {off: coef for off, coef in zip(self._offsets, sol) if coef != 0}

    def _system_matrix(self):
        rows = []
        used_taylor_terms = []
        for deriv_order in range(self.max_order_taylor):
            taylor_terms_for_deriv_order = self._multinomial_powers(deriv_order)
            for term in taylor_terms_for_deriv_order:
                rows.append(self._system_matrix_row(term))
                used_taylor_terms.append(term)
                if not self._rows_are_linearly_independent(rows):
                    rows.pop()
                    used_taylor_terms.pop()
                if len(rows) == len(self._offsets):
                    return np.array(rows), used_taylor_terms
        raise Exception('Not enough terms. Try to increase max_order.')

    def _system_matrix_row(self, powers):
        row = []
        for a in self._offsets:
            value = 1
            for i, power in enumerate(powers):
                value *= a[i] ** power
            row.append(value)
        return row

    def _multinomial_powers(self, the_sum):
        """Returns all tuples of a given dimension that add up to the_sum."""
        all_combs = list(product(range(the_sum + 1), repeat=self.ndims))
        return list(filter(lambda tpl: sum(tpl) == the_sum, all_combs))

    def _rows_are_linearly_independent(self, matrix):
        """Checks the linear independence of the rows of a matrix."""
        matrix = np.array(matrix).astype(float)
        return np.linalg.matrix_rank(matrix) == len(matrix)

    def _convert_partials_to_dict(self, expr):
        self._check_sanity(expr)

        partials = {}

        def parse_mul(mul):
            w = 1
            p = [0] * self.ndims
            for arg in mul.args:
                if type(arg) == DerivativeSymbol:
                    p[arg.axis] = arg.degree
                else:
                    w *= int(arg)
            return tuple(p), w

        if type(expr) == Add:
            for term in expr.args:
                if type(term) == DerivativeSymbol:
                    p = [0] * self.ndims
                    w = 1
                    p[term.axis] = term.degree
                elif type(term) == Mul:
                    p, w = parse_mul(term)
                partials[tuple(p)] = w
        elif type(expr) == Mul:
            p, w = parse_mul(expr)
        elif type(expr) == DerivativeSymbol:
            p = [0] * self.ndims
            w = 1
            p[expr.axis] = expr.degree
        else:
            raise TypeError('Unable to convert object to dict.')
        partials[tuple(p)] = w
        return partials

    def _check_sanity(self, expr):

        def is_valid_mul(ex):
            for arg in ex.args:
                if arg.is_number:
                    continue
                if isinstance(arg, Expr) and not type(arg) == DerivativeSymbol:
                    return False
            return True

        def is_valid_add(ex):
            for arg in ex.args:
                if type(arg) == Mul:
                    if not is_valid_mul(arg):
                        return False
                elif type(arg) == DerivativeSymbol:
                    pass
                else:
                    return False
            return True

        err_msg = 'Expression is not in required form.'
        if type(expr) == DerivativeSymbol:
            return
        elif type(expr) == Mul:
            if not is_valid_mul(expr):
                raise ValueError(err_msg)
        elif type(expr) == Add:
            if not is_valid_add(expr):
                raise ValueError(err_msg)


class Stencil1D(GenericStencil):

    def __init__(self, deriv, offsets, dx, symbolic=False):
        data = self._calculate_stencil_data(deriv, offsets, dx, symbolic)
        super(Stencil1D, self).__init__(data)

    def expand_dims(self, ndims, axis):
        new_data = {}
        for off, coeff in self._data.items():
            m_off = [0] * ndims
            m_off[axis] = off
            new_data[tuple(m_off)] = coeff
        return GenericStencil(new_data)

    def get_boundary_size(self):
        return max(np.abs(self.offsets))

    def get_num_points_side(self):
        offs = self.offsets
        return abs(min(filter(lambda off: off <= 0, offs))), max(filter(lambda off: off >= 0, offs))

    def _calculate_stencil_data(self, deriv, offsets, dx, symbolic=False):
        A = self._build_matrix(offsets)
        rhs = self._build_rhs(deriv, offsets)
        if not symbolic:
            coefs = np.linalg.solve(A, rhs) * (dx ** (-deriv))
        else:
            sol = sympy.linsolve((Matrix(A), Matrix(rhs)))
            coefs = list(sol)[0]

        return ({off: coef for off, coef in zip(offsets, coefs)})

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


class SymmetricStencil1D(Stencil1D):

    def __init__(self, deriv, dx, acc=2, symbolic=False):
        assert acc % 2 == 0
        num_central = 2 * math.floor((deriv + 1) / 2) - 1 + acc
        p = num_central // 2
        offsets = np.array(list(range(-p, p + 1)))
        super(SymmetricStencil1D, self).__init__(deriv, offsets, dx, symbolic)
        self.acc = acc


class ForwardStencil1D(Stencil1D):
    def __init__(self, deriv, dx, acc=2, symbolic=False):
        assert acc % 2 == 0
        num_coefs = 2 * math.floor((deriv + 1) / 2) - 1 + acc
        if deriv % 2 == 0:
            num_coefs = num_coefs + 1
        offsets = np.array(list(range(0, num_coefs)))
        super(ForwardStencil1D, self).__init__(deriv, offsets, dx, symbolic)
        self.acc = acc


class BackwardStencil1D(Stencil1D):
    def __init__(self, deriv, dx, acc=2, symbolic=False):
        assert acc % 2 == 0
        num_coefs = 2 * math.floor((deriv + 1) / 2) - 1 + acc
        if deriv % 2 == 0:
            num_coefs = num_coefs + 1
        offsets = - np.array(list(range(0, num_coefs)))
        super(BackwardStencil1D, self).__init__(deriv, offsets[::-1], dx, symbolic)
        self.acc = acc