class Combinable:

    def __init__(self):
        self.add_handler = Add
        self.mul_handler = Mul

    def __add__(self, other):
        return self.add_handler(self, other)

    def __radd__(self, other):
        return self.add_handler(other, self)

    def __mul__(self, other):
        return self.mul_handler(self, other)

    def __rmul__(self, other):
        return self.mul_handler(other, self)

    def __sub__(self, other):
        return self.add_handler(self, self.mul_handler(-1, other))

    def __rsub__(self, other):
        return self.add_handler(other, self.mul_handler(-1, self))

    def __neg__(self):
        return self.mul_handler(-1, self)


class Numberlike(Combinable):

    def __init__(self, value):
        super(Numberlike, self).__init__()
        self.value = value

    def apply(self, target, operation):
        return operation(self.value, target)

    def __repr__(self):
        return '%s(%s)' % (self.__class__.__name__, str(self.value))

    def __str__(self):
        return str(self.value)


class Operation(Combinable):

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
        return '%s(%s, %s)' % (self.__class__.__name__, self.left, self.right)

    def _needs_wrapping(self, arg):
        return not arg.__class__.__module__.startswith('findiff')

    def __call__(self, target, *args, **kwargs):
        return self.apply(target, *args, **kwargs)


class Mul(Operation):

    def operation(self, a, b):
        return a * b

    def apply(self, target, *args, **kwargs):

        for side in [self.right, self.left]:
            if not isinstance(side, Numberlike):
                res = side.apply(target, *args, **kwargs)
            else:
                res = side.apply(target, self.operation)
            target = res
        return res



class Add(Operation):

    def operation(self, a, b):
        return a + b

    def apply(self, target, *args, **kwargs):

        if type(self.right) != Numberlike:
            right_result = self.right.apply(target, *args, **kwargs)
        else:
            right_result = self.right.apply(target, self.operation)

        if type(self.left) != Numberlike:
            left_result = self.left.apply(target, *args, **kwargs)
        else:
            left_result = self.left.apply(target, self.operation)

        return left_result + right_result

