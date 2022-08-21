class Node:
    pass


class Numberlike(Node):

    def __init__(self, value):
        self.value = value

    def apply(self, target, operation):
        return operation(self.value, target)

    def __repr__(self):
        return '%s(%s)' % (Numberlike.__name__, str(self.value))

    def __str__(self):
        return str(self.value)


class Operation(Node):

    operation = None

    def __init__(self, left, right):
        if self._is_numberlike(left):
            self.left = Numberlike(left)
        else:
            self.left = left
        if self._is_numberlike(right):
            self.right = Numberlike(right)
        else:
            self.right = right

    def __repr__(self):
        return 'Mul(%s, %s)' % (self.left, self.right)

    def __str__(self):
        return '%s(%s, %s)' % (self.__class__.__name__, self.left, self.right)

    def contains_type(self, typ):
        for side in [self.left, self.right]:
            if isinstance(side, Operation):
                contains = side.contains_type(typ)
            else:
                contains = type(side) == typ
            if contains:
                return True
        return False

    def replace(self, tester, replacer):

        if isinstance(self.left, Operation):
            self.left.replace(tester, replacer)
        else:
            if tester(self.left):
                self.left = replacer(self.left)

        if isinstance(self.right, Operation):
            self.right.replace(tester, replacer)
        else:
            if tester(self.right):
                self.right = replacer(self.right)

    def _is_numberlike(self, arg):
        return not isinstance(arg, Node)


class Mul(Operation):

    def operation(self, a, b):
        return a * b

    def apply(self, target, *args, **kwargs):

        for side in [self.right, self.left]:
            if type(side) != Numberlike:
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

