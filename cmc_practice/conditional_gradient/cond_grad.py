import numpy as np

'''
class Functional():
    def __init__(self, input_str):
        input_str_tmp = input_str.replace('<', '(').replace('>', ')').replace(',', ')#(')
        #self.func = input_str.replace('<', '(').replace('>', ')').replace(',', ')#(')
        self.func = []
        self.coefs = []
        is_num = 0
        for chr in input_str_tmp:
            if chr in actions:
                if is_num != 0:
                    self.func.append(is_num)
                    is_num = 0
                self.func.append(chr)
            elif chr.isalpha():
                if is_num != 0:
                    self.func.append(is_num)
                    is_num = 0
                self.func.append(chr)
                if chr != 'u':
                    self.coefs.append(chr)
            else:
                is_num = is_num*10 + int(chr)
        if input_str_tmp[len(input_str_tmp) - 1].isdigit():
            self.func.append(is_num)

    def rpn(self):
        self.func_rpn = []
        operands = []
        func_reversed = self.func.copy()
        func_reversed.reverse()
        #print(func_reversed)
        while func_reversed:
            elem = func_reversed.pop()
            if elem in actions:
                if len(operands) == 0:
                    operands.append(elem)
                else:
                    if (elem in ['*', '/', '#'] and operands[len(operands) - 1] in ['*', '/', '#', '(']) or (elem in ['+', '-'] and operands[len(operands) - 1] in ['+', '-', '(']) or (elem in ['+', '-'] and operands[len(operands) - 1] in ['*', '/', '#', '(']):
                        self.func_rpn.append(operands.pop())
                        operands.append(elem)
                    elif (elem in ['*', '/', '#'] and operands[len(operands) - 1] in ['+', '-', '(']) or elem == '(':
                        operands.append(elem)
                    elif elem == ')':
                        while operands[len(operands) - 1] != '(':
                            self.func_rpn.append(operands.pop())
                        operands.pop()
            elif type(elem) != 'str':
                self.func_rpn.append(elem)
        while operands:
            self.func_rpn.append(operands.pop())
'''


class Functional():
    def __init__(self, input_func, input_lims):
        self.func = input_func
        self.limits = input_lims
        self.u_k = np.random.random(5) #vector of normally distributed values from [0.0, 1.0)

    def Gradient(self, point):
        grad = []
        dim = len(point)
        h = 0.0001 #обдумать этот моментик
        for pos, num in enumerate(point):
            dimension_f = np.array(point)
            dimension_b = np.array(point)
            dimension_f[pos] += h #шаг вперед
            dimension_b[pos] -= h #шаг назад
            print(dimension_f)
            print(dimension_b)
            print(self.func(dimension_f.tolist()))
            print(self.func(dimension_b.tolist()))
            grad.append((self.func(dimension_f.tolist()) - self.func(dimension_b.tolist())) / (2 * h))
        return grad

    def Hessian(self, point):
        hess = []
        dim = len(point)
        h = 0.0001 #обдумать этот моментик
        for pos1, num1 in enumerate(point):
            partial_derivative = []
            for pos2, num2 in enumerate(point):
                dimension_f = np.array(point)
                dimension_m = np.array(point)
                dimension_b = np.array(point)
                if pos1 == pos2:
                    dimension_f[pos1] = num1 + h #шаг вперед
                    dimension_m[pos1] = num1 #центральная точка
                    dimension_b[pos1] = num1 - h #шаг назад
                    partial_derivative.append((self.func(dimension_f.tolist()) + self.func(dimension_b.tolist()) - 2*self.func(dimension_m.tolist())) / (h**2))
                else:
                    dimension_f[pos1] += h #шаг вперед, координата 1
                    dimension_b[pos1] -= h #шаг назад, координата 1
                    dimension_f[pos2] += h #фиксируем координату 2
                    dimension_b[pos2] += h #фиксируем координату 2
                    der1 = (self.func(dimension_f.tolist()) - self.func(dimension_b.tolist())) / (2 * h) #производная по координате 1
                    dimension_f[pos1] = num1
                    dimension_b[pos1] = num1
                    print(der1)
                    dimension_f[pos2] -= h #шаг вперед, координата 2
                    dimension_b[pos2] -= h #шаг назад, координата 2
                    der2 = der1 - (self.func(dimension_f.tolist()) - self.func(dimension_b.tolist())) / (2 * h) #производная по координате 2
                    partial_derivative.append(der2)
            hess.append(partial_derivative)
        return hess



# M A I N
funct = Functional(lambda x: x[0]**3 + (x[0]**2)*(x[1]**2) + x[1]**3, lambda y: 2 - y[0] + 1 - y[1] - y[0])
print(funct.Gradient([2.0, 5.0]))
print(funct.Hessian([2.0, 5.0]))
