import numpy as np

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



# M A I N
u_initial = np.random.random(5) #vector of normally distributed values from [0.0, 1.0)
#print(u_initial)
actions = ['+', '-', '*', '/', '#', '<', '>', '(', ')'] #math operations

print("Введите функционал для минизации, используя < и > для скалярного произведения и не пропуская знак умножения и без пробелов")

funct = Functional(input())
print(funct.func)
funct.rpn()
print(funct.func_rpn)
