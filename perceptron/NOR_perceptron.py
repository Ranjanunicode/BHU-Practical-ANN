# perceptron for OR Logic Gate

import numpy as np

# define Unite Step Function
def unitStep(v):
    if v >=0:
        return 1
    else:
        return 0

# design Perceptron Model
def perceptrorModel(x,w,b):
    v = np.dot(w,x) + b
    y = unitStep(v)
    return y

# NOT logic Function
# wnot = -1, bnot = 0.5
def NOT_logicFunction(x):
    wnot = -1
    bnot = 0.5
    return perceptrorModel(x, wnot, bnot)

# AND logic Function
# w1 = 1, w2 = 1, b = -1.5
def OR_logicFunction(x):
    w = np.array([1,1])
    b = -0.5
    return perceptrorModel(x, w, b)

# NOR logic Function
def NOR_logicFunction(x):
    output_OR = OR_logicFunction(x)
    output_NOT = NOT_logicFunction(output_OR)
    return output_NOT

# testing the perceptron Model
test1 = np.array([0,0])
test2 = np.array([0,1])
test3 = np.array([1,0])
test4 = np.array([1,1])

# printing the outputs
print("NOR({}, {}) = {}".format(0, 0, NOR_logicFunction(test1)))
print("NOR({}, {}) = {}".format(0, 1, NOR_logicFunction(test2)))
print("NOR({}, {}) = {}".format(1, 0, NOR_logicFunction(test3)))
print("NOR({}, {}) = {}".format(1, 1, NOR_logicFunction(test4)))