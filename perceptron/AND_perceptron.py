# perceptron for AND Logic Gate

import numpy as np

# define Unite Step Function
def unitStep(v):
    if v >= 0:
        return 1
    else:
        return 0

# design Perceptron Model
def perceptrorModel(x,w,b):
    v = np.dot(w,x) + b
    y = unitStep(v)
    return y

# AND logic Function
# w1 = 1, w2 = 1, b = -1.5
def AND_logicFunction(x):
    w = np.array([1,1])
    b = -1.5
    return perceptrorModel(x,w,b)
# testing the perceptron Model
test1 = np.array([0,0])
test2 = np.array([0,1])
test3 = np.array([1,0])
test4 = np.array([1,1])

# printing the outputs
print("AND({}, {}) = {}".format(0,0,AND_logicFunction(test1)))
print("AND({}, {}) = {}".format(0,1,AND_logicFunction(test2)))
print("AND({}, {}) = {}".format(1,0,AND_logicFunction(test3)))
print("AND({}, {}) = {}".format(1,1,AND_logicFunction(test4)))