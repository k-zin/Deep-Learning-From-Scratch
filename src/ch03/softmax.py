import numpy as np
import matplotlib.pylab as plt

a = np.array([0.3, 2.9, 4.0])

exp_a = np.exp(a)  # 指数関数
# print(exp_a)

sum_exp_a = np.sum(exp_a)
# print(sum_exp_a)

y = exp_a / sum_exp_a
# print(y)


# def softmax(a):
#     exp_a = np.exp(a)
#     sum_exp_a = np.sum(exp_a)
#     y = exp_a / sum_exp_a
#     return y


a = np.array([1010, 1000, 990])
np.exp(a) / np.sum(np.exp(a))
# print(np.exp(a) / np.sum(np.exp(a)))

c = np.max(a)
# print(a - c)

# print(np.exp(a - c) / np.sum(np.exp(a - c)))


def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


print(softmax(a))
