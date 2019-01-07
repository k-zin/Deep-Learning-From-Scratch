import sys
import os
# sys.path.append('c:\\workspace\\Python\\machinelearning\\deep-learning-from-scratch\\src\\dataset')
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '\..\dataset')
import mnist

(x_train, t_train), (x_test, t_test) = mnist.load_mnist(
    flatten=True, normalize=False)

print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)