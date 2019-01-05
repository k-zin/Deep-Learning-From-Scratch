import numpy as np

A = np.array([[1, 2], [3, 4]])

# print(A)
# print(A.shape)
# print(A.dtype)

B = np.array([[3, 0], [0, 6]])

# print(A + B)
# print(A * B)

C = np.array([10, 20])

# print(A * C)

X = np.array([[51, 55], [14, 19], [0, 4]])

# print(X)
# print(X[0])
# print(X[0][1])

# for row in X:
#     print(row)

X = X.flatten()  # Xを1次元配列に変換

print(X)
print(X[np.array([0, 2, 4])])  # インデックスが0,2,4番目の要素を取得
print(X > 15)
print(X[X > 15])
