import numpy as np

t = 60

V = np.array([300, 450, 500, 520, 523, 528, 540, 550, 563, 567, 576, 587, 600, 605, 615, 624, 631, 644, 650])
X = np.array([0, 0, 0, 33, 83, 92, 83, 92, 77, 101, 112, 102, 87, 100, 110, 115, 97, 104, 98])

R = X / t
del_R = np.sqrt(X) / t

X1 = np.array([3, 2, 3, 2, 0, 1, 1, 1, 2, 0, 2, 0, 2, 2, 2, 2, 3, 3, 1, 1, 0, 2, 2, 0, 2, 1, 1, 0, 1, 2, 1, 0,
               2, 0, 5, 0, 3, 1, 3, 1, 3, 1, 1, 1, 3, 2, 0, 2, 1, 3, 2, 2, 1, 0, 3, 1, 5, 4, 0, 3, 1, 0, 1, 1,
               1, 3, 3, 4, 2, 0, 0, 1, 0, 1, 2, 2, 0, 4, 1, 1, 1, 0, 0, 2, 3, 3, 1, 3, 1, 1, 4, 0, 2, 0, 1, 3,
               2, 2, 3, 1])

X2 = np.array([4, 3, 5, 4, 1, 4, 4, 6, 4, 4, 6, 3, 5, 4, 3, 2, 2, 0, 4, 4, 4, 4, 4, 5, 3, 4, 5, 2, 3, 2, 1, 3,
               1, 0, 4, 3, 1, 0, 3, 6, 2, 3, 6, 3, 0, 4, 3, 4, 7, 5, 0, 6, 5, 4, 6, 1, 2, 4, 1, 6, 4, 3, 4, 3,
               3, 2, 2, 5, 1, 6, 4, 2, 1, 3, 3, 0, 1, 3, 4, 2, 4, 5, 2, 2, 2, 3, 4, 0, 5, 2, 2, 5, 1, 2, 6, 1,
               2, 5, 4, 0])

X4 = np.array([12, 10, 3, 12, 6, 6, 7, 9, 13, 7, 4, 7, 10, 5, 9, 5, 8, 7, 11, 6, 11, 9, 6, 4, 10, 8, 7, 7, 8, 7,
               7, 7, 3, 10, 5, 11, 5, 8, 4, 10, 6, 6, 4, 7, 4, 9, 6, 6, 5, 10, 4, 7, 6, 10, 5, 4, 4, 6, 7, 11, 3,
               6, 5, 8, 7, 11, 3, 8, 7, 6, 1, 6, 11, 6, 11, 10, 5, 6, 9, 12, 6, 11, 4, 8, 7, 7, 10, 4, 7, 6, 4,
               9, 9, 17, 10, 10, 10, 3, 7, 9])
