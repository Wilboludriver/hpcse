#!/usr/bin/env python3


import numpy as np
import matplotlib.pyplot as plt


def Spline(x, x_data, y_data, f):

    i = 0
    while x > x_data[i+1]:
        i += 1

    f_i = f[i]
    f_next = f[i+1]
    D_x = x_data[i+1]-x_data[i]

    return f_i * ((x_data[i+1]-x)**3 / (6*D_x)) + \
        f_next * ((x-x_data[i])**3/(6*D_x)) + \
        (((y_data[i+1]-y_data[i]) / D_x) - (f_next-f_i)*(D_x/6)) * (x-x_data[i]) + \
        (y_data[i]-f_i*(D_x**2/6))


if __name__ == "__main__":

    x_data = np.loadtxt("data/x_data")
    y_data = np.loadtxt("data/y_data")
    f_dense = np.loadtxt("b_dense")
    f_symmetric = np.loadtxt("b_symmetric")
    f_tridiagonal = np.loadtxt("b_tridiagonal")

    f_dense = np.concatenate(([0], f_dense, [0]))
    f_symmetric = np.concatenate(([0], f_symmetric, [0]))
    f_tridiagonal = np.concatenate(([0], f_tridiagonal, [0]))

    x = np.linspace(9, 16, 1000)
    y_dense = np.empty(x.size)
    y_symmetric = np.empty(x.size)
    y_tridiagonal = np.empty(x.size)

    for i in range(x.size):
        y_dense[i] = Spline(x[i], x_data, y_data, f_dense)

    for i in range(x.size):
        y_symmetric[i] = Spline(x[i], x_data, y_data, f_symmetric)
    
    for i in range(x.size):
        y_tridiagonal[i] = Spline(x[i], x_data, y_data, f_tridiagonal)

    fig, ax = plt.subplots()
    ax.plot(x_data, y_data, label="Ref")
    ax.plot(x, y_dense, label="dense")
    ax.plot(x, y_symmetric, label="symmetric")
    ax.plot(x, y_tridiagonal, label="tridiagonal")
    ax.set(xlim=[11,15])
    ax.legend()
    plt.show()