from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt


E, tau = 1, 10**(-3)


def dirac(x_):
    ''' implements the dirac delta function '''
    if x_ == 0:
        return 1
    return 0


def r(t_):
    ''' defines relaxation function '''
    return E * np.heaviside(t_, 1)


def f(t_):
    ''' defines force function '''
    res_arr = []
    for x in t_:
        def rr(t_1):
            return (E * np.heaviside(x - t_1, 1) + dirac(x - t_1)) * x * 2
        res_arr.append(integrate.quad(rr, 0, x)[0])
    return res_arr


def function_graph(func, n, a, b, x_name, y_name, title):
    ''' paint a graph of function "func" with number of points "n" and scopes by x "a, b" '''
    fig = plt.subplots()
    t = np.linspace(a, b, n)
    plt.title(title)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.plot(t, func(t))
    plt.show()


function_graph(r, 100, 1.5, 2, "t", "R(t)", None)
function_graph(f, 100, 1.5, 2, "t", "F(t)", None)
