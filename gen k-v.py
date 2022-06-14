from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma, betainc


E, tau = 1, 10**(-3)
m, n = 0.75, 0.2
t_l = 2
t_d = 3
lam = 2


def dirac(x_):
    ''' implements the dirac delta function '''
    if x_ == 0:
        return 1
    return 0


def r(t_):
    ''' defines relaxation function '''
    return E * (1/gamma(1-m)*(t_/tau)**(-m) + 1/gamma(1-n)*(t_/tau)**(-n))


def f_l(t_, m_, n_, t_l_, t_d_, tau_):
    ''' defines force function '''
    res_arr = []
    for x in t_:
        if x <= t_l_:
            temp = E * gamma(lam + 1) * (x/t_l_)**lam * (1/gamma(lam + 1 - m_)*(x/tau_)**(-m_)
                                                         + 1/gamma(lam + 1 - n_)*(x/tau_)**(-n_))
            res_arr.append(temp)
        if t_l_ < x <= t_d_+t_l_:
            temp = E*gamma(lam + 1)*(x/t_l_)**lam*(betainc(lam, 1-m_, t_l_/x)/gamma(lam + 1 - m_)*(x/tau_)**(-m_)
                                                   + betainc(lam, 1-n_, t_l_/x)/gamma(lam + 1 - n_)*(x/tau_)**(-n_))
            res_arr.append(temp)
    return res_arr


def function_graph(func, n_, a, b, x_name, y_name, title):
    ''' paint a graph of function "func" with number of points "n" and scopes by x "a, b" '''
    fig = plt.subplots()
    t = np.linspace(a, b, n_)
    plt.title(title)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.plot(t, func(t))
    plt.show()


#function_graph(r, 100, 1.5, 3, "t", "R(t)", None)
#function_graph(f_l, 100, 1.5, 3, "t", "F(t)", None)
fig = plt.subplots()
t1 = np.linspace(0.001, 1, 1000)
t2 = np.linspace(1, 10, 1000)
plt.title(None)
plt.xlabel("t")
plt.ylabel("F(t)")
# plt.plot(t1, f_l(t1, m, n, 1, 9, tau), color='b')
plt.plot(t1, f_l(t1, m, 0, 1, 9, tau), label='β=0')
plt.plot(t1, f_l(t1, m, 0.1, 1, 9, tau), label='β=0.1')
plt.plot(t1, f_l(t1, m, 0.2, 1, 9, tau), label='β=0.2')
plt.plot(t1, f_l(t1, m, 0.5, 1, 9, tau), label='β=0.5')
plt.plot(t1, f_l(t1, m, 0.8, 1, 9, tau), label='β=0.8')
plt.plot(t1, f_l(t1, m, 1, 1, 9, tau), label='β=1')
# plt.plot(t1, f_l(t1, m, 0.2, 1, 9, tau), color='b', label='tau=0.001s')
# plt.plot(t1, f_l(t1, m, 0.2, 1, 9, tau*10), linestyle='--', color='b', label='tau=0.01s')
# plt.plot(t1, f_l(t1, m, 0.2, 1, 9, tau*100), linestyle='-.', color='b', label='tau=0.1s')
plt.yscale('log')
plt.xscale('log')
#plt.plot(t, f_l(t, m, 0, 1, 9, tau), color='r')
#plt.plot(t, f_l(t, m, 0, 1, 9, tau*10), linestyle='--', color='r')
#plt.plot(t, f_l(t, m, 0, 1, 9, tau*100), linestyle='-.', color='r')
#plt.plot(t, f_l(t, m, 0.2, 1, 9, tau), color='b', label='tau=0.001s')
#plt.plot(t, f_l(t, m, 0.2, 1, 9, tau*10), linestyle='--', color='b', label='tau=0.01s')
#plt.plot(t, f_l(t, m, 0.2, 1, 9, tau*100), linestyle='-.', color='b', label='tau=0.1s')
#plt.text(5, 1.1, 'β=0', c='r')
#plt.text(5, 0.5, 'β=0.2', c='b')
plt.legend()
plt.show()

plt.title(None)
plt.xlabel("t")
plt.ylabel("F(t)")
# plt.plot(t1, f_l(t2, m, n, 1, 9, tau))
plt.plot(t1, f_l(t2, m, 0, 1, 9, tau), label='β=0')
plt.plot(t1, f_l(t2, m, 0.1, 1, 9, tau), label='β=0.1')
plt.plot(t1, f_l(t2, m, 0.2, 1, 9, tau), label='β=0.2')
plt.plot(t1, f_l(t2, m, 0.5, 1, 9, tau), label='β=0.5')
plt.plot(t1, f_l(t2, m, 0.8, 1, 9, tau), label='β=0.8')
plt.plot(t1, f_l(t2, m, 1, 1, 9, tau), label='β=1')
plt.xscale('log')
plt.legend()
plt.show()
