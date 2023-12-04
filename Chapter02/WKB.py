import numpy as np
from scipy import integrate
from matplotlib import pyplot as plt

e = 1.602e-19
m0 = 9.11e-31
h_ = 6.626e-34/(2*np.pi)

# 定义势垒函数 V
def V(x):
    if(0.5e-9 < x <=1.0e-9):
        return (x*1e9-0.5)*e
    elif(1.0e-9 < x <= 1.5e-9):
        return 0.5*e
    else:
        return 0
def func_I(x, E=0):
    temp = max((V(x)-E)/e, .0)
    return np.sqrt(2*temp)

def WKB_trans(func_I, E, x1, x2):
    I = integrate.quad(func_I, x1, x2, args=(E),
                       limit=10000, epsabs=0)
    return np.sqrt(np.exp((-2*np.sqrt(m0*e)/h_)*I[0]))
E_mesh = np.linspace(0, 0.6*e, 121)
T_mesh = []
for E in E_mesh:
    T_mesh.append(WKB_trans(func_I, E, 0, 2.5e-9))
plt.figure(figsize=(4.5, 4))
plt.plot(E_mesh/e, T_mesh, c='r')
plt.plot([0.5, 0.5], [-0.5, 1.00], c='#000000', linestyle='--')
plt.plot([-0.03, 0.5], [1.0, 1.0], c='#000000', linestyle='--')
plt.xlabel('能量/eV')
plt.ylabel('投射系数')
plt.xlim(-0.03, 0.62)
plt.xlim(-0.05, 1.05)
plt.show()




