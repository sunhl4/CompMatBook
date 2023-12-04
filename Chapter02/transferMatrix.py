import numpy as np
from matplotlib import  pyplot as plt
# 计算第n， n+1层之间的边界矩阵
def Calculate_Boundary_Condition_Matrix(deltan):
    bcMatrix = (1/2) * np.array([[1+deltan, 1-deltan],
                                 [1-deltan, 1+deltan]])
    return bcMatrix

# 计算第n层的传递矩阵
def Calculate_Propagation_Matrix(kn, dmn):
    propMatrix = np.array([[np.exp(-1j*kn*dmn), 0],[0, np.exp(1j*kn*dmn)]])
    return propMatrix

def divide(a, b):
    if (b == 0):
        return complex(np.log(float('inf')**a.real), np.log(float('inf')**a.imag))
    else:
        return a/b

def TransmissionVsEnergyPlot(MF, VM, DM, energy):
    tfrac = np.zeros(len(energy)) #初始化透射系数
    # 常量声明
    hbar = 1.005 * 1e-34
    mo = 9.1095 * 1e-31
    q = 1.602 * 1e-19
    s = (2 * q * mo * 1e-18) / (hbar**2)
    # 主循环
    for m in range (len(energy)):
        # 初始化第一层的边界条件矩阵
        # 计算第一层的 K 值
        k1 = np.sqrt(complex(s * MF[0] * (energy[m] - VM[0])))
        # 计算第二层的 k 值
        k2 = np.sqrt(complex(s * MF[1] * (energy[m] - VM[1])))
        # 计算第一层的 delta
        delta1 = (k2 / k1) * (MF[0] / MF[1])
        trans = Calculate_Boundary_Condition_Matrix(delta1)
        # 对剩余各层
        for n in range(len(MF) - 2):
            n += 1
            #   计算第 n 层
            kn = np.sqrt(complex(s * MF[n] * (energy[m] - VM[n])))
            # 计算下一层的 k
            kn1 = np.sqrt(complex(s * MF[n+1] * (energy[m] - VM[n+1])))
            # 计算 deltan
            deltan = (kn1 /kn) * (MF[n] / MF[n+1])
            LayerMatrixn = np.dot(Calculate_Propagation_Matrix(kn, DM[n]),
                                  Calculate_Boundary_Condition_Matrix(deltan))
            trans = np.dot(trans, LayerMatrixn)
        # 求各能量下的透射系数
        tfrac[m] = 1-abs(trans[1, 0])**2 / abs(trans[0, 0])**2
    # 画出透射系数随能量变化的曲线图
    plt.plot(energy, tfrac, c='r')
    plt.xlabel('E(eV)')
    plt.ylabel('T')
    plt.show()

# 定义各层参数并求解
MF = [1, 1, 1, 1, 1];   # 各层中电子的有效质量（单位：mo）
VM = [0, 0.5, 0, 0.5, 0];   # 各层势垒高度（单位：eV）
DM = [0, 0.5, 0.5, 0.5, 0];  #各层厚度（单位：nm）
energy = np.linspace(0, 2, 400)

TransmissionVsEnergyPlot(MF, VM, DM, energy)





