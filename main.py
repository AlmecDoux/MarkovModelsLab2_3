import math
import numpy as np
from matplotlib import pyplot as plt

N = 100
start_property = [0.9, 0.1]
mtrx = np.array([[0.9, 0.1], [0.2, 0.8]])
v = [-1, 1]
m = 3
q = 3


# Генерация цепи Маркова и получение матрицы вероятности переходов

def generate_next_value_chain(x):
    if x == v[0]:
        return np.random.choice(v, p=mtrx[0])
    elif x == v[1]:
        return np.random.choice(v, p=mtrx[1])


def generate_chain(N):
    chain_ = []
    next_value = np.random.choice(v, p=start_property)
    chain_.append(next_value)
    for _ in np.arange(0, N - 1):
        next_value = generate_next_value_chain(next_value)
        chain_.append(next_value)
    return chain_


def generate_psp(psp_array):
    for _ in range(0, 20):
        psp_array.append((psp_array[len(psp_array) - 1] + psp_array[len(psp_array) - 3]) % 2)
    for i in range(0, len(psp_array)):
        if psp_array[i] == 0:
            psp_array[i] = -1
    return psp_array


chain = generate_chain(N)


def D(m1, m2):
    return m1 - m2


def FM_phasing_manipulation(signal, q, gauss):
    return 4 * q * (signal_transformation(signal) + np.random.uniform(0, 1, 1) / (pow(2 * q, 1 / 2)))


chain = normal_form(chain)
result_chain = []
for s in chain:
    result_chain.append(phasing_manipulation(s))

print(result_chain)
