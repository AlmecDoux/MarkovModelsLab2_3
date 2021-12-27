import math
import numpy as np
from matplotlib import pyplot as plt

N = 100
start_property = [0.9, 0.1]
mtrx = np.array([[0.9, 0.1], [0.2, 0.8]])
v = [-1, 1]
m = 3
q = 3
d = {
    (1, 1): 0,
    (1, -1): 0,
    (-1, 1): 0,
    (-1, -1): 0,
}


# Генерация цепи Маркова и получение матрицы вероятности переходов

def generate_next_value_chain(x):
    if x == v[0]:
        return np.random.choice(v, p=mtrx[0])
    elif x == v[1]:
        return np.random.choice(v, p=mtrx[1])


def generate_chain(N):
    chain = []
    next_value = np.random.choice(v, p=start_property)
    chain.append(next_value)
    for _ in np.arange(0, N - 1):
        tmp = next_value
        next_value = generate_next_value_chain(next_value)
        d[(tmp, next_value)] += 1
        chain.append(next_value)
    return chain


def generate_psp(psp_array):
    for _ in range(0, 20):
        psp_array.append((psp_array[len(psp_array) - 1] + psp_array[len(psp_array) - 3]) % 2)
    for i in range(0, len(psp_array)):
        if psp_array[i] == 0:
            psp_array[i] = -1
    return psp_array


chain = generate_chain(N)
count_one = len(list(filter(lambda x: x == 1, chain)))
count_m_one = len(list(filter(lambda x: x == -1, chain)))
p_matrix = np.array([
    [d[(1, 1)] / count_one, d[(1, -1)] / count_one],
    [d[(-1, 1)] / count_m_one, d[(-1, -1)] / count_m_one]
])

# Получение ПСП
psp_array = generate_psp([1, 1, 1])
L = 2 ** m - 1
psp_array = psp_array[:L]


def add_noise(chain_, psp_array_):
    noise_chain = []
    for chain_char in chain_:
        for noise_char in psp_array_:
            print('noise_char', noise_char)
            print('chain_char', chain_char)
            print(noise_char + chain_char)
            noise_chain.append((noise_char + chain_char) % 2)
    return noise_chain


def delete_noise(noise_chain, psp_):
    clean_chain = []
    k = 0
    for noise_chain_part in noise_chain:
        if k == len(psp_):
            k = 0
        print(k)
        clean_chain.append((noise_chain_part + psp_[k]) % 2)
        k = k + 1
    return clean_chain

def D(m1, m2):
    return m1 - m2

def normal_form(chain):
    for i in range(0, len(chain)):
        if chain[i] == 0:
            chain[i] = 1
        else:
            chain[i] = -1
    return chain

def signal_transformation(s):
    if s == v[0]:
        return 1
    else:
        return -1


def phasing_manipulation(s):
    return 4 * q * (signal_transformation(s) + np.random.uniform(0, 1, 1) / (pow(2 * q, 1 / 2)))



chain = normal_form(chain)
result_chain = []
for s in chain:
    result_chain.append(phasing_manipulation(s))

print(result_chain)