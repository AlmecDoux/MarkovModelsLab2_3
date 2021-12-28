import math
import numpy as np
from matplotlib import pyplot as plt

N = 10 ** 6
start_property = [0.5, 0.5]
mtrx = np.array([[0.6, 0.4], [0.2, 0.8]])
v = [-1, 1]


def generate_next_value_chain(x):
    if x == v[0]:
        return np.random.choice(v, p=mtrx[0])
    elif x == v[1]:
        return np.random.choice(v, p=mtrx[1])


def generate_chain(N_):
    chain_ = []
    next_value = np.random.choice(v, p=start_property)
    chain_.append(next_value)
    for _ in np.arange(0, N_ - 1):
        next_value = generate_next_value_chain(next_value)
        chain_.append(next_value)
    return chain_


def signal_transformation(signal):
    if signal == v[0]:
        return 1
    else:
        return -1


def FM_phasing_manipulation(signal, q_, gauss_):
    return 4 * q_ * (signal_transformation(signal) + gauss_ / (math.sqrt(2 * q_)))


def CHM_phasing_manipulation(signal, q_, gauss_):
    return 2 * q_ * (signal_transformation(signal) + gauss_ / (math.sqrt(q_)))


def dB_at_times(x):
    return pow(10, x / 10)


def solver(x):
    if x > (np.log(start_property[1] / start_property[0])):
        return -1
    else:
        return 1


def property_chain(result_chain_):
    k = 0
    for i in range(0, len(result_chain_)):
        chain_for_plot.append(solver(result_chain_[i]))
        if solver(result_chain_[i]) != chain[i]:
            k = k + 1
        chain_for_plot.append(solver(result_chain_[i]))
    return k / len(result_chain_)


Q = [-6, -3, 0, 3, 6]
chain_for_plot = []
q_times = []

# Генерация цепи Маркова
chain = generate_chain(N)
gauss = np.random.standard_normal(len(chain))

# Перевод dB в разы
for q in Q:
    q_times.append(dB_at_times(q))

# ЧМ Модуляция
print('ЧМ Модуляция: ')
chm_modul_result = []
for q in q_times:
    result_chain = []
    for i in range(0, len(chain)):
        result_chain.append(CHM_phasing_manipulation(chain[i], q, gauss[i]))

    chm_modul_result.append(property_chain(result_chain))
print(chm_modul_result)

# ФМ Модуляция
print('ФМ Модуляция: ')
fm_modul_result = []
for q in q_times:
    result_chain = []
    for i in range(0, len(chain)):
        result_chain.append(FM_phasing_manipulation(chain[i], q, gauss[i]))

    fm_modul_result.append(property_chain(result_chain))
print(fm_modul_result)


def draw_plot():
    # График первых 100 элементов информационной последовательности
    plt.plot(chain[:100])
    plt.show()

    # График первых 100 элементов последовательности на выходе решающего устройства приемника.
    plt.plot(chain_for_plot[:100])
    plt.show()

    # График зависимости вероятности ошибки распознавания ЧМ-сигналов
    plt.plot(Q, chm_modul_result)
    plt.show()

    # График зависимости вероятности ошибки распознавания ФМ-сигналов
    plt.plot(Q, fm_modul_result)
    plt.show()


draw_plot()
