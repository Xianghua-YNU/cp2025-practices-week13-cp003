#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
薛定谔方程 - 方势阱能级计算

本模块实现了一维方势阱中粒子能级的计算方法。
"""

import numpy as np
import matplotlib.pyplot as plt

# 物理常数
HBAR = 1.0545718e-34  # 约化普朗克常数 (J·s)
ELECTRON_MASS = 9.1094e-31  # 电子质量 (kg)
EV_TO_JOULE = 1.6021766208e-19  # eV -> J


def calculate_y_values(E_values, V, w, m):
    E_joule = E_values * EV_TO_JOULE
    alpha_sq = w**2 * m * E_joule / (2 * HBAR**2)

    y1 = np.tan(np.sqrt(alpha_sq))
    y2 = np.sqrt((V - E_values) / E_values)
    y3 = -np.sqrt(E_values / (V - E_values))

    y1 = np.clip(y1, -10, 10)
    y2 = np.clip(y2, -10, 10)
    y3 = np.clip(y3, -10, 10)

    return y1, y2, y3


def plot_energy_functions(E_values, y1, y2, y3):
    fig = plt.figure(figsize=(10, 6))
    plt.plot(E_values, y1, label=r'$y_1 = \tan(\sqrt{\frac{w^2 m E}{2\hbar^2}})$', color='blue')
    plt.plot(E_values, y2, label=r'$y_2 = \sqrt{(V - E)/E}$ (even)', linestyle='--', color='orange')
    plt.plot(E_values, y3, label=r'$y_3 = -\sqrt{E/(V - E)}$ (odd)', linestyle='-.', color='green')
    plt.axhline(0, color='gray', linestyle=':')
    plt.xlabel("Energy E (eV)")
    plt.ylabel("Function Value")
    plt.title("Transcendental Equation Functions for Finite Square Well")
    plt.ylim(-10, 10)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    return fig


def transcendental_eq(E, V, w, m, parity):
    E_joule = E * EV_TO_JOULE
    alpha = np.sqrt(w**2 * m * E_joule / (2 * HBAR**2))

    if parity == 'even':
        lhs = np.tan(alpha)
        rhs = np.sqrt((V - E) / E)
    else:
        lhs = np.tan(alpha)
        rhs = -np.sqrt(E / (V - E))
    return lhs - rhs


def find_energy_level_bisection(n, V, w, m, precision=0.001, E_min=0.001, E_max=None):
    if E_max is None:
        E_max = V - 0.001
    parity = 'even' if n % 2 == 0 else 'odd'
    count = 0
    roots = []
    E_vals = np.linspace(E_min, E_max, 10000)
    f_vals = transcendental_eq(E_vals, V, w, m, parity)

    for i in range(len(E_vals) - 1):
        if np.sign(f_vals[i]) != np.sign(f_vals[i + 1]):
            a, b = E_vals[i], E_vals[i + 1]
            while b - a > precision:
                c = (a + b) / 2
                if np.sign(transcendental_eq(a, V, w, m, parity)) != np.sign(transcendental_eq(c, V, w, m, parity)):
                    b = c
                else:
                    a = c
            roots.append((a + b) / 2)
            count += 1
            if count > n:
                return roots[n]
    return None


def main():
    V = 20.0  # eV
    w = 1e-9  # m
    m = ELECTRON_MASS

    E_values = np.linspace(0.001, V - 0.001, 1000)
    y1, y2, y3 = calculate_y_values(E_values, V, w, m)
    fig = plot_energy_functions(E_values, y1, y2, y3)
    plt.savefig('energy_functions.png', dpi=300)
    plt.show()

    energy_levels = []
    for n in range(6):
        energy = find_energy_level_bisection(n, V, w, m)
        energy_levels.append(energy)
        print(f"能级 {n}: {energy:.3f} eV")

    print("\n参考能级值:")
    reference_levels = [0.318, 1.270, 2.851, 5.050, 7.850, 11.215]
    for n, ref in enumerate(reference_levels):
        print(f"能级 {n}: {ref:.3f} eV")

    # 计算相对误差
    print("\n相对误差:")
    for n, (calc, ref) in enumerate(zip(energy_levels, reference_levels)):
        rel_error = abs(calc - ref) / ref * 100
        print(f"能级 {n}: {rel_error:.2f}%")


if __name__ == "__main__":
    main()
