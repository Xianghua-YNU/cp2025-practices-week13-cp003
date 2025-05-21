#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
拉格朗日点 - 地球-月球系统L1点位置计算

本模块实现了求解地球-月球系统L1拉格朗日点位置的数值方法。
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

# 物理常数
G = 6.674e-11  # 万有引力常数 (m^3 kg^-1 s^-2)
M = 5.974e24   # 地球质量 (kg)
m = 7.348e22   # 月球质量 (kg)
R = 3.844e8    # 地月距离 (m)
omega = 2.662e-6  # 月球角速度 (s^-1)


def lagrange_equation(r):
    """
    L1拉格朗日点位置方程
    
    参数:
        r (float): 从地心到L1点的距离 (m)
    
    返回:
        float: 方程左右两边的差值，当r是L1点位置时返回0
    """
    # TODO: 实现L1点位置方程 (约5行代码)
    # [STUDENT_CODE_HERE]
    # 提示: 方程应该包含地球引力、月球引力和离心力的平衡关系
    
    # 计算地球引力项
    earth_gravity = G * M / (r ** 2)
    # 计算月球引力项（注意方向）
    moon_gravity = G * m / ((R - r) ** 2)
    # 计算离心力项
    centrifugal_force = omega ** 2 * r
    # 返回力平衡方程结果
    return earth_gravity - moon_gravity - centrifugal_force


def lagrange_equation_derivative(r):
    """
    L1拉格朗日点位置方程的导数，用于牛顿法
    
    参数:
        r (float): 从地心到L1点的距离 (m)
    
    返回:
        float: 方程对r的导数值
    """
    # TODO: 实现L1点位置方程的导数 (约5-10行代码)
    # [STUDENT_CODE_HERE]
    # 提示: 对lagrange_equation函数求导
    
    # 地球引力项的导数
    earth_deriv = -2 * G * M / (r ** 3)
    # 月球引力项的导数
    moon_deriv = 2 * G * m / ((R - r) ** 3)
    # 离心力项的导数
    centrifugal_deriv = omega ** 2
    return earth_deriv + moon_deriv - centrifugal_deriv

def newton_method(f, df, x0, tol=1e-8, max_iter=100):
    """
    使用牛顿法（切线法）求解方程f(x)=0
    
    参数:
        f (callable): 目标方程，形式为f(x)=0
        df (callable): 目标方程的导数
        x0 (float): 初始猜测值
        tol (float): 收敛容差
        max_iter (int): 最大迭代次数
    
    返回:
        tuple: (近似解, 迭代次数, 收敛标志)
    """
    # TODO: 实现牛顿法 (约15行代码)
    # [STUDENT_CODE_HERE]
    # 提示: 迭代公式为 x_{n+1} = x_n - f(x_n)/df(x_n)
    
    x = x0
    iterations = 0
    converged = False
    
    for _ in range(max_iter):
        fx = f(x)
        if abs(fx) < tol:
            converged = True
            break
        
        dfx = df(x)
        if abs(dfx) < 1e-14:
            break
        
        x -= fx / dfx
        iterations += 1
        
        if abs(fx) < tol:
            converged = True
            break
            
    return x, iterations, converged


def secant_method(f, a, b, tol=1e-8, max_iter=100):
    """
    使用弦截法求解方程f(x)=0
    
    参数:
        f (callable): 目标方程，形式为f(x)=0
        a (float): 区间左端点
        b (float): 区间右端点
        tol (float): 收敛容差
        max_iter (int): 最大迭代次数
    
    返回:
        tuple: (近似解, 迭代次数, 收敛标志)
    """
    # TODO: 实现弦截法 (约15行代码)
    # [STUDENT_CODE_HERE]
    # 提示: 迭代公式为 x_{n+1} = x_n - f(x_n)*(x_n-x_{n-1})/(f(x_n)-f(x_{n-1}))
    
    x0, x1 = a, b
    f0, f1 = f(x0), f(x1)
    iterations = 0
    converged = False
    
    for _ in range(max_iter):
        if abs(f1) < tol:
            converged = True
            break
            
        if abs(f1 - f0) < 1e-14:
            break
            
        x_next = x1 - f1 * (x1 - x0) / (f1 - f0)
        x0, x1 = x1, x_next
        f0, f1 = f1, f(x1)
        iterations += 1
        
        if abs(f1) < tol:
            converged = True
            break
            
    return x1, iterations, converged


def plot_lagrange_equation(r_min, r_max, num_points=1000):
    """
    绘制L1拉格朗日点位置方程的函数图像
    
    参数:
        r_min (float): 绘图范围最小值 (m)
        r_max (float): 绘图范围最大值 (m)
        num_points (int): 采样点数
    
    返回:
        matplotlib.figure.Figure: 绘制的图形对象
    """
    # TODO: 实现绘制方程图像的代码 (约15行代码)
    # [STUDENT_CODE_HERE]
    # 提示: 在合适的范围内绘制函数图像，标记零点位置
    
    r_values = np.linspace(r_min, r_max, num_points)
    f_values = [lagrange_equation(r) for r in r_values]
    
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    
    # 绘制函数曲线
    ax.plot(r_values/R, f_values, label='L1 Equation')
    
    # 标记零点位置
    zero_crossings = np.where(np.diff(np.sign(f_values)))[0]
    for idx in zero_crossings:
        r_zero = r_values[idx] - f_values[idx] * (r_values[idx+1] - r_values[idx]) / (f_values[idx+1] - f_values[idx])
        ax.plot(r_zero/R, 0, 'ro', label='Zero Crossing')
    
    ax.axhline(0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('Normalized Distance (Earth-Moon distance)')
    ax.set_ylabel('Equation Value')
    ax.set_title('L1 Lagrange Point Equation')
    ax.legend()
    ax.grid(True)
    
    return fig


def main():
    """
    主函数，执行L1拉格朗日点位置的计算和可视化
    """
    # 1. 绘制方程图像，帮助选择初值
    r_min = 3.0e8  # 搜索范围下限 (m)，约为地月距离的80%
    r_max = 3.8e8  # 搜索范围上限 (m)，接近地月距离
    fig = plot_lagrange_equation(r_min, r_max)
    plt.savefig('lagrange_equation.png', dpi=300)
    plt.show()
    
    # 2. 使用牛顿法求解
    print("\n使用牛顿法求解L1点位置:")
    r0_newton = 3.5e8  # 初始猜测值 (m)，大约在地月距离的90%处
    r_newton, iter_newton, conv_newton = newton_method(lagrange_equation, lagrange_equation_derivative, r0_newton)
    if conv_newton:
        print(f"  收敛解: {r_newton:.8e} m")
        print(f"  迭代次数: {iter_newton}")
        print(f"  相对于地月距离的比例: {r_newton/R:.6f}")
    else:
        print("  牛顿法未收敛!")
    
    # 3. 使用弦截法求解
    print("\n使用弦截法求解L1点位置:")
    a, b = 3.2e8, 3.7e8  # 初始区间 (m)
    r_secant, iter_secant, conv_secant = secant_method(lagrange_equation, a, b)
    if conv_secant:
        print(f"  收敛解: {r_secant:.8e} m")
        print(f"  迭代次数: {iter_secant}")
        print(f"  相对于地月距离的比例: {r_secant/R:.6f}")
    else:
        print("  弦截法未收敛!")
    
    # 4. 使用SciPy的fsolve求解
    print("\n使用SciPy的fsolve求解L1点位置:")
    r0_fsolve = 3.5e8  # 初始猜测值 (m)
    r_fsolve = optimize.fsolve(lagrange_equation, r0_fsolve)[0]
    print(f"  收敛解: {r_fsolve:.8e} m")
    print(f"  相对于地月距离的比例: {r_fsolve/R:.6f}")
    
    # 5. 比较不同方法的结果
    if conv_newton and conv_secant:
        print("\n不同方法结果比较:")
        print(f"  牛顿法与弦截法的差异: {abs(r_newton-r_secant):.8e} m ({abs(r_newton-r_secant)/r_newton*100:.8f}%)")
        print(f"  牛顿法与fsolve的差异: {abs(r_newton-r_fsolve):.8e} m ({abs(r_newton-r_fsolve)/r_newton*100:.8f}%)")
        print(f"  弦截法与fsolve的差异: {abs(r_secant-r_fsolve):.8e} m ({abs(r_secant-r_fsolve)/r_secant*100:.8f}%)")


if __name__ == "__main__":
    main()
