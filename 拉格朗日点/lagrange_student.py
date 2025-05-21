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
    
    moon_dist = max(R - r, 1e-6)  
    earth_gravity = G * M / r**2
    moon_gravity = G * m / moon_dist**2
    centrifugal = omega**2 * r
    return earth_gravity - moon_gravity - centrifugal

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
    
    moon_dist = max(R - r, 1e-6)
    earth_deriv = -2 * G * M / r**3
    moon_deriv = 2 * G * m / moon_dist**3
    return earth_deriv + moon_deriv - omega**2

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
    
    def newton_method(f, df, x0, tol=1e-10, max_iter=100):
    """改进的阻尼牛顿法实现"""
    x = np.clip(x0, 0.1*R, 0.95*R)  # 物理范围约束[3,6](@ref)
    prev_fx = np.inf
    damping = 1.0  # 初始阻尼因子
    
    for i in range(max_iter):
        try:
            fx = f(x)
            dfx = df(x)
        except:
            return x, i, False
        
        # 收敛条件增强：函数值与步长双重检查[1,3](@ref)
        if abs(fx) < tol and abs(x - prev_x) < 0.1*tol*R:
            return x, i+1, True
        
        # 动态阻尼调整（网页2、6方法）
        if abs(fx) >= abs(prev_fx):  
            damping *= 0.5  # 函数值未下降时增强阻尼
            damping = max(damping, 0.1)  # 最小阻尼限制
        else:
            damping = min(damping*1.1, 1.0)  # 恢复阻尼
        
        # 步长计算与限制
        delta = fx / (dfx + 1e-14)  # 避免零除
        max_step = 0.1 * R * damping  # 动态最大步长
        delta = np.clip(delta, -max_step, max_step)
        
        prev_x = x
        prev_fx = fx
        x = np.clip(x - delta, 0.1*R, 0.95*R)  # 物理约束
        
    return x, i+1, abs(fx) < tol


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
    
    x0, x1 = np.clip(a, 0.6*R, 0.95*R), np.clip(b, 0.6*R, 0.95*R)
    f0, f1 = f(x0), f(x1)
    
    # 强制初始区间有效性
    for _ in range(5):  # 最多尝试5次调整区间
        if f0 * f1 < 0:
            break
        x1 = x0 + 0.1*(x1 - x0)
        f1 = f(x1)
    else:
        return x1, 0, False

    for i in range(max_iter):
        # 数值稳定性处理
        delta = (x1 - x0) / (f1 - f0 + 1e-14)
        x_new = x1 - f1 * delta
        x_new = np.clip(x_new, 0.6*R, 0.95*R)
        
        f_new = f(x_new)
        if abs(f_new) < tol:
            return x_new, i+1, True
            
        # 选择保留符号相反的端点
        if f_new * f1 < 0:
            x0, f0 = x1, f1
        x1, f1 = x_new, f_new
        
    return x1, i+1, False


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
    
    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111)
    
    # 理论解标记
    mass_ratio = m/(M + m)
    theory_r = R * (1 - (mass_ratio/3)**(1/3))
    ax.axvline(theory_r/R, color='green', linestyle='--', label='Theoretical L1')
    
    ax.plot(r_values/R, f_values, label='L1 Equation')
    
    # 精确零点计算
    zero_crossings = np.where(np.diff(np.sign(f_values)))[0]
    for idx in zero_crossings:
        r0, r1 = r_values[idx], r_values[idx+1]
        f0, f1 = f_values[idx], f_values[idx+1]
        r_zero = r0 - f0*(r1 - r0)/(f1 - f0)
        ax.plot(r_zero/R, 0, 'ro', markersize=8, label='Numerical Solution' if idx==0 else "")
    
    ax.axhline(0, color='k', linestyle=':', alpha=0.5)
    ax.set_xlabel('Normalized Distance (Earth-Moon distance)')
    ax.set_ylabel('Equation Value')
    ax.set_title('L1 Lagrange Point Equation (Updated)')
    ax.legend(loc='upper left')
    ax.grid(True)
    return fig

def main():
    # 精确理论解计算
    mass_ratio = m/(M + m)
    r0_approx = R * (1 - (mass_ratio/3)**(1/3))
    
    # 绘图范围优化
    r_min, r_max = 0.6*R, 0.95*R
    fig = plot_lagrange_equation(r_min, r_max)
    plt.savefig('lagrange_equation_v2.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 牛顿法求解（提升收敛精度）
    print("\n=== 牛顿法求解 ===")
    r_newton, iter_newton, conv_newton = newton_method(
        lagrange_equation, 
        lagrange_equation_derivative,
        r0_approx,
        tol=1e-10,
        max_iter=100
    )
    
    # 弦截法求解（优化初始区间）
    print("\n=== 弦截法求解 ===")
    a, b = 0.7*R, 0.9*R  # 确保包含理论解[9](@ref)
    r_secant, iter_secant, conv_secant = secant_method(
        lagrange_equation, a, b,
        tol=1e-10,
        max_iter=100
    )
    
    # SciPy验证（增加容错）
    print("\n=== SciPy验证 ===")
    try:
        r_fsolve = optimize.fsolve(
            lagrange_equation, 
            r0_approx,
            fprime=lagrange_equation_derivative,
            xtol=1e-10
        )[0]
    except:
        r_fsolve = optimize.root_scalar(
            lagrange_equation,
            bracket=[0.7*R, 0.95*R],
            xtol=1e-10
        ).root

    # 结果对比（增加单位转换）
    if conv_newton and conv_secant:
        theory_m = R * (1 - (mass_ratio/3)**(1/3))
        print(f"\n理论解: {theory_m/R:.6f}R ({theory_m/1000:.2f} km)")
        
        print("\n=== 数值解对比 ===")
        results = {
            "Newton": r_newton,
            "Secant": r_secant,
            "SciPy": r_fsolve
        }
        for name, val in results.items():
            print(f"{name:8} {val/R:.6f}R 误差: {abs(val - theory_m)/1000:.4f} km")


if __name__ == "__main__":
    main()
