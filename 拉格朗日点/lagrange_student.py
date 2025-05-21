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
    
    moon_distance = R - r
    if moon_distance <= 1e-6 or r <= 1e6:  # 物理约束保护
        return np.inf
    earth_gravity = G * M / r**2
    moon_gravity = G * m / moon_distance**2
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
    
    moon_distance = R - r
    if moon_distance <= 1e-6 or r <= 1e6:
        return np.inf
    earth_deriv = -2 * G * M / r**3
    moon_deriv = 2 * G * m / moon_distance**3
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
    
    x = min(x0, 0.99*R)  # 初始约束
    for i in range(max_iter):
        try:
            fx = f(x)
            dfx = df(x)
        except:
            return x, i, False
            
        if abs(fx) < tol:
            return x, i+1, True
            
        if abs(dfx) < 1e-14:
            break
            
        delta = fx / dfx
        delta = np.clip(delta, -0.01*R, 0.01*R)  # 步长限制
        x = max(1e6, min(0.999*R, x - delta))  # 物理范围约束
        
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
    
    x0, x1 = a, b
    f0, f1 = f(x0), f(x1)
    for i in range(max_iter):
        if f0*f1 > 0:  # 确保区间有效性
            x1 = 0.5*(x0 + x1)
            f1 = f(x1)
            continue
            
        delta = (x1 - x0)/(f1 - f0 + 1e-14)
        x_next = x1 - f1 * delta
        x_next = max(1e6, min(0.999*R, x_next))  # 物理约束
        
        f_next = f(x_next)
        if abs(f_next) < tol:
            return x_next, i+1, True
            
        x0, x1 = x1, x_next
        f0, f1 = f1, f_next
        
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
    # 1. 计算天体力学近似初始值
    mass_ratio = m / (M + m)
    r0_approx = R * (1 - (mass_ratio/3)**(1/3))  # 理论近似公式[1,4](@ref)

    # 2. 绘制方程图像（调整范围避免R边界）
    r_min = 0.7 * R  # 约2.69e8米
    r_max = 0.95 * R # 约3.65e8米
    fig = plot_lagrange_equation(r_min, r_max)
    plt.savefig('lagrange_equation.png', dpi=300)
    plt.show()

    # 3. 牛顿法求解（使用理论初始值）
    print("\n=== 牛顿法求解L1点 ===")
    r_newton, iter_newton, conv_newton = newton_method(
        lagrange_equation, 
        lagrange_equation_derivative, 
        r0_approx,
        tol=1e-8,
        max_iter=100
    )
    if conv_newton:
        print(f"收敛解: {r_newton:.8e} m")
        print(f"迭代次数: {iter_newton}")
        print(f"相对地月距离: {r_newton/R:.6f}")
    else:
        print("警告：牛顿法未收敛！")

    # 4. 弦截法求解（优化初始区间）
    print("\n=== 弦截法求解L1点 ===")
    a, b = 0.6*R, 0.9*R  # 确保包含零点[6,8](@ref)
    r_secant, iter_secant, conv_secant = secant_method(
        lagrange_equation, 
        a, 
        b,
        tol=1e-8,
        max_iter=100
    )
    if conv_secant:
        print(f"收敛解: {r_secant:.8e} m")
        print(f"迭代次数: {iter_secant}")
        print(f"相对地月距离: {r_secant/R:.6f}")
    else:
        print("警告：弦截法未收敛！")

    # 5. SciPy验证（统一初始值）
    print("\n=== SciPy fsolve验证 ===")
    r_fsolve = optimize.fsolve(lagrange_equation, r0_approx)[0]
    print(f"SciPy解: {r_fsolve:.8e} m")
    print(f"相对地月距离: {r_fsolve/R:.6f}")

    # 6. 综合对比（含理论值）
    if conv_newton and conv_secant:
        print("\n=== 结果对比 ===")
        # 理论值计算[11](@ref)
        theory_val = R * (1 - (mass_ratio/3)**(1/3))  
        
        # 各方法结果汇总
        results = {
            "牛顿法": r_newton,
            "弦截法": r_secant,
            "SciPy": r_fsolve,
            "理论值": theory_val
        }
        
        # 按距离排序输出
        sorted_results = sorted(results.items(), key=lambda x: x[1])
        for name, val in sorted_results:
            print(f"{name:8} {val/R:.6f}R ({val:.3e} m)")
        
        # 误差分析
        print("\n=== 相对误差 ===")
        for name, val in results.items():
            if name != "理论值":
                error = abs((val - theory_val)/theory_val)*100
                print(f"{name:8} 误差: {error:.6f}%")


if __name__ == "__main__":
    main()
