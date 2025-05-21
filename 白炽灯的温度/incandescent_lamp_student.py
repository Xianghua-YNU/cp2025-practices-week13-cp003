#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
白炽灯温度优化 - 学生代码模板

请根据项目说明实现以下函数，完成白炽灯效率与最优温度的计算。
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.optimize import minimize_scalar

# 设置中文字体显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 物理常数定义
H = 6.62607015e-34  # 普朗克常数 [J·s]
C = 299792458       # 光速 [m/s] 
K_B = 1.380649e-23  # 玻尔兹曼常数 [J/K]

# 可见光波长范围 [m]
VISIBLE_LIGHT_MIN = 380e-9  # 380 nm (紫光)
VISIBLE_LIGHT_MAX = 780e-9  # 780 nm (红光)

def planck_law(wavelength, temperature):
    """
    普朗克黑体辐射公式计算
    
    参数:
        wavelength: 波长 [m], 可以是单个值或numpy数组
        temperature: 黑体温度 [K]
        
    返回:
        辐射强度 [W/(m²·m)]
    """
    # 计算指数部分的参数
    arg = H * C / (wavelength * K_B * temperature)
    
    # 对极大值进行截断处理（避免exp溢出）
    max_exp_arg = 500  # exp(700) 已经是非常大的数了
    arg = np.where(arg > max_exp_arg, max_exp_arg, arg)
    # 计算分子部分: 2hc²/λ⁵
    numerator = 2.0 * H * C**2 / (wavelength**5)
    # 计算指数部分: exp(hc/λk_BT)
    exponent = np.exp(arg)
    # 完整的普朗克公式
    return numerator / (exponent - 1.0)

def calculate_visible_power_ratio(temperature):
    """
    计算可见光功率占总辐射功率的比例
    
    参数:
        temperature: 黑体温度 [K]
        
    返回:
        可见光效率 [0-1]
    """
    # 定义被积函数
    def intensity_func(wavelength):
         # 对极小的波长值返回0（避免数值问题）
        if wavelength < 1e-12:  # 1皮米以下的波长不考虑
            return 0.0
        return planck_law(wavelength, temperature)
    
    # 计算可见光波段积分 (380-780nm)
    visible_power, _ = integrate.quad(intensity_func, VISIBLE_LIGHT_MIN, VISIBLE_LIGHT_MAX)
    # 计算全波段积分 (1nm-10000nm)
    total_power, _ = integrate.quad(intensity_func, 1e-9, 10000e-9)
    
    return visible_power / total_power

def plot_efficiency_vs_temperature(temp_range):
    """
    绘制效率-温度关系曲线
    
    参数:
        temp_range: 温度范围数组 [K]
        
    返回:
        fig: 图形对象
        temps: 温度数组
        effs: 效率数组
    """
    # 计算各温度对应的效率
    efficiencies = np.array([calculate_visible_power_ratio(T) for T in temp_range])
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(temp_range, efficiencies, 'b-', linewidth=2)
    
    # 标记最大效率点
    max_idx = np.argmax(efficiencies)
    max_temp = temp_range[max_idx]
    max_eff = efficiencies[max_idx]
    ax.plot(max_temp, max_eff, 'ro', markersize=8)
    ax.text(max_temp, max_eff*0.95, 
           f'峰值效率: {max_eff:.4f}\n温度: {max_temp:.1f} K',
           ha='center', va='top')
    
    # 设置图形属性
    ax.set_title('白炽灯可见光效率 vs 温度', fontsize=14)
    ax.set_xlabel('温度 [K]', fontsize=12)
    ax.set_ylabel('可见光效率', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    fig.tight_layout()
    
    return fig, temp_range, efficiencies

def find_optimal_temperature():
    """
    寻找最优工作温度
    
    返回:
        optimal_temp: 最优温度 [K]
        optimal_eff: 最大效率 [0-1]
    """
    # 定义目标函数（求负值以便使用最小化函数）
    def objective(T):
        return -calculate_visible_power_ratio(T)
    
    # 使用有界优化算法
    result = minimize_scalar(
        objective,
        bounds=(1000, 10000),  # 温度搜索范围
        method='bounded',
        options={'xatol': 1.0}  # 温度精度1K
    )
    
    return result.x, -result.fun

def main():
    """主程序"""
    print("白炽灯效率分析程序运行中...")
    
    # 1. 绘制效率-温度曲线
    temp_range = np.linspace(1000, 10000, 200)  # 200个温度点
    fig, temps, effs = plot_efficiency_vs_temperature(temp_range)
    plt.savefig('efficiency_curve.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. 计算最优温度
    opt_temp, opt_eff = find_optimal_temperature()
    print(f"\n计算结果:")
    print(f"理论最优温度: {opt_temp:.1f} K")
    print(f"最大可见光效率: {opt_eff:.4f} ({opt_eff*100:.2f}%)")
    
    # 3. 与实际白炽灯比较
    real_temp = 2700  # 典型白炽灯工作温度
    real_eff = calculate_visible_power_ratio(real_temp)
    print(f"\n实际参数:")
    print(f"工作温度: {real_temp} K")
    print(f"实际效率: {real_eff:.4f} ({real_eff*100:.2f}%)")
    print(f"效率提升空间: {(opt_eff-real_eff)*100:.2f}%")
    
    # 4. 绘制对比图
    plt.figure(figsize=(10, 6))
    plt.plot(temps, effs, 'b-', label='效率曲线')
    plt.plot(opt_temp, opt_eff, 'ro', label=f'最优温度 {opt_temp:.1f}K')
    plt.plot(real_temp, real_eff, 'gs', label=f'实际温度 {real_temp}K')
    plt.xlabel('温度 [K]')
    plt.ylabel('可见光效率')
    plt.title('白炽灯效率优化分析')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig('comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()
