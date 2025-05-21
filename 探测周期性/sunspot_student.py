#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
太阳黑子周期性分析 - 学生代码模板

请根据项目说明实现以下函数，完成太阳黑子效率与最优温度的计算。
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
def load_sunspot_data(url):
    """
    从本地文件读取太阳黑子数据
    
    参数:
        url (str): 本地文件路径
        
    返回:
        tuple: (years, sunspots) 年份和太阳黑子数
    """
    # TODO: 使用np.loadtxt读取数据，只保留第2(年份)和3(太阳黑子数)列
    # [STUDENT_CODE_HERE]
    data = np.loadtxt(ur1)
    years = data[:, 0]  # 第一列是年份
    sunspots = data[:, 1]  # 第二列是太阳黑子数
    raise NotImplementedError("请在 {} 中实现此函数".format(__file__))
    return years, sunspots

def plot_sunspot_data(years, sunspots):
    """
    绘制太阳黑子数据随时间变化图
    
    参数:
        years (numpy.ndarray): 年份数组
        sunspots (numpy.ndarray): 太阳黑子数数组
    """
    # TODO: 实现数据可视化
    # [STUDENT_CODE_HERE]
    plt.figure(figsize=(12, 6))
    plt.plot(years, sunspots)
    plt.title('Sunspot Number Over Time')
    plt.xlabel('Year')
    plt.ylabel('Sunspot Number')
    plt.grid(True)
    plt.savefig('sunspot_time_series.png')
    plt.show()
    raise NotImplementedError("请在 {} 中实现此函数".format(__file__))

def compute_power_spectrum(sunspots):
    """
    计算太阳黑子数据的功率谱
    
    参数:
        sunspots (numpy.ndarray): 太阳黑子数数组
        
    返回:
        tuple: (frequencies, power) 频率数组和功率谱
    """
    # TODO: 实现傅里叶变换和功率谱计算
    # [STUDENT_CODE_HERE]
    N = len(sunspots)
    yf = fft(sunspots)
    xf = fftfreq(N, 1)[:N//2]  # 每月一个数据点
    power = 2/N * np.abs(yf[0:N//2])
    raise NotImplementedError("请在 {} 中实现此函数".format(__file__))
    return frequencies, power

def plot_power_spectrum(frequencies, power):
    """
    绘制功率谱图
    
    参数:
        frequencies (numpy.ndarray): 频率数组
        power (numpy.ndarray): 功率谱数组
    """
    # TODO: 实现功率谱可视化
    # [STUDENT_CODE_HERE]
    plt.figure(figsize=(12, 6))
    plt.plot(1/frequencies[1:], power[1:])  # 转换为周期
    plt.title('Power Spectrum of Sunspot Numbers')
    plt.xlabel('Period (months)')
    plt.ylabel('Power')
    plt.xlim(0, 200)  # 限制周期范围
    plt.grid(True)
    plt.savefig('sunspot_power_spectrum.png')
    plt.show()
    raise NotImplementedError("请在 {} 中实现此函数".format(__file__))

def find_main_period(frequencies, power):
    """
    找出功率谱中的主周期
    
    参数:
        frequencies (numpy.ndarray): 频率数组
        power (numpy.ndarray): 功率谱数组
        
    返回:
        float: 主周期（月）
    """
    # TODO: 实现主周期检测
    # [STUDENT_CODE_HERE]
    # 跳过第一个频率（无穷大周期）
    idx = np.argmax(power[1:]) + 1
    main_period = 1/frequencies[idx]
    raise NotImplementedError("请在 {} 中实现此函数".format(__file__))
    return main_period

def main():
    # 数据文件路径
    data = "sunspot_data.txt"
    
    # 1. 加载并可视化数据
    years, sunspots = load_sunspot_data(data)
    plot_sunspot_data(years, sunspots)
    
    # 2. 傅里叶变换分析
    frequencies, power = compute_power_spectrum(sunspots)
    plot_power_spectrum(frequencies, power)
    
    # 3. 确定主周期
    main_period = find_main_period(frequencies, power)
    print(f"\nMain period of sunspot cycle: {main_period:.2f} months")
    print(f"Approximately {main_period/12:.2f} years")

if __name__ == "__main__":
    main()
