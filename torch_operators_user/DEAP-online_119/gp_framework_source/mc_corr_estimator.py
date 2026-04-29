import re, os
import gc, sys

username = os.getlogin()
sys.path.append(rf'/home/{username}/torch_operators_user')
from torch_operators_my import *
import torch
import numpy as np
from scipy.stats import norm
import pandas as pd
# torch.cuda.is_available() 
mc_device = "cuda:0"
torch.manual_seed(100)

__default_n_time = 1024
__default_n_stock = 4096


def gen_field_mc(n_time=__default_n_time, n_stock=__default_n_stock):
    """
    使用蒙特卡洛模拟生成基础字段数据
    Args:
        n_time: 时间步
        n_stock: 股票数量

    Returns: torch.tensor

    """
    data = torch.rand(size=(n_time, n_stock), device=mc_device, dtype=torch.float32)
    data = torch.clamp(data, min=1e-6, max=1.0 - 1e-6)  # 确保不包含 0（可选）

    return data


def generate_correlated_uniform_data_from_corr(correlation_matrix, T, N):
    """
    生成多个具有指定相关系数矩阵的 0-1 均匀分布数据。

    Args:
        correlation_matrix (torch.Tensor): base_factor_num x base_factor_num 的相关系数矩阵
        T (int): 时间维度
        N (int): 字段维度
        device (str): 设备 ('cuda' or 'cpu')

    Returns:
        torch.Tensor: 生成的多个具有相关系数矩阵的 0-1 均匀分布数据矩阵
    """
    # 计算协方差矩阵，假设标准差为1
    base_factor_num = correlation_matrix.shape[0]
    std_array = torch.ones(base_factor_num, device=mc_device)  # 假设标准差为1
    cov_matrix = torch.outer(std_array, std_array) * correlation_matrix  # 构建协方差矩阵

    # 添加一个小常数，确保协方差矩阵是正定的
    epsilon = 1e-6
    cov_matrix += torch.eye(cov_matrix.shape[0], device=mc_device) * epsilon

    # 使用多元正态分布生成数据
    dist = torch.distributions.MultivariateNormal(torch.zeros(base_factor_num,  device=mc_device),
                                                  covariance_matrix=cov_matrix)

    normal_data = dist.sample((T, N))

    # 将数据从标准正态分布映射到均匀分布 [0, 1]
    standard_normal = torch.distributions.Normal(0, 1)
    uniform_data = torch.tensor(standard_normal.cdf(normal_data), dtype=torch.float32, device=mc_device)

    uniform_data = torch.clamp(uniform_data, 0, 1)

    return uniform_data


def extract_ops_and_fields(expr):
    # 匹配函数名（以字母开头，后跟字母或数字，下接括号）
    ops = re.findall(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', expr)

    # 匹配字段名（变量名），排除函数名，排除数字和括号
    fields = re.findall(r'\b([A-Za-z_][A-Za-z0-9_]*)\b', expr)

    # 去除函数名、数字等非变量字段
    func_names_set = set(ops)
    fields_filtered = set([f for f in fields if f not in func_names_set and not f.isdigit()])

    return ops, fields_filtered


def cross_sectional_corr_mean(x: torch.Tensor, y: torch.Tensor) -> float:
    """
    矢量化计算两个 [n_time, n_stock] 形状 tensor 的横截面 Pearson 相关性均值，忽略 NaN。

    Returns:
        float: 所有时间点横截面相关性的平均值（忽略 NaN）
    """
    assert x.shape == y.shape, "两个输入 tensor 形状必须一致"

    # 有效位置 mask
    valid_mask = ~(torch.isnan(x) | torch.isnan(y))
    x_masked = torch.where(valid_mask, x, torch.tensor(0.0, device=x.device))
    y_masked = torch.where(valid_mask, y, torch.tensor(0.0, device=y.device))

    # 有效元素个数：每个时间点的有效股票数量
    count = valid_mask.sum(dim=1)

    # 过滤掉有效数目 < 2 的时间点
    valid_time_mask = count >= 2

    # 中心化（mean over stocks），注意只对有效值求平均
    x_mean = (x_masked.sum(dim=1) / count.clamp(min=1)).unsqueeze(1)
    y_mean = (y_masked.sum(dim=1) / count.clamp(min=1)).unsqueeze(1)

    x_centered = torch.where(valid_mask, x - x_mean, torch.tensor(0.0, device=x.device))
    y_centered = torch.where(valid_mask, y - y_mean, torch.tensor(0.0, device=y.device))

    # 协方差与标准差
    cov = (x_centered * y_centered).sum(dim=1)
    x_std = torch.sqrt((x_centered ** 2).sum(dim=1))
    y_std = torch.sqrt((y_centered ** 2).sum(dim=1))
    denom = x_std * y_std

    # 避免除以0
    corr = torch.where(denom == 0, torch.tensor(0.0, device=x.device), cov / denom)

    # 只保留有效时间点的相关系数
    if valid_time_mask.sum() == 0:
        return float("nan")

    return corr[valid_time_mask].mean().item()


def trans_temp_field(exp):
    """
    防止字段冲突
    Args:
        exp: 表达式

    Returns:

    """
    field_name_set = extract_ops_and_fields(exp)[1]
    field_name_dict = {i: f'_{i}' for i in field_name_set}

    for ky in field_name_dict.keys():
        exp = re.sub(ky, field_name_dict[ky], exp)

    return exp


def cal_corr_estimate(exp1, exp2, n_time=__default_n_time, n_stock=__default_n_stock):
    """
    通过蒙特卡洛估计相关性
    Args:
        n_stock: 股票数量
        n_time: 时间步
        exp1: 因子表达式1
        exp2: 因子表达式2

    Returns:

    """
    exp1 = trans_temp_field(exp1)
    exp2 = trans_temp_field(exp2)

    field_name_set = extract_ops_and_fields(exp1)[1] | extract_ops_and_fields(exp2)[1]
    # print(field_name_set)
    # 创建一个局部变量字典来存储模拟数据
    local_vars = {}

    for field_name in field_name_set:
        sim_data = gen_field_mc()  # 假设这个函数返回适当的数据
        local_vars[field_name] = sim_data

    # 初始化因子数据
    local_vars['factor_data1'] = torch.empty(size=(n_time, n_stock), device=mc_device)
    local_vars['factor_data2'] = torch.empty(size=(n_time, n_stock), device=mc_device)

    # 执行表达式
    exec(rf"factor_data1 = {exp1}", globals(), local_vars)
    exec(rf"factor_data2 = {exp2}", globals(), local_vars)

    factor_data1 = local_vars['factor_data1']
    factor_data2 = local_vars['factor_data2']

    # print(factor_data1)
    gc.collect()
    corr_estimated = cross_sectional_corr_mean(factor_data1, factor_data2)

    return corr_estimated if not np.isnan(corr_estimated) else 0  # 假设你想返回这两个值


def cal_corr_estimate_field_related(text_exp1: str, text_exp2: str, correlation_matrix: pd.DataFrame):
    """
    生成虚拟字段，并基于给定的相关系数矩阵生成相关的均匀分布数据。

    Args:
        text_exp1 (str): 第一个文本表达式，用于提取字段和操作
        text_exp2 (str): 第二个文本表达式，用于提取字段和操作
        correlation_matrix (pd.DataFrame): 相关系数矩阵（字段之间的相关性）

    Returns:
        uniform_data (torch.Tensor): 生成的均匀分布数据
        data_map (dict): 字段名称到生成数据的映射字典
    """
    local_vars = dict()
    op1, fields_filtered1 = extract_ops_and_fields(text_exp1)
    op2, fields_filtered2 = extract_ops_and_fields(text_exp2)
    merge_field = list(set(fields_filtered1 | fields_filtered2))
    
    # 替换掉表达式中标准化后的部分
    sub_field_map = {f"cs_normalize({i})": i for i in merge_field}
    for ky in sub_field_map.keys():
        text_exp1 = text_exp1.replace(ky, sub_field_map[ky])
        text_exp2 = text_exp2.replace(ky, sub_field_map[ky])

    valid_fields = [field for field in merge_field if field in list(correlation_matrix.index)]
    invalid_fields = list(set(merge_field) - set(valid_fields))
    
    correlation_matrix_selected = correlation_matrix.loc[valid_fields, valid_fields]
    correlation_matrix_selected = torch.tensor(correlation_matrix_selected.values, dtype=torch.float32, device=mc_device)
    uniform_data = generate_correlated_uniform_data_from_corr(correlation_matrix_selected, __default_n_time, __default_n_stock)

    for idx, name in enumerate(valid_fields):
        # 能够被检查到的字段
        local_vars[name] = uniform_data[:, :, idx]
    for idx, name in enumerate(invalid_fields):
        # 不能被检查到的字段
        local_vars[name] = gen_field_mc(__default_n_time, __default_n_stock)

    local_vars['factor_data1'] = torch.empty(size=(__default_n_time, __default_n_stock), device=mc_device)
    local_vars['factor_data2'] = torch.empty(size=(__default_n_time, __default_n_stock), device=mc_device)
    
    # 执行表达式
    exec(rf"factor_data1 = {text_exp1}", globals(), local_vars)
    exec(rf"factor_data2 = {text_exp2}", globals(), local_vars)

    factor_data1 = local_vars['factor_data1']
    factor_data2 = local_vars['factor_data2']

    # print(factor_data1)
    gc.collect()
    corr_estimated = cross_sectional_corr_mean(factor_data1, factor_data2)
    

    return corr_estimated if not np.isnan(corr_estimated) else 0
