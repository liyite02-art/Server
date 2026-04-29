import logging
from typing import List, Literal
import numpy as np
from sklearn.mixture import GaussianMixture
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.distributions as dist
from torch.optim import Adam
# import numba
# from numba import njit, prange
#from arch import arch_model
import scipy

EPS = 1e-9


def get_nan_tensor(shape, device, dtype):
    return torch.full(shape, torch.nan, dtype=dtype, device=device)
def get_e_tensor(shape, device, dtype,e:float):
    return torch.full(shape, e, dtype=dtype, device=device)

# ------------------------------DEAP框架发生冲突的算子重定义------------------------------ #

def Max(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.maximum(x, y)


def Min(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.minimum(x, y)


def Abs(x: torch.Tensor) -> torch.Tensor:
    return torch.abs(x)

def Pow(x: torch.Tensor, e: float) -> torch.Tensor:
    if e <= 1:
        return torch.pow(x.abs(), e)
    else:
        return torch.pow(x, e)

# ------------------------------逐元素算子------------------------------ #


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.add(x, y)


def sub(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sub(x, y)


def mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.mul(x, y)


def div(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.where(torch.abs(y) > EPS, torch.divide(x, y), torch.nan)


def max(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.maximum(x, y)


def min(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.minimum(x, y)


def abs(x: torch.Tensor) -> torch.Tensor:
    return torch.abs(x)


def neg(x: torch.Tensor) -> torch.Tensor:
    return torch.negative(x)


def sqrt(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.abs(x))


def log(x: torch.Tensor) -> torch.Tensor:
    return torch.where(torch.abs(x) > EPS, torch.log(torch.abs(x)), torch.nan)


def pow(x: torch.Tensor, e: float) -> torch.Tensor:
    if e <= 1:
        return torch.pow(x.abs(), e)
    else:
        return torch.pow(x, e)


def nentexpo(x: torch.Tensor, e: float) -> torch.Tensor:
    """
    将输入张量中的每个元素作为指数，计算底数e的幂
    
    参数:
        x (torch.Tensor): 输入张量，形状为任意维度
        e (float): 指数运算的底数
    
    返回:
        torch.Tensor: 与输入x形状相同的张量，每个元素为e的对应元素次方
    
    示例:
        >>> x = torch.tensor([1.0, 2.0, 3.0])
        >>> exponent(x, 2.0)
        tensor([2., 4., 8.])
    """
    # 将底数转换为与输入张量相同设备和数据类型的标量
    base = torch.tensor(e, dtype=x.dtype, device=x.device)
    # 计算每个元素的指数幂
    return torch.pow(base, x)


def exp(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(x)

def taylor_log_diff(x: torch.Tensor) -> torch.Tensor:
    '''
    适用于1附近的数
    '''
    return (abs(x) - 1 - log(abs(x))) * 2 - pow(log(abs(x)), 2)

def taylor_exp_diff(x: torch.Tensor) -> torch.Tensor:
    '''
    适用于1附近的数
    '''
    return (exp(x) - 1 - x) * 2 - pow(x, 2)

def loop_max_min(y:torch.Tensor) ->torch.Tensor:
    
    zmax=ts_sum(y,240)
    zmin=ts_sum(y,240)
    for i in range(240):
       
        z=ts_sum(y,i+1)
        zmax=question_xmy(zmax,z,zmax,z)
        zmin=question_xmy(zmin,z,z,zmin)
    return zmax,zmin
    

def inv(x: torch.Tensor) -> torch.Tensor:
    return torch.where(torch.abs(x) > EPS, torch.divide(1., x), torch.nan)


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    return F.sigmoid(x)

def erf(x: torch.Tensor) -> torch.Tensor:
    return torch.erf(x)

def sign(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x)


def sign_square(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.pow(torch.abs(x), 2)

def fill_with_quantile(x: torch.Tensor, q: float = 0.8) -> torch.Tensor:
    # 计算输入张量中所有元素的 0.8 分位数，忽略 NaN 值
    quantile_value = torch.nanquantile(x, q)
    
    # 创建一个与输入张量形状相同的张量，并用分位数值填充
    result = torch.full_like(x, quantile_value)
    
    return result

# 返回x是否大于value的bool矩阵
def is_gt(x: torch.Tensor, value: any) -> torch.Tensor:
    return x > value


# 返回x是否小于value的bool矩阵
def is_lt(x: torch.Tensor, value: any) -> torch.Tensor:
    return x < value

def is_zero(x: torch.Tensor) -> torch.Tensor:
    mask=x==0.0
    x[mask]=float('nan')
    return x

def is_nan(x: torch.Tensor) -> torch.Tensor:
    mask=torch.isnan(x[237:])
    x[237:][mask]=0.0
    return x

# 取2个bool矩阵的交集
def bool_and2(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    return x1 & x2


# 取2个bool矩阵的交集
def bool_or2(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    return x1 | x2


# 只保留x中cond为True的部分，其余置为空值
def filt_cond(x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
    return torch.where(cond, x, torch.nan)

def fill_xy_cond(cond: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.where(cond, x, y)

def question_xn0(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    return torch.where(x > 0, y, z)

def question_xisnot0(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    return torch.where(x != 0, y, z)


def question_xmy(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return torch.where(x > y, z, w)

def question_xisy(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return torch.where(x == y, z, w)


def question_xm0(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    return torch.where(x < 0, y, z)

def reverse_cond(x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
    return torch.where(cond, x, -x)

def fillna(x: torch.Tensor, fill_value: any, valid_mask: torch.Tensor = None) -> torch.Tensor:
    res = torch.where(x.isfinite(), x, fill_value)
    if valid_mask is not None:
        res[~valid_mask] = torch.nan
    return res


def ceil_scale(x: torch.Tensor, n: int):
    '''
    将x的每个元素向上取整到1/n的倍数
    '''
    return torch.ceil(x * n) / n


def floor_scale(x: torch.Tensor, n: int):
    '''
    将x的每个元素向下取整到1/n的倍数
    '''
    return torch.floor(x * n) / n


def round_zero_scale(x: torch.Tensor, n: int):
    '''
    将x的每个元素向0的两侧取整到1/n的倍数
    '''
    return torch.sign(x) * torch.floor(torch.abs(x) * n) / n


def studentT_log_prob(x:torch.Tensor, df=5):
    """
    求自由度为df的t分布在输入值处对应的概率密度
    """
    assert df>=1, "t分布的自由度必须大于等于1"
    student_t = dist.StudentT(df, loc=0.0, scale=1.0)
    mask = x.isnan() | x.isinf()
    x_filled = torch.where(mask, 0.0, x)
    prob_filled = student_t.log_prob(x_filled)
    prob = torch.where(mask, torch.nan, prob_filled)
    return prob

def normal_cdf(x:torch.Tensor, loc=0.0, scale=1.0):
    """
    求标准正态分布在输入值处的累计概率分布
    """
    # 构造正态分布
    normal_dist = dist.Normal(loc=0.0, scale=1.0)
    # 处理缺失值（对应的方法中无法自动处理缺失值）
    mask = x.isnan()|x.isinf()
    x_filled = torch.where(mask, 0.0, x)
    cdf_filled = normal_dist.cdf(x_filled)
    cdf = torch.where(mask, torch.nan, cdf_filled)
    return cdf


def normal_log_prob(x:torch.Tensor, loc=0.0, scale=1.0):
    """
    求标准正态分布在输入值处的对数概率密度
    """        
    # 构造正态分布
    normal_dist = dist.Normal(loc=loc, scale=scale)
    # 处理缺失值（对应的方法中无法自动处理缺失值）
    mask = x.isnan()|x.isinf()
    x_filled = torch.where(mask, 0.0, x)
    prob_filled = normal_dist.log_prob(x_filled)
    prob = torch.where(mask, torch.nan, prob_filled)
    return prob

def pow_xy(x: torch.Tensor, e: any) -> torch.Tensor:
    return torch.pow(x, e)


def boxcox(x: torch.Tensor,lamda:float)-> torch.Tensor:
    '''
    对序列进行boxcox变换
    序列:x,原则上要先转换为正数,可以加上一个正值
    参数lamda:为0时为对数变化
    试运行中,无误将注册至表内
    '''
    res = get_nan_tensor(shape=x.shape, device=x.device, dtype=x.dtype)
    if lamda!=0:
        res=(torch.pow(x,lamda)-1)/lamda
    else:
        res=torch.log(x)
    return res

def mul_element_wise(x: torch.Tensor, k) -> torch.Tensor:
    """对张量x进行逐元素乘法, 所乘元素为k
    """
    return k * x


def rs_Hurst(x: torch.Tensor)-> torch.Tensor:
    """
    算子试运行中,若无问题将注册至表内,20250320
    使用R/S重标极差法分析计算给定序列X的Hurst指数。
    试运行1:目前时间复杂度较高,且算子逻辑是对全序列求解Husrt指数,预计单条基础字段回测计算为12min,个股对input的序列仅能计算出一个值代表全频率的自相关性,目前在思考其他具有统计含义的切片滚动算法
    试运行2:改进了一下回归的算法,不需要采用遍历分开回归,但必须把序列补齐,不然回归的beta系数会全返回为nan,好处是表达式回测快了非常多,目前大约1-2min可以跑完
    备注:全序列的Husrt指数在数学上的条件期望非常复杂,对收益率绝对值(波动率)求解Hurst指数时,能获得回测结果较优的label4,,使用时需注意假设条件与含义
    参考文章:https://zhuanlan.zhihu.com/p/38282038
    参数:
        X (torch.Tensor): 输入的时间序列数据
        
    返回:
        Tensor: 计算得到的Hurst指数。
    """
    def mean(X):
        return torch.mean(X,dim=0,keepdim=True)

    def deviation(X, mean):
        return X - mean

    def cumulative_deviation(Y):
        return torch.cumsum(Y, dim=0)

    def rescaled_range(X):
        Z = deviation(X, mean(X))
        C= cumulative_deviation(Z)
        R1 = C.max(dim=0,keepdim=True)[0]
        R2=C.min(dim=0,keepdim=True)[0]
        R=R1-R2
        S = torch.std(X,dim=0,keepdim=True)

        iszero=S==0.0
        S=torch.where(iszero,torch.tensor(1.0),S)
        RS=R/S
        iszero=RS==0.0
        RS=torch.where(iszero,torch.tensor(1.0),RS)
        return R / S

    # 对不同长度子序列进行R/S分析
    
    # isnan=torch.isnan(x)
    # x=torch.where(isnan,torch.tensor(0.0),x)
    # non_zero_rows = ~(x==0).all(dim=1)
    # # x=x[non_zero_rows]
    # res = get_nan_tensor(shape=x.shape, device=x.device, dtype=x.dtype)
    res = get_nan_tensor(shape=(1,x.shape[1]), device=x.device, dtype=x.dtype)
    # 均值填充列
    isnan=torch.isnan(x)
    cols_means=torch.nanmean(x,dim=0)
    cols_means_expaned=cols_means.unsqueeze(0).repeat(x.size(0),1)
    x=torch.where(isnan,cols_means_expaned,x)
    isnan=torch.isnan(x)
    x=torch.where(isnan,torch.tensor(0.0),x)
    data=x
    lengths = []
    rs_values = []
    for n in range(2, len(data)//2 + 1):
        if len(data) % n == 0:
            R_S = 0
            for i in range(n):
                sub_X = data[i*len(data)//n : (i+1)*len(data)//n,:]
                rs = rescaled_range(sub_X)
                if rs is not None:
                    R_S += rs
            if R_S is None and R_S==0:
                continue
            R_S /= n
        else:
            continue
        lengths.append(torch.log(torch.tensor(len(sub_X[0]),dtype=torch.float32)))
        rs_values.append(torch.log(R_S))

    # 线性拟合求斜率，即为Hurst指数
    A = torch.stack([torch.ones_like(torch.stack(lengths)), torch.stack(lengths)], dim=1).to(x.device)
    b = torch.stack(rs_values,dim=1).to(x.device)
    # 直接取回归系数，不单独对每列求回归，但可能某些列为空或inf时整体返回会变成Nan，前面需保证不能有这些值，不用遍历的话时间复杂度小一些
    beta = torch.linalg.lstsq(A,b[0][:,:])
    hurst_index=beta.solution[1]
    res[:,:]=hurst_index
    # for dt in range(int(b.shape[2])):
    #     solution= torch.linalg.lstsq(A,b[0][:,dt]).solution
    #     hurst_index = solution[1].item()
    #     res[:,dt]=hurst_index
    return res

def arctan(x: torch.Tensor) -> torch.Tensor:
    return torch.arctan(x)

def logical_and(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    元素级逻辑与运算, 支持广播机制, 显式转换非布尔输入, 并处理 NaN
    参数:
        x: 输入张量, 非零元素视为True,零视为False
        y: 输入张量, 形状需与x可广播
    返回:
        布尔张量, True表示对应位置x和y同时为真
    """
    x_bool = torch.where(torch.isnan(x), False, x != 0)
    y_bool = torch.where(torch.isnan(y), False, y != 0)
    return torch.logical_and(x_bool, y_bool)


def logical_and_n(*tensors: torch.Tensor) -> torch.Tensor:
    """
    N个张量的逐元素逻辑与运算, 支持广播机制, 显式转换非布尔输入, 并处理 NaN
    参数:
        tensors: 多个输入张量, 需可广播到相同形状
    返回:
        布尔张量, True表示所有输入在对应位置均为真
    """
    assert len(tensors) >= 2, "至少需要两个输入张量"
    result = tensors[0]
    for t in tensors[1:]:
        result = torch.logical_and(result, t)
    return result


def logical_or(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    元素级逻辑或运算, 支持广播机制, 显式转换非布尔输入, 并处理 NaN
    参数:
        x: 输入张量, 非零元素视为True, 零视为False  
        y: 输入张量, 形状需与x可广播
    返回:
        布尔张量, True表示对应位置x或y至少一个为真
    """
    x_bool = torch.where(torch.isnan(x), False, x != 0)
    y_bool = torch.where(torch.isnan(y), False, y != 0)
    return torch.logical_or(x_bool, y_bool)


def logical_not(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    元素级逻辑非运算, 支持广播机制, 显式转换非布尔输入, 并处理 NaN
    参数:
        x: 输入张量, 非零元素视为True, 零视为False
    返回:
        布尔张量, True对应输入为False的位置
    """
    x_bool = torch.where(torch.isnan(x), False, x != 0)
    y_bool = torch.where(torch.isnan(y), False, y != 0)
    return torch.logical_not(x_bool,y_bool)


def logical_xor(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    元素级逻辑异或运算, 支持广播机制, 显式转换非布尔输入, 并处理 NaN
    参数:
        x: 输入张量, 非零元素视为True, 零视为False  
        y: 输入张量, 形状需与x可广播
    返回:
        布尔张量, True表示对应位置x和y真假性不同
    """
    x_bool = torch.where(torch.isnan(x), False, x != 0)
    y_bool = torch.where(torch.isnan(y), False, y != 0)
    return torch.logical_xor(x_bool, y_bool)


def approx_eq(x: torch.Tensor, y: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """
    考虑浮点误差的近似相等判断
    参数:
        x: 输入张量, 任意数值类型
        y: 输入张量, 形状需与x可广播
        eps: 允许的绝对误差阈值, 默认为全局变量EPS, 即1e-9
    返回:
        布尔张量, True表示|x - y| < eps或同为NaN
    """
    # 处理NaN的特殊情况
    both_nan = torch.isnan(x) & torch.isnan(y)
    # 处理无限值的特殊情况
    inf_eq = (torch.isinf(x) & torch.isinf(y)) & (torch.sign(x) == torch.sign(y))
    # 常规数值比较
    value_close = torch.abs(x - y) < eps
    return (value_close | both_nan | inf_eq)


def compare_and_return(
    a: torch.Tensor, 
    b: torch.Tensor,
    c: torch.Tensor, 
    d: torch.Tensor,
    e: torch.Tensor,
    eps: float = EPS
) -> torch.Tensor:
    """
    对于a, b矩阵进行逐元素大小比较, 并在返回的矩阵中填充c, d, e中对应元素的值
    若 a_i_j > b_i_j + EPS, return_i_j = c_i_j
    若 abs(a_i_j - b_i_j) < EPS, return_i_j = d_i_j
    若 a_i_j < b_i_j - EPS, return_i_j = e_i_j
    参数:
        a: 输入张量, 任意数值类型, 需要与b进行逐元素大小比较的矩阵
        b: 输入张量, 形状需与a相同, 需要与a进行逐元素大小比较的矩阵
        c: 输入张量, 形状需与a相同, 当a中待比较元素大于b中对应位置元素时, 将最终返回矩阵对应位置元素设定为该矩阵中对应位置元素
        d: 输入张量, 形状需与a相同, 当a中待比较元素等于b中对应位置元素时, 将最终返回矩阵对应位置元素设定为该矩阵中对应位置元素
        e: 输入张量, 形状需与a相同, 当a中待比较元素小于b中对应位置元素时, 将最终返回矩阵对应位置元素设定为该矩阵中对应位置元素
        eps: 允许的绝对误差阈值, 默认为全局变量EPS, 即1e-9
    返回:
        输出张量, 形状与a相同, 针对a和b中非NaN位置均进行了运算
    """
    # 创建空值掩码（任一输入为空则标记为True）
    nan_mask = torch.isnan(a) | torch.isnan(b)
    
    # 处理无穷大（将inf转换为极值避免溢出）
    a_finite = torch.where(torch.isinf(a), torch.sign(a)*1e30, a)
    b_finite = torch.where(torch.isinf(b), torch.sign(b)*1e30, b)
    
    # 计算差值并处理边界条件
    delta = a_finite - b_finite
    abs_delta = torch.abs(delta)
    
    # 构建条件分支掩码
    cond_gt = delta > eps          # 条件1：a显著大于b
    cond_eq = abs_delta <= eps      # 条件2：a与b近似相等
    cond_lt = delta < -eps         # 条件3：a显著小于b
    
    # 逐元素选择结果（使用torch.where链式选择）
    result = torch.where(cond_gt, c, 
                torch.where(cond_eq, d, e))
    
    # 恢复空值位置
    return torch.where(nan_mask, torch.tensor(torch.nan), result)


import math
from typing import Any

def create_like(input_tensor: torch.Tensor, target: Any, dtype=None, device=None) -> torch.Tensor:
    """
    生成与输入张量形状相同且所有值为目标值的张量
    
    参数:
        input_tensor: 参考张量，决定输出形状和设备(如果未指定)
        target: 目标值，支持以下类型：
                - 标量：int, float, bool
                - 特殊值：NaN, Inf, -Inf
                - 张量：允许标量张量
        dtype: 强制指定输出类型（可选）
        device: 强制指定设备（可选）
    
    返回:
        形状与input_tensor相同，值全为target的张量
    
    示例:
        >>> x = torch.randn(2, 3)
        >>> create_like(x, math.nan)  # 全NaN的浮点张量
        >>> create_like(x, True)      # 全True的布尔张量
        >>> create_like(x, 1e-5)      # 全0.00001的浮点张量
    """
    # 设备处理逻辑
    device = device or input_tensor.device
    
    # 类型推断逻辑
    if dtype is None:
        if isinstance(target, torch.Tensor):
            dtype = target.dtype
        elif isinstance(target, bool):
            dtype = torch.bool
        elif isinstance(target, int):
            dtype = torch.int64
        elif isinstance(target, float) or (isinstance(target, (int, float)) and 
                                          (math.isinf(target) or math.isnan(target))):
            dtype = torch.float32
        else:
            dtype = input_tensor.dtype

    # 特殊值转换逻辑
    if isinstance(target, float) and math.isnan(target):
        value = torch.tensor(float('nan'), dtype=dtype, device=device)
    elif isinstance(target, float) and math.isinf(target):
        value = torch.tensor(float('inf'), dtype=dtype, device=device) if target > 0 else torch.tensor(float('-inf'), dtype=dtype, device=device)
    else:
        try:
            value = torch.tensor(target, dtype=dtype, device=device)
        except Exception as e:
            raise ValueError(f"无法将目标值 {target} 转换为类型 {dtype}") from e

    # 生成最终张量
    return torch.full(
        size=input_tensor.shape,
        fill_value=value.item(),
        dtype=dtype,
        device=device,
        layout=input_tensor.layout  # 保持存储格式一致
    )


def t_cdf(x: torch.Tensor, df) -> torch.Tensor:
    """
    :param x: torch.Tensor 形状为(T,N), T为时间步数, N为股票数量
    :param df: t分布的自由度
    :return: 形状为(T,N), x在该t分布下对应的累积概率
    :rtype: torch.Tensor
    """
    
    cdf = scipy.stats.t.cdf(x.cpu().numpy(), df)
    res = get_nan_tensor(shape=x.shape, device=x.device, dtype=x.dtype)
    res[:] = torch.from_numpy(cdf)
    # res.to(x.device)
    return res

# ------------------------------辅助算子------------------------------ #


# 返回展开的滑动窗口 2D -> 3D
def _unfold(x: torch.Tensor, d: int) -> torch.Tensor:
    res = get_nan_tensor(x.shape + (d,), device=x.device, dtype=x.dtype)
    res[d - 1:] = x.unfold(0, d, 1)
    return res


def corr_row(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    计算两个相同形状的张量 x 和 y 每一行之间的皮尔逊相关系数。
    
    参数:
    - x: 形状为 (m, n) 的张量
    - y: 形状为 (m, n) 的张量
    
    返回:
    - 相关系数矩阵，形状为 (m, n)，其中第 (i, j) 个元素表示 x[i, :] 和 y[i, :] 之间的皮尔逊相关系数
    """
    # 确保输入张量具有相同的形状
    assert x.shape == y.shape, "x和y必须有相同的形状"
    
    m, n = x.shape
    
    # 标准化处理：减去均值除以标准差
    x_normalized = (x - x.mean(dim=1, keepdim=True)) / x.std(dim=1, keepdim=True)
    y_normalized = (y - y.mean(dim=1, keepdim=True)) / y.std(dim=1, keepdim=True)
    
    # 计算每一对行之间的相关系数
    corr_matrix = torch.zeros((m, n), dtype=x.dtype, device=x.device)
    for i in range(m):
        corr_value = torch.dot(x_normalized[i], y_normalized[i]) / (n - 1)
        corr_matrix[i] = corr_value.repeat(n)
    
    return corr_matrix


def corr_col(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:

    """
    计算两个相同形状的张量 x 和 y 每一列之间的皮尔逊相关系数。
    
    参数:
    - x: 形状为 (m, n) 的张量
    - y: 形状为 (m, n) 的张量
    
    返回:
    - 相关系数矩阵，形状为 (m, n)，其中第 (i, j) 个元素表示 x[:, j] 和 y[:, j] 之间的皮尔逊相关系数
    """
    # 确保输入张量具有相同的形状
    assert x.shape == y.shape, "x和y必须有相同的形状"
    
    m, n = x.shape
    
    # 标准化处理：减去均值除以标准差
    x_normalized = (x - x.mean(dim=0, keepdim=True)) / x.std(dim=0, keepdim=True)
    y_normalized = (y - y.mean(dim=0, keepdim=True)) / y.std(dim=0, keepdim=True)
    
    # 计算每一对列之间的相关系数
    corr_matrix = torch.zeros((m, n), dtype=x.dtype, device=x.device)
    for j in range(n):
        corr_value = torch.dot(x_normalized[:, j], y_normalized[:, j]) / (m - 1)
        corr_matrix[:, j] = corr_value.repeat(m)
    
    return corr_matrix

# 最后一个维度上求统计量
def _sum(x: torch.Tensor) -> torch.Tensor:
    return x.nansum(dim=-1)


def _mean(x: torch.Tensor) -> torch.Tensor:
    return x.nanmean(dim=-1)

def _median(x: torch.Tensor) -> torch.Tensor:
    return x.nanmedian(dim=-1)

def _stddev(x: torch.Tensor) -> torch.Tensor:
    x_demean = x - x.nanmean(dim=-1, keepdim=True)
    x_std = torch.sqrt(torch.nanmean(torch.pow(x_demean, 2), dim=-1))
    return x_std

def _keepdimstddev(x: torch.Tensor) -> torch.Tensor:
    x_demean = x - x.nanmean(dim=-1, keepdim=True)
    x_std = torch.sqrt(torch.nanmean(torch.pow(x_demean, 2), dim=-1,keepdim = True))
    return x_std

def _skew(x: torch.Tensor):
    x_demeaned = x - x.nanmean(dim=-1, keepdim=True)
    x_std = torch.sqrt(torch.nanmean(torch.pow(x_demeaned, 2), dim=-1, keepdim=True))
    zscore = torch.pow(div(x_demeaned, x_std), 3)
    return _mean(zscore)


def _kurt(x: torch.Tensor):
    x_demeaned = x - x.nanmean(dim=-1, keepdim=True)
    x_var = torch.nanmean(torch.pow(x_demeaned, 2), dim=-1, keepdim=True)
    x_4 = torch.pow(x_demeaned, 4)
    return div(_mean(x_4), torch.pow(_mean(x_var), 2)) - 3


def _qua(x: torch.Tensor, q: float):
    return div(x.nanquantile(q, dim=-1), x.nanmean(dim=-1))


# 集中度
def _conc(x: torch.Tensor, q: any) -> torch.Tensor:
    if q <= 1:
        return div(_sum(torch.pow(x.abs(), q)), torch.pow(_sum(x.abs()), q))
    else:
        return div(_sum(torch.pow(x, q)), torch.pow(_sum(x), q))


# 求最后一个维度的最小值 跳过空值 支持将非valid_mask的元素置为空值
def _nanmin(x: torch.Tensor, valid_mask: torch.Tensor = None) -> torch.Tensor:
    xc = x.clone()
    if valid_mask is None:
        valid_mask = torch.isfinite(x)
    xc[~valid_mask] = torch.finfo(x.dtype).max
    res = xc.min(dim=-1).values
    res[valid_mask.sum(dim=-1) == 0] = torch.nan
    return res



# 求最后一个维度的最大值 跳过空值 支持将非valid_mask的元素置为空值
def _nanmax(x: torch.Tensor, valid_mask: torch.Tensor = None) -> torch.Tensor:
    xc = x.clone()
    if valid_mask is None:
        valid_mask = torch.isfinite(x)
    xc[~valid_mask] = torch.finfo(x.dtype).min
    res = xc.max(dim=-1).values
    res[valid_mask.sum(dim=-1) == 0] = torch.nan
    return res

# 求最后一个维度的中位数 跳过空值 支持将非valid_mask的元素置为空值
def _nanmed(x: torch.Tensor, valid_mask: torch.Tensor = None) -> torch.Tensor:
    xc = x.clone()
    if valid_mask is None:
        valid_mask = torch.isfinite(x)
    xc[~valid_mask] = torch.nan

    res = torch.nanmedian(xc, dim=-1).values
    # torch的中位数求法,在个数为偶数时,不会求平均,待优化
    # xc_np = xc.cpu().numpy()
    # import numpy as np
    # res_np = np.nanmedian(xc_np, axis=-1)
    # res = torch.tensor(res_np, device=x.device)

    res[valid_mask.sum(dim=-1) == 0] = torch.nan
    return res


# 求最后一个维度的最小值位置 跳过空值 支持将非valid_mask的元素置为空值
def _nanargmin(x: torch.Tensor, valid_mask: torch.Tensor = None) -> torch.Tensor:
    xc = x.clone()
    if valid_mask is None:
        valid_mask = torch.isfinite(x)
    xc[~valid_mask] = torch.finfo(x.dtype).max
    res = xc.argmin(dim=-1).to(dtype=x.dtype)
    res[valid_mask.sum(dim=-1) == 0] = torch.nan
    return res


# 求最后一个维度的最大值位置 跳过空值 支持将非valid_mask的元素置为空值
def _nanargmax(x: torch.Tensor, valid_mask: torch.Tensor = None) -> torch.Tensor:
    xc = x.clone()
    if valid_mask is None:
        valid_mask = torch.isfinite(x)
    xc[~valid_mask] = torch.finfo(x.dtype).min
    res = xc.argmax(dim=-1).to(dtype=x.dtype)
    res[valid_mask.sum(dim=-1) == 0] = torch.nan
    return res


# 对最后一个维度排序 跳过空值 排名从1开始 支持将非valid_mask的元素置为空值 pct返回比例
def _nanrank(x: torch.Tensor, valid_mask: torch.Tensor = None, pct: bool = True) -> torch.Tensor:
    ranks = x.clone()
    if valid_mask is None:
        valid_mask = torch.isfinite(x)
    ranks[~valid_mask] = torch.finfo(x.dtype).max
    ranks = torch.argsort(torch.argsort(ranks, dim=-1), dim=-1)
    ranks = ranks.to(dtype=x.dtype) + 1
    if pct:
        valid_counts = valid_mask.sum(dim=-1, keepdim=True)
        ranks = ranks / valid_counts
    ranks[~valid_mask] = torch.nan
    return ranks

# 对最后一个维度排序 跳过空值 排名从1开始 支持将非valid_mask的元素置为空值 pct返回比例
def _my_nanrank(x: torch.Tensor, valid_mask: torch.Tensor = None, pct: bool = True) -> torch.Tensor:
    ranks = x.clone()
    if valid_mask is None:
        valid_mask = torch.isfinite(x)
    ranks[~valid_mask] = torch.finfo(x.dtype).max
    ranks = torch.argsort(torch.argsort(ranks, dim=-1), dim=-1)
    ranks = ranks.to(dtype=x.dtype) + 1
    
    ranks[~valid_mask] = torch.nan
    return ranks
def _weighted_skew(x: torch.Tensor, weights: torch.Tensor):
    
    mask = weights.isnan()|weights.isinf()|x.isnan()|x.isinf()
    x[mask] = torch.nan
    weights[mask] = torch.nan
    
    weights = div(weights, torch.nansum(weights, dim=-1, keepdim=True))
    
    x_demeaned = x - x.nanmean(dim=-1, keepdim=True)
    x_std = torch.sqrt(torch.nanmean(torch.pow(x_demeaned, 2), dim=-1, keepdim=True))  
    numerator = weights * torch.pow(x_demeaned, 3)
    denominator = torch.pow(x_std, 3)
    zscore = div(numerator, denominator)
    return _mean(zscore)


def _nancoss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x_std = torch.sqrt(torch.nansum(torch.pow(x, 2), dim=-1))
    y_std = torch.sqrt(torch.nansum(torch.pow(y, 2), dim=-1))

    return div((x * y).nansum(dim=-1),(x_std * y_std))


def _get_multi_x(x: torch.Tensor, param_lis: list, func, *args) -> torch.Tensor:
    """对每支股票,在param_lis中的不同参数下调用时间序列函数func计算多个结果
    
    :param torch.Tensor x: 形状为 (T, N)，其中 T 是时间步数,N 是股票数
    :param list param_lis: 待遍历参数的所有值
    :param func: 时序算子的函数,第一个参数为x,第二个参数为param_lis的元素对应的参数,返回(T, N)张量
    :params *args: func需要的其他参数,在前两个参数之后传入func
    :return: 不同参数的计算结果，形状为(T, N, K), K为param_lis的长度
    :rtype: torch.Tensor
    """
    temp_lis = []
    for k in param_lis:
        temp = func(x, k, *args)
        temp_lis.append(temp)
    res = torch.stack(temp_lis, dim=-1)
    return res

def sort_by_another(x: torch.Tensor, by: torch.Tensor, d: int, descending=False) -> torch.Tensor:
    """滚动d长度的窗口，基于窗口内by的升序or降序序列，对窗口内的x进行排序，输出x对应排序后的结果

    :param torch.Tensor x: 形状为 (T, N)，其中 T 是时间步数,N 是股票数
    :param torch.Tensor by: 形状为 (T, N)，其中 T 是时间步数,N 是股票数
    :param int d: 窗口长度
    :param bool descending: 是否按by的窗口降序去排x
    :return: x依据by排序后的窗口结果，形状为(T-d+1, N, d)
    :rtype: torch.Tensor
    """
    assert x.shape == by.shape, "The shapes of x and by must be equal"
    assert d <= x.shape[0], 'd should not be bigger than x.shape[0]'
    # x_by = torch.stack([x,by], dim=-1) # (T,N,2)
    # x_by_unfold = x_by.unfold(0, d, 1) # (T-d+1,N,2,d)
    x_unfold = x.unfold(0, d, 1) # (T-d+1,N,d)
    by_unfold = by.unfold(0, d, 1) # (T-d+1,N,d)
    by_unfold = torch.where(torch.isnan(by_unfold) | torch.isinf(by_unfold), torch.nanmean(by_unfold, dim=-1, keepdim=True), by_unfold)
    x_unfold = torch.where(torch.isnan(x_unfold) | torch.isinf(x_unfold), torch.nanmean(x_unfold, dim=-1, keepdim=True), x_unfold)
    sorted_index = torch.argsort(by_unfold, dim=-1, descending=descending)
    sorted_x_unfold = torch.gather(x_unfold, dim=-1, index=sorted_index)
    return sorted_x_unfold # (T-d+1,N,d)

def fill_with_mean(x: torch.Tensor) -> torch.Tensor:
    mask = x.isnan() | x.isinf()
    return torch.where(mask, x.nanmean(dim = -1, keepdim = True), x)

def indicator(x: torch.Tensor) -> torch.Tensor:
    return (x > 0).to(torch.float32)
# ------------------------------时序算子------------------------------ #


def ts_conc(x: torch.Tensor, d: int, q: any) -> torch.Tensor:
    x_unfold = _unfold(x, d)
    return _conc(x_unfold, q)


def ts_mean_selq(x: torch.Tensor, cond_var: torch.Tensor, d: int, low_q: float, high_q: float) -> torch.Tensor:
    x_unfold = _unfold(x, d)
    cond_unfold = _unfold(cond_var, d)
    low_value = torch.nanquantile(cond_unfold, low_q, dim=-1, keepdim=True)
    high_value = torch.nanquantile(cond_unfold, high_q, dim=-1, keepdim=True)
    cond = (cond_unfold >= low_value) & (cond_unfold <= high_value)
    x_unfold[~cond] = torch.nan
    return _mean(x_unfold)

def ts_mean_diff_selq(x: torch.Tensor, cond_var: torch.Tensor, d: int, low_q: float, high_q: float) -> torch.Tensor:
    x_unfold = _unfold(x, d)
    cond_unfold = _unfold(cond_var, d)
    low_value = torch.nanquantile(cond_unfold, low_q, dim=-1, keepdim=True)
    high_value = torch.nanquantile(cond_unfold, high_q, dim=-1, keepdim=True)
    
    cond_1 = cond_unfold < low_value
    cond_2 = cond_unfold > high_value
    return _mean(filt_cond(x_unfold, cond_1)) - _mean(filt_cond(x_unfold, cond_2))

def ts_stddev_selq(x: torch.Tensor, cond_var: torch.Tensor, d: int, low_q: float, high_q: float) -> torch.Tensor:
    x_unfold = _unfold(x, d)
    cond_unfold = _unfold(cond_var, d)
    low_value = torch.nanquantile(cond_unfold, low_q, dim=-1, keepdim=True)
    high_value = torch.nanquantile(cond_unfold, high_q, dim=-1, keepdim=True)
    cond = (cond_unfold >= low_value) & (cond_unfold <= high_value)
    x_unfold[~cond] = torch.nan
    return _stddev(x_unfold)

def ts_stddev_diff_selq(x: torch.Tensor, cond_var: torch.Tensor, d: int, low_q: float, high_q: float) -> torch.Tensor:
    x_unfold = _unfold(x, d)
    cond_unfold = _unfold(cond_var, d)
    low_value = torch.nanquantile(cond_unfold, low_q, dim=-1, keepdim=True)
    high_value = torch.nanquantile(cond_unfold, high_q, dim=-1, keepdim=True)
    
    cond_1 = cond_unfold < low_value
    cond_2 = cond_unfold > high_value
    return div(_stddev(filt_cond(x_unfold, cond_1)), _stddev(filt_cond(x_unfold, cond_2)))

def ts_skew_selq(x: torch.Tensor, cond_var: torch.Tensor, d: int, low_q: float, high_q: float) -> torch.Tensor:
    x_unfold = _unfold(x, d)
    cond_unfold = _unfold(cond_var, d)
    low_value = torch.nanquantile(cond_unfold, low_q, dim=-1, keepdim=True)
    high_value = torch.nanquantile(cond_unfold, high_q, dim=-1, keepdim=True)
    cond = (cond_unfold >= low_value) & (cond_unfold <= high_value)
    x_unfold[~cond] = torch.nan
    return _skew(x_unfold)

def ts_skew_diff_selq(x: torch.Tensor, cond_var: torch.Tensor, d: int, low_q: float, high_q: float) -> torch.Tensor:
    x_unfold = _unfold(x, d)
    cond_unfold = _unfold(cond_var, d)
    low_value = torch.nanquantile(cond_unfold, low_q, dim=-1, keepdim=True)
    high_value = torch.nanquantile(cond_unfold, high_q, dim=-1, keepdim=True)
    
    cond_1 = cond_unfold < low_value
    cond_2 = cond_unfold > high_value
    return _skew(filt_cond(x_unfold, cond_1)) - _skew(filt_cond(x_unfold, cond_2))

def ts_kurt_selq(x: torch.Tensor, cond_var: torch.Tensor, d: int, low_q: float, high_q: float) -> torch.Tensor:
    x_unfold = _unfold(x, d)
    cond_unfold = _unfold(cond_var, d)
    low_value = torch.nanquantile(cond_unfold, low_q, dim=-1, keepdim=True)
    high_value = torch.nanquantile(cond_unfold, high_q, dim=-1, keepdim=True)
    cond = (cond_unfold >= low_value) & (cond_unfold <= high_value)
    x_unfold[~cond] = torch.nan
    return _kurt(x_unfold)

def ts_kurt_diff_selq(x: torch.Tensor, cond_var: torch.Tensor, d: int, low_q: float, high_q: float) -> torch.Tensor:
    x_unfold = _unfold(x, d)
    cond_unfold = _unfold(cond_var, d)
    low_value = torch.nanquantile(cond_unfold, low_q, dim=-1, keepdim=True)
    high_value = torch.nanquantile(cond_unfold, high_q, dim=-1, keepdim=True)
    
    cond_1 = cond_unfold < low_value
    cond_2 = cond_unfold > high_value
    return _kurt(filt_cond(x_unfold, cond_1)) - _kurt(filt_cond(x_unfold, cond_2))

def ts_qua_selq(x: torch.Tensor, cond_var: torch.Tensor, d: int, q: float, low_q: float, high_q: float) -> torch.Tensor:
    x_unfold = _unfold(x, d)
    cond_unfold = _unfold(cond_var, d)
    low_value = torch.nanquantile(cond_unfold, low_q, dim=-1, keepdim=True)
    high_value = torch.nanquantile(cond_unfold, high_q, dim=-1, keepdim=True)
    cond = (cond_unfold >= low_value) & (cond_unfold <= high_value)
    x_unfold[~cond] = torch.nan
    return _qua(x_unfold, q)

def ts_sum_selq(x: torch.Tensor, cond_var: torch.Tensor, d: int, low_q: float, high_q: float) -> torch.Tensor:
    x_unfold = _unfold(x, d)
    cond_unfold = _unfold(cond_var, d)
    low_value = torch.nanquantile(cond_unfold, low_q, dim=-1, keepdim=True)
    high_value = torch.nanquantile(cond_unfold, high_q, dim=-1, keepdim=True)
    cond = (cond_unfold >= low_value) & (cond_unfold <= high_value)
    x_unfold[~cond] = torch.nan
    return _sum(x_unfold)

def ts_prodsum_selq(x: torch.Tensor, cond_var: torch.Tensor, d: int, low_q: float, high_q: float) -> torch.Tensor:
    x_unfold = _unfold(x, d)
    cond_unfold = _unfold(cond_var, d)
    low_value = torch.nanquantile(cond_unfold, low_q, dim=-1, keepdim=True)
    high_value = torch.nanquantile(cond_unfold, high_q, dim=-1, keepdim=True)
    cond = (cond_unfold >= low_value) & (cond_unfold <= high_value)
    
    res = (filt_cond(x_unfold, cond) * filt_cond(cond_unfold, cond)).nansum(dim=-1)
    res[:d-1] = torch.nan
    return res

def ts_corr_selq(x: torch.Tensor, cond_var: torch.Tensor, d: int, low_q: float, high_q: float) -> torch.Tensor:
    x_unfold = _unfold(x, d)
    cond_unfold = _unfold(cond_var, d)
    low_value = torch.nanquantile(cond_unfold, low_q, dim=-1, keepdim=True)
    high_value = torch.nanquantile(cond_unfold, high_q, dim=-1, keepdim=True)
    cond = (cond_unfold >= low_value) & (cond_unfold <= high_value)
    
    x_unfold = filt_cond(x_unfold, cond)
    cond_unfold = filt_cond(cond_unfold, cond)   
    x_demean = x_unfold - x_unfold.nanmean(dim=-1, keepdim=True)
    cond_demean = cond_unfold - cond_unfold.nanmean(dim=-1, keepdim=True)
    
    return _nancoss(x_demean, cond_demean)

def ts_mean_flip(x: torch.Tensor, cond_var: torch.Tensor, d: int, low_q: float, high_q: float) -> torch.Tensor:
    x_unfold = _unfold(x, d)
    x_demean = x_unfold - x_unfold.nanmean(dim=-1, keepdim=True)
    
    cond_unfold = _unfold(cond_var, d)
    low_value = torch.nanquantile(cond_unfold, low_q, dim=-1, keepdim=True)
    high_value = torch.nanquantile(cond_unfold, high_q, dim=-1, keepdim=True)
    cond = (cond_unfold >= low_value) & (cond_unfold <= high_value)
    
    return _mean(reverse_cond(x_demean, cond))


def ts_stddev_flip(x: torch.Tensor, cond_var: torch.Tensor, d: int, low_q: float, high_q: float) -> torch.Tensor:
    x_unfold = _unfold(x, d)
    x_demean = x_unfold - x_unfold.nanmean(dim=-1, keepdim=True)
    
    cond_unfold = _unfold(cond_var, d)
    low_value = torch.nanquantile(cond_unfold, low_q, dim=-1, keepdim=True)
    high_value = torch.nanquantile(cond_unfold, high_q, dim=-1, keepdim=True)
    cond = (cond_unfold >= low_value) & (cond_unfold <= high_value)
    
    return _stddev(x_demean) - _stddev(reverse_cond(x_demean, cond))


def ts_skew_flip(x: torch.Tensor, cond_var: torch.Tensor, d: int, low_q: float, high_q: float) -> torch.Tensor:
    x_unfold = _unfold(x, d)
    x_demean = x_unfold - x_unfold.nanmean(dim=-1, keepdim=True)
    
    cond_unfold = _unfold(cond_var, d)
    low_value = torch.nanquantile(cond_unfold, low_q, dim=-1, keepdim=True)
    high_value = torch.nanquantile(cond_unfold, high_q, dim=-1, keepdim=True)
    cond = (cond_unfold >= low_value) & (cond_unfold <= high_value)
    
    return _skew(x_demean) - _skew(reverse_cond(x_demean, cond))


def ts_kurt_flip(x: torch.Tensor, cond_var: torch.Tensor, d: int, low_q: float, high_q: float) -> torch.Tensor:
    x_unfold = _unfold(x, d)
    x_demean = x_unfold - x_unfold.nanmean(dim=-1, keepdim=True)
    
    cond_unfold = _unfold(cond_var, d)
    low_value = torch.nanquantile(cond_unfold, low_q, dim=-1, keepdim=True)
    high_value = torch.nanquantile(cond_unfold, high_q, dim=-1, keepdim=True)
    cond = (cond_unfold >= low_value) & (cond_unfold <= high_value)
    
    return _kurt(x_demean) - _kurt(reverse_cond(x_demean, cond))


def ts_qua_flip(x: torch.Tensor, cond_var: torch.Tensor, d: int, q: float, low_q: float, high_q: float) -> torch.Tensor:
    x_unfold = _unfold(x, d)
    x_demean = x_unfold - x_unfold.nanmean(dim=-1, keepdim=True)
    
    cond_unfold = _unfold(cond_var, d)
    low_value = torch.nanquantile(cond_unfold, low_q, dim=-1, keepdim=True)
    high_value = torch.nanquantile(cond_unfold, high_q, dim=-1, keepdim=True)
    cond = (cond_unfold >= low_value) & (cond_unfold <= high_value)
    
    return _qua(x_demean, q) - _qua(reverse_cond(x_demean, cond), q)


def ts_corr_flip(x: torch.Tensor, cond_var: torch.Tensor, d: int, low_q: float, high_q: float) -> torch.Tensor:
    x_unfold = _unfold(x, d)
    cond_unfold = _unfold(cond_var, d)
    
    x_demean = x_unfold - x_unfold.nanmean(dim=-1, keepdim=True)
    cond_demean = cond_unfold - cond_unfold.nanmean(dim=-1, keepdim=True)
    
    low_value = torch.nanquantile(cond_unfold, low_q, dim=-1, keepdim=True)
    high_value = torch.nanquantile(cond_unfold, high_q, dim=-1, keepdim=True)
    cond = (cond_unfold >= low_value) & (cond_unfold <= high_value)
    
    return _nancoss(reverse_cond(x_demean, cond), cond_demean)

def ts_counter(x: torch.Tensor, d: int) -> torch.Tensor:
    res = get_nan_tensor(shape = x.shape, device=x.device, dtype=x.dtype)
    unfolded = x.unfold(dimension=0, size=d, step=1)
    non_nan_counts = (~torch.isnan(unfolded)).sum(dim=-1)               
    res[d - 1:] = non_nan_counts                                         
    return res

def ts_delay(x: torch.Tensor, d: int) -> torch.Tensor:
    res = get_nan_tensor(shape=x.shape, device=x.device, dtype=x.dtype)
    res[d:] = x[:-d]
    return res

def ts_shift(x: torch.Tensor, d: int) -> torch.Tensor:
    res = get_nan_tensor(shape=x.shape, device=x.device, dtype=x.dtype)
    res[:-d] = x[d:]
    return res

def ts_pct(x: torch.Tensor) -> torch.Tensor:
    """
    计算时间序列的百分比变化：(下一时间步的值 - 当前值) / 当前值
    输入形状: [T, N] (T为时间步数，N为特征数)
    输出形状: [T, N]，最后一个时间步用 NaN 填充
    """
    # 初始化全 NaN 张量（与输入保持相同形状和设备）
    res = torch.full_like(x, torch.nan)
    
    # 当时间步数 >= 2 时，计算前 T-1 步的百分比变化
    if x.size(0) >= 2:
        current = x[:-1]  # 当前值：[T-1, N]
        next_val = x[1:]  # 下一时间步的值：[T-1, N]
        pct_change = (next_val - current) / current
        res[:-1] = pct_change  # 最后一位保持 NaN
    
    return res

def ts_delta(x: torch.Tensor, d: int) -> torch.Tensor:
    res = get_nan_tensor(shape=x.shape, device=x.device, dtype=x.dtype)
    res[d:] = x[d:] - x[: -d]
    return res


def ts_corr(x: torch.Tensor, y: torch.Tensor, d: int) -> torch.Tensor:
    res = get_nan_tensor(shape=x.shape, device=x.device, dtype=x.dtype)
    x_unfold = x.unfold(0, d, 1)
    y_unfold = y.unfold(0, d, 1)

    x_demean = x_unfold - x_unfold.nanmean(dim=-1, keepdim=True)
    y_demean = y_unfold - y_unfold.nanmean(dim=-1, keepdim=True)

    x_std = torch.sqrt(torch.nansum(torch.pow(x_demean, 2), dim=-1))
    y_std = torch.sqrt(torch.nansum(torch.pow(y_demean, 2), dim=-1))

    numerator = (x_demean * y_demean).nansum(dim=-1)
    denominator = x_std * y_std
    res[d - 1:] = numerator / denominator

    res[d - 1:][(x_std < EPS) | (y_std < EPS)] = torch.nan
    return res

def ts_cos_similar(x: torch.Tensor, y: torch.Tensor, d: int) -> torch.Tensor:
    res = get_nan_tensor(shape=x.shape, device=x.device, dtype=x.dtype)
    x_unfold = x.unfold(0, d, 1)
    y_unfold = y.unfold(0, d, 1)

    x_demean = x_unfold 
    
    y_demean = y_unfold

    x_std = torch.sqrt(torch.nansum(torch.pow(x_demean, 2), dim=-1))
    y_std = torch.sqrt(torch.nansum(torch.pow(y_demean, 2), dim=-1))

    numerator = (x_demean * y_demean).nansum(dim=-1)
    denominator = x_std * y_std
    res[d - 1:] = numerator / denominator

    res[d - 1:][(x_std < EPS) | (y_std < EPS)] = torch.nan
    return res

def ts_rankcorr(x: torch.Tensor, y: torch.Tensor, d: int) -> torch.Tensor:
    res = get_nan_tensor(shape=x.shape, device=x.device, dtype=x.dtype)
    x_unfold = _nanrank(x.unfold(0, d, 1), pct=False)
    y_unfold = _nanrank(y.unfold(0, d, 1), pct=False)

    x_demean = x_unfold - x_unfold.nanmean(dim=-1, keepdim=True)
    y_demean = y_unfold - y_unfold.nanmean(dim=-1, keepdim=True)

    x_std = torch.sqrt(torch.nansum(torch.pow(x_demean, 2), dim=-1))
    y_std = torch.sqrt(torch.nansum(torch.pow(y_demean, 2), dim=-1))

    numerator = (x_demean * y_demean).nansum(dim=-1)
    denominator = x_std * y_std
    res[d - 1:] = numerator / denominator

    res[d - 1:][(x_std < EPS) | (y_std < EPS)] = torch.nan
    return res


def ts_cov(x: torch.Tensor, y: torch.Tensor, d: int) -> torch.Tensor:
    res = get_nan_tensor(shape=x.shape, device=x.device, dtype=x.dtype)
    x_windows = x.unfold(0, d, 1)
    y_windows = y.unfold(0, d, 1)

    x_mean = x_windows.nanmean(dim=-1, keepdim=True)
    y_mean = y_windows.nanmean(dim=-1, keepdim=True)
    x_demeaned = x_windows - x_mean
    y_demeaned = y_windows - y_mean

    res[d - 1:] = (x_demeaned * y_demeaned).nansum(dim=-1) / d
    return res


def ts_autocorr(x: torch.Tensor, d: int, shift: int) -> torch.Tensor:
    res = get_nan_tensor(shape=x.shape, device=x.device, dtype=x.dtype)
    res[shift:] = ts_corr(x[shift:], x[:-shift], d)
    return res

def ts_autocos(x: torch.Tensor, d: int, shift: int) -> torch.Tensor:
    res = get_nan_tensor(shape=x.shape, device=x.device, dtype=x.dtype)
    res[shift:] = ts_cos_similar(x[shift:], x[:-shift], d)
    return res


def ts_decay_linear(x: torch.Tensor, d: int) -> torch.Tensor:
    w = torch.arange(1, d + 1, dtype=torch.float32, device=x.device)
    w = w / w.sum()
    x_unfolded = x.unfold(dimension=0, size=d, step=1)
    w = w.view(1, 1, -1)
    res = torch.nansum(x_unfolded * w, dim=-1)
    result = get_nan_tensor(shape=x.shape, device=x.device, dtype=x.dtype)
    result[d - 1:] = res
    return result


def ts_rank(x: torch.Tensor, d: int, pct: bool = True) -> torch.Tensor:
    res = get_nan_tensor(shape=x.shape, device=x.device, dtype=x.dtype)
    res[d - 1:] = _nanrank(x.unfold(0, d, 1), pct=pct)[..., -1]
    return res

def ts_min(x: torch.Tensor, d: int) -> torch.Tensor:
    res = get_nan_tensor(shape=x.shape, device=x.device, dtype=x.dtype)
    res[d - 1:] = _nanmin(x.unfold(0, d, 1))
    return res

def ts_max(x: torch.Tensor, d: int) -> torch.Tensor:
    res = get_nan_tensor(shape=x.shape, device=x.device, dtype=x.dtype)
    res[d - 1:] = _nanmax(x.unfold(0, d, 1))
    return res

def ts_med(x: torch.Tensor, d: int) -> torch.Tensor:
    res = get_nan_tensor(shape=x.shape, device=x.device, dtype=x.dtype)
    res[d - 1:] = _nanmed(x.unfold(0, d, 1))
    return res

def ts_argmin(x: torch.Tensor, d: int, pct: bool = True) -> torch.Tensor:
    res = get_nan_tensor(shape=x.shape, device=x.device, dtype=x.dtype)
    res[d - 1:] = d - 1 - _nanargmin(x.unfold(0, d, 1))
    if pct:
        res = (res + 1) / d
    return res


def ts_argmax(x: torch.Tensor, d: int, pct: bool = True) -> torch.Tensor:
    res = get_nan_tensor(shape=x.shape, device=x.device, dtype=x.dtype)
    res[d - 1:] = d - 1 - _nanargmax(x.unfold(0, d, 1))
    if pct:
        res = (res + 1) / d
    return res


def ts_sum(x: torch.Tensor, d: int) -> torch.Tensor:
    res = get_nan_tensor(shape=x.shape, device=x.device, dtype=x.dtype)
    unfolded = x.unfold(dimension=0, size=d, step=1)
    res[d - 1:] = unfolded.nansum(dim=-1, keepdim=False)
    return res


def ts_mean(x: torch.Tensor, d: int) -> torch.Tensor:
    res = get_nan_tensor(shape=x.shape, device=x.device, dtype=x.dtype)
    unfolded = x.unfold(dimension=0, size=d, step=1)
    res[d - 1:] = unfolded.nanmean(dim=-1)
    return res


def ts_gmean(x: torch.Tensor, d: int) -> torch.Tensor:
    '''
    几何平均值
    '''
    res = torch.full((x.shape[0], x.shape[1]), float('nan'), 
                      dtype=torch.float32, device=x.device)

    x_unfold = x.unfold(0, d, 1)
    
    # 分离正负值 (保持维度)
    pos_mask = x_unfold > 0
    neg_mask = x_unfold < 0
    
    # 计算正数部分几何平均
    pos_vals = torch.where(pos_mask, x_unfold, torch.nan)
    log_pos = torch.log(pos_vals)
    g_pos = torch.exp(torch.nanmean(log_pos, dim=-1))
    
    # 计算负数部分几何平均（取绝对值后恢复符号）
    neg_vals = torch.where(neg_mask, x_unfold.abs(), torch.nan)
    log_neg = torch.log(neg_vals)
    g_neg = -torch.exp(torch.nanmean(log_neg, dim=-1))
    
    # 合并结果（加权平均）
    valid_counts = (~torch.isnan(g_pos)).float() + (~torch.isnan(g_neg)).float()
    gmean = torch.where(valid_counts > 0, 
                       (torch.nan_to_num(g_pos, 0) + torch.nan_to_num(g_neg, 0)) / valid_counts,
                       torch.nan)
    
    # 拼接最终结果
    res[d-1:] = gmean

    return res


def ts_hmean(x: torch.Tensor, d: int):
    '''
    调和平均值
    '''
    res = torch.full((x.shape[0], x.shape[1]), float('nan'), 
                      dtype=torch.float32, device=x.device)

    x_unfold = x.unfold(0, d, 1)  # (T-d+1, C, d)
    
    # 分离正负值 (保持维度)
    pos_mask = x_unfold > 0
    neg_mask = x_unfold < 0
    
    # 计算正数部分调和平均
    pos_vals = torch.where(pos_mask, x_unfold, torch.nan)
    inv_pos = 1 / pos_vals
    h_pos = torch.nansum(pos_mask.float(), dim=-1) / torch.nansum(inv_pos, dim=-1)
    
    # 计算负数部分调和平均（取绝对值后恢复符号）
    neg_vals = torch.where(neg_mask, x_unfold.abs(), torch.nan)
    inv_neg = 1 / neg_vals
    h_neg = -torch.nansum(neg_mask.float(), dim=-1) / torch.nansum(inv_neg, dim=-1)
    
    # 合并结果（加权平均）
    valid_counts = (~torch.isnan(h_pos)).float() + (~torch.isnan(h_neg)).float()
    hmean = torch.where(valid_counts > 0, 
                       (torch.nan_to_num(h_pos, 0) + torch.nan_to_num(h_neg, 0)) / valid_counts,
                       torch.nan)

    res[d-1: ] = hmean

    return res

def ts_gmean_amean_diff(x: torch.Tensor, d: int) -> torch.Tensor:
    res = ts_gmean(x, d) - ts_mean(x, d)
    return res

def ts_hmean_amean_diff(x: torch.Tensor, d: int) -> torch.Tensor:
    res = ts_hmean(x, d) - ts_mean(x, d)
    return res

def ts_gmean_hmean_diff(x: torch.Tensor, d: int) -> torch.Tensor:
    res = ts_gmean(x, d) - ts_hmean(x, d)
    return res

def ts_quantile(x: torch.Tensor, q: float, d: int) -> torch.Tensor:
    res = get_nan_tensor(shape=x.shape, device=x.device, dtype=x.dtype)
    res[d - 1:] = torch.nanquantile(x.unfold(0, d, 1), q, dim=-1)
    return res


def ts_qua(x: torch.Tensor, q: float, d: int) -> torch.Tensor:
    return ts_quantile(x, q, d) / ts_mean(x, d)


def ts_return(x: torch.Tensor, d: int) -> torch.Tensor:
    numerator = ts_delta(x, d)
    denominator = ts_delay(x, d)
    res = div(numerator, denominator)
    return res


def ts_mean_return(x: torch.Tensor, d: int) -> torch.Tensor:
    return ts_mean(ts_return(x, 1), d)


def ts_product(x: torch.Tensor, d: int) -> torch.Tensor:
    res = get_nan_tensor(shape=x.shape, device=x.device, dtype=x.dtype)
    unfolded = x.unfold(0, d, 1)
    res[d - 1:] = fillna(unfolded, 1).prod(dim=-1)
    return res


def ts_stddev(x: torch.Tensor, d: int) -> torch.Tensor:
    res = get_nan_tensor(shape=x.shape, device=x.device, dtype=x.dtype)
    x_unfold = x.unfold(0, d, 1)
    x_demean = x_unfold - x_unfold.nanmean(dim=-1, keepdim=True)
    x_std = torch.sqrt(torch.nanmean(torch.pow(x_demean, 2), dim=-1))
    res[d - 1:] = x_std
    return res

def ts_var(x: torch.Tensor, d: int) -> torch.Tensor:
    res = get_nan_tensor(shape=x.shape, device=x.device, dtype=x.dtype)
    x_unfold = x.unfold(0, d, 1)
    x_demean = x_unfold - x_unfold.nanmean(dim=-1, keepdim=True)
    x_var = torch.nanmean(torch.pow(x_demean, 2), dim=-1)
    res[d - 1:] = x_var
    return res


def ts_demean(x: torch.Tensor, d: int):
    return x - ts_mean(x, d)

def ts_mad_mean(x: torch.Tensor, d: int) -> torch.Tensor:
    central = ts_demean(x, d)
    return ts_mean(abs(central), d)

def ts_mad_med(x: torch.Tensor, d: int) -> torch.Tensor:
    central = x - ts_med(x, d)
    return ts_med(abs(central), d)


def ts_zscore(x: torch.Tensor, d: int):
    x_unfold = _unfold(x, d)
    x_demeaned = x_unfold - x_unfold.nanmean(dim=-1, keepdim=True)
    x_std = torch.sqrt(torch.nanmean(torch.pow(x_demeaned, 2), dim=-1, keepdim=True))
    zscore = div(x_demeaned, x_std)
    return zscore[..., -1]


def ts_skew(x: torch.Tensor, d: int):
    x_unfold = _unfold(x, d)
    return _skew(x_unfold)


def ts_kurt(x: torch.Tensor, d: int):
    x_unfold = _unfold(x, d)
    return _kurt(x_unfold)


def ts_regbeta(x: torch.Tensor, y: torch.Tensor, d: int) -> torch.Tensor:
    res = get_nan_tensor(shape=x.shape, device=x.device, dtype=x.dtype)
    x_unfold = x.unfold(0, d, 1)
    y_unfold = y.unfold(0, d, 1)
    x_mean = x_unfold.nanmean(dim=-1, keepdim=True)
    y_mean = y_unfold.nanmean(dim=-1, keepdim=True)
    x_demean = x_unfold - x_mean
    y_demean = y_unfold - y_mean
    std = torch.nansum(torch.pow(x_demean, 2), dim=-1)
    res[d - 1:] = torch.where(std < EPS, torch.nan, torch.nansum(x_demean * y_demean, dim=-1) / std)
    return res


def ts_regres(x: torch.Tensor, y: torch.Tensor, d: int) -> torch.Tensor:
    beta = ts_regbeta(x, y, d)
    return y - beta * x


def ts_cross_entropy(x: torch.Tensor, y: torch.Tensor, d: int, buckets: int = 10):
    assert x.shape == y.shape, "Input tensors must have the same shape."
    assert d > 0 and d < x.shape[0], "Window size must be positive."
    assert d >= buckets, "Number of buckets must be less than or equal to window size."
    res = torch.full(x.shape, torch.nan, device=x.device, dtype=x.dtype)

    x_unfold = x.unfold(0, d, 1)
    y_unfold = y.unfold(0, d, 1)

    # 均分区间
    mask_x = torch.isnan(x_unfold)  # 创建掩码
    mask_y = torch.isnan(y_unfold)  # 创建掩码
    x_max = torch.max(x_unfold.masked_fill(mask_x, float('-inf')), dim=-1)[0]  # 忽略NaN，填充为负无穷
    x_min = torch.min(x_unfold.masked_fill(mask_x, float('inf')), dim=-1)[0]  # 会返回一个元组，第一个元素是值，第二个元素是索引
    y_max = torch.max(y_unfold.masked_fill(mask_y, float('-inf')), dim=-1)[0]  # 忽略NaN，填充为负无穷
    y_min = torch.min(y_unfold.masked_fill(mask_y, float('inf')), dim=-1)[0]  # 会返回一个元组，第一个元素是值，第二个元素是索引

    # 计算最大值和最小值之间的均分区间 (buckets个区间)
    bucket_edges = torch.linspace(0, 1, buckets + 1, device=x.device)  # shape: (buckets+1,)
    # 对每个最大最小区间进行线性变换，得到实际区间的边界
    bucket_edges_x = x_min.unsqueeze(-1) + bucket_edges.view(1, 1, -1) * (x_max - x_min).unsqueeze(-1)
    bucket_edges_y = y_min.unsqueeze(-1) + bucket_edges.view(1, 1, -1) * (y_max - y_min).unsqueeze(-1)

    hists_x = torch.stack([torch.nansum(
        (x_unfold >= bucket_edges_x[..., i].unsqueeze(-1)) & (x_unfold < bucket_edges_x[..., i + 1].unsqueeze(-1)),
        dim=-1)
        for i in range(buckets)], dim=-1)  # stack的dim=-1是指在最后一维上进行堆叠
    hists_y = torch.stack([torch.nansum(
        (y_unfold >= bucket_edges_y[..., i].unsqueeze(-1)) & (y_unfold < bucket_edges_y[..., i + 1].unsqueeze(-1)),
        dim=-1)
        for i in range(buckets)], dim=-1)  # stack的dim=-1是指在最后一维上进行堆叠

    probs_x = hists_x / torch.nansum(hists_x, dim=-1, keepdim=True)
    probs_y = hists_y / torch.nansum(hists_y, dim=-1, keepdim=True)

    cross_entropy = -torch.nansum(probs_x * torch.log2(probs_y + 1e-9), dim=-1) / buckets + torch.tensor(buckets,
                                                                                                         device=x.device,
                                                                                           dtype=x.dtype).log2()
    res[d - 1:] = cross_entropy
    return res

def ts_ewma(x:torch.Tensor, d:int,alpha=None)->torch.Tensor:
    alpha = 1- 2/(d+1) if alpha is None else alpha
    assert alpha > 0 and alpha < 1, "Alpha must be in (0, 1)."
    assert d > 0 and d < x.shape[0], "Window size must be positive."
    res = torch.full(x.shape, float('nan'), device=x.device, dtype=x.dtype)
    res[0] = x[0]
    weight = (1-alpha)**torch.arange(d, 0, -1, device=x.device) # (d,) 
    x_unfold = x.unfold(0, d, 1)
    #print(x_unfold.shape, weight.shape)
    x_weighted = torch.nansum(x_unfold * weight, dim=-1) / torch.nansum(weight)
    #print(x_weighted.shape)
    res[d-1:] = x_weighted
    return res

def ts_entropy(x: torch.Tensor, d: int, buckets: int = 10) -> torch.Tensor:
    assert d > 0 and d < x.shape[0], "Window size must be positive."
    assert d >= buckets, "Number of buckets must be less than or equal to window size."
    res = torch.full(x.shape, torch.nan, device=x.device, dtype=x.dtype)

    x_unfold = x.unfold(0, d, 1)  # shape: (days-d+1, stocks,d)

    # 均分区间
    mask = torch.isnan(x_unfold)  # 创建掩码
    x_max = torch.max(x_unfold.masked_fill(mask, float('-inf')), dim=-1)[0]  # 忽略NaN，填充为负无穷
    x_min = torch.min(x_unfold.masked_fill(mask, float('inf')), dim=-1)[0]  # 会返回一个元组，第一个元素是值，第二个元素是索引
    # 计算最大值和最小值之间的均分区间 (buckets个区间)
    bucket_edges = torch.linspace(0, 1, buckets + 1, device=x.device)  # shape: (buckets+1,)
    # 对每个最大最小区间进行线性变换，得到实际区间的边界
    bucket_edges = x_min.unsqueeze(-1) + bucket_edges.view(1, 1, -1) * (x_max - x_min).unsqueeze(-1)
    # print(bucket_edges.shape) # (days-d+1, stocks, buckets-1)

    hists = torch.stack([torch.sum(
        (x_unfold >= bucket_edges[..., i].unsqueeze(-1)) & (x_unfold < bucket_edges[..., i + 1].unsqueeze(-1)), dim=-1)
        for i in range(buckets)], dim=-1)  # stack的dim=-1是指在最后一维上进行堆叠

    probs = hists / torch.sum(hists, dim=-1, keepdim=True)

    entropy = -torch.sum(probs * torch.log2(probs + 1e-9), dim=-1) / buckets + torch.tensor(buckets, device=x.device,
                                                                                            dtype=x.dtype).log2()

    res[d - 1:] = entropy
    return res


def ts_kl_diverg(x: torch.Tensor, y: torch.Tensor, d: int, buckets: int = 10):
    assert x.shape == y.shape, "Input tensors must have the same shape."
    assert d > 0 and d < x.shape[0], "Window size must be positive."
    assert d >= buckets, "Number of buckets must be less than or equal to window size."

    res = torch.full(x.shape, torch.nan, device=x.device, dtype=x.dtype)

    x_unfold = x.unfold(0, d, 1)
    y_unfold = y.unfold(0, d, 1)

    # 均分区间
    mask_x = torch.isnan(x_unfold)  # 创建掩码
    mask_y = torch.isnan(y_unfold)  # 创建掩码
    x_max = torch.max(x_unfold.masked_fill(mask_x, float('-inf')), dim=-1)[0]  # 忽略NaN，填充为负无穷
    x_min = torch.min(x_unfold.masked_fill(mask_x, float('inf')), dim=-1)[0]  # 会返回一个元组，第一个元素是值，第二个元素是索引
    y_max = torch.max(y_unfold.masked_fill(mask_y, float('-inf')), dim=-1)[0]  # 忽略NaN，填充为负无穷
    y_min = torch.min(y_unfold.masked_fill(mask_y, float('inf')), dim=-1)[0]  # 会返回一个元组，第一个元素是值，第二个元素是索引

    # 计算最大值和最小值之间的均分区间 (buckets个区间)
    bucket_edges = torch.linspace(0, 1, buckets + 1, device=x.device)  # shape: (buckets+1,)
    # 对每个最大最小区间进行线性变换，得到实际区间的边界
    bucket_edges_x = x_min.unsqueeze(-1) + bucket_edges.view(1, 1, -1) * (x_max - x_min).unsqueeze(-1)
    bucket_edges_y = y_min.unsqueeze(-1) + bucket_edges.view(1, 1, -1) * (y_max - y_min).unsqueeze(-1)

    hists_x = torch.stack([torch.nansum(
        (x_unfold >= bucket_edges_x[..., i].unsqueeze(-1)) & (x_unfold < bucket_edges_x[..., i + 1].unsqueeze(-1)),
        dim=-1)
        for i in range(buckets)], dim=-1)  # stack的dim=-1是指在最后一维上进行堆叠
    hists_y = torch.stack([torch.nansum(
        (y_unfold >= bucket_edges_y[..., i].unsqueeze(-1)) & (y_unfold < bucket_edges_y[..., i + 1].unsqueeze(-1)),
        dim=-1)
        for i in range(buckets)], dim=-1)  # stack的dim=-1是指在最后一维上进行堆叠

    probs_x = hists_x / torch.nansum(hists_x, dim=-1, keepdim=True)
    probs_y = hists_y / torch.nansum(hists_y, dim=-1, keepdim=True)

    eps = torch.tensor(1e-8, device=x.device, dtype=x.dtype)

    kl = torch.nansum(probs_x * (torch.log2(probs_x + eps) - torch.log2(probs_y + eps)), dim=-1) / buckets

    res[d - 1:] = kl

    return res


def ts_log_diff(x: torch.Tensor, d: int) -> torch.Tensor:
    assert d > 0 and d < x.shape[0], "Window size must be positive."
    log = torch.log(x)
    res = log[d:] - log[:-d]
    res = torch.cat([torch.full((d, x.shape[1]), float('nan'), device=x.device, dtype=x.dtype), res], dim=0)
    return res


def ts_diff_rows(x: torch.Tensor) -> torch.Tensor:
    # 创建一个与输入张量形状相同但所有元素均为 NaN 的张量
    result = torch.full_like(x, float('nan'), device=x.device, dtype=x.dtype)
    
    # 第一行直接复制输入张量的第一行
    result[0] = x[0]
    
    # 对于后续的每一行，计算当前行减去前一行的结果
    for i in range(1, x.shape[0]):
        result[i] = x[i] - x[i - 1]
    
    return result


def ts_reg_betase(x: torch.Tensor, y: torch.Tensor, d: int):
    assert x.shape == y.shape, "x and y shapes must be the same"
    assert 2 < d <= x.shape[0], "OLS lens must be greater than 2"
    res = torch.full(x.shape, torch.nan, dtype=x.dtype, device=x.device)

    x_unfold = x.unfold(0, d, 1).view(-1, d)  # (T-d+1)*N,d
    y_unfold = y.unfold(0, d, 1).view(-1, d)

    # 填充
    x_mask = x_unfold.isnan() | x_unfold.isinf()
    y_mask = y_unfold.isnan() | y_unfold.isinf()
    x_mean = torch.nanmean(x_unfold, dim=-1, keepdim=True)
    y_mean = torch.nanmean(y_unfold, dim=-1, keepdim=True)

    x_unfold = torch.where(x_mask, x_mean, x_unfold)
    y_unfold = torch.where(y_mask, y_mean, y_unfold)

    x_demean = (x_unfold - x_mean).unsqueeze(-1)
    y_demean = (y_unfold - y_mean).unsqueeze(-1)

    beta = torch.matmul(torch.linalg.pinv(torch.matmul(x_demean.transpose(1, 2), x_demean)),
                        torch.matmul(x_demean.transpose(1, 2), y_demean))
    y_hat = torch.matmul(x_demean, beta).squeeze(-1)
    
    rss = torch.sum((y_demean.squeeze(-1) - y_hat) ** 2, dim=-1, keepdim=True)

    var_x = torch.sum(x_demean.squeeze(-1) ** 2, dim=-1, keepdim=True)

    sebeta = torch.sqrt(rss / (var_x * (d - 2))).view(-1, x.shape[1])

    res[d - 1:, :] = sebeta

    return res

def ts_regbeta(x: torch.Tensor, y: torch.Tensor, d: int) -> torch.Tensor:
    x_unfold = _unfold(x, d)
    y_unfold = _unfold(y, d)
    x_demean = x_unfold - x_unfold.nanmean(dim=-1, keepdim=True)
    y_demean = y_unfold - y_unfold.nanmean(dim=-1, keepdim=True)
    return div(torch.nansum(x_demean * y_demean, dim=-1), torch.nansum(torch.pow(x_demean, 2), dim=-1))


def ts_dist_to_high(x: torch.Tensor, d: int) -> torch.Tensor:
    """
    计算当前数据值距离过去d个时间段内最高点的时间距离
    """
    # 初始化结果张量为 NaN
    res = get_nan_tensor(shape=x.shape, device=x.device, dtype=x.dtype)
    x_unfold = x.unfold(0, d, 1)  # 形状为 (T-d+1, N, d)

    mask = ~torch.isnan(x_unfold)
    filled_x = torch.where(mask, x_unfold, torch.tensor(float('-inf'), device=x.device))
    max_indices = filled_x.argmax(dim=-1)  # 形状为 (T-d+1, N)

    time_distance = (d - 1) - max_indices  # 形状为 (T-d+1, N)
    
    all_nan_mask = ~mask.any(dim=-1)  # 形状为 (T-d+1, N)
    time_distance = torch.where(all_nan_mask, 
                               torch.tensor(float('nan'), device=x.device),
                               time_distance.float())

    res[d-1:] = time_distance

    return res


def ts_dist_to_min(x: torch.Tensor, d: int) -> torch.Tensor:
    """
    计算当前数据值距离过去d个时间段内最低点的时间距离
    """
    # 初始化结果张量为 NaN
    res = get_nan_tensor(shape=x.shape, device=x.device, dtype=x.dtype)
    x_unfold = x.unfold(0, d, 1)  # 形状为 (T-d+1, N, d)

    mask = ~torch.isnan(x_unfold)
    filled_x = torch.where(mask, x_unfold, torch.tensor(float('-inf'), device=x.device))
    min_indices = filled_x.argmin(dim=-1)  # 形状为 (T-d+1, N)

    time_distance = (d - 1) - min_indices  # 形状为 (T-d+1, N)
    
    all_nan_mask = ~mask.any(dim=-1)  # 形状为 (T-d+1, N)
    time_distance = torch.where(all_nan_mask, 
                               torch.tensor(float('nan'), device=x.device),
                               time_distance.float())

    res[d-1:] = time_distance

    return res


def ts_reg_r2(x, y, d: int):
    """
    对于截面上每一只股票，在过去d个时间步内，以y作因变量，x作自变量，进行OLS回归并求预测的R2
    x (torch.Tensor): 回归的自变量，形状为(T, N)的tensor，T是时间步数，N是股票数;
    y (torch.Tensor): 回归的因变量，形状为(T, N)的tensor，T是时间步数，N是股票数;

    注：本函数还没有完全解决因变量是一个常数的情况
    """
    assert x.shape == y.shape, "shapes of inputs x and y must match"
    # assert x.dtype == float, "dtype of x must be float"
    assert d > 2, "time window d must be bigger than 2 since we are conducting OLS"
    assert d <= x.shape[0], "time window d must be smaller than x.shape[0]"

    T, N = x.shape
    res = torch.full(x.shape, torch.nan, dtype=x.dtype, device=x.device)

    # 提取滑动窗口并合并维度，得到的x和y是大小为((T-d+1)*N, d)的矩阵
    x = x.unfold(0, d, 1).view(-1, d)
    y = y.unfold(0, d, 1).view(-1, d)

    # 对x和y先进行均值填充，再进行去均值化处理
    """
    值得注意的是，值全部为nan的回看窗口内的数据没有被填充，因此在计算矩阵逆的那一步并不会报错
    """
    mean_values_x = torch.nanmean(x, dim=-1, keepdim=True)
    x_filled = torch.where(torch.isnan(x), mean_values_x, x)
    x_demeaned = x_filled - torch.nanmean(x_filled, dim=-1, keepdim=True)
    x_demeaned = x_demeaned.unsqueeze(-1)

    mean_values_y = torch.nanmean(y, dim=-1, keepdim=True)
    y_filled = torch.where(torch.isnan(y), mean_values_y, y)
    y_demeaned = y_filled - torch.nanmean(y_filled, dim=-1, keepdim=True)
    y_demeaned = y_demeaned.unsqueeze(-1)

    # 使用矩阵运算
    XtX = torch.bmm(x_demeaned.transpose(1, 2), x_demeaned)
    XtY = torch.bmm(x_demeaned.transpose(1, 2), y_demeaned)
    XtX_inv = torch.pinverse(XtX)
    beta = torch.bmm(XtX_inv, XtY)

    # 预测值
    y_pred = torch.bmm(x_demeaned, beta).squeeze(-1)  # (T-d+1)*N, d
    y_demeaned = y_demeaned.squeeze(-1)  # (T-d+1)*N, d

    # 计算R²
    ss_tot = torch.sum(y_demeaned ** 2, dim=1)
    ss_res = torch.sum((y_demeaned - y_pred) ** 2, dim=1)
    r_squared = (1 - (ss_res / ss_tot)).view(-1, N)
    
    res[d - 1:, :] = r_squared

    return res

def ts_reg_resse(x: torch.Tensor, y: torch.Tensor, d: int):
    assert x.shape == y.shape, "x and y shapes must be the same"
    assert 2 < d <= x.shape[0], "OLS lens must be greater than 2"
    res = torch.full(x.shape, torch.nan, dtype=x.dtype, device=x.device)

    x_unfold = x.unfold(0, d, 1).view(-1, d)  # (T-d+1)*N,d
    y_unfold = y.unfold(0, d, 1).view(-1, d)

    # 填充
    x_mask = x_unfold.isnan() | x_unfold.isinf()
    y_mask = y_unfold.isnan() | y_unfold.isinf()
    x_mean = torch.nanmean(x_unfold, dim=-1, keepdim=True)
    y_mean = torch.nanmean(y_unfold, dim=-1, keepdim=True)

    x_unfold = torch.where(x_mask, x_mean, x_unfold)
    y_unfold = torch.where(y_mask, y_mean, y_unfold)

    x_demean = (x_unfold - x_mean).unsqueeze(-1)
    y_demean = (y_unfold - y_mean).unsqueeze(-1)

    beta = torch.matmul(torch.linalg.pinv(torch.matmul(x_demean.transpose(1, 2), x_demean)),
                        torch.matmul(x_demean.transpose(1, 2), y_demean))
    y_hat = torch.matmul(x_demean, beta).squeeze(-1)

    rs_var = torch.var(y_demean.squeeze(-1) - y_hat, dim=-1, unbiased=True, keepdim=True)
    rssse = torch.sqrt(rs_var).view(-1, x.shape[1])
    res[d - 1:, :] = rssse
    return res

def ts_regsse(x: torch.Tensor, y: torch.Tensor, d: int) -> torch.Tensor:
    res = get_nan_tensor(x.shape, x.device, x.dtype)
    x_unfold = _unfold(x, d)
    y_unfold = _unfold(y, d)
    x_demean = x_unfold - x_unfold.nanmean(dim=-1, keepdim=True)
    y_demean = y_unfold - y_unfold.nanmean(dim=-1, keepdim=True)

    var = torch.pow(x_demean,2).nanmean(dim=-1, keepdim=True)
    beta = div((x_demean * y_demean).nansum(dim=-1,keepdim=True), var)
    res[d-1:] = torch.pow((y_demean - beta * x_demean),2).nansum(dim=-1)[d-1:]
    return res

def ts_reg_tval(x, y, d: int):
    """
    对于截面上每一只股票，在过去d个时间步内，以y作因变量，x作自变量，进行OLS回归并求预测的t统计量
    x (torch.Tensor): 回归的自变量，形状为(T, N)的tensor，T是时间步数，N是股票数;
    y (torch.Tensor): 回归的因变量，形状为(T, N)的tensor，T是时间步数，N是股票数;

    注：本函数还没有完全解决因变量是一个常数的情况
    """
    assert x.shape == y.shape, "shapes of inputs x and y must match"
    # assert x.dtype == float, "dtype of x must be float"
    assert d > 2, "time window d must be bigger than 2 since we are conducting OLS"
    assert d <= x.shape[0], "time window d must be smaller than x.shape[0]"

    T, N = x.shape
    res = torch.full(x.shape, torch.nan, dtype=x.dtype, device=x.device)

    # 提取滑动窗口并合并维度，得到的x和y是大小为((T-d+1)*N, d)的矩阵
    x = x.unfold(0, d, 1).view(-1, d)
    y = y.unfold(0, d, 1).view(-1, d)

    # 对x和y先进行均值填充，再进行去均值化处理
    """
    值得注意的是，值全部为nan的回看窗口内的数据没有被填充（仍为nan），因此在计算矩阵逆的那一步并不会报错
    """
    mean_values_x = torch.nanmean(x, dim=-1, keepdim=True)
    x_filled = torch.where(torch.isnan(x), mean_values_x, x)
    x_demeaned = x_filled - torch.nanmean(x_filled, dim=-1, keepdim=True)
    x_demeaned = x_demeaned.unsqueeze(-1)

    mean_values_y = torch.nanmean(y, dim=-1, keepdim=True)
    y_filled = torch.where(torch.isnan(y), mean_values_y, y)
    y_demeaned = y_filled - torch.nanmean(y_filled, dim=-1, keepdim=True)
    y_demeaned = y_demeaned.unsqueeze(-1)  # ((T-d+1)*N, d, 1)

    # 使用矩阵运算
    XtX = torch.bmm(x_demeaned.transpose(1, 2), x_demeaned)
    XtY = torch.bmm(x_demeaned.transpose(1, 2), y_demeaned)
    XtX_inv = torch.pinverse(XtX)
    beta = torch.bmm(XtX_inv, XtY)  # ((T-d+1)*N, k, 1)

    # 预测值
    y_pred = torch.bmm(x_demeaned, beta).squeeze(-1)  # ((T-d+1)*N, d)
    y_demeaned = y_demeaned.squeeze(-1)  # ((T-d+1)*N, d)
    beta = beta.squeeze(-1)

    # print(y_pred.shape)
    # print(y_demeaned.shape)

    residuals_var = torch.var(y_demeaned - y_pred, unbiased=False, dim=1, keepdim=True)
    standard_error = torch.sqrt(residuals_var / d)
    t_values = beta / standard_error

    t_values = t_values.view(-1, N)

    res[d - 1:, :] = t_values

    return res 


def ts_scale(x: torch.Tensor, d: int, constant: int = 0) -> torch.Tensor:
    assert d > 0 and d < x.shape[0], "Window size must be positive."
    res = torch.full(x.shape, float('nan'), device=x.device, dtype=x.dtype)
    x_unfold = x.unfold(0, d, 1)
    # 较旧的版本没有nanmax和nanmin
    # 在滑动窗口中查找最大值和最小值（忽略NaN）
    mask = torch.isnan(x_unfold)  # 创建掩码
    x_max = torch.max(x_unfold.masked_fill(mask, float('-inf')), dim=-1)[0]  # 忽略NaN，填充为负无穷
    x_min = torch.min(x_unfold.masked_fill(mask, float('inf')), dim=-1)[0]  # 会返回一个元组，第一个元素是值，第二个元素是索引
    # print(x_max.shape, x_min.shape)
    # print(x_max.shape, x_min.shape)
    res[d - 1:] = (x[d - 1:] - x_min) / (x_max - x_min) + constant
    return res


def ts_spread(x: torch.Tensor, d: int, low_q: float, high_q: float) -> torch.Tensor:
    '''
    计算滑动窗口里，求前q分位数和后q分位数的均值的差值
    '''

    assert d > 0 and d < x.shape[0], "Window size must be positive."
    res = torch.full(x.shape, float('nan'), device=x.device, dtype=x.dtype)
    x_unfold = x.unfold(0, d, 1)
    x_low_q = torch.nanquantile(x_unfold, low_q, dim=-1, keepdim=True)  # (days-d+1, stocks)
    x_high_q = torch.nanquantile(x_unfold, 1 - high_q, dim=-1, keepdim=True)  # (days-d+1, stocks)

    # 创建掩码，标记符合条件的元素
    mask_low = (x_unfold <= x_low_q)
    mask_high = (x_unfold >= x_high_q)

    # 将不符合条件的元素置为 NaN（通过掩码）
    x_low_q_filtered = x_unfold.clone()
    x_low_q_filtered[~mask_low] = float('nan')  # 将不符合条件的元素置为 NaN

    x_high_q_filtered = x_unfold.clone()
    x_high_q_filtered[~mask_high] = float('nan')  # 将不符合条件的元素置为 NaN

    # 计算符合条件的元素的均值
    x_low_q_mean = torch.nanmean(x_low_q_filtered, dim=-1) if low_q * d >= 1 else torch.full(x_unfold.shape[:-1], 0.,
                                                                                             device=x.device,
                                                                                             dtype=x.dtype)
    x_high_q_mean = torch.nanmean(x_high_q_filtered, dim=-1) if high_q * d >= 1 else torch.full(x_unfold.shape[:-1], 0.,
                                                                                                device=x.device,
                                                                                                dtype=x.dtype)
    # x_low_q_mean = torch.nanmean(x_unfold*(x_unfold <= x_low_q).to(torch.int), dim=-1)
    # print(x_unfold*(x_unfold <= x_low_q).to(torch.int)[0,0])
    # x_high_q_mean = torch.nanmean(x_unfold*(x_unfold >= x_high_q).to(torch.int), dim=-1) #会有0值

    res[d - 1:] = x_high_q_mean - x_low_q_mean
    return res

def ts_spread_std(x: torch.Tensor, d: int, low_q: float, high_q: float) -> torch.Tensor:
    x_unfold = _unfold(x, d)
    low_value = torch.nanquantile(x_unfold, low_q, dim=-1, keepdim=True)
    high_value = torch.nanquantile(x_unfold, high_q, dim=-1, keepdim=True)
    
    cond_1 = x_unfold < low_value
    cond_2 = x_unfold > high_value
    return _stddev(filt_cond(x_unfold, cond_1)) - _stddev(filt_cond(x_unfold, cond_2))

def ts_spread_selq(x: torch.Tensor, cond_var: torch.Tensor, d: int, low_q: float, high_q: float):
    assert d > 0 and d < x.shape[0], "Window size must be positive."
    res = torch.full(x.shape, float('nan'), device=x.device, dtype=x.dtype)
    x_unfold = x.unfold(0, d, 1)
    cond_var_unfold = cond_var.unfold(0, d, 1)
    
    high = cond_var_unfold.masked_fill(cond_var_unfold.isnan(), float('-inf')).argsort(dim=-1, descending=True)
    low = cond_var_unfold.masked_fill(cond_var_unfold.isnan(), float('inf')).argsort(dim=-1, descending=False)

    # gather函数，根据索引从原始tensor中取值,返回的tensor的shape和索引的shape一样
    # print(high.shape, low.shape)
    x_high_q = x_unfold.gather(-1, high[..., round(high_q * d) - 1 if high_q * d >= 0.5 else 0].unsqueeze(-1))
    x_low_q = x_unfold.gather(-1, low[..., round(low_q * d) - 1 if low_q * d >= 0.5 else 0].unsqueeze(-1))

    mask_low = (x_unfold <= x_low_q)
    mask_high = (x_unfold >= x_high_q)

    x_low_q_filtered = x_unfold.clone()
    x_low_q_filtered[~mask_low] = float('nan')
    x_high_q_filtered = x_unfold.clone()
    x_high_q_filtered[~mask_high] = float('nan')

    x_low_q_mean = torch.nanmean(x_low_q_filtered, dim=-1) if low_q * d >= 1 else torch.full(x_unfold.shape[:-1], 0.,
                                                                                             device=x.device,
                                                                                             dtype=x.dtype)
    x_high_q_mean = torch.nanmean(x_high_q_filtered, dim=-1) if high_q * d >= 1 else torch.full(x_unfold.shape[:-1], 0.,
                                                                                                device=x.device,
                                                                                                dtype=x.dtype)

    res[d - 1:] = x_high_q_mean - x_low_q_mean
    return res


def ts_spreadw_selq(x: torch.Tensor, cond_var: torch.Tensor, d: int, low_q: float, high_q: float):
    assert d > 0 and d < x.shape[0], "Window size must be positive."
    res = torch.full(x.shape, float('nan'), device=x.device, dtype=x.dtype)
    x_unfold = x.unfold(0, d, 1)
    cond_var_unfold = cond_var.unfold(0, d, 1)

    high = cond_var_unfold.masked_fill(cond_var_unfold.isnan(), float('-inf')).argsort(dim=-1, descending=True)
    low = cond_var_unfold.masked_fill(cond_var_unfold.isnan(), float('inf')).argsort(dim=-1, descending=False)
    # gather函数，根据索引从原始tensor中取值,返回的tensor的shape和索引的shape一样
    # print(high.shape, low.shape)
    x_high_q = x_unfold.gather(-1, high[..., round(high_q * d) - 1 if high_q * d >= 0.5 else 0].unsqueeze(-1))
    x_low_q = x_unfold.gather(-1, low[..., round(low_q * d) - 1 if low_q * d >= 0.5 else 0].unsqueeze(-1))
    # print(low[...,int(low_q*(d-1))].shape)
    # print(x_high_q.shape, x_low_q.shape)
    mask_low = (x_unfold <= x_low_q)
    mask_high = (x_unfold >= x_high_q)

    x_low_q_filtered = x_unfold.clone()
    x_low_q_filtered[~mask_low] = float('nan')
    cond_var_low = cond_var_unfold.clone()
    cond_var_low[~mask_low] = float('nan')
    cond_var_low_weight = cond_var_low.abs() / torch.nansum(cond_var_low.abs(), dim=-1, keepdim=True)
    x_low_q_weighted = x_low_q_filtered * cond_var_low_weight

    x_high_q_filtered = x_unfold.clone()
    x_high_q_filtered[~mask_high] = float('nan')
    cond_var_high = cond_var_unfold.clone()
    cond_var_high[~mask_high] = float('nan')
    cond_var_high_weight = cond_var_high.abs() / torch.nansum(cond_var_high.abs(), dim=-1, keepdim=True)
    x_high_q_weighted = x_high_q_filtered * cond_var_high_weight

    x_low_q_mean = torch.nanmean(x_low_q_weighted, dim=-1) if low_q * d >= 1 else torch.full(x_unfold.shape[:-1], 0.,
                                                                                             device=x.device,
                                                                                             dtype=x.dtype)
    x_high_q_mean = torch.nanmean(x_high_q_weighted, dim=-1) if high_q * d >= 1 else torch.full(x_unfold.shape[:-1], 0.,
                                                                                                device=x.device,
                                                                                                dtype=x.dtype)

    res[d - 1:] = x_high_q_mean - x_low_q_mean
    return res


def ts_sumprod_selq(x, y, cond_var, d, low_q, high_q):
    """
    计算过去d个时间步内cond_var的分位数，筛选出分位数在low_q到high_q之间的样本，计算x和y(时序上标准化过后，可看成一个系数)对应位置乘积之和。

    参数:
        x (torch.Tensor): 形状为(T, N)的tensor，T是时间步数，N是股票数。
        cond_var (torch.Tensor): 形状为(T, N)的tensor，T是时间步数，N是股票数。
        y (torch.Tensor): 形状为(T, N)的tensor，T是时间步数，N是股票数。
        d (int): 过去的时间步数。
        low_q (float): 低分位数，范围[0, 1]。
        high_q (float): 高分位数，范围[0, 1]。

    返回:
        torch.Tensor: 计算结果的张量，形状为(T - d + 1, N)，即从第d分钟开始有有效值。
    """
    # 获取时间步数 T 和股票数 N
    T, N = x.shape
    res = torch.full(x.shape, torch.nan, dtype=x.dtype, device=x.device)

    # 使用 unfold 提取过去 d 个时间步的数据，得到形状为 (T-d+1, d, N) 的 tensor
    x_unfold = x.unfold(0, d, 1).permute(0, 2, 1)  # (T-d+1, d, N)
    cond_var_unfold = cond_var.unfold(0, d, 1).permute(0, 2, 1)  # (T-d+1, d, N)
    y_unfold = y.unfold(0, d, 1).permute(0, 2, 1)  # (T-d+1, d, N)

    # 对于每个时间窗口，计算 cond_var 的低分位数和高分位数
    cond_var_low = torch.quantile(cond_var_unfold, low_q, dim=1)  # (T-d+1, N)
    cond_var_high = torch.quantile(cond_var_unfold, high_q, dim=1)  # (T-d+1, N)

    # 构造 mask，检查每个时间窗口内 cond_var 是否在 low_q 和 high_q 之间
    mask = (cond_var_unfold >= cond_var_low.unsqueeze(1)) & (
            cond_var_unfold <= cond_var_high.unsqueeze(1))  # （T-d+1, d, N）

    # 使用 mask 筛选出符合条件的 x 和 y，然后计算乘积之和
    # 对于每个时间步（过去 d 个时间步内），计算 x 和 y(标准化后) 对应位置的乘积并求和
    y_sum = y_unfold.sum(dim=1, keepdim=True)
    y_normalized = y_unfold / y_sum
    product_sum = (x_unfold * y_normalized * mask).sum(dim=1)  # 计算乘积之和

    res[d - 1:] = product_sum

    return res


def ts_partial_corr(
        x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, d: int) -> torch.Tensor:

    assert x.shape == y.shape == z.shape, "Input tensors must have the same shape."
 
    res = torch.full(x.shape, float('nan'), device=x.device, dtype=x.dtype)
    
    x_error = ts_regres(z, x, d)
    y_error = ts_regres(z, y, d)

    partial_corr = ts_corr(x_error, y_error, d)
    
    res[d - 1:] = partial_corr

    return res


def ts_triple_corr(
        x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, d: int) -> torch.Tensor:
    """
    标准化后三个变量的三阶交互的中心矩

    Args:
        x (torch.Tensor): Input tensor of shape (days, stocks).
        y (torch.Tensor): Input tensor of shape (days, stocks).
        z (torch.Tensor): Input tensor of shape (days, stocks).
        d (int): Sliding window size.
    
    Returns:
        torch.Tensor: Tensor of triple correlations with shape (days, stocks).
    """
    assert x.shape == y.shape == z.shape, "Input tensors must have the same shape."

    res = torch.full(x.shape, float('nan'), device=x.device, dtype=x.dtype)

    x_unfold = x.unfold(0, d, 1)  # 对第0维展开，窗口大小为d，步长为1
    y_unfold = y.unfold(0, d, 1)  # 形成(days, stocks, sliding_window_size)的tensor
    z_unfold = z.unfold(0, d, 1)  # 改变的是视图，不改变原始数据

    x_mean = x_unfold.nanmean(dim=-1)  # keepdim默认为False，不保留维度
    y_mean = y_unfold.nanmean(dim=-1)
    z_mean = z_unfold.nanmean(dim=-1)

    x_demean = x_unfold - x_mean.unsqueeze(-1)
    y_demean = y_unfold - y_mean.unsqueeze(-1)
    z_demean = z_unfold - z_mean.unsqueeze(-1)

    x_std = torch.sqrt(torch.nansum(x_demean ** 2, dim=-1))
    y_std = torch.sqrt(torch.nansum(y_demean ** 2, dim=-1))
    z_std = torch.sqrt(torch.nansum(z_demean ** 2, dim=-1))
    print(x_std.shape, y_std.shape, z_std.shape)
    numerator = torch.nansum(x_demean * y_demean * z_demean, dim=-1)
    denominator = x_std * y_std * z_std

    triple_corr = numerator / denominator

    res[d - 1:] = triple_corr
    res[d - 1:][(x_std < 1e-9) | (y_std < 1e-9)] = torch.nan

    return res


def ts_reg_tvalue(x: torch.Tensor, y: torch.Tensor, d: int) -> torch.Tensor:
    x_unfold = _unfold(x, d)
    y_unfold = _unfold(y, d)
    x_demean = x_unfold - x_unfold.nanmean(dim=-1, keepdim=True)
    y_demean = y_unfold - y_unfold.nanmean(dim=-1, keepdim=True)

    var = torch.pow(x_demean,2).nanmean(dim=-1)
    beta = div((x_demean * y_demean).nansum(dim=-1), var) 
    sse = torch.pow((y_demean - beta.unsqueeze(-1) * x_demean),2).nansum(dim=-1)
    std_beta = torch.sqrt(div(sse, var*(d - 2)))
    
    return div(beta, std_beta)   


def ts_reg_rsquare(x: torch.Tensor, y: torch.Tensor, d: int) -> torch.Tensor:
    x_unfold = _unfold(x, d)
    y_unfold = _unfold(y, d)
    x_demean = x_unfold - x_unfold.nanmean(dim=-1, keepdim=True)
    y_demean = y_unfold - y_unfold.nanmean(dim=-1, keepdim=True)
    
    var = torch.pow(x_demean,2).nanmean(dim=-1, keepdim=True)
    beta = div((x_demean * y_demean).nansum(dim=-1,keepdim=True), var)
    return 1 - div(torch.pow((y_demean - beta * x_demean),2).nansum(dim=-1), torch.pow(y_demean,2).nansum(dim=-1))



def ts_weighted_skew(x: torch.Tensor, weights: torch.Tensor, d: int):
    x_unfold = _unfold(x, d)
    weights_unfold = _unfold(weights, d)
    return _weighted_skew(x_unfold, weights_unfold)

def ts_weighted_kurt(x: torch.Tensor, weights: torch.Tensor, d: int):
    x_unfold = _unfold(x, d)
    weights_unfold = _unfold(weights, d)
    
    x_demeaned = x_unfold - x_unfold.nanmean(dim=-1, keepdim=True)
    x_4 = div((torch.pow(x_demeaned, 4) * weights_unfold).nansum(dim = -1), weights_unfold.nansum(dim = -1))
    x_var = div((torch.pow(x_demeaned, 2) * weights_unfold).nansum(dim = -1), weights_unfold.nansum(dim = -1))
    return div(x_4, torch.pow(x_var, 2)) - 3


def ts_first_deriv(x:torch.Tensor, d:int):
    assert 2 < d <= x.shape[0]
    
    res = torch.full(x.shape, torch.nan, dtype=x.dtype, device=x.device)
    x_unfold = x.unfold(0,d,1).view(-1,d) # 先是展开，然后其实是对最后一个维度操作，所以前两维度可以合并
    timeline = torch.tile(torch.arange(d,dtype=x.dtype, device=x.device),(x_unfold.shape[0],1))
    # 填充
    x_mask = x_unfold.isnan()|x_unfold.isinf()
    #y_mask = timeline.isnan()|y_unfold.isinf()
    x_mean = torch.nanmean(x_unfold, dim=-1,keepdim=True)
    y_mean = torch.nanmean(timeline, dim=-1,keepdim=True)
    
    x_unfold = torch.where(x_mask,x_mean,x_unfold)
    #y_unfold = torch.where(y_mask,y_mean,y_unfold)
    
    x_demean = (x_unfold - x_mean).unsqueeze(-1)
    y_demean = (timeline - y_mean).unsqueeze(-1)    
    #print(x_demean.shape)
    beta = torch.matmul(1. / torch.matmul(x_demean.transpose(1,2),x_demean), torch.matmul(x_demean.transpose(1,2),y_demean)).squeeze(-1)
    res[d-1:,:] = beta.view(-1, x.shape[1])
    return res

def ts_hurst(x: torch.Tensor, d: int) -> torch.Tensor:
    res = torch.full(x.shape, torch.nan, dtype=x.dtype, device=x.device)
    
    # 滑动窗口展开
    x_unfold = x.unfold(0, d, 1)  # [N-d+1, d]
    
    # 计算窗口内均值
    x_mean = x_unfold.nanmean(dim=-1, keepdim=True)  # [N-d+1, 1]
    
    # 计算去均值序列 X_t = x_t - x̄
    x_demean = x_unfold - x_mean
    
    # 计算累积偏差序列 S_t
    x_cumsum = torch.cumsum(x_unfold - x_mean, dim=-1)  # (N-d+1, d)
    
    mask = ~torch.isnan(x_cumsum)  # 找到非 NaN 值的 mask
    R_d = torch.where(mask, x_cumsum, torch.tensor(-float("inf"), device=x.device)).max(dim=-1)[0] - \
    torch.where(mask, x_cumsum, torch.tensor(float("inf"), device=x.device)).min(dim=-1)[0]
    
    # 计算标准差 S(d)
    S_d = torch.sqrt(torch.nansum(torch.pow(x_demean, 2), dim=-1) / (d - 1))  # [N-d+1]
    
    # 计算 R/S 比率
    RS = R_d / S_d
    
    # 计算 log(R/S) 和 log(d)
    log_RS = torch.log(RS)
    log_d = torch.full_like(log_RS, torch.log(torch.tensor(d, dtype=x.dtype, device=x.device)))
    
    # 计算 Hurst 指数：H = log(R/S) / log(d)
    hurst = log_RS / log_d
    
    # 赋值
    res[d - 1:] = hurst
    res[d - 1:][S_d < 1e-9] = torch.nan  # 避免 S_d 过小导致除零
    
    return res

import torch

def ts_fill(x: torch.Tensor, t: int ) -> torch.Tensor:
    """
    对每支股票，若当前时间步是 NaN，向前回看 t 个时间步并填充第一个非 NaN 值。
    
    Args:
    - x (torch.Tensor): 输入张量，形状为 (T, N)，其中 T 是时间步数，N 是股票数。
    - t (int): 向前回看的时间步数。
    
    Returns:
    - torch.Tensor: 填充后的张量，形状与输入相同。
    """

    if t == 0:
        return x
    
    if t == 1:
        result = x.clone()

        result[1:] = x[0:-1]

        filled_tensor = torch.where(torch.isnan(x), result, x)

        return filled_tensor

    else:
        return ts_fill(ts_fill(x,1),t-1)
    



def ts_second_deriv(x:torch.Tensor, d1:int,d2=None):
    if d2 == None:
        d2 = d1
    d = d1+d2-1
    assert 2 < d <= x.shape[0]
    
    res = torch.full(x.shape, torch.nan, dtype=x.dtype, device=x.device)
    x_unfold = x.unfold(0,d,1).unfold(-1,d1,1).contiguous().view(-1,d1)
    #print(x_unfold.shape)
    timeline = torch.tile(torch.arange(d1,dtype=x.dtype, device=x.device),(x_unfold.shape[0],1))
    # 填充
    x_mask = x_unfold.isnan()|x_unfold.isinf()
    #y_mask = timeline.isnan()|y_unfold.isinf()
    x_mean = torch.nanmean(x_unfold, dim=-1,keepdim=True)
    y_mean = torch.nanmean(timeline, dim=-1,keepdim=True)
    
    x_unfold = torch.where(x_mask,x_mean,x_unfold)
    #y_unfold = torch.where(y_mask,y_mean,y_unfold)
    
    x_demean = (x_unfold - x_mean).unsqueeze(-1)
    y_demean = (timeline - y_mean).unsqueeze(-1)    
    #print(x_demean.shape)
    beta = torch.matmul(1. / torch.matmul(x_demean.transpose(1,2),x_demean), torch.matmul(x_demean.transpose(1,2),y_demean)).squeeze(-1).view(-1,d2)
    
    timeline = torch.tile(torch.arange(d2,dtype=x.dtype, device=x.device),(beta.shape[0],1))
    beta_mean = torch.nanmean(beta, dim=-1,keepdim=True)
    y_mean = torch.nanmean(timeline, dim=-1,keepdim=True)
    x_demean = (beta - beta_mean).unsqueeze(-1)
    y_demean = (timeline - y_mean).unsqueeze(-1)  
    
    beta2 = torch.matmul(1. / torch.matmul(x_demean.transpose(1,2),x_demean), torch.matmul(x_demean.transpose(1,2),y_demean)).squeeze(-1)
    res[d-1:,:] = beta2.view(-1, x.shape[1])
    return res


def ts_garch(x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
    """
    构建 GARCH 模型，处理收益率序列和可选的外生变量。

    参数:
    - x: torch.Tensor，形状为 (T,)，表示收益率序列。
    - y: torch.Tensor，形状为 (T, N)，表示外生变量序列，默认为 None。

    返回:
    - final_cond_var: torch.Tensor，形状为 (T,)，表示条件方差序列。
    """
    T = x.shape[0]
    N = y.shape[1] if y.ndim>1 else 0         

    # 初始化参数
    omega = torch.tensor(0.01, dtype=torch.float32, requires_grad=True, device=x.device)
    alpha = torch.tensor(0.1, dtype=torch.float32, requires_grad=True, device=x.device)
    beta = torch.tensor(0.8, dtype=torch.float32, requires_grad=True, device=x.device)
    gamma = torch.zeros(N, requires_grad=True, device=x.device) if N > 0 else None

    params = [omega, alpha, beta]
    if gamma is not None:
        params.append(gamma)
        
    optimizer = Adam(params, lr=0.01)

    def garch_likelihood():
        cond_var = torch.empty(T, dtype=torch.float32, device=x.device)

        # 初始条件方差
        if (alpha + beta).item() < 1:
            cond_var[0] = omega / (1 - (alpha + beta))
        else:
            return torch.tensor(float('inf'), device=x.device)

        for t in range(1, T):
            prev_ret_sq = x[t - 1] ** 2
            prev_cond_var = cond_var[t - 1]

            # 将外生变量 y 纳入模型
            exog_effect = torch.sum(y[t-1] * gamma) if N > 0 else 0.0
            
            new_var = omega + alpha*prev_ret_sq + beta*prev_cond_var + exog_effect
            cond_var[t] = new_var

        # 过滤 NaN 值
        valid_mask = ~torch.isnan(cond_var)
        x_valid = x[valid_mask]
        cond_var_valid = cond_var[valid_mask]

        log_likelihood = -0.5 * torch.sum(torch.log(cond_var_valid) + (x_valid ** 2) / cond_var_valid)
        return -log_likelihood

    for _ in range(300): 
        optimizer.zero_grad()
        loss = garch_likelihood()
        loss.backward()
        optimizer.step()

    # 计算最终条件方差序列
    final_cond_var = torch.empty(T, dtype=torch.float32, device=x.device)
    if (alpha + beta).item() < 1:
        final_cond_var[0] = omega / (1 - (alpha + beta))
    else:
        return torch.full((T,), float('nan'), dtype=torch.float32, device=x.device)

    for t in range(1, T):
        prev_ret_sq = x[t - 1] ** 2
        prev_cond_var = final_cond_var[t - 1]
        exog_effect = torch.sum(y[t - 1] * torch.tensor([0.05, -0.03], device=x.device))
        final_cond_var[t] = omega + alpha * prev_ret_sq + beta * prev_cond_var + exog_effect

    return final_cond_var
# ts_garch算子更新尚未完成，可能存在梯度消失或爆炸的问题，待解决



def ts_multiregbeta(x_list: list, y: torch.Tensor, d: int) -> torch.Tensor:
    # 数据预处理
    x = torch.stack([xx if xx.ndim==2 else xx.unsqueeze(1) for xx in x_list], dim=2)  # [T, N, F]
    T, N, F = x.shape
    y = y.view(T, N)  # 确保y形状为 [T, N]

    # 滚动窗口生成（关键修改）
    x_unfold = x.unfold(0, d, 1)               # [T-d+1, N, d, F]
    y_unfold = y.unfold(0, d, 1).unsqueeze(-1) # [T-d+1, N, d, 1]

    # 去均值计算
    x_mean = torch.nanmean(x_unfold, dim=2, keepdim=True)  # [T-d+1, N, 1, F]
    y_mean = torch.nanmean(y_unfold, dim=2, keepdim=True)  # [T-d+1, N, 1, 1]

    x_demean = x_unfold - x_mean  # [T-d+1, N, d, F]
    y_demean = y_unfold - y_mean  # [T-d+1, N, d, 1]

    # 矩阵运算（修正后的维度对齐）
    xTx = torch.matmul(x_demean.transpose(2,3), x_demean)  # [T-d+1, N, F, F] 
    xTy = torch.matmul(x_demean.transpose(2,3), y_demean)  # [T-d+1, N, F, 1]

    # 后续计算保持不变
    beta = torch.linalg.pinv(xTx) @ xTy
    return beta.squeeze(-1).mean(dim=-1)
# ts_multiregbeta更新尚未完成，列表读取多个基础字段待完善

def ts_multireg_2d(
    x1: torch.Tensor,
    x2: torch.Tensor,
    y: torch.Tensor,
    d: int, ret_type: Literal[0, 1, 2, 3]
) -> torch.Tensor:
    """
    y~(x1, x2)做带截距的线性回归
    Params: 
        - x1, x2, y 为(T, N) tensor, T为时间序列长度, N为股票数
        - d为滚动窗口长度
        - ret_type为0, 1, 2, 3分别返回截距, beta1, beta2, 残差
    Return: (T, N) Tensor, 前d个以及矩阵不可逆填充Nan
    """

    raise NotImplementedError  # 不是 为什么跑不了啊？

    assert x1.shape == x2.shape == y.shape
    T, N = x1.shape

    # 确保使用浮点类型 否则无法求逆与存储NaN
    x1 = x1.float()
    x2 = x2.float()
    y = y.float()

    ones = torch.ones_like(x1)
    X_all = torch.stack([ones, x1, x2], dim=-1)  # (T, N, 3)

    X_win = X_all.unfold(0, d, 1).permute(0, 1, 3, 2)  # (T-d+1, N, d, 3)
    y_win = y.unfold(0, d, 1)                          # (T-d+1, N, d)

    Xt = X_win.transpose(2, 3)                        # (T-d+1, N, 3, d)
    XtX = Xt @ X_win                                  # (T-d+1, N, 3, 3)
    Xty = Xt @ y_win.unsqueeze(-1)                    # (T-d+1, N, 3, 1)

    result = torch.full((T, N), float('nan'), device=x1.device)

    try:
        beta = torch.linalg.solve(XtX, Xty).squeeze(-1)  # (T-d+1, N, 3)
    except RuntimeError:
        # Use pinv fallback if singular matrices are likely
        beta = torch.linalg.pinv(XtX) @ Xty
        beta = beta.squeeze(-1)

    if ret_type in [0, 1, 2]:
        result[d-1:] = beta[:, :, ret_type]
    elif ret_type == 3:
        x1_last = x1[d-1:]
        x2_last = x2[d-1:]
        y_last = y[d-1:]
        y_hat = beta[:, :, 0] + beta[:, :, 1] * x1_last + beta[:, :, 2] * x2_last
        result[d-1:] = y_last - y_hat

    return result

def ts_multiregsse(y: torch.Tensor, d: int, *xs) -> torch.Tensor:

    x = torch.stack(xs,dim = 1)  

    x_unfold = fill_with_mean(_unfold(x, d))
    y_unfold = fill_with_mean(_unfold(y, d)).unsqueeze(1)  

    x_demean_T = (x_unfold - torch.nanmean(x_unfold, dim=-1, keepdim=True)).transpose(1, 2)
    y_demean_T = (y_unfold - torch.nanmean(y_unfold, dim=-1, keepdim=True)).transpose(1, 2)

    xTx = torch.matmul(x_demean_T, x_demean_T.transpose(2, 3))  
    xTy = torch.matmul(x_demean_T, y_demean_T.transpose(2, 3)) 

    beta = torch.linalg.pinv(xTx) @ xTy  
    y_hat =  (x_demean_T.transpose(2, 3) @ beta).squeeze(-1)

    return torch.pow(y_demean_T.transpose(2, 3).squeeze(-1) - y_hat, 2).nansum(dim=-1) 


def ts_multiregrsquare(y: torch.Tensor, d: int, *xs) -> torch.Tensor:

    x = torch.stack(xs,dim = 1)  

    x_unfold = fill_with_mean(_unfold(x, d))
    y_unfold = fill_with_mean(_unfold(y, d)).unsqueeze(1)

    x_demean_T = (x_unfold - torch.nanmean(x_unfold, dim=-1, keepdim=True)).transpose(1, 2)
    y_demean_T = (y_unfold - torch.nanmean(y_unfold, dim=-1, keepdim=True)).transpose(1, 2)

    xTx = torch.matmul(x_demean_T, x_demean_T.transpose(2, 3))  
    xTy = torch.matmul(x_demean_T, y_demean_T.transpose(2, 3)) 

    beta = torch.linalg.pinv(xTx) @ xTy  
    y_hat =  (x_demean_T.transpose(2, 3) @ beta).squeeze(-1)
    sse = torch.pow(y_demean_T.transpose(2, 3).squeeze(-1) - y_hat, 2).nansum(dim=-1) 
    return 1 - div(sse, torch.pow(y_demean_T.transpose(2, 3).squeeze(-1), 2).nansum(dim=-1))


def ts_multiregfvalue(y: torch.Tensor, d: int, *xs) -> torch.Tensor:

    x = torch.stack(xs,dim = 1)  

    x_unfold = fill_with_mean(_unfold(x, d))
    y_unfold = fill_with_mean(_unfold(y, d)).unsqueeze(1)

    x_demean_T = (x_unfold - torch.nanmean(x_unfold, dim=-1, keepdim=True)).transpose(1, 2)
    y_demean_T = (y_unfold - torch.nanmean(y_unfold, dim=-1, keepdim=True)).transpose(1, 2)

    xTx = torch.matmul(x_demean_T, x_demean_T.transpose(2, 3))  
    xTy = torch.matmul(x_demean_T, y_demean_T.transpose(2, 3)) 

    beta = torch.linalg.pinv(xTx) @ xTy  
    y_hat =  (x_demean_T.transpose(2, 3) @ beta).squeeze(-1)
    sse = torch.pow(y_demean_T.transpose(2, 3).squeeze(-1) - y_hat, 2).nansum(dim=-1) 
    r2 = 1 - div(sse, torch.pow(y_demean_T.transpose(2, 3).squeeze(-1), 2).nansum(dim=-1))
    return div(r2 / len(xs), (1 - r2) / (d - len(xs) - 1))

# ts_gmm_2ndmean (optimized version)
def ts_gmm_2ndmean(x: torch.Tensor, d: int, em_iters: int = 10) -> torch.Tensor:
    """
    Use a sliding window to fit a 2-component GMM (based on the EM algorithm) 
    to the time series data of each stock, and return the short-term price jump mean 
    (i.e., the mean of the second component).
    
    Parameters:
        x: Input tensor of shape [num_minutes, num_stocks]
        d: Sliding window size
        em_iters: Number of EM algorithm iterations (default is 10, can be adjusted as needed)
    
    Returns:
        res: A tensor with the same shape as x, where only the last row of each window 
             contains the GMM result for that window, and all other values are NaN.
    """
    res = get_nan_tensor(shape=x.shape, device=x.device, dtype=x.dtype)
    if x.shape[0] < d:
        return res

    # 1. Extract sliding windows using unfold, shape will be [num_windows, num_stocks, d]
    windows = x.unfold(dimension=0, size=d, step=1)
    num_windows, n_stocks, _ = windows.shape
    B = num_windows * n_stocks
    # Reshape windows to 2D: each row represents d time points for a stock in a window
    windows = windows.reshape(B, d)  # shape: [B, d]

    # 2. Handle NaN values on GPU: for each window, fill NaNs with the mean of that window
    # Compute the mean for each row (ignoring NaNs), NaN will be returned if all values are NaN
    window_means = torch.nanmean(windows, dim=1, keepdim=True)  # shape: [B, 1]
    windows_filled = torch.where(torch.isnan(windows), window_means, windows)  # [B, d]

    # 3. Initialize parameters for the 2-component GMM (processing each window independently)
    # Initial means: the first component gets the median, the second component gets the 80th percentile
    medians = torch.median(windows_filled, dim=1).values  # shape: [B]
    # !! prior knowledge: the short-term price jump may be the 80th percentile, which can be adjusted
    perc80 = torch.quantile(windows_filled, 0.8, dim=1)     # shape: [B]
    var_init = torch.var(windows_filled, dim=1, unbiased=False)  # shape: [B]
    var_init = var_init.clamp(min=1e-6)  # Prevent zero variance

    # All parameters are of shape [B, 2]
    means = torch.stack([medians, perc80], dim=1) # Initial means
    variances = var_init.unsqueeze(1).repeat(1, 2) # Initial variances
    weights = torch.full((B, 2), 0.5, dtype=x.dtype, device=x.device) # Equal weights

    # Expand window data to [B, d, 1] for later broadcasting
    X = windows_filled.unsqueeze(2)  # shape: [B, d, 1]

    # 4. EM algorithm iterations (all parallelized on GPU)
    for _ in range(em_iters):
        # E-step: Compute posterior probabilities (responsibilities)
        # Expand means and variances for broadcasting
        mu = means.unsqueeze(1)         # [B, 1, 2]
        sigma2 = variances.unsqueeze(1)   # [B, 1, 2]
        # Compute the Gaussian distribution coefficients
        coef = 1.0 / torch.sqrt(2 * torch.pi * sigma2)  # [B, 1, 2]
        # Compute the exponential part
        exp_term = torch.exp(-0.5 * ((X - mu) ** 2) / sigma2)  # [B, d, 2]
        likelihood = coef * exp_term  # [B, d, 2]
        # Weight by the mixture weights, note that weights need to be expanded
        weighted_likelihood = weights.unsqueeze(1) * likelihood  # [B, d, 2]
        # Normalize by summing across the components, compute the denominator for posterior probabilities
        normalizer = weighted_likelihood.sum(dim=2, keepdim=True)  # [B, d, 1]
        # Compute responsibilities, add 1e-10 to avoid division by zero
        resp = weighted_likelihood / (normalizer + 1e-10)  # [B, d, 2]

        # M-step: Update parameters
        # Calculate the responsibility sums for each component: [B, 2]
        resp_sum = resp.sum(dim=1)
        # Update means: weighted average
        means = (resp * X).sum(dim=1) / (resp_sum + 1e-10)  # [B, 2]
        # Update variances: weighted variance
        variances = (resp * (X - means.unsqueeze(1))**2).sum(dim=1) / (resp_sum + 1e-10)  # [B, 2]
        # Update weights: average responsibility for each component
        weights = resp_sum / d  # [B, 2]

    # 5. Take the mean of the second component as the short-term price jump mean
    second_component_mean = means[:, 1]  # [B]
    # Reshape to [num_windows, num_stocks]
    gmm_means = second_component_mean.reshape(num_windows, n_stocks)

    # 6. Place the computed result into res, with the last time point of each window filled with the GMM result
    res[d-1:, :] = gmm_means.to(x.device)
    return res

# ts_gmm_1stmean (optimized version)
def ts_gmm_1stmean(x: torch.Tensor, d: int, em_iters: int = 10) -> torch.Tensor:
    """
    Use a sliding window to fit a 2-component GMM (based on the EM algorithm) 
    to the time series data of each stock, and return the long-term stable mean 
    (i.e., the mean of the first component).
    
    Parameters:
        x: Input tensor of shape [num_minutes, num_stocks]
        d: Sliding window size
        em_iters: Number of EM algorithm iterations (default is 10, can be adjusted as needed)
    
    Returns:
        res: A tensor with the same shape as x, where only the last row of each window 
             contains the GMM result for that window, and all other values are NaN.
    """
    res = get_nan_tensor(shape=x.shape, device=x.device, dtype=x.dtype)
    if x.shape[0] < d:
        return res

    # 1. Extract sliding windows using unfold, shape will be [num_windows, num_stocks, d]
    windows = x.unfold(dimension=0, size=d, step=1)
    num_windows, n_stocks, _ = windows.shape
    B = num_windows * n_stocks
    # Reshape windows to 2D: each row represents d time points for a stock in a window
    windows = windows.reshape(B, d)  # shape: [B, d]

    # 2. Handle NaN values on GPU: for each window, fill NaNs with the mean of that window
    # Compute the mean for each row (ignoring NaNs), NaN will be returned if all values are NaN
    window_means = torch.nanmean(windows, dim=1, keepdim=True)  # shape: [B, 1]
    windows_filled = torch.where(torch.isnan(windows), window_means, windows)  # [B, d]

    # 3. Initialize parameters for the 2-component GMM (processing each window independently)
    # Initial means: the first component gets the median, the second component gets the 80th percentile
    medians = torch.median(windows_filled, dim=1).values  # shape: [B]
    # !! prior knowledge: the short-term price jump may be the 80th percentile, which can be adjusted
    perc80 = torch.quantile(windows_filled, 0.8, dim=1)     # shape: [B]
    var_init = torch.var(windows_filled, dim=1, unbiased=False)  # shape: [B]
    var_init = var_init.clamp(min=1e-6)  # Prevent zero variance

    # All parameters are of shape [B, 2]
    means = torch.stack([medians, perc80], dim=1) # Initial means
    variances = var_init.unsqueeze(1).repeat(1, 2) # Initial variances
    weights = torch.full((B, 2), 0.5, dtype=x.dtype, device=x.device) # Equal weights

    # Expand window data to [B, d, 1] for later broadcasting
    X = windows_filled.unsqueeze(2)  # shape: [B, d, 1]

    # 4. EM algorithm iterations (all parallelized on GPU)
    for _ in range(em_iters):
        # E-step: Compute posterior probabilities (responsibilities)
        # Expand means and variances for broadcasting
        mu = means.unsqueeze(1)         # [B, 1, 2]
        sigma2 = variances.unsqueeze(1)   # [B, 1, 2]
        # Compute the Gaussian distribution coefficients
        coef = 1.0 / torch.sqrt(2 * torch.pi * sigma2)  # [B, 1, 2]
        # Compute the exponential part
        exp_term = torch.exp(-0.5 * ((X - mu) ** 2) / sigma2)  # [B, d, 2]
        likelihood = coef * exp_term  # [B, d, 2]
        # Weight by the mixture weights, note that weights need to be expanded
        weighted_likelihood = weights.unsqueeze(1) * likelihood  # [B, d, 2]
        # Normalize by summing across the components, compute the denominator for posterior probabilities
        normalizer = weighted_likelihood.sum(dim=2, keepdim=True)  # [B, d, 1]
        # Compute responsibilities, add 1e-10 to avoid division by zero
        resp = weighted_likelihood / (normalizer + 1e-10)  # [B, d, 2]

        # M-step: Update parameters
        # Calculate the responsibility sums for each component: [B, 2]
        resp_sum = resp.sum(dim=1)
        # Update means: weighted average
        means = (resp * X).sum(dim=1) / (resp_sum + 1e-10)  # [B, 2]
        # Update variances: weighted variance
        variances = (resp * (X - means.unsqueeze(1))**2).sum(dim=1) / (resp_sum + 1e-10)  # [B, 2]
        # Update weights: average responsibility for each component
        weights = resp_sum / d  # [B, 2]

    # 5. Take the mean of the first component as the long-term stable mean
    first_component_mean = means[:, 0]  # [B]
    # Reshape to [num_windows, num_stocks]
    gmm_means = first_component_mean.reshape(num_windows, n_stocks)

    # 6. Place the computed result into res, with the last time point of each window filled with the GMM result
    res[d-1:, :] = gmm_means.to(x.device)
    return res

# Obtain the long-term stable standard deviation (optimized version)
def ts_gmm_1ststd(x: torch.Tensor, d: int, em_iters: int = 10) -> torch.Tensor:
    """
    Use a sliding window to fit a 2-component GMM (based on the EM algorithm) 
    to the time series data of each stock, and return the long-term stable standard deviation 
    (i.e., the standard deviation of the first component).
    
    Parameters:
        x: Input tensor of shape [num_minutes, num_stocks]
        d: Sliding window size
        em_iters: Number of EM algorithm iterations (default is 10, can be adjusted as needed)
    
    Returns:
        res: A tensor with the same shape as x, where only the last row of each window 
             contains the GMM result for that window, and all other values are NaN.
    """
    res = get_nan_tensor(shape=x.shape, device=x.device, dtype=x.dtype)
    if x.shape[0] < d:
        return res

    # 1. Extract sliding windows using unfold, shape will be [num_windows, num_stocks, d]
    windows = x.unfold(dimension=0, size=d, step=1)
    num_windows, n_stocks, _ = windows.shape
    B = num_windows * n_stocks
    # Reshape windows to 2D: each row represents d time points for a stock in a window
    windows = windows.reshape(B, d)  # shape: [B, d]

    # 2. Handle NaN values on GPU: for each window, fill NaNs with the mean of that window
    window_means = torch.nanmean(windows, dim=1, keepdim=True)  # shape: [B, 1]
    windows_filled = torch.where(torch.isnan(windows), window_means, windows)  # [B, d]

    # 3. Initialize parameters for the 2-component GMM (processing each window independently)
    medians = torch.median(windows_filled, dim=1).values  # shape: [B]
    perc80 = torch.quantile(windows_filled, 0.8, dim=1)     # shape: [B]
    var_init = torch.var(windows_filled, dim=1, unbiased=False)  # shape: [B]
    var_init = var_init.clamp(min=1e-6)  # Prevent zero variance

    # All parameters are of shape [B, 2]
    means = torch.stack([medians, perc80], dim=1)  # Initial means
    variances = var_init.unsqueeze(1).repeat(1, 2)  # Initial variances
    weights = torch.full((B, 2), 0.5, dtype=x.dtype, device=x.device)  # Equal weights

    # Expand window data to [B, d, 1] for later broadcasting
    X = windows_filled.unsqueeze(2)  # shape: [B, d, 1]

    # 4. EM algorithm iterations (all parallelized on GPU)
    for _ in range(em_iters):
        # E-step: Compute posterior probabilities (responsibilities)
        mu = means.unsqueeze(1)         # [B, 1, 2]
        sigma2 = variances.unsqueeze(1)   # [B, 1, 2]
        coef = 1.0 / torch.sqrt(2 * torch.pi * sigma2)  # [B, 1, 2]
        exp_term = torch.exp(-0.5 * ((X - mu) ** 2) / sigma2)  # [B, d, 2]
        likelihood = coef * exp_term  # [B, d, 2]
        weighted_likelihood = weights.unsqueeze(1) * likelihood  # [B, d, 2]
        normalizer = weighted_likelihood.sum(dim=2, keepdim=True)  # [B, d, 1]
        resp = weighted_likelihood / (normalizer + 1e-10)  # [B, d, 2]

        # M-step: Update parameters
        resp_sum = resp.sum(dim=1)
        means = (resp * X).sum(dim=1) / (resp_sum + 1e-10)  # [B, 2]
        variances = (resp * (X - means.unsqueeze(1))**2).sum(dim=1) / (resp_sum + 1e-10)  # [B, 2]
        weights = resp_sum / d  # [B, 2]

    # 5. Take the standard deviation of the first component as the long-term stable standard deviation
    first_component_std = torch.sqrt(variances[:, 0])  # [B]
    gmm_stds = first_component_std.reshape(num_windows, n_stocks)

    # 6. Place the computed result into res, with the last time point of each window filled with the GMM result
    res[d-1:, :] = gmm_stds.to(x.device)
    return res

# Obtain the short-term price jump standard deviation (optimized version)
def ts_gmm_2ndstd(x: torch.Tensor, d: int, em_iters: int = 10) -> torch.Tensor:
    """
    Use a sliding window to fit a 2-component GMM (based on the EM algorithm) 
    to the time series data of each stock, and return the short-term price jump standard deviation 
    (i.e., the standard deviation of the second component).
    
    Parameters:
        x: Input tensor of shape [num_minutes, num_stocks]
        d: Sliding window size
        em_iters: Number of EM algorithm iterations (default is 10, can be adjusted as needed)
    
    Returns:
        res: A tensor with the same shape as x, where only the last row of each window 
             contains the GMM result for that window, and all other values are NaN.
    """
    res = get_nan_tensor(shape=x.shape, device=x.device, dtype=x.dtype)
    if x.shape[0] < d:
        return res

    # 1. Extract sliding windows using unfold, shape will be [num_windows, num_stocks, d]
    windows = x.unfold(dimension=0, size=d, step=1)
    num_windows, n_stocks, _ = windows.shape
    B = num_windows * n_stocks
    # Reshape windows to 2D: each row represents d time points for a stock in a window
    windows = windows.reshape(B, d)  # shape: [B, d]

    # 2. Handle NaN values on GPU: for each window, fill NaNs with the mean of that window
    window_means = torch.nanmean(windows, dim=1, keepdim=True)  # shape: [B, 1]
    windows_filled = torch.where(torch.isnan(windows), window_means, windows)  # [B, d]

    # 3. Initialize parameters for the 2-component GMM (processing each window independently)
    medians = torch.median(windows_filled, dim=1).values  # shape: [B]
    perc80 = torch.quantile(windows_filled, 0.8, dim=1)     # shape: [B]
    var_init = torch.var(windows_filled, dim=1, unbiased=False)  # shape: [B]
    var_init = var_init.clamp(min=1e-6)  # Prevent zero variance

    # All parameters are of shape [B, 2]
    means = torch.stack([medians, perc80], dim=1)  # Initial means
    variances = var_init.unsqueeze(1).repeat(1, 2)  # Initial variances
    weights = torch.full((B, 2), 0.5, dtype=x.dtype, device=x.device)  # Equal weights

    # Expand window data to [B, d, 1] for later broadcasting
    X = windows_filled.unsqueeze(2)  # shape: [B, d, 1]

    # 4. EM algorithm iterations (all parallelized on GPU)
    for _ in range(em_iters):
        # E-step: Compute posterior probabilities (responsibilities)
        mu = means.unsqueeze(1)         # [B, 1, 2]
        sigma2 = variances.unsqueeze(1)   # [B, 1, 2]
        coef = 1.0 / torch.sqrt(2 * torch.pi * sigma2)  # [B, 1, 2]
        exp_term = torch.exp(-0.5 * ((X - mu) ** 2) / sigma2)  # [B, d, 2]
        likelihood = coef * exp_term  # [B, d, 2]
        weighted_likelihood = weights.unsqueeze(1) * likelihood  # [B, d, 2]
        normalizer = weighted_likelihood.sum(dim=2, keepdim=True)  # [B, d, 1]
        resp = weighted_likelihood / (normalizer + 1e-10)  # [B, d, 2]

        # M-step: Update parameters
        resp_sum = resp.sum(dim=1)
        means = (resp * X).sum(dim=1) / (resp_sum + 1e-10)  # [B, 2]
        variances = (resp * (X - means.unsqueeze(1))**2).sum(dim=1) / (resp_sum + 1e-10)  # [B, 2]
        weights = resp_sum / d  # [B, 2]

    # 5. Take the standard deviation of the second component as the short-term price jump standard deviation
    second_component_std = torch.sqrt(variances[:, 1])  # [B]
    gmm_stds = second_component_std.reshape(num_windows, n_stocks)

    # 6. Place the computed result into res, with the last time point of each window filled with the GMM result
    res[d-1:, :] = gmm_stds.to(x.device)
    return res

def ts_min_max_norm(x, d):
    """
    对二维时间序列 x 进行滑动窗口 min-max 归一化（逐列计算），不填补 NaN。

    参数：
    x: 2D Tensor, 形状 (N, M)，其中 N 是时间步数，M 是特征数量
    d: 滑动窗口大小

    返回：
    res: 2D Tensor, 形状 (N, M)，前 d-1 行会被自动计算为首个窗口的归一化值
    """
    N, M = x.shape  

    x_unfold = x.unfold(0, d, 1)  

    x_min = torch.amin(x_unfold, dim=1)  
    x_max = torch.amax(x_unfold, dim=1) 

    x_last = x_unfold[:, -1, :]  

    norm = (x_last - x_min) / (x_max - x_min)
    norm[x_max == x_min] = 0.5 

    res = torch.cat([norm[:1].repeat(d-1, 1), norm], dim=0)  

    return res



def ts_autocorr_reg(x: torch.Tensor, d: int, k: int, beta: bool=True) -> torch.Tensor:
    '''
    获取各股票各时点1-k阶自相关系数对ln1-lnk回归的斜率或截距
    '''
    assert len(x) - k >= d, "The order of autocorrelation can only be len(x)-d."
    assert d >= 10*k, "To calculate k-order autocorrelation, it is advisable to use at least 10*k samples."
    assert k > 2, "OLS needs more than 2 samples."
    res = get_nan_tensor(shape=x.shape, device=x.device, dtype=x.dtype)
    tensor_lis = []
    # 求各时点的1-k阶自相关系数
    for i in range(1,k+1):
        tensor_lis.append(ts_autocorr(x, d, i)[k+d-len(x):])
    # 用1-k阶自相关系数对ln1-lnk回归
    Y = torch.stack(tensor_lis, dim=-1)
    X = torch.log(torch.arange(1,k+1,device=x.device, dtype=x.dtype)).view(1,1,k).repeat(Y.shape[0], Y.shape[1], 1)
    X_mean = X.nanmean(dim=-1, keepdim=True)
    Y_mean = Y.nanmean(dim=-1, keepdim=True)
    
    X_demean = X - X_mean
    Y_demean = Y - Y_mean
    std = torch.nansum(torch.pow(X_demean, 2), dim=-1)
    res[k+d-len(x):] = torch.where(std < EPS, torch.nan, torch.nansum(X_demean * Y_demean, dim=-1) / std)
    if beta:
        return res
    else:
        alpha = get_nan_tensor(shape=x.shape, device=x.device, dtype=x.dtype)
        alpha[k+d-len(x):] = Y.nanmean(dim=-1) - res[k+d-len(x):] * X.nanmean(dim=-1)
        return alpha


def ts_norm2interval(x: torch.Tensor, d: int, up: int, down: int) -> torch.Tensor:
    """滚动窗口标准化至特定区间

    :param torch.Tensor x: 形状为(T,N), T为时间步数, N为股票数量
    :param int d: 滚动窗口长度
    :param int up: 标准化后区间的上界
    :param int down: 标准化后区间的下界
    :return: 标准化后的张量, 形状为(T,N)
    :rtype: torch.Tensor
    """
    res = get_nan_tensor(shape=x.shape, device=x.device, dtype=x.dtype)
    unfolded = x.unfold(0, d, 1)
    # unfolded = torch.where(unfolded.isinf() | unfolded.isnan(), torch.nanmean(unfolded, dim=-1, keepdim=True), unfolded)
    mask = unfolded.isnan()
    mins = torch.amin(unfolded.masked_fill(mask, float('inf')), dim=-1)
    maxes = torch.amax(unfolded.masked_fill(mask, float('-inf')), dim=-1)
    res[d-1:] = (up - down) * (x[d-1:] - mins) / (maxes - mins) + down
    return res

#在过去d个时间步内，根据每期y较前一期的变化判断x的累加情况（y较前一期大，上一期x加本期x；y较前一期小，上一期x减本期x；y较前一期不变，上一期x加0）
def ts_obp(x: torch.Tensor, y: torch.Tensor, d: int) -> torch.Tensor:
    assert x.shape == y.shape, "x and y must have the same shape."
    assert d > 0 and d < x.shape[0], "Window size must be positive and less than the time steps."

    # 初始化结果张量
    res = torch.full(x.shape, float('nan'), device=x.device, dtype=x.dtype)
    # 使用 unfold 提取滑动窗口
    x_unfold = x.unfold(0, d, 1) 
    y_unfold = y.unfold(0, d, 1) 
    # 获取前一个元素和当前元素
    y_prev = y_unfold[:, :, :-1]
    y_curr = y_unfold[:, :, 1:]
    x_curr = x_unfold[:, :, 1:]
    # 条件掩码
    less_mask = y_prev < y_curr # 当前元素大于前一个元素
    greater_mask = y_prev > y_curr # 当前元素小于前一个元素

    x_modified = torch.where(less_mask, x_curr, torch.where(greater_mask, -x_curr, torch.zeros_like(x_curr)))
    row_sum = x_unfold[:, :, 0] + x_modified.sum(dim=2)
    res[d - 1:, :] = row_sum
    return res


#在过去d个时间步内，根据每期y较前一期的变化判断x的累减情况（y较前一期大，上一期x减本期x；y较前一期小，上一期x加本期x；y较前一期不变，上一期x加0）
def ts_obn(x: torch.Tensor, y: torch.Tensor, d: int) -> torch.Tensor:
    assert x.shape == y.shape, "x and y must have the same shape."
    assert d > 0 and d < x.shape[0], "Window size must be positive and less than the time steps."

    # 初始化结果张量
    res = torch.full(x.shape, float('nan'), device=x.device, dtype=x.dtype)
    # 使用 unfold 提取滑动窗口
    x_unfold = x.unfold(0, d, 1) 
    y_unfold = y.unfold(0, d, 1) 
    # 获取前一个元素和当前元素
    y_prev = y_unfold[:, :, :-1]
    y_curr = y_unfold[:, :, 1:]
    x_curr = x_unfold[:, :, 1:]
    # 条件掩码
    less_mask = y_prev < y_curr # 当前元素大于前一个元素
    greater_mask = y_prev > y_curr # 当前元素小于前一个元素

    x_modified = torch.where(less_mask, -x_curr, torch.where(greater_mask, x_curr, torch.zeros_like(x_curr)))
    row_sum = x_unfold[:, :, 0] + x_modified.sum(dim=2)
    res[d - 1:, :] = row_sum
    return res


#在过去d个时间步内，根据每期y较前一期的变化判断x的正向累乘情况（y较前一期大，上一期累积值乘以本期x再除以上一期x得到新累积值；y较前一期小或等于，上一期累积值乘1得到新累积值）
def ts_obpm(x: torch.Tensor, y: torch.Tensor, d: int) -> torch.Tensor:
    assert x.shape == y.shape, "x and y must have the same shape."
    assert d > 0 and d < x.shape[0], "Window size must be positive and less than the time steps."

    # 初始化结果张量
    res = torch.full(x.shape, float('nan'), device=x.device, dtype=x.dtype)
    # 使用 unfold 提取滑动窗口
    x_unfold = x.unfold(0, d, 1) 
    y_unfold = y.unfold(0, d, 1) 
    # 获取前一个元素和当前元素
    y_prev = y_unfold[:, :, :-1]
    y_curr = y_unfold[:, :, 1:]
    x_curr = x_unfold[:, :, 1:]
    x_curr2 = x_unfold[:, :, :-1]
    # 条件掩码
    less_mask = y_prev < y_curr # 当前元素大于前一个元素
    greater_mask = y_prev > y_curr # 当前元素小于前一个元素

    x_modified = x_curr / x_curr2
    x_modified2 = torch.where(less_mask, x_modified,  torch.ones_like(x_curr))

    row_sum = x_unfold[:, :, 0] * x_modified2.prod(dim=2)
    res[d - 1:, :] = row_sum
    return res

#在过去d个时间步内，根据每期y较前一期的变化判断x的负向累乘情况（y较前一期大或等于，上一期累积值乘1得到新累积值；y较前一期小，上一期累积值乘以本期x除以上一期x得到新累积值）
def ts_obnm(x: torch.Tensor, y: torch.Tensor, d: int) -> torch.Tensor:
    assert x.shape == y.shape, "x and y must have the same shape."
    assert d > 0 and d < x.shape[0], "Window size must be positive and less than the time steps."

    # 初始化结果张量
    res = torch.full(x.shape, float('nan'), device=x.device, dtype=x.dtype)
    # 使用 unfold 提取滑动窗口
    x_unfold = x.unfold(0, d, 1) 
    y_unfold = y.unfold(0, d, 1) 
    # 获取前一个元素和当前元素
    y_prev = y_unfold[:, :, :-1]
    y_curr = y_unfold[:, :, 1:]
    x_curr = x_unfold[:, :, 1:]
    x_curr2 = x_unfold[:, :, :-1]
    # 条件掩码
    less_mask = y_prev < y_curr # 当前元素大于前一个元素
    greater_mask = y_prev > y_curr # 当前元素小于前一个元素

    x_modified = x_curr / x_curr2
    x_modified2 = torch.where(greater_mask, x_modified,  torch.ones_like(x_curr))

    row_sum = x_unfold[:, :, 0] * x_modified2.prod(dim=2)
    res[d - 1:, :] = row_sum
    return res


def ts_cond_quantile_mean(x: torch.Tensor, y: torch.Tensor, d: int, low_q: float, high_q: float) -> torch.Tensor:
    assert x.shape == y.shape, "x and y must have the same shape."

    res = torch.full(x.shape, float('nan'), device=x.device, dtype=x.dtype)

    # 滑动窗口展开
    x_unfold = x.unfold(0, d, 1)
    y_unfold = y.unfold(0, d, 1)
    
    # 计算每个窗口内x的分位数排名
    x_ranks = _nanrank(x_unfold)
    
    # 创建筛选掩码
    mask = (x_ranks >= low_q) & (x_ranks <= high_q)
    
    # 在y上应用掩码，不满足条件的设为NaN
    y_masked = y_unfold.clone()
    y_masked[~mask] = torch.nan
    
    # 计算每列非NaN值的均值 (沿窗口维度)
    res[d - 1:, :] = y_masked.nanmean(dim=-1)
    
    return res

def ts_cond_quantile_stddev(x: torch.Tensor, y: torch.Tensor, d: int, low_q: float, high_q: float) -> torch.Tensor:
    assert x.shape == y.shape, "x and y must have the same shape."

    res = torch.full(x.shape, float('nan'), device=x.device, dtype=x.dtype)
    
    # 滑动窗口展开
    x_unfold = x.unfold(0, d, 1)
    y_unfold = y.unfold(0, d, 1)
    
    # 计算每个窗口内x的分位数排名
    x_ranks = _nanrank(x_unfold)
    
    # 创建筛选掩码
    mask = (x_ranks >= low_q) & (x_ranks <= high_q)
    
    # 在y上应用掩码，不满足条件的设为NaN
    y_masked = y_unfold.clone()
    y_masked[~mask] = torch.nan
    
    # 计算每列非NaN值的均值 (沿窗口维度)
    res[d - 1:, :] = _stddev(y_masked)
    
    return res

def ts_cond_quantile_skew(x: torch.Tensor, y: torch.Tensor, d: int, low_q: float, high_q: float) -> torch.Tensor:
    assert x.shape == y.shape, "x and y must have the same shape."

    res = torch.full(x.shape, float('nan'), device=x.device, dtype=x.dtype)
    
    # 滑动窗口展开
    x_unfold = x.unfold(0, d, 1)
    y_unfold = y.unfold(0, d, 1)
    
    # 计算每个窗口内x的分位数排名
    x_ranks = _nanrank(x_unfold)
    
    # 创建筛选掩码
    mask = (x_ranks >= low_q) & (x_ranks <= high_q)
    
    # 在y上应用掩码，不满足条件的设为NaN
    y_masked = y_unfold.clone()
    y_masked[~mask] = torch.nan
    
    # 计算每列非NaN值的均值 (沿窗口维度)
    res[d - 1:, :] = _skew(y_masked)
    
    return res

def ts_cond_quantile_kurt(x: torch.Tensor, y: torch.Tensor, d: int, low_q: float, high_q: float) -> torch.Tensor:
    assert x.shape == y.shape, "x and y must have the same shape."

    res = torch.full(x.shape, float('nan'), device=x.device, dtype=x.dtype)
    
    x_unfold = x.unfold(0, d, 1)
    y_unfold = y.unfold(0, d, 1)
    
    x_ranks = _nanrank(x_unfold)
    
    mask = (x_ranks >= low_q) & (x_ranks <= high_q)
    
    y_masked = y_unfold.clone()
    y_masked[~mask] = torch.nan
    
    res[d - 1:, :] = _kurt(y_masked)
    
    return res


def ts_cond_quantile_corr(x: torch.Tensor, y: torch.Tensor, d: int, low_q: float, high_q: float) -> torch.Tensor:
    assert x.shape == y.shape, "x and y must have the same shape."

    res = torch.full(x.shape, float('nan'), device=x.device, dtype=x.dtype)
    
    x_unfold = x.unfold(0, d, 1)
    y_unfold = y.unfold(0, d, 1)
    
    x_ranks = _nanrank(x_unfold)
    
    mask = (x_ranks >= low_q) & (x_ranks <= high_q)

    x_masked = x_unfold.clone()
    x_masked[~mask] = torch.nan
    y_masked = y_unfold.clone()
    y_masked[~mask] = torch.nan

    x_demean = x_masked - x_masked.nanmean(dim=-1, keepdim=True)
    y_demean = y_masked - y_masked.nanmean(dim=-1, keepdim=True)

    x_std = torch.sqrt(torch.nansum(torch.pow(x_demean, 2), dim=-1))
    y_std = torch.sqrt(torch.nansum(torch.pow(y_demean, 2), dim=-1))
    
    numerator = (x_demean * y_demean).nansum(dim=-1)
    denominator = x_std * y_std
    res[d - 1:] = numerator / denominator

    res[d - 1:][(x_std < EPS) | (y_std < EPS)] = torch.nan
    
    return res

def ts_cond_quantile_sum(x: torch.Tensor, y: torch.Tensor, d: int, low_q: float, high_q: float) -> torch.Tensor:
    assert x.shape == y.shape, "x and y must have the same shape."

    res = torch.full(x.shape, float('nan'), device=x.device, dtype=x.dtype)

    # 滑动窗口展开
    x_unfold = x.unfold(0, d, 1)
    y_unfold = y.unfold(0, d, 1)
    
    # 计算每个窗口内x的分位数排名
    x_ranks = _nanrank(x_unfold)
    
    # 创建筛选掩码
    mask = (x_ranks >= low_q) & (x_ranks <= high_q)
    
    # 在y上应用掩码，不满足条件的设为NaN
    y_masked = y_unfold.clone()
    y_masked[~mask] = torch.nan
    
    # 计算每列非NaN值的求和 (沿窗口维度)
    res[d - 1:, :] = torch.nansum(y_masked, dim=-1)
    
    return res


def ts_rs_Hurst(x: torch.Tensor, d: int)-> torch.Tensor:
    """
    修正算子试运行中,若无问题将调整原算子不足之处,20250327
    使用R/S重标极差法分析计算给定序列X以d步长作切分后,各步长内对应x子序列的的Hurst指数。
    修正测试中:已修正原算子，主要问题在于,采用原算子求回归需将空值进行填充,原算子采用均值填充时会取到未来数据导致结果有偏,目前算子采用对当天有数据的部分求Hurst指数
    参数:
        X (torch.Tensor): 输入的时间序列数据,d (int) 为需要计算的序列步长，
        注:d不能太短,以30min为例,是对每天作30min切分,对每30min的序列计算Hurst指数,如果太短则回归可能过拟合

        
    返回:
        Tensor: 计算得到的Hurst指数,每天每个d内都能计算出1个值,代表这个d时间范围内的Hurst水平
    """
    # 对不同长度子序列进行R/S分析
    # 相对旧算子，这里不采用填充
    res_final = get_nan_tensor(shape=x.shape, device=x.device, dtype=x.dtype)
    mask = ~torch.isnan(x).all(dim=1)
    # 使用掩码过滤张量
    x_inday= x[mask]
    len=int(x.shape[0]-x_inday.shape[0])
    unfolded=x_inday.unfold(0,d,d)

    data=unfolded
    for dimen in range(int(data.shape[0])):
        dt=data[dimen].t()
        res1=rs_Hurst(dt)
        res_final[(dimen*d+len):((dimen+1)*d+len),:]=res1
    isnan=torch.isnan(res_final[len:,:])
    res_final_inday=torch.where(isnan,res1[0],res_final[len:,:])
    res_final[len:,:]=res_final_inday
    res_final[res_final==0.0]=float('nan')
    return res_final

import builtins
from torch.nn.functional import pad

def ts_hawkes_pred_exp_fast(x: torch.Tensor, d: int) -> torch.Tensor:
    T, N = x.shape
    device = x.device
    
    # 处理全NaN情况
    if torch.isnan(x).all():
        return torch.zeros_like(x)
    
    # 替换NaN为0并确保float32类型
    x_no_nan = torch.where(torch.isnan(x), torch.tensor(0.0, device=device), x).float()
    
    # 初始化输出
    expectations = torch.zeros((T, N), device=device)
    
    if d <= 0 or T < 1:
        return expectations
    
    # 修正1：确保d不超过T
    d = builtins.min(d, T)
    
    # 正确填充方案
    padded_x = torch.nn.functional.pad(x_no_nan, (0, 0, d-1, 0), value=0.0)
    
    # 展开滑动窗口
    x_unfold = padded_x.unfold(0, d, 1)          # 形状 (T-d+1, N, d)
    x_unfold = x_unfold.permute(0, 2, 1)         # 形状 (T-d+1, d, N)
    
    # 计算统计量
    mu = x_unfold.mean(dim=1)                    # (T-d+1, N)
    alpha = x_unfold.var(dim=1)                  # (T-d+1, N)
    alpha = torch.clamp(alpha, min=1e-8, max=1e4)
    beta = 1.0 / (mu + 1e-8)
    beta = torch.clamp(beta, min=1e-8, max=1e4)
    
    # 构造时间差矩阵 (1, d, 1)
    time_diff = (torch.arange(d, 0, -1, device=device) - 1).float().view(1, d, 1)
    
    # 计算衰减权重
    decay_weights = alpha.unsqueeze(1) * torch.exp(-beta.unsqueeze(1) * time_diff)
    
    # 加权求和
    intensity = mu + (x_unfold * decay_weights).sum(dim=1)
    intensity = torch.clamp(intensity, min=0, max=1e4)
    
    # 修正2：确保赋值长度匹配
    valid_length = builtins.min(T - (d-1), intensity.shape[0])
    expectations[d-1:d-1+valid_length] = intensity[:valid_length]
    
    # 处理前d-1个点
    for i in range(d-1):
        if i >= T:
            break
        hist_data = x_no_nan[builtins.max(0,i-10):i] if i > 0 else x_no_nan[[i]]
        expectations[i] = hist_data.mean(dim=0)
    
    return expectations

def ts_hawkes_pred_powerlaw_fast(x: torch.Tensor, d: int, power: float = 1.5) -> torch.Tensor:
    """
    优化的幂律衰减核霍克斯过程预测
    
    Args:
        x: 输入事件计数张量，形状为 (T, N)
        d: 滑动窗口长度
        power: 幂律指数（需 >1）
    
    Returns:
        预测的事件强度张量，形状为 (T, N)
    """
    T, N = x.shape
    device = x.device
    
    # 处理全NaN情况
    if torch.isnan(x).all():
        return torch.zeros_like(x)
    
    # 替换NaN为0并确保float32类型
    x_no_nan = torch.where(torch.isnan(x), torch.tensor(0.0, device=device), x).float()
    
    # 初始化输出
    expectations = torch.zeros((T, N), device=device)
    
    if d <= 0 or T < 1:
        return expectations
    
    # 确保窗口不超过数据长度
    #d = min(d, T)
    
    # 填充初始数据
    padded_x = torch.nn.functional.pad(x_no_nan, (0, 0, d-1, 0), value=0.0)
    
    # 展开滑动窗口 (T-d+1, d, N)
    x_unfold = padded_x.unfold(0, d, 1).permute(0, 2, 1)
    
    # 计算统计量
    mu = x_unfold.mean(dim=1)                    # (T-d+1, N)
    alpha = x_unfold.var(dim=1)                  # (T-d+1, N)
    alpha = torch.clamp(alpha, min=1e-8, max=1e4)
    
    # 构造时间差矩阵 (1, d, 1)
    time_diff = (torch.arange(d, 0, -1, device=device) - 1 + 1.0).float().view(1, d, 1)
    
    # 计算幂律衰减权重 (T-d+1, d, N)
    decay_weights = alpha.unsqueeze(1) * torch.pow(time_diff, -power)
    
    # 加权求和
    intensity = mu + (x_unfold * decay_weights).sum(dim=1)
    intensity = torch.clamp(intensity, min=0, max=1e4)
    
    # 填充结果
    expectations[d-1:] = intensity[:T-d+1]
    
    # 处理前d-1个点（简化计算）
    for i in range(d-1):
        if i >= T:
            break
        hist_len = i
        if hist_len == 0:
            expectations[i] = x_no_nan[i]
        else:
            history = x_no_nan[:i]
            mu_partial = history.mean(dim=0)
            alpha_partial = history.var(dim=0)
            alpha_partial = torch.clamp(alpha_partial, min=1e-8, max=1e4)
            
            time_diff_partial = (torch.arange(hist_len, 0, -1, device=device) - 1 + 1.0).float().view(-1, 1)
            decay_weights_partial = alpha_partial * torch.pow(time_diff_partial, -power)
            
            expectations[i] = mu_partial + (history * decay_weights_partial).sum(dim=0)
    
    return expectations

def ts_count_above_threshold(x: torch.Tensor, d: int, f: float) -> torch.Tensor:
    """
    计算每个时间点前d分钟内x值大于阈值f的个数
    
    Args:
        x: 输入张量，形状为(T, N)，表示T个时间点N只股票的数值
        d: 滑动窗口长度
        f: 比较阈值
    
    Returns:
        输出张量，形状与x相同，每个位置值为前d分钟x>f的计数
    """
    T, N = x.shape
    device = x.device
    
    # 初始化输出
    result = torch.zeros_like(x, dtype=torch.float32)
    
    if d <= 0 or T < 1:
        return result
    
    # 处理NaN（视为不满足条件）
    x_no_nan = torch.where(torch.isnan(x), torch.tensor(-float('inf'), device=device), x)
    
    # 填充初始数据
    padded_x = torch.nn.functional.pad(x_no_nan, (0, 0, d-1, 0), value=-float('inf'))
    
    # 展开滑动窗口 (T, d, N)
    x_unfold = padded_x.unfold(0, d, 1).permute(0, 2, 1)
    
    # 计算每个窗口内x>f的个数
    counts = (x_unfold > f).sum(dim=1, dtype=torch.float32)
    
    # 赋值结果
    result[d-1:] = counts[:T-d+1]
    
    # 处理前d-1个时间点
    for i in range(d-1):
        if i >= T:
            break
        window = x_no_nan[builtins.max(0,i-d+1):i+1]  # 可用数据可能少于d个
        result[i] = (window > f).sum(dim=0, dtype=torch.float32)
    
    return result

def ts_ema(x: torch.Tensor, d: int) -> torch.Tensor:
    """
    计算单只股票的指数移动平均（Exponential Moving Average, EMA），
    支持前向填充（Forward Fill）处理空值（NaN）。
    
    参数:
    x (torch.Tensor): 单只股票的时间序列张量，形状为 [时间步长]
    d (int): 时间窗口长度（分钟），决定EMA的平滑周期
    
    返回:
    torch.Tensor: 与输入x形状相同的张量，前d-1分钟为NaN，从第d分钟开始为有效EMA值
    """
    if x.size(0) == 0:
        return torch.empty_like(x)
    
    # 前向填充（Forward Fill）处理空值
    x_filled = x.clone()
    for t in range(1, x.size(0)):
        if torch.isnan(x_filled[t]):
            x_filled[t] = x_filled[t-1]
    
    res = torch.full_like(x, float('nan'))  # 初始化为NaN
    
    alpha = 2 / (d + 1)  # 平滑系数
    
    # 从第d分钟开始计算
    for t in range(d-1, x.size(0)):
        if t == d-1:
            # 初始窗口的EMA等于第一个有效值（直接取填充后的数据）
            res[t] = x_filled[t]
        else:
            # 递归计算EMA：当前值 = alpha*新数据 + (1-alpha)*前值
            res[t] = alpha * x_filled[t] + (1 - alpha) * res[t-1]
    
    return res


def ts_ridgebeta(x: torch.Tensor, y: torch.Tensor, d: int, lambda_: float = 0.1) -> torch.Tensor:
    """
    使用滑动窗口计算岭回归系数。
    
    参数:
        x: 输入特征张量，形状为(T, N)，T为时间步数，N为股票数
        y: 目标变量张量，形状为(T, N)
        d: 滑动窗口大小
        lambda_: 正则化参数，默认为0.1
        
    返回:
        形状为(T, N)的回归系数张量
    """
    assert x.shape == y.shape, "x和y的形状必须相同"
    assert d > 2, "窗口大小必须大于2"
    
    res = torch.full(x.shape, torch.nan, dtype=x.dtype, device=x.device)
    
    # 滑动窗口展开
    x_unfold = x.unfold(0, d, 1).view(-1, d)  # (T-d+1)*N, d
    y_unfold = y.unfold(0, d, 1).view(-1, d)  # (T-d+1)*N, d
    
    # 处理缺失值
    x_mask = x_unfold.isnan() | x_unfold.isinf()
    y_mask = y_unfold.isnan() | y_unfold.isinf()
    x_mean = torch.nanmean(x_unfold, dim=-1, keepdim=True)
    y_mean = torch.nanmean(y_unfold, dim=-1, keepdim=True)
    
    x_unfold = torch.where(x_mask, x_mean, x_unfold)
    y_unfold = torch.where(y_mask, y_mean, y_unfold)
    
    # 中心化
    x_demean = (x_unfold - x_mean).unsqueeze(-1)  # (T-d+1)*N, d, 1
    y_demean = (y_unfold - y_mean).unsqueeze(-1)  # (T-d+1)*N, d, 1
    
    # 计算岭回归系数: beta = (X'X + λI)^(-1)X'y
    identity = torch.eye(d, device=x.device).unsqueeze(0).expand(x_demean.size(0), -1, -1)
    xTx = torch.matmul(x_demean.transpose(1, 2), x_demean) + lambda_ * identity
    xTy = torch.matmul(x_demean.transpose(1, 2), y_demean)
    beta = torch.matmul(torch.linalg.pinv(xTx), xTy)
    
    # 重塑结果
    res[d-1:] = beta.view(-1, x.shape[1]).squeeze(-1)
    
    return res


def ts_ridgepred(x: torch.Tensor, y: torch.Tensor, d: int, lambda_: float = 0.1) -> torch.Tensor:
    """
    使用岭回归进行预测。
    
    参数:
        x: 输入特征张量，形状为(T, N)，T为时间步数，N为股票数
        y: 目标变量张量，形状为(T, N)
        d: 滑动窗口大小
        lambda_: 正则化参数，默认为0.1
        
    返回:
        形状为(T, N)的预测值张量
    """
    assert x.shape == y.shape, "x和y的形状必须相同"
    assert d > 2, "窗口大小必须大于2"
    
    res = torch.full(x.shape, torch.nan, dtype=x.dtype, device=x.device)
    
    # 滑动窗口展开
    x_unfold = x.unfold(0, d, 1).view(-1, d)  # (T-d+1)*N, d
    y_unfold = y.unfold(0, d, 1).view(-1, d)  # (T-d+1)*N, d
    
    # 处理缺失值
    x_mask = x_unfold.isnan() | x_unfold.isinf()
    y_mask = y_unfold.isnan() | y_unfold.isinf()
    x_mean = torch.nanmean(x_unfold, dim=-1, keepdim=True)
    y_mean = torch.nanmean(y_unfold, dim=-1, keepdim=True)
    
    x_unfold = torch.where(x_mask, x_mean, x_unfold)
    y_unfold = torch.where(y_mask, y_mean, y_unfold)
    
    # 中心化
    x_demean = (x_unfold - x_mean).unsqueeze(-1)  # (T-d+1)*N, d, 1
    y_demean = (y_unfold - y_mean).unsqueeze(-1)  # (T-d+1)*N, d, 1
    
    # 计算岭回归系数: beta = (X'X + λI)^(-1)X'y
    identity = torch.eye(d, device=x.device).unsqueeze(0).expand(x_demean.size(0), -1, -1)
    xTx = torch.matmul(x_demean.transpose(1, 2), x_demean) + lambda_ * identity
    xTy = torch.matmul(x_demean.transpose(1, 2), y_demean)
    beta = torch.matmul(torch.linalg.pinv(xTx), xTy)
    
    # 计算预测值
    y_pred = torch.matmul(x_demean, beta)
    
    # 重塑结果
    res[d-1:] = y_pred.view(-1, x.shape[1]).squeeze(-1) + y_mean.squeeze(-1)
    
    return res


def ts_newey_west_se(x: torch.Tensor, y: torch.Tensor, d: int, lag: int = 4) -> torch.Tensor:
    assert x.shape == y.shape, "x and y shapes must be the same"
    assert 2 < d <= x.shape[0], "OLS lens must be greater than 2"
    assert lag >= 0, "lag must be a non-negative integer"

    # 计算回归系数和预测值
    x_unfold = x.unfold(0, d, 1).view(-1, d)  # 展开自变量
    y_unfold = y.unfold(0, d, 1).view(-1, d)  # 展开因变量

    x_mask = x_unfold.isnan() | x_unfold.isinf()
    y_mask = y_unfold.isnan() | y_unfold.isinf()
    x_mean = torch.nanmean(x_unfold, dim=-1, keepdim=True)
    y_mean = torch.nanmean(y_unfold, dim=-1, keepdim=True)

    x_unfold = torch.where(x_mask, x_mean, x_unfold)
    y_unfold = torch.where(y_mask, y_mean, y_unfold)

    x_demean = (x_unfold - x_mean).unsqueeze(-1)
    y_demean = (y_unfold - y_mean).unsqueeze(-1)

    beta = torch.matmul(torch.linalg.pinv(torch.matmul(x_demean.transpose(1, 2), x_demean)),
                        torch.matmul(x_demean.transpose(1, 2), y_demean))
    y_hat = torch.matmul(x_demean, beta).squeeze(-1)

    # 计算残差
    residuals = y_demean.squeeze(-1) - y_hat
    n_obs = residuals.shape[0]
    k = residuals.shape[1]  # 特征数量

    # 计算协方差矩阵（Newey-West调整）
    gamma_0 = torch.matmul(residuals.transpose(0, 1), residuals) / n_obs
    nw_cov = gamma_0.clone()

    for l in range(1, lag + 1):
        gamma_l = torch.matmul(residuals[:-l].transpose(0, 1), residuals[l:]) / n_obs
        nw_cov += (1 - l / (lag + 1)) * gamma_l

    # 计算Newey-West调整后的标准误差
    sebeta = torch.sqrt(torch.diagonal(torch.matmul(torch.matmul(torch.linalg.pinv(x_demean.transpose(1, 2), x_demean), nw_cov), torch.linalg.pinv(x_demean.transpose(1, 2), x_demean)))) 

    # 结果填充
    res = torch.full(x.shape, torch.nan, dtype=x.dtype, device=x.device)
    res[d - 1:, :] = sebeta.view(-1, k)  # 填充标准误差到结果张量

    return res

def ts_reg_Fvalue(x: torch.Tensor, y: torch.Tensor, d: int):
    
    r2 = ts_reg_rsquare(x, y, d)
    return (d-2) * r2 / (1-r2)

def ts_multireg_alpha(x: torch.Tensor, y: torch.Tensor, d: int, param_lis: list, func, *args) -> torch.Tensor:
    assert x.shape == y.shape, "x和y形状必须相同"
    assert d > len(param_lis), "窗口长度d必须大于特征数"
    assert d > len(param_lis)+1, "OLS条件"
    xstack = _get_multi_x(x, param_lis, func, *args)
    xstack = torch.cat([torch.ones_like(x).unsqueeze(-1),xstack], dim=-1)
    res = get_nan_tensor(x.shape, device=x.device, dtype=x.dtype)
    xstack_unfold = xstack.unfold(0, d, 1) # (T-d+1,N,k+1,d)

    y_unfold = y.unfold(0, d, 1)
    y_unfold = y_unfold.unsqueeze(-1) # (T-d+1,N,d,1)

    # mask_x = torch.isnan(xstack_unfold) | torch.isinf(xstack_unfold)
    # mask_y = torch.isnan(y_unfold) | torch.isinf(y_unfold)
    mask_x = torch.isinf(xstack_unfold)
    mask_y = torch.isinf(y_unfold)
    # xstack_unfold = torch.where(mask_x, torch.nanmean(xstack_unfold, dim=-1, keepdim=True), xstack_unfold)
    # y_unfold = torch.where(mask_y, torch.nanmean(y_unfold, dim=-1, keepdim=True), y_unfold)
    xstack_unfold = torch.where(mask_x, 0.0, xstack_unfold)
    y_unfold = torch.where(mask_y, 0.0, y_unfold)

    xxt = torch.matmul(xstack_unfold, xstack_unfold.transpose(2,3))
    xy = torch.matmul(xstack_unfold, y_unfold)
    beta = torch.matmul(torch.linalg.pinv(xxt), xy) # (T-d+1,N,k+1,1)
    alpha = beta.squeeze(-1)[:,:,0] # (T-d+1,N)
    res[d-1:] = alpha
    return res



def ts_sort_diffcumsum(x: torch.Tensor, by: torch.Tensor, d: int) -> torch.Tensor:
    """先基于by的窗口升序、降序序列分别对窗口内相应的x排序，分别求x排序后的累加序列，对两个累加序列作差后再度累加，取最新值

    :param torch.Tensor x: 形状为 (T, N)，其中 T 是时间步数,N 是股票数
    :param torch.Tensor by: 形状为 (T, N)，其中 T 是时间步数,N 是股票数
    :param int d: 窗口长度
    """
    res = get_nan_tensor(shape=x.shape, device=x.device, dtype=x.dtype)
    x_ascending = torch.cumsum(sort_by_another(x, by, d), dim=-1)
    x_descending = torch.cumsum(sort_by_another(x, by, d, True), dim=-1)
    x_diff_cumsum = torch.cumsum(x_ascending - x_descending, dim=-1)
    res[d-1:] = x_diff_cumsum[:,:,-1]
    return res

def ts_sort_diff_tailsubhead(x: torch.Tensor, by: torch.Tensor, d: int) -> torch.Tensor:
    """先基于by的窗口升序、降序序列分别对窗口内相应的x排序，分别求x排序后的累加序列，对两个累加序列作差后计算差值序列前后半均值之差
    
    :param torch.Tensor x: 形状为 (T, N)，其中 T 是时间步数,N 是股票数
    :param torch.Tensor by: 形状为 (T, N)，其中 T 是时间步数,N 是股票数
    :param int d: 窗口长度
    """
    assert d >= 2, 'd should be 2 at least'
    res = get_nan_tensor(shape=x.shape, device=x.device, dtype=x.dtype)
    x_ascending = torch.cumsum(sort_by_another(x, by, d), dim=-1)
    x_descending = torch.cumsum(sort_by_another(x, by, d, True), dim=-1)
    x_diff = x_ascending - x_descending
    if d % 2 == 0:
        res[d-1:] = torch.nanmean(x_diff[:,:,:int(d/2)], dim=-1) - torch.nanmean(x_diff[:,:,int(d/2):], dim=-1)
    else:
        res[d-1:] = torch.nanmean(x_diff[:,:,:int((d-1)/2)], dim=-1) - torch.nanmean(x_diff[:,:,int((d+1)/2):], dim=-1)
    return res

def ts_max_change(x: torch.Tensor, least_d: int, drawdown=True) -> torch.Tensor:
    """返回截至当前时点，输入指标的最大跌幅or涨幅
    
    """
    
    T = x.shape[0]
    assert least_d<=T, "least_d should not be bigger than x.shape[0]"
    res = get_nan_tensor(shape=x.shape, device=x.device, dtype=x.dtype)
    if drawdown:
        for i in range(least_d, T):
            window = x[:i,:]
            window = torch.where(torch.isnan(window) | torch.isinf(window), torch.nanmean(window, dim=0, keepdim=True), window)
            res[i-1] = torch.max(window, dim=0)[0] / window[-1] - 1
    else:
        for i in range(least_d, T):
            window = x[:i,:]
            window = torch.where(torch.isnan(window) | torch.isinf(window), torch.nanmean(window, dim=0, keepdim=True), window)
            res[i-1] = window[-1] / (torch.min(window, dim=0)[0]) - 1
    return res

# Newey-West standard error calculation
# This function computes the Newey-West standard error for a time series regression
def ts_reg_NWbetase(x: torch.Tensor, y: torch.Tensor, d: int, L: int = None):
    assert x.shape == y.shape, "x and y shapes must be the same"
    assert 2 < d <= x.shape[0], "OLS lens must be greater than 2"
    T, N = x.shape
    res = torch.full((T, N), torch.nan, dtype=x.dtype, device=x.device)

    # If L is not provided, calculate it using the empirical formula
    if L is None:
        L = int(4 * (T / 100) ** (2 / 9))
    
    # Unfold into sliding windows of size d, shape (T-d+1, N, d)
    x_unfold = x.unfold(0, d, 1)
    y_unfold = y.unfold(0, d, 1)

    # Handle missing and infinite values
    x_mask = torch.isnan(x_unfold) | torch.isinf(x_unfold)
    y_mask = torch.isnan(y_unfold) | torch.isinf(y_unfold)

    x_mean = torch.nanmean(x_unfold, dim=2, keepdim=True)
    y_mean = torch.nanmean(y_unfold, dim=2, keepdim=True)

    x_unfold = torch.where(x_mask, x_mean, x_unfold)
    y_unfold = torch.where(y_mask, y_mean, y_unfold)

    # Demean the data
    x_demean = x_unfold - x_mean
    y_demean = y_unfold - y_mean

    # Compute regression coefficients beta
    XY = torch.sum(x_demean * y_demean, dim=2)  # (T-d+1, N)
    XX = torch.sum(x_demean ** 2, dim=2)
    beta = XY / XX  # (T-d+1, N)

    # Calculate residuals
    residual = y_demean - beta.unsqueeze(2) * x_demean  # (T-d+1, N, d)

    # Compute S0
    S0 = torch.sum((residual ** 2) * (x_demean ** 2), dim=2) / d  # (T-d+1, N)

    # Compute weights for Newey-West adjustment
    w = 1 - torch.arange(1, L + 1, dtype=torch.float32, device=x.device) / (L + 1)  # (L,)

    # Compute S1 with autocovariance terms
    S1 = torch.zeros_like(S0)
    for l in range(1, L + 1):
        if l >= d:  # Skip if lag exceeds window size
            continue
        # Slice the residual and x_demean for lag l
        res_lag = residual[..., :-l] * residual[..., l:]
        x_lag = x_demean[..., :-l] * x_demean[..., l:]
        product = res_lag * x_lag
        sum_product = torch.sum(product, dim=2)  # (T-d+1, N)
        S1 += w[l - 1] * sum_product
    S1 = 2 * S1 / d  # Account for symmetric lags

    # Compute Newey-West adjusted variance
    S_hat = S0 + S1

    # Calculate standard error of beta
    var_x = XX  # sum of squared x_demean, already computed
    sebeta = torch.sqrt(S_hat * d / (var_x ** 2 * (d - 2)))

    # Assign results to the output tensor
    res[d - 1:, :] = sebeta

    return res

def ts_demean_sum(x: torch.Tensor, d: int) -> torch.Tensor:
    """滚动窗口去均值再求和，窗口内所有元素去均值时减去的均值是此窗口均值，而非各自窗口的均值
    
    """
    res = get_nan_tensor(x.shape, device=x.device, dtype=x.dtype)
    x_unfold = x.unfold(0, d, 1) # (T-d+1,N,d)
    x_mean = x_unfold.nanmean(dim=-1, keepdim=True)
    x_demean = x_unfold - x_mean
    res[d-1:] = x_demean.nansum(dim=-1)
    return res

def ts_dfa(x: torch.Tensor, d: int, slis: list, k: int) -> torch.Tensor:
    """去趋势波动分析
    :params int d: 计算去均值之和的窗口长度
    :params list slis: 进一步分割时间序列时，每个子窗口的长度s的集,升序输入
    :params int k: 进一步分割时间序列时，每个大窗口内长度s的子窗口的数量
    """
    T, N = x.shape
    assert d <= T, "d shouldn't be bigger than T"
    assert len(slis) > 2, "OLS needs more than 2 samples"
    
    try:
        slis[-1]*k <= T
    except:
        
        raise ValueError("ks should not be bigger than T")
    
    cum_x = ts_demean_sum(x, d) # (T,N)
    temp_res_lis, sstack_lis = [], []
    
    for s in slis:
        ks = k*s
        temp_res = get_nan_tensor(x.shape, device=x.device, dtype=x.dtype) # (T,N)
        cum_x_unfold = cum_x.unfold(0, ks, 1) # (T-ks+1,N,ks)
        cum_x_unfold = cum_x_unfold.reshape(T-ks+1,N,k,s) # (T-ks+1,N,k,s)
        cum_x_unfold_var = (cum_x_unfold - cum_x_unfold.nanmean(dim=-1, keepdim=True))**2 # (T-ks+1,N,k,s)
        std = cum_x_unfold_var.nanmean(dim=(-1,-2))**0.5 # (T-ks+1,N)
        temp_res[ks-1:] = std
        temp_res_lis.append(temp_res)
        sstack_lis.append(torch.ones_like(x)*s)
    y = torch.log(torch.stack(temp_res_lis, dim=-1)) # (T,N,S),S=len(slis)
    sstack = torch.log(torch.stack(sstack_lis, dim=-1)) # (T,N,S),S=len(slis)
    y_demean = y - y.nanmean(dim=-1, keepdim=True)
    sstack_demean = sstack - sstack.nanmean(dim=-1, keepdim=True)
    numerator = (y_demean * sstack_demean).nansum(dim=-1)
    denominator = (sstack_demean**2).nansum(dim=-1)
    res = numerator / denominator # 反映时间序列的自相似性

    return res

def ts_poly2_trend(x:torch.Tensor,d:int):
    assert 3 < d <= x.shape[0] #样本量（期数）需大于解释变量个数(2+1)

    b = torch.full(x.shape,torch.nan,dtype=x.dtype,device=x.device) #存储系数b
    r2 = torch.full(x.shape,torch.nan,dtype=x.dtype,device=x.device) #存储系数R^2
    x_unfold = x.unfold(0,d,1).view(-1,d) #先将x展开，由于仅对最后一个维度操作，所以前两个维度可以合并
    timeline = torch.tile(torch.arange(d,dtype=x.dtype, device=x.device),(x_unfold.shape[0],1)) #时间轴
    timeline_squared = timeline**2 #时间轴的平方项

    #缺失值填充（被解释变量）
    x_mask = x_unfold.isnan()|x_unfold.isinf()
    x_mean = torch.nanmean(x_unfold, dim=-1,keepdim=True)
    x_unfold = torch.where(x_mask,x_mean,x_unfold)

    #构造解释变量矩阵
    X = torch.stack([torch.ones_like(timeline), timeline, timeline_squared], dim=-1)
    
    # 计算回归系数
    XTX_inv = torch.linalg.pinv(torch.matmul(X.transpose(1, 2), X)) #(X^TX)^-1
    XTy = torch.matmul(X.transpose(1, 2), x_unfold.unsqueeze(-1)) #X^Ty
    beta = torch.matmul(XTX_inv, XTy).squeeze(-1) #beta = (X^TX)^-1 X^Ty

    # 计算预测值
    y_pred = torch.matmul(X, beta.unsqueeze(-1)).squeeze(-1)

    # 计算 R2 系数
    ss_res = torch.sum((x_unfold - y_pred) ** 2, dim=-1)
    ss_tot = torch.sum((x_unfold - x_mean) ** 2, dim=-1)
    r2[d - 1:, :] = 1 - (ss_res / ss_tot).view(-1, x.shape[1])

    # 提取一次项系数
    b[d - 1:, :] = beta[:, 1].view(-1, x.shape[1])

    #得到最终的修正趋势
    product = b * r2
    return product

def ts_multiregression_getparams(y:torch.Tensor,x:List[torch.Tensor],d:int,i:int):
    '''
    获取多元线性回归中, 指定变量的回归系数
    y: torch.Tensor, 因变量
    x: List[torch.Tenor], 自变量序列,有序
    d: int, 估计多元线性回归所用回溯窗口长度
    i: int, 表示获取第i个自变量的系数, 从1开始取
    '''
    n = len(x) #解释变量个数
    T,_ = y.shape #样本长度.股票个数
    assert n+1 < d <= T #回溯期比解释变量数目+1还要大,但不能超过样本期

    y_unfold = y.unfold(0,d,1).view(-1,d)
    x_unfold = [xs.unfold(0,d,1).view(-1,d) for xs in x]

    params = torch.full(y.shape,torch.nan,dtype=y.dtype,device=y.device) #存储回归系数
    
    def fillna_mean(var:torch.Tensor):
        #按照回测期均值替换缺失值
        var_mask = var.isnan()|var.isinf()
        var_mean = torch.nanmean(var,dim=-1,keepdim=True)
        var = torch.where(var_mask,var_mean,var)
        return var
    
    y_unfold = fillna_mean(y_unfold)
    x_unfold = [fillna_mean(xs) for xs in x_unfold] #缺失值填充
    
    x_unfold.insert(0,torch.ones_like(x_unfold[0]))
    X = torch.stack(x_unfold,dim=-1) 

    #计算回归系数
    XTX_inv = torch.linalg.pinv(torch.matmul(X.transpose(1, 2), X)) #(X^TX)^-1
    XTy = torch.matmul(X.transpose(1, 2), y_unfold.unsqueeze(-1)) #X^Ty
    beta = torch.matmul(XTX_inv, XTy).squeeze(-1) #beta = (X^TX)^-1 X^Ty

    #提取感兴趣的回归系数
    params[d-1:,:] = beta[:,i].view(-1,y.shape[1])

    return params

def ts_VAR_get_params(y: List[torch.Tensor], d: int, r: int, i: int, j: int, t: int) -> torch.Tensor:
    '''
    获取VAR模型中,指定变量的回归系数
    y: List[torch.Tensor], 所有内生变量的序列, 有序
    d: 估计VAR模型所用的回溯期
    r: VAR模型的滞后阶数
    i,j,t: 代表取第i个变量对第j个变量滞后t期的回归系数, 三个参数都从1开始取
    '''
    n = len(y)
    T,_ = y[0].shape
    assert n*r+1 < d <= T
    assert 0 < i <= n
    assert 0 < j <= n
    assert i != j
    assert 0 < t <= r

    Y = y[i-1] #被解释变量

    #生成滞后变量集合
    lagged_y = []
    for t in range(1,r + 1):
        lag_t = []  
        for ys in y:
            ys_lag = torch.roll(ys, shifts=t, dims=0)
            ys_lag[:t] = float('nan')  # 前t行填充NaN
            lag_t.append(ys_lag)
        lagged_y += lag_t
    
    params = ts_multiregression_getparams(Y,lagged_y,d,(t-1)*n+j)
    return params


def ts_ridgeregression_pred(y: torch.Tensor, x: List[torch.Tensor], d: int, 
                            lambda_: float = 100.0):
    """
    优化计算速度的岭回归预测函数
    """
    device = y.device
    dtype = y.dtype
    n = len(x)
    T, N = y.shape
    assert n+1 < d <= T
    
    pred = torch.full_like(y, torch.nan)
    
    # 数据预处理：直接用0填充缺失值
    def fill_zero(tensor):
        """直接用0填充缺失值"""
        return torch.where(torch.isnan(tensor) | torch.isinf(tensor), 
                         torch.zeros_like(tensor), 
                         tensor)
    
    # 处理输入数据
    y_processed = fill_zero(y)
    x_processed = [fill_zero(xi) for xi in x]
    
    X = torch.ones((T-d+1, N, d, n+1), device=device, dtype=dtype)
    for j, xi in enumerate(x_processed, 1):
        X[..., j] = xi.unfold(0, d, 1).movedim(-1, 2)
    
    y_unfold = y_processed.unfold(0, d, 1).movedim(-1, 2) # [T-d+1, N, d, 1]
    
    try:
        Xt = X.transpose(2, 3)  # [T-d+1, N, n+1, d]
        XtX = torch.matmul(Xt, X)  # [T-d+1, N, n+1, n+1]
        Xty = torch.matmul(Xt, y_unfold.unsqueeze(-1))  # [T-d+1, N, n+1, 1]
        
        # 添加正则化项
        I = torch.eye(n+1, device=device, dtype=dtype)
        I = I.expand(T-d+1, N, n+1, n+1)
        XtX_reg = XtX + lambda_ * I
        
        # 使用LU分解
        batch_size = (T-d+1) * N
        XtX_flat = XtX_reg.reshape(batch_size, n+1, n+1)
        Xty_flat = Xty.reshape(batch_size, n+1, 1)

        try:
            LU, pivots, info = torch.linalg.lu_factor_ex(XtX_flat)
            valid_batch = info == 0
            
            if valid_batch.all():
                # 所有批次都成功
                beta = torch.linalg.lu_solve(LU, pivots, Xty_flat)
            else:
                # 对于失败的批次，使用linalg.lu
                beta = torch.empty_like(Xty_flat)
                beta[valid_batch] = torch.linalg.lu_solve(LU[valid_batch], pivots[valid_batch], Xty_flat[valid_batch])
                P, L, U = torch.linalg.lu(XtX_flat[~valid_batch])
                y = torch.triangular_solve(torch.matmul(P, Xty_flat[~valid_batch]), L, upper=False)[0]
                failed_beta = torch.triangular_solve(y, U, upper=True)[0]
                beta[~valid_batch] = failed_beta

        except Exception as e:
            print(f"LU分解完全失败，使用伪逆: {str(e)}")
            XtX_pinv = torch.linalg.pinv(XtX_flat)
            beta = torch.matmul(XtX_pinv, Xty_flat)

        beta = beta.reshape(T-d+1, N, n+1, 1)
        
        # 逐时间点计算预测值
        for t in range(d-1, T):
            idx = t - (d-1)  # 对应于beta中的索引
            X_t = X[idx]  # 当前时间点的特征矩阵 [N, d, n+1]
            X_last_t = X_t[:, -1, :]  # 取最后一个时间步的特征 [N, n+1]
            beta_t = beta[idx]  # 当前时间点的系数 [N, n+1, 1]
            
            # 计算预测值
            pred[t] = torch.bmm(X_last_t.unsqueeze(1), beta_t).squeeze()
        
    except Exception as e:
        print(f"计算过程中出错: {str(e)}")
    
    return pred



def ts_ridgeregression_res(y: torch.Tensor, x: List[torch.Tensor], d: int, 
                                lambda_: float = 100.0):
    """
    计算岭回归的残差（实际值减去预测值）
    
    参数:
    - y: 因变量
    - x: 自变量列表
    - d: 滑动窗口大小
    - lambda_: 正则化参数
    
    返回:
    - 残差张量，与y同形状
    """
    pred = ts_ridgeregression_pred(y, x, d, lambda_)
    
    # 计算残差
    resid = y - pred
    
    return resid


def ts_max_drawdown(X:torch.Tensor,d:int):
    '''
    计算一定时间窗口内的最大回撤率(对变量本身计算)
    '''
    results = torch.full(X.shape,torch.nan,dtype=X.dtype,device=X.device)
    X_unfold = X.unfold(0,d,1)
    rolling_max = torch.cummax(X_unfold,dim=2)[0]
    drawdown = (rolling_max-X_unfold)/rolling_max
    max_drawdown = torch.max(drawdown,dim=2)[0]
    results[d-1:,:] = max_drawdown
    return results

def ts_ewmv(x: torch.Tensor, d: int, alpha=None) -> torch.Tensor:
    """
    alpha: 平滑因子(可选)，未指定时自动计算为 1-2/(d+1)
    """
    alpha = 1 - 2/(d+1) if alpha is None else alpha
    assert alpha > 0 and alpha < 1, "alpha必须在(0, 1)区间内"
    assert d > 0 and d < x.shape[0], "窗口大小必须为正数且小于输入长度"
    
    # First calculate EWMA
    ewma = torch.full(x.shape, float('nan'), device=x.device, dtype=x.dtype)
    ewma[0] = x[0]
    weight = (1-alpha)**torch.arange(d, 0, -1, device=x.device)  # (d,)
    x_unfold = x.unfold(0, d, 1)
    ewma_unfold = torch.nansum(x_unfold * weight, dim=-1) / torch.nansum(weight)
    ewma[d-1:] = ewma_unfold
    
    # Then calculate EWMV using the EWMA values
    res = torch.full(x.shape, float('nan'), device=x.device, dtype=x.dtype)
    x_sq_unfold = (x_unfold - ewma_unfold.unsqueeze(-1))**2  # squared deviations
    x_var_weighted = torch.nansum(x_sq_unfold * weight, dim=-1) / torch.nansum(weight)
    res[d-1:] = x_var_weighted
    
    return res

def ts_ewmstd(x: torch.Tensor, d: int, alpha=None) -> torch.Tensor:
    ewmvar = ts_ewmv(x, d, alpha)
    return torch.sqrt(ewmvar)

def ts_downside_volatility(x: torch.Tensor,d:int)-> torch.Tensor:
    x_demean = cs_demean(x)
    mask = x_demean < 0
    rv = ts_stddev(torch.where(mask,x_demean,torch.nan), d)
    return div(x,rv)

def ts_upside_volatility(x: torch.Tensor,d:int)-> torch.Tensor:
    x_demean = cs_demean(x)
    mask = x_demean > 0
    rv = ts_stddev(torch.where(mask,x_demean,torch.nan), d)
    return div(x,rv)

def ts_all_volatility(x: torch.Tensor,d:int)-> torch.Tensor:
    x_demean = cs_demean(x)
    rv = ts_stddev(x_demean, d)
    return div(x,rv)

def ts_adaptive_sampling(
    x: torch.Tensor,
    y: torch.Tensor,
    base_window: int = 20,
    state_window: int = 60,
    quantile_level: float = 0.7,
    trend_coef: float = 1.2,
    noise_coef: float = 0.8
) -> torch.Tensor:
 
    assert x.shape == y.shape, "输入张量形状必须一致"
    T, N = x.shape
    
    # 第一层：动态分位数阈值
    y_unfold = y.unfold(0, state_window, 1)
    quantiles = y_unfold.quantile(quantile_level, dim=2)
    # 手动填充分位数
    pad_quant = state_window - 1
    quantiles = torch.cat([quantiles[0:1].expand(pad_quant, N), quantiles], dim=0)
    
    # 第二层：趋势方向调整
    # 短期均值计算及填充
    ma_short = y.unfold(0, base_window, 1).mean(dim=2)
    pad_short = base_window - 1
    ma_short = torch.cat([ma_short[0:1].expand(pad_short, N), ma_short], dim=0)
    
    # 长期均值计算及填充
    ma_long = y.unfold(0, state_window, 1).mean(dim=2)
    pad_long = state_window - 1
    ma_long = torch.cat([ma_long[0:1].expand(pad_long, N), ma_long], dim=0)
    
    trend_mask = (ma_short > ma_long).float()
    trend_adj = trend_coef * trend_mask + (1 - trend_mask)
    
    # 第三层：波动状态分类
    rolling_std = y.unfold(0, base_window, 1).std(dim=2)
    pad_std = base_window - 1
    rolling_std = torch.cat([rolling_std[0:1].expand(pad_std, N), rolling_std], dim=0)
    vol_level = rolling_std.median(dim=0).values
    vol_state = (rolling_std > vol_level).float()
    
    # 合成动态阈值
    dynamic_threshold = quantiles * trend_adj * (vol_state * trend_coef + (1 - vol_state) * noise_coef)
    
    # 生成触发掩码
    trigger_mask = y >= dynamic_threshold
    
    # 构建传播索引
    time_idx = torch.arange(T, device=x.device)[:, None].expand(-1, N)
    trigger_idx = torch.where(trigger_mask, time_idx, -1).cummax(dim=0).values.clamp(min=0)
    
    # 前值传播采样
    return torch.where(trigger_mask, x, x.gather(0, trigger_idx))


def ts_mean_gt_pct(x: torch.Tensor, y: torch.Tensor, threshold: float, d: int) -> torch.Tensor:
    """
    切割类算子，计算在y上threshold分位数条件下，x的均值（加速版：使用 unfold 向量化）

    Args:
        x (torch.Tensor): 待计算的变量 (T, N)
        y (torch.Tensor): 用于切割的变量 (T, N)
        threshold (float): 分位数阈值
        d (int): 回溯窗口大小

    Returns:
        torch.Tensor: 条件均值张量 (T, N)，前d行为 NaN
    """
    T, N = x.shape
    if T < d:
        raise ValueError("T must be >= d")
    
    # 展开时间维度：变成 (T - d + 1, d, N)
    x_unfold = x.unfold(0, d, 1)  # shape: (T - d + 1, N, d)
    y_unfold = y.unfold(0, d, 1)  # shape: (T - d + 1, N, d)

    # 计算 y 每列分位数门槛（按最后一维：窗口）
    thresh = torch.quantile(y_unfold, threshold, dim=2, keepdim=True)  # (T - d + 1, N, 1)

    # 生成掩码
    mask = y_unfold > thresh  # (T - d + 1, N, d)

    # 应用掩码到 x
    selected = x_unfold * mask  # (T - d + 1, N, d)
    count = mask.sum(dim=2).clamp(min=1)  # 避免除 0

    # 求均值
    mean = selected.sum(dim=2) / count  # (T - d + 1, N)


    # 构造完整结果（前 d 行设为 NaN）
    res = torch.full((T, N), float('nan'), device=x.device)
    res[d - 1:] = mean

    return res


def ts_stddev_gt_pct(x: torch.Tensor, y: torch.Tensor, threshold: float, d: int) -> torch.Tensor:
    """
    切割类算子，计算在 y 上 threshold 分位数条件下，x 的标准差（向量化版本）

    Args:
        x (torch.Tensor): 待计算的变量 (T, N)
        y (torch.Tensor): 用于切割的变量 (T, N)
        threshold (float): 分位数阈值（0~1）
        d (int): 回溯窗口大小

    Returns:
        torch.Tensor: 条件标准差张量，形状为 (T, N)
    """
    T, N = x.shape
    if T < d:
        raise ValueError("T must be >= d")
    
    # 使用 unfold 滑动窗口 -> (T - d + 1, N, d)
    x_unfold = x.unfold(0, d, 1)  # (T - d + 1, d, N) -> (T - d + 1, N, d)
    y_unfold = y.unfold(0, d, 1)

    # 计算分位数阈值 (T - d + 1, N, 1)
    thresh = torch.quantile(y_unfold, threshold, dim=2, keepdim=True)

    # 掩码 (T - d + 1, N, d)
    mask = y_unfold > thresh

    # 应用掩码到 x
    selected = x_unfold * mask
    count = mask.sum(dim=2).clamp(min=1)

    # 计算均值
    mean = selected.sum(dim=2) / count  # (T - d + 1, N)

    # 广播均值并计算偏差平方
    mean_broadcasted = mean.unsqueeze(2)  # (T - d + 1, N, 1)
    var = ((selected - mean_broadcasted) * mask) ** 2

    # 计算标准差
    std = torch.sqrt(var.sum(dim=2) / count)

    # 构造完整输出
    res = torch.full((T, N), float('nan'), device=x.device)
    res[d - 1:] = std

    return res


def ts_sum_gt_pct(x: torch.Tensor, y: torch.Tensor, threshold: float, d: int) -> torch.Tensor:
    """
    切割类算子，计算在 y 下 threshold 分位数条件下，x 的标准差（向量化版本）

    Args:
        x (torch.Tensor): 待计算的变量 (T, N)
        y (torch.Tensor): 用于切割的变量 (T, N)
        threshold (float): 分位数阈值（0~1）
        d (int): 回溯窗口大小

    Returns:
        torch.Tensor: 条件标准差张量，形状为 (T, N)
    """
    T, N = x.shape
    if T < d:
        raise ValueError("T must be >= d")
    
    # 使用 unfold 滑动窗口 -> (T - d + 1, N, d)
    x_unfold = x.unfold(0, d, 1) # (T - d + 1, d, N) -> (T - d + 1, N, d)
    y_unfold = y.unfold(0, d, 1)

    # 计算分位数阈值 (T - d + 1, N, 1)
    thresh = torch.quantile(y_unfold, threshold, dim=2, keepdim=True)

    # 掩码 (T - d + 1, N, d)
    mask = y_unfold > thresh

    # 应用掩码到 x
    selected = x_unfold * mask
    count = mask.sum(dim=2).clamp(min=1)

    # 计算均值
    mean = selected.sum(dim=2) / count  # (T - d + 1, N)

    # 广播均值并计算偏差平方
    mean_broadcasted = mean.unsqueeze(2)  # (T - d + 1, N, 1)
    var = ((selected - mean_broadcasted) * mask) ** 2

    # 计算标准差
    std = torch.sqrt(var.sum(dim=2) / count)

    # 构造完整输出
    res = torch.full((T, N), float('nan'), device=x.device)
    res[d - 1:] = std

    return res


def ts_sum_gt_pct(x: torch.Tensor, y: torch.Tensor, threshold: float, d: int) -> torch.Tensor:
    """
    切割类算子，计算在 y 下 threshold 分位数条件下，x 的求和（向量化加速版）

    Args:
        x (torch.Tensor): 待计算变量 (T, N)
        y (torch.Tensor): 用于切割判断的变量 (T, N)
        threshold (float): 分位点（0~1）
        d (int): 回溯窗口大小

    Returns:
        torch.Tensor: 条件求和结果，形状为 (T, N)，前 d 行为 NaN
    """
    T, N = x.shape
    if T < d:
        raise ValueError("T must be >= d")

    # 展开窗口：变成 (T - d + 1, d, N) -> (T - d + 1, N, d)
    x_unfold = x.unfold(0, d, 1)
    y_unfold = y.unfold(0, d, 1)

    # 分位数阈值（每列每窗口）
    thresh = torch.quantile(y_unfold, threshold, dim=2, keepdim=True)  # (T - d + 1, N, 1)

    # 条件掩码：y > 分位数
    mask = y_unfold > thresh  # (T - d + 1, N, d)

    # 应用掩码到 x，并求和
    selected = x_unfold * mask  # (T - d + 1, N, d)
    sum_result = selected.sum(dim=2)  # (T - d + 1, N)

    # 构造最终结果
    res = torch.full((T, N), float('nan'), device=x.device)
    res[d - 1:] = sum_result

    return res

def ts_psychology(x: torch.Tensor, d: int) -> torch.Tensor:
    """
    计算过去d分钟内上涨的分钟数占比
    :param x: torch.Tensor, 价格的一维时间序列
    :param d: int, 窗口长度
    :return: torch.Tensor, 心理指标序列
    """
    res = torch.full(x.shape, torch.nan, dtype=x.dtype, device=x.device)
    up_flag = (x[1:] > x[:-1]).float()
    up_unfold = up_flag.unfold(0, d, 1)
    up_count = torch.nansum(up_unfold, dim=-1)
    res[d:] = up_count / d
    return res


def ts_supersmoother_filter(x: torch.Tensor, N: int, d: int) -> torch.Tensor:
    """
    2-pole SuperSmoother
    - 遇缺口自动重启，重启值为前两步有效输入值的均值
    """
    res = torch.full_like(x, float('nan'))
    T = x.shape[0]
    if T < d:
        return res

    pi = x.new_tensor(torch.pi)
    w = 1.4142 * pi / N          # √2·π/N
    a1 = torch.exp(-w)
    b1 = 2 * a1 * torch.cos(w)
    b2 = -a1 * a1
    c1 = 1.0 - b1 - b2

    y_prev1 = torch.full_like(x[0], float('nan'))
    y_prev2 = torch.full_like(x[0], float('nan'))

    for t in range(T):
        if t < d - 1:
            continue

        good_in = ~(torch.isnan(x[t]) | torch.isnan(x[t - 1]))
        need_init = good_in & (torch.isnan(y_prev1) | torch.isnan(y_prev2))
        if need_init.any():
            # 重新启动计算的列，用两条输入的平均做第一次输出，并初始化状态
            y_init = (x[t] + x[t - 1]) / 2.0
            res[t][need_init] = y_init[need_init]

            y_prev2[need_init] = x[t - 1][need_init]
            y_prev1[need_init] = y_init[need_init]

        # 若x无nan，则正常递归
        can_calc = good_in & ~(torch.isnan(y_prev1) | torch.isnan(y_prev2))
        if can_calc.any():
            y_curr = (
                c1 * (x[t] + x[t - 1]) / 2
                + b1 * y_prev1
                + b2 * y_prev2
            )
            res[t][can_calc] = y_curr[can_calc]
            y_prev2[can_calc] = y_prev1[can_calc]
            y_prev1[can_calc] = y_curr[can_calc]
    return res


def ts_apen(x: torch.Tensor, m: int, d: int, r: float = 0.15) -> torch.Tensor:
    """
    计算输入序列的滚动近似熵，反映序列的可重复性，返回以该时间点为末尾、长度为 d 的窗口所计算的近似熵值
    """
    T, *rest = x.shape
    n_series = int(torch.prod(torch.tensor(rest))) if rest else 1
    res = torch.full_like(x, float("nan")).view(T, n_series)
    if d < m + 1:
        return torch.full_like(x, float("nan"))
    x2d = x.view(T, n_series)

    for t in range(d - 1, T):
        win = x2d[t - d + 1 : t + 1]  
        mu = torch.nanmean(win, dim=0)                         
        std = torch.sqrt(torch.nanmean((win - mu) ** 2, dim=0)) 
        tol = r * std                                          
        valid = ~torch.isnan(win).any(dim=0) & ~torch.isnan(tol)  #取合法索引列
        if not valid.any():
            continue
        wv = win[:, valid]    
        tv = tol[valid]       
        Nv = wv.shape[1]
        wvt = wv.transpose(0, 1)  
        emb_m   = wvt.unfold(1, m, 1)    
        emb_mp1 = wvt.unfold(1, m + 1, 1)
        #计算 φ^(m)
        def _phi(embed: torch.Tensor, n_vec: int, tol_arr: torch.Tensor):
            diff = (embed.unsqueeze(2) - embed.unsqueeze(1)).abs() 
            dist = diff.amax(dim=-1)    # Chebyshev 距离                          
            sim  = dist <= tol_arr[:, None, None]     # 判断是否相似         
            C    = sim.sum(dim=2).to(embed.dtype) / n_vec       
            C    = torch.where(C > EPS, C, torch.full_like(C, float("nan")))
            return torch.nanmean(torch.log(C), dim=1)
               
        n_vec_m   = d - m + 1
        n_vec_mp1 = d - m
        phi_m   = _phi(emb_m,   n_vec_m,   tv)
        phi_mp1 = _phi(emb_mp1, n_vec_mp1, tv)
        apen    = phi_m - phi_mp1                               
        res[t, valid] = apen
    return res.view(x.shape)






# ------------------------------截面算子------------------------------ #


# 横截面排序 排名从1开始 支持自定义valid_mask pct返回比例
def cs_rank(x: torch.Tensor, valid_mask: torch.Tensor = None, pct: bool = True) -> torch.Tensor:
    ranks = x.clone()
    if valid_mask is None:
        valid_mask = torch.isfinite(x)
    ranks[~valid_mask] = torch.finfo(x.dtype).max
    ranks = torch.argsort(torch.argsort(ranks, dim=-1), dim=-1)
    ranks = ranks.to(dtype=x.dtype) + 1
    if pct:
        valid_counts = valid_mask.sum(dim=-1, keepdim=True)
        ranks = ranks / valid_counts
    ranks[~valid_mask] = torch.nan
    return ranks


def cs_orth(x:torch.Tensor, y:torch.Tensor):
    mask = torch.isnan(x) | torch.isinf(x) |torch.isnan(y) | torch.isinf(y)
    x[mask] = torch.mean(x[~mask])
    y[mask] = torch.mean(y[~mask])
    rej = torch.nansum(x * y, dim=1, keepdim=True) / torch.nansum(y**2, dim=1, keepdim=True) * y
    res = x - rej
    return res

def cs_schmidt(x: torch.Tensor, y: List[torch.Tensor]) -> torch.Tensor:
    def clean(t: torch.Tensor) -> torch.Tensor:
        mask = torch.isnan(t) | torch.isinf(t)
        if mask.any():
            t[mask] = torch.mean(t[~mask])
        return t
    x = clean(x.clone())
    y = [clean(ys) for ys in y]
    res = x.clone()
    for ys in y:
        res -= (torch.sum(x * ys, dim=1, keepdim=True) / torch.sum(ys**2, dim=1, keepdim=True).clamp(min=1e-12)) * ys
    return res

def cs_schmidt_last(x: List[torch.Tensor]) -> torch.Tensor:
    def clean(t: torch.Tensor) -> torch.Tensor:
        mask = torch.isnan(t) | torch.isinf(t)
        if mask.any():
            t[mask] = torch.mean(t[~mask])
        return t
    x = [clean(xs) for xs in x]
    for i in range(1,len(x)):
        res = x[i].clone()
        for j in range(i):
            res -= (torch.sum(x[i] * x[j], dim=1, keepdim=True) / torch.sum(x[j]**2, dim=1, keepdim=True).clamp(min=1e-12)) * x[j]
        x[i] = res
    return x[-1]

def cs_schmidt_sum(x: List[torch.Tensor]) -> torch.Tensor:
    def clean(t: torch.Tensor) -> torch.Tensor:
        mask = torch.isnan(t) | torch.isinf(t)
        if mask.any():
            t[mask] = torch.mean(t[~mask])
        return t
    x = [clean(xs) for xs in x]
    for i in range(1,len(x)):
        res = x[i].clone()
        for j in range(i):
            res -= (torch.sum(x[i] * x[j], dim=1, keepdim=True) / torch.sum(x[j]**2, dim=1, keepdim=True).clamp(min=1e-12)) * x[j]
        x[i] = res
    return sum(x)

# def cs_schmidt_sumadj(x: List[torch.Tensor]) -> torch.Tensor:
#     def clean(t: torch.Tensor) -> torch.Tensor:
#         mask = torch.isnan(t) | torch.isinf(t)
#         if mask.any():
#             t[mask] = torch.mean(t[~mask])
#         return t
#     x = [clean(xs) for xs in x]
#     for i in range(1,len(x)):
#         res = x[i].clone()
#         for j in range(i):
#             res -= (torch.sum(x[i] * x[j], dim=1, keepdim=True) / torch.sum(x[j]**2, dim=1, keepdim=True).clamp(min=1e-12)) * x[j]
#         x[i] = res
#     return sum(x[:-1]) - x[-1]

def cs_regpred(x: torch.Tensor, y: torch.Tensor):
    mask = torch.isnan(x) | torch.isinf(x) | torch.isnan(y) | torch.isinf(y)
    x[mask] = torch.mean(x[~mask])
    y[mask] = torch.mean(y[~mask])
    ones = torch.ones_like(x)
    X = torch.stack([ones, x], dim=2)
    # print(X[0])
    # pinv本质是奇异值分解，对于矩阵X，pinv(X) = V * pinv(S) * U'，pinv(S)是S的伪逆，非零奇异值的倒数放到对角线上，其余元素置为零
    beta = torch.matmul(torch.linalg.pinv(torch.matmul(X.transpose(1, 2), X)),
                        torch.matmul(X.transpose(1, 2), y.unsqueeze(2)))
    y_hat = torch.matmul(X, beta).squeeze()
    y_hat[mask] = torch.nan
    return y_hat


def cs_leftright_range1(x: torch.Tensor) -> torch.Tensor:
    """
    计算每行[j-5, j+5]范围内列的均值（自动处理边界）
    
    参数：
        x (Tensor): 输入二维矩阵，形状为(N, C)
    
    返回：
        Tensor: 与x形状相同的矩阵，每位置为滑动窗口均值
    """
    # 生成有效值掩码（0=NaN，1=有效值）
    nan_mask = torch.where(torch.isnan(x), 0.0, 1.0)
    assert x.dim() == 2, "输入需为二维张量"
    N, C = x.shape
    device = x.device
    
    # 生成列索引和范围边界
    j_indices = torch.arange(C, device=device)
    left = torch.clamp(j_indices - 5, min=0)
    right = torch.clamp(j_indices + 5, max=C-1)
    
    # 生成滑动窗口掩码 (C, C)
    col_grid = torch.arange(C, device=device).view(1, -1).expand(C, -1)
    mask = (col_grid >= left.view(-1, 1)) & (col_grid <= right.view(-1, 1))
    mask = mask.float()  # 转换为浮点型以便矩阵乘法
    
    # 计算加权和与有效元素数
    # 修正加权和计算
    x= x.masked_fill(torch.isnan(x), 0.0)
    sum_per_j = (x * nan_mask) @ mask.T  # 将NaN置零后求和
    count_per_j = (nan_mask @ mask.T)    # 实际有效元素数
    
    # 计算均值并处理零除（理论不会触发）
    return sum_per_j / count_per_j.clamp_min(1)
def cs_regres(x: torch.Tensor, y: torch.Tensor):
    # 把inf和nan替换为均值
    mask = torch.isnan(x) | torch.isinf(x) | torch.isnan(y) | torch.isinf(y)
    x[mask] = torch.mean(x[~mask])
    y[mask] = torch.mean(y[~mask])
    ones = torch.ones_like(x)
    X = torch.stack([ones, x], dim=2)
    # print(X.shape)
    # 回归系数的表达式：beta = (X'X)^(-1)X'Y
    beta = torch.matmul(torch.linalg.pinv(torch.matmul(X.transpose(1, 2), X)),
                        torch.matmul(X.transpose(1, 2), y.unsqueeze(2)))
    # print(beta)
    y_hat = torch.matmul(X, beta).squeeze()
    # print(y_hat.shape)
    res = y - y_hat
    res[mask] = torch.nan
    return res

# def pprint(x: torch.Tensor):
#     print(x)
#     print(x.size())
#     return x




def cs_normalize(x: torch.tensor):
    mean = torch.nanmean(x, axis=1, keepdims=True)
    std = torch.nanmean(x, axis=1, keepdims=True)
    return (x - mean) / std

def cs_demean(x: torch.tensor):
    mean = torch.nanmean(x, axis=1, keepdims=True)
    return (x - mean)

def cs_min(x: torch.Tensor, valid_mask: torch.Tensor = None) -> torch.Tensor:
    xc = x.clone()
    if valid_mask is None:
        valid_mask = torch.isfinite(x)
    xc[~valid_mask] = torch.finfo(x.dtype).max
    res = xc.min(dim=-1, keepdim=True).values
    res[valid_mask.sum(dim=-1) == 0] = torch.nan
    return res

def cs_sum(x: torch.Tensor) -> torch.Tensor:
    """
    将输入张量的每行元素替换为该行的中位数。

    参数:
        x (torch.Tensor): 输入张量，支持一维或二维。

    返回:
        torch.Tensor: 与输入形状相同的张量，每行的所有元素均为该行的中位数。

    示例:
        >>> x = torch.tensor([[1, 3, 5], [2, 4, 6]])
        >>> cs_median(x)
        tensor([[3, 3, 3],
                [4, 4, 4]])

        >>> x = torch.tensor([1, 3, 5])
        >>> cs_median(x)
        tensor([3, 3, 3])
    """
    original_shape = x.shape
    # 处理一维输入，转换为二维
    if x.dim() == 1:
        x = x.unsqueeze(0)
    # 计算每行的中位数
    medians = torch.sum(x, dim=1).values
    # 扩展中位数至原形状
    medians_expanded = medians.unsqueeze(1).expand(-1, x.size(1))
    # 恢复一维输入的原始形状
    if len(original_shape) == 1:
        medians_expanded = medians_expanded.squeeze(0)
    return medians_expanded

def cs_median(x: torch.Tensor) -> torch.Tensor:
    """
    将输入张量的每行元素替换为该行的中位数。

    参数:
        x (torch.Tensor): 输入张量，支持一维或二维。

    返回:
        torch.Tensor: 与输入形状相同的张量，每行的所有元素均为该行的中位数。

    示例:
        >>> x = torch.tensor([[1, 3, 5], [2, 4, 6]])
        >>> cs_median(x)
        tensor([[3, 3, 3],
                [4, 4, 4]])

        >>> x = torch.tensor([1, 3, 5])
        >>> cs_median(x)
        tensor([3, 3, 3])
    """
    original_shape = x.shape
    # 处理一维输入，转换为二维
    if x.dim() == 1:
        x = x.unsqueeze(0)
    # 计算每行的中位数
    medians = torch.median(x, dim=1).values
    # 扩展中位数至原形状
    medians_expanded = medians.unsqueeze(1).expand(-1, x.size(1))
    # 恢复一维输入的原始形状
    if len(original_shape) == 1:
        medians_expanded = medians_expanded.squeeze(0)
    return medians_expanded
    
def cs_max(x: torch.Tensor, valid_mask: torch.Tensor = None) -> torch.Tensor:
    xc = x.clone()
    if valid_mask is None:
        valid_mask = torch.isfinite(x)
    xc[~valid_mask] = torch.finfo(x.dtype).min
    res = xc.max(dim=-1, keepdim=True).values
    res[valid_mask.sum(dim=-1) == 0] = torch.nan

    return res

def cs_market(x:torch.Tensor, weight:torch.Tensor=None):
    '''
    得到全市场因子(排除自身)
    '''
    if weight is None:
        weight = torch.ones(x.shape, device=x.device)
    return ((x * weight).nansum(dim=1,keepdim=True) - x*weight) / (weight.nansum(dim=1,keepdim=True) - weight) 

def cs_multireg_pred(x: torch.Tensor, y: torch.Tensor, func, param_lis, beta_rolling_mean_window: int=None, *args) -> torch.Tensor:
    """求截面多元OLS回归的预测值，用于预测的回归系数可选择是否先取移动平均

    :param torch.Tensor x: 形状为(T, N)，其中 T 是时间步数,N 是股票数
    :param torch.Tensor y: 形状为(T, N)
    :param int=None beta_rolling_mean_window: 用回归系数计算预测y值时先对系数作移动平均的计算窗口大小，默认为None，表示不作移动平均直接预测
    :param *args: func除x和窗口长度外，需要的其它参数
    :return: 截面回归预测值，形状为(T, N)
    :rtype: torch.Tensor
    """
    # res = get_nan_tensor(shape=x.shape, device=x.device, dtype=x.dtype)
    # 输入的y已经shift了
    # x = x[:-shift]
    # y = y[shift:]
    x = _get_multi_x(x, param_lis, func, *args) # 形状为(T, N，K)，其中 T 是时间步数,N 是股票数,K是自变量个数
    T, N, K = x.shape
    assert N > K+1, "OLS needs more than K+1 samples"
    y = y.unsqueeze(-1) # [T, N, 1]
    x = torch.cat([x, torch.ones(T, N, 1, device=x.device, dtype=x.dtype)], dim=-1) # 添加常数项,(T,N,K+1)
    # 填充inf
    x = torch.where(x.isinf() | x.isnan(), torch.nanmean(x, dim=0, keepdim=True), x)
    y = torch.where(y.isinf() | y.isnan(), torch.nanmean(y, dim=0, keepdim=True), y)
    # 计算beta
    xT = x.transpose(1, 2) # [T, K+1, N]
    xTx = torch.matmul(xT, x) # [T, K+1, K+1]
    xTy = torch.matmul(xT, y) # [T, K+1, 1]
    beta = torch.matmul(torch.linalg.pinv(xTx), xTy) # [T, K+1, 1]
    # beta = torch.linalg.solve(xTx,xTy)
    if beta_rolling_mean_window is not None:
        beta = ts_ewma(beta, beta_rolling_mean_window) # [T, K+1, 1],有T-beta_rolling_mean_window+1个值
    y_pred = torch.matmul(x, beta) # [T, N, 1]
    y_pred = y_pred.squeeze(-1) # [T,N]
    
    return y_pred





def cs_winsorize(x: torch.Tensor) -> torch.Tensor:
    """
    对(T, N)形状的股票数据进行横截面4σ标准差截尾处理
    参数：
        x : 输入张量，形状为[T, N]，T为时间点，N为股票数
    返回：
        处理后的张量，极值被替换为均值±4σ边界值
    """
    # 计算横截面统计量（忽略NaN）
    mu = torch.nanmean(x, dim=1, keepdim=True)   # 形状[T, 1]
    sigma = _keepdimstddev(x) # 形状[T, 1]
    
    # 设置最小标准差阈值防止除零
    sigma = torch.clamp(sigma, min=1e-6)
    
    # 计算截尾边界
    upper = mu + 4 * sigma
    lower = mu - 4 * sigma

    # 执行截尾操作
    return torch.clamp(x, min=lower, max=upper)


def cs_mad_winsorize(x: torch.Tensor, n: float = 3) -> torch.Tensor:
    """
    使用MAD方法进行横截面异常值处理
    
    参数:
    x: torch.Tensor - 输入数据
    n: float - MAD的倍数，默认为3倍
    
    返回:
    torch.Tensor - 经过MAD方法处理后的数据
    """
    res = x.clone()
    median = torch.nanmedian(x, dim=1, keepdim=True).values
    abs_dev = torch.abs(x - median)
    mad = torch.nanmedian(abs_dev, dim=1, keepdim=True).values
    upper_bound = median + n * mad
    lower_bound = median - n * mad
    
    res = torch.minimum(res, upper_bound)
    res = torch.maximum(res, lower_bound)
    
    return res

def cs_mean_distance(distance_type: str, *args: torch.Tensor) -> torch.tensor:
    """输入多个特征，以此求截面上其余各股与自身的欧氏距离或曼哈顿距离的均值
    
    :param distance_type: str 距离类型，可选ecld（欧氏距离）或mht（曼哈顿距离）
    :param *args: torch.Tensor 用于计算各股票间距离的特征，均为(T,N)张量，共k个
    :return: 形状为(T,N), 每个截面上，其余股票与自身的平均距离
    :rtype: torch.Tensor
    """
    shape = args[0].shape
    feature_lis = []
    for x in args:
        assert x.shape == shape, "All tensors should have the same shape"
        x = torch.where(x.isnan() | x.isinf(), torch.nanmean(x, dim=1, keepdim=True), x)
        x[x.isnan()] = 0.0
        feature_lis.append(x)
    all_feature = torch.stack(feature_lis, dim=-1) # (T,N,K)
    res = get_nan_tensor(all_feature.shape[:2], all_feature.device, all_feature.dtype)
    n = shape[1]
    k = all_feature.shape[-1]
    if distance_type == 'ecld':
        for i in range(n):
            diff = all_feature - all_feature[:,i,:].unsqueeze(1)
            distances = torch.sqrt(torch.nansum(diff**2, dim=-1)) / k # (T,N)
            distance_mean = torch.sum(distances, dim=-1) / (n-1)
            res[:,i] = distance_mean
    elif distance_type == 'mht':
        for i in range(n):
            diff = all_feature - all_feature[:,i,:].unsqueeze(1)
            distances = torch.nanmean(torch.abs(diff), dim=-1) # (T,N)
            distance_mean = torch.sum(distances, dim=-1) / (n-1)
            res[:,i] = distance_mean

    return res

def bucket(x: torch.Tensor, start: float, end: float, step: float) -> torch.Tensor:
    """
    高效正确的分桶函数
    
    参数:
        x: 输入张量
        start: 起始值 (包含)
        end: 结束值 (包含)
        step: 步长
        
    返回:
        分桶编号(从1开始)，超出范围则为nan
    """
    # 生成桶边界 (确保包含end)
    boundaries = torch.arange(start, end + step, step, device=x.device)
    boundaries[-1] = end  # 强制最后一个边界为end
    
    # 计算每个元素所属的桶索引
    # (x - boundaries)产生广播矩阵，寻找第一个大于等于0的位置
    diffs = x.unsqueeze(-1) - boundaries.unsqueeze(0)
    bucket_indices = torch.sum(diffs >= 0, dim=-1)
    
    # 创建有效掩码
    valid_mask = (x >= start) & (x <= end)
    
    # 组合结果 (有效值: bucket_indices+1, 无效值: nan)
    result = torch.where(valid_mask, bucket_indices.float(), torch.nan)
    
    return result

def cs_group_rank(
    x: torch.Tensor, 
    bucket: torch.Tensor, 
    pct: bool = True
) -> torch.Tensor:
    """
    完全向量化的分组排名计算，无显式循环。
    支持逐行独立分组排序，兼容 NaN/Inf 处理。

    参数:
        x: 输入数据 [N, M]
        bucket: 分组标签 [N, M]（同形状，NaN 表示无效）
        pct: 是否返回百分比排名

    返回:
        排名结果 [N, M]，无效位置为 NaN
    """
    # 初始化结果并过滤无效值
    result = torch.full_like(x, float('nan'))
    valid_mask = (~torch.isnan(x)) & (~torch.isinf(x)) & (~torch.isnan(bucket))
    
    # 生成唯一标识符：行号 * 组数 + 组号
    group_ids = (torch.arange(x.shape[0], device=x.device)[:, None] * 
                (bucket.max().int() + 1) + bucket.int()) * valid_mask
    unique_groups = torch.unique(group_ids[valid_mask])
    
    # 对每个分组并行计算排名
    for group in unique_groups:
        mask = (group_ids == group)
        group_x = x[mask]
        
        # 计算当前分组的排名（稳定排序）
        ranks = torch.argsort(torch.argsort(group_x)) + 1
        
        if pct:
            ranks = ranks.float() / mask.sum()
        
        result[mask] = ranks
    
    return result

def cs_group_zscore(x: torch.Tensor, bucket: torch.Tensor) -> torch.Tensor:
    """
    完全向量化的分组标准化计算，无显式循环。
    支持逐行独立分组标准化，兼容 NaN/Inf 处理。

    参数:
        x: 输入数据 [N, M]
        bucket: 分组标签 [N, M]（同形状，NaN 表示无效）
        eps: 防止除零的小常数

    返回:
        Z-Score 结果 [N, M]，无效位置为 NaN
    """
    # 初始化结果并过滤无效值
    result = torch.full_like(x, float('nan'))
    valid_mask = (~torch.isnan(x)) & (~torch.isinf(x)) & (~torch.isnan(bucket))
    
    # 生成唯一标识符：行号 * 组数 + 组号
    group_ids = (torch.arange(x.shape[0], device=x.device)[:, None] * 
                (bucket.max().int() + 1) + bucket.int()) * valid_mask
    unique_groups = torch.unique(group_ids[valid_mask])
    
    # 对每个分组并行计算
    for group in unique_groups:
        mask = (group_ids == group) & valid_mask
        group_x = x[mask]
        
        if group_x.numel() < 2:  # 样本不足
            continue
            
        # 计算均值和标准差
        mean = group_x.mean()
        std = group_x.std()
        
        # 处理标准差过小的情况
        if std < EPS:
            result[mask] = torch.nan
        else:
            result[mask] = (group_x - mean) / std
    
    return result


def cs_group_neutralize(x: torch.Tensor, bucket: torch.Tensor) -> torch.Tensor:
    """
    完全向量化的分组中性化计算，无显式循环。
    支持逐行独立分组中性化，兼容 NaN/Inf 处理。

    参数:
        x: 输入数据 [N, M]
        bucket: 分组标签 [N, M]（同形状，NaN 表示无效）

    返回:
        中性化结果 [N, M]，无效位置为 NaN
    """
    # 初始化结果并过滤无效值
    result = torch.full_like(x, float('nan'))
    valid_mask = (~torch.isnan(x)) & (~torch.isinf(x)) & (~torch.isnan(bucket))
    
    # 生成唯一标识符：行号 * 组数 + 组号
    group_ids = (torch.arange(x.shape[0], device=x.device)[:, None] * 
                (bucket.max().int() + 1) + bucket.int()) * valid_mask
    unique_groups = torch.unique(group_ids[valid_mask])
    
    # 对每个分组并行计算
    for group in unique_groups:
        mask = (group_ids == group) & valid_mask
        group_x = x[mask]
        
        if group_x.numel() == 0:  # 空组
            continue
            
        # 中性化计算
        result[mask] = group_x - group_x.mean()
    
    return result


def cs_dense_rank(x: torch.Tensor, valid_mask: torch.Tensor = None, pct: bool = True) -> torch.Tensor:
    """横截面排序 排名从1开始 支持自定义valid_mask pct返回比例
    处理重复值：相同的值获得相同的排名（密集排名）
    Args:
        dense: 是否使用密集排名。True时重复值后的排名连续(如[1,2,2,3]), False时跳过重复值的排名(如[1,2,2,4])
        valid_mask: 有效值掩码，默认为非空值
        pct: 是否返回百分比排名
    """
    x_dtype = torch.float32 if not torch.is_floating_point(x) else x.dtype
    x = x.to(x_dtype)
    
    if valid_mask is None:
        valid_mask = torch.isfinite(x)
    
    ranks = torch.zeros_like(x)
    
    max_value = torch.finfo(x_dtype).max
    temp_x = torch.where(valid_mask, x, max_value)
    
    sorted_indices = torch.argsort(temp_x, dim=-1)
    
    sorted_x = torch.gather(temp_x, -1, sorted_indices)
    
    not_equal = (sorted_x[..., 1:] != sorted_x[..., :-1])
    not_equal = torch.cat([torch.ones_like(not_equal[..., :1]), not_equal], dim=-1)
    dense_rank_values = torch.cumsum(not_equal, dim=-1)
    
    rank_indices = torch.argsort(sorted_indices, dim=-1)
    ranks = torch.gather(dense_rank_values, -1, rank_indices)
    
    if pct:
        max_ranks = ranks.max(dim=-1, keepdim=True).values
        ranks = ranks / max_ranks

    ranks = ranks.to(torch.float32)
    ranks[~valid_mask] = torch.nan
    
    return ranks

def cs_abs_deviation(x: torch.Tensor, valid_mask: any = None) -> torch.Tensor:
    ''' 计算每个因子的横截面绝对偏差 
        即每个股票的值减去横截面的中位数（去偏），再取绝对值
    '''
    csad = x.clone()
    if valid_mask is None:
        valid_mask = torch.isfinite(x)
        
    med = torch.nanmedian(x, dim=1, keepdim=True).values
    csad = torch.abs(x - med)
    csad[~valid_mask] = torch.nan

    return csad

def cs_group_rank_neutralize(original: torch.Tensor, neutralizer: torch.Tensor, group_num: int = 5) -> torch.tensor:

    #check size match
    if original.shape == neutralizer.shape:
        
        valid_mask = original.isfinite()&neutralizer.isfinite()
        original = original.masked_fill(~valid_mask, torch.nan)
        neutralizer = neutralizer.masked_fill(~valid_mask, torch.nan)

        get_q = torch.linspace(0, 1, group_num+1).to('cuda')
        quantiles = torch.nanquantile(neutralizer, q = get_q, dim = 1).T
        rank_mat = torch.zeros_like(original)
        for i in range(group_num):
            q_1 = quantiles[:, i:i+1]
            q_2 = quantiles[:, i+1:i+2]
            if i == group_num - 1:
                mask = neutralizer.ge(q_1)&neutralizer.le(q_2)
            else:
                mask = neutralizer.ge(q_1)&neutralizer.lt(q_2)
            rank_mat += cs_dense_rank(original, mask, True).nan_to_num()
        return rank_mat
            
    else:
        raise ValueError("size not match")
    
def cs_salience(x:torch.Tensor)->torch.Tensor:
    '''
    计算在横截面上的显著性水平
    算子试运行中20250402,测试无误将注册至表内
    '''
    market_mean=torch.nanmean(x, axis=1, keepdims=True)
    res=abs(x-market_mean)/(0.01+abs(x)+abs(market_mean))
    return res

def cs_marketcorr(x:torch.Tensor,d:int)->torch.Tensor:
    '''
    截面相关性计算
    计算在横截面上与市场中所有股票的(皮尔逊)相关性均值情况
    参数d为选取多少回看区间作为相关性矩阵的维度,一般看样本个数对相关性计算的影响
    算子试运行中20250402,目前若采用滚动的方式计算则时间复杂度过大,目前正在考虑用分域切片的方式计算(cs_marketcorr2),或取隔夜数据后进行日内120/240区间内数据进行计算
    '''
    def cs_corr_matrix(x:torch.Tensor):
        corr_matrix = torch.corrcoef(x_final)
        corr_mean=torch.nanmean(corr_matrix,dim=0)
        return corr_mean
    
    
    
    res=get_nan_tensor(x.shape,x.device,x.dtype)
    x_unfold=x.unfold(0,d,1)
    # x_unfold=x.unfold(0,d,d)

    for i in range(x_unfold.shape[0]):
        x_final=x_unfold[i]
        corr_mean=cs_corr_matrix(x_final)
        res[d-1+i:]=corr_mean
        # res[d*i:d*(i+1)]=corr_mean
    # print(corr_mean)
    # mask = ~torch.isnan(x).all(dim=1)
    # # 使用掩码过滤张量
    # x_inday= x[mask]
    # mask = ~torch.isnan(x_inday).all(dim=0)
    # x_final= x_inday[:,mask]
    return res

def cs_orth(x:torch.Tensor, y:torch.Tensor):
    mask = torch.isnan(x) | torch.isinf(x) |torch.isnan(y) | torch.isinf(y)
    x[mask] = torch.mean(x[~mask])
    y[mask] = torch.mean(y[~mask])
    rej = torch.nansum(x * y, dim=1, keepdim=True) / torch.nansum(y**2, dim=1, keepdim=True) * y
    res = x - rej
    return res

def cs_orth_add(x:torch.Tensor, y:torch.Tensor):
    return torch.add(y, cs_orth(x,y))

def cs_orth_sub(x:torch.Tensor, y:torch.Tensor):
    return torch.sub(y, cs_orth(x,y))
def neg(x: torch.Tensor) -> torch.Tensor:
    return torch.negative(x)

def cs_orth_mul(x:torch.Tensor, y:torch.Tensor):
    return torch.mul(y, cs_orth(x,y))

def div(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.where(torch.abs(y) > EPS, torch.divide(x, y), torch.nan)
def cs_orth_div(x:torch.Tensor, y:torch.Tensor):
    return div(y, cs_orth(x,y))
def inv(x: torch.Tensor) -> torch.Tensor:
    return torch.where(torch.abs(x) > EPS, torch.divide(1., x), torch.nan)

def cs_regres(x: torch.Tensor, y: torch.Tensor):
    # 把inf和nan替换为均值
    mask = torch.isnan(x) | torch.isinf(x) | torch.isnan(y) | torch.isinf(y)
    x[mask] = torch.mean(x[~mask])
    y[mask] = torch.mean(y[~mask])
    ones = torch.ones_like(x)
    X = torch.stack([ones, x], dim=2)
    # print(X.shape)
    # 回归系数的表达式：beta = (X'X)^(-1)X'Y
    beta = torch.matmul(torch.linalg.pinv(torch.matmul(X.transpose(1, 2), X)),
                        torch.matmul(X.transpose(1, 2), y.unsqueeze(2)))
    # print(beta)
    y_hat = torch.matmul(X, beta).squeeze()
    # print(y_hat.shape)
    res = y - y_hat
    res[mask] = torch.nan
    return res

def cs_regpred(x: torch.Tensor, y: torch.Tensor):
    mask = torch.isnan(x) | torch.isinf(x) | torch.isnan(y) | torch.isinf(y)
    x[mask] = torch.mean(x[~mask])
    y[mask] = torch.mean(y[~mask])
    ones = torch.ones_like(x)
    X = torch.stack([ones, x], dim=2)
    # print(X[0])
    # pinv本质是奇异值分解，对于矩阵X，pinv(X) = V * pinv(S) * U'，pinv(S)是S的伪逆，非零奇异值的倒数放到对角线上，其余元素置为零
    beta = torch.matmul(torch.linalg.pinv(torch.matmul(X.transpose(1, 2), X)),
                        torch.matmul(X.transpose(1, 2), y.unsqueeze(2)))
    y_hat = torch.matmul(X, beta).squeeze()
    y_hat[mask] = torch.nan
    return y_hat

def cs_marketcorr2(x:torch.Tensor,d:int)->torch.Tensor:
    '''
    截面相关性计算2
    计算在横截面上与市场中所有股票的(皮尔逊)相关性均值情况
    参数d为选取多少回看区间作为相关性矩阵的维度,一般看样本个数对相关性计算的影响
    算子试运行中20250402,以切片分域的方式求解截面相关性,可以降低时间复杂度,但容易取到未来数据
    '''
    def cs_corr_matrix(x:torch.Tensor):
        mask = ~torch.isnan(x).all(dim=0)
        # 使用掩码过滤张量
        x_final= x[:,mask]
        corr_matrix = torch.corrcoef(x_final)
        corr_mean=torch.nanmean(corr_matrix,dim=0)
        return corr_mean
    
    
    
    res=get_nan_tensor(x.shape,x.device,x.dtype)
    # mask = ~torch.isnan(x).all(dim=1)
    # # 使用掩码过滤张量
    # x= x[mask]
    x_unfold=x.unfold(0,d,d)
    # x_unfold=x.unfold(0,d,d)

    for i in range(x_unfold.shape[0]):
        x_final=x_unfold[i]
        corr_mean=cs_corr_matrix(x_final)
        res[d*i:d*(i+1)]=corr_mean
    isnan=torch.isnan(res[d*(i+1):,:])
    res_final_inday=torch.where(isnan,corr_mean,res[d*(i+1):,:])
    res[d*(i+1):,:]=res_final_inday
    return res

def cs_highlowtag(x:torch.tensor,low:float,high:float,Zero:bool)->torch.Tensor:
    '''
    算子试运行中,若无问题将注册至表内,20250411
    在每一个横截面上取[low,high]区域的个股标记为1,其余为NaN
    使用须知:以相对大小作特征进行标记,不适用于所有字段,可以类比FF三因子模型对股票组合的分组方式(市值等)来理解,并在算子标记后与原字段相乘得到分组后的股票。
    建议搭配截面算子（时序上可能对于个股,不同时段会偶尔出现被分在组外的情况）
    如果需要对多个组合进行切片,可以将参数Zero设置成True,即保证不在组内的个股为0,并与另一组相加后,搭配is_gt算子进行二次标记
    如果需要对多个条件下进行分组,可以考虑使用乘法进行筛选,即同时为1的/一部分为1的与都不为1的
    '''
    
    res=get_nan_tensor(x.shape,x.device,x.dtype)
    # 计算分位数
    quantile_values_low = torch.nanquantile(x, q=low, dim=1,keepdim=True)
    quantile_values_high = torch.nanquantile(x, q=high, dim=1,keepdim=True)
    mask=(x>quantile_values_low)& (x<quantile_values_high)
    res[mask]=1.0
    if Zero:
        res[237:]=torch.nan_to_num(res[237:],0.0)
    return res

def cs_group_demean(x_tensor: torch.Tensor, y_tensor: torch.Tensor) -> torch.Tensor:

    """
    将输入的两个张量（x, y）转换为 Polars DataFrame，执行去均值操作，再转回张量。

    pandas源代码改的，暂时有些不必要的代码，但不影响运行的时间复杂度
    
    参数:
        x_tensor: 输入特征张量，形状为 (time_steps, num_stocks)
        y_tensor: 输入多分类张量，形状为 (time_steps, num_stocks)，元素为数值
    
    返回:
        demeaned_tensor: 去均值后的特征张量，形状与 x_tensor 相同
    """
    import polars as pl
    import numpy as np

    def demean_by_sector(x: pl.DataFrame, y: pl.DataFrame) -> pl.DataFrame:
        # 添加时间索引列
        x = x.with_row_count("time_idx")
        y = y.with_row_count("time_idx")
        
        # 保存原始股票列顺序（排除时间索引列）
        original_columns = [col for col in x.columns if col != "time_idx"]
        
        # 将数据转换为长格式
        x_long = x.melt(id_vars="time_idx", variable_name="stock", value_name="x_value")
        y_long = y.melt(id_vars="time_idx", variable_name="stock", value_name="sector")
        
        # 合并特征数据和分类数据
        combined = x_long.join(y_long, on=["time_idx", "stock"])
        
        # 计算每个时间点-板块组合的均值
        sector_means = combined.group_by(["time_idx", "sector"]).agg(
            pl.col("x_value").mean().alias("group_mean")
        )
        
        # 将板块均值合并回原始数据
        combined = combined.join(sector_means, on=["time_idx", "sector"])
        
        # 计算去均值后的值
        combined = combined.with_columns(
            (pl.col("x_value") - pl.col("group_mean")).alias("demeaned")
        )
        
        # 将数据转换回宽格式并保持原始列顺序
        result = combined.pivot(
            index="time_idx",
            columns="stock",
            values="demeaned",
            aggregate_function=None
        ).sort("time_idx").select(["time_idx"] + original_columns)
        
        # 移除临时添加的时间索引列
        return result.drop("time_idx")

    # 1. 确保张量在 CPU 上
    x_np = x_tensor.cpu().numpy()
    y_np = y_tensor.cpu().numpy()
    
    # 2. 创建列名列表（假设股票名称为 stock_0, stock_1, ..., stock_{num_stocks-1}）
    num_stocks = x_tensor.shape[1]
    stocks = [f"stock_{i}" for i in range(num_stocks)]
    
    # 3. 转换为 Polars DataFrame
    x_df = pl.DataFrame(x_np, stocks)
    y_df = pl.DataFrame(y_np, stocks)
    
    # 4. 执行去均值操作
    demeaned_df = demean_by_sector(x_df, y_df)
    
    # 5. 转换回 NumPy 数组并确保形状正确
    demeaned_np = demeaned_df.to_numpy()
    # 确保形状与输入一致（时间点 × 股票数）
    assert demeaned_np.shape == x_np.shape, "输出形状与输入不一致"
    
    # 6. 转换回张量（可选：返回到原始设备）
    demeaned_tensor = torch.from_numpy(demeaned_np).to(x_tensor.device)
    
    return demeaned_tensor



def cs_group_demean_2(x, y):
    """
    对每个时间点的x在y的每个板块上进行去均值操作
    
    参数:
    x (Tensor): 形状为 (T, N, ...) 的张量，其中T是时间轴，N是股票数量
    y (Tensor): 形状与x相同的分类张量，每个元素表示对应的股票在该时间点的板块
    
    返回:
    Tensor: 形状与x相同的去均值后的张量
    """
    assert x.device.type == "cuda", "Input x must be on GPU!"
    assert y.device.type == "cuda", "Input y must be on GPU!"

    # 获取所有唯一的分类
    classes = torch.unique(y)
    assert len(classes)>100, f"{classes} classes"
    # 初始化结果张量
    z = torch.zeros_like(x)
    
    for c in classes:
        # 创建当前分类的掩码
        mask = (y == c)
        
        # 计算当前分类在每个时间点的总和（沿股票维度求和）
        sum_x = (x * mask).sum(dim=1, keepdim=True)
        # 计算当前分类在每个时间点的股票数量
        count = mask.sum(dim=1, keepdim=True).float()
        
        # 计算均值（避免除以零，但掩码为0时不会影响结果）
        mean_c = sum_x / count
        
        # 计算当前分类的去均值部分
        demeaned_c = x - mean_c
        demeaned_c_part = demeaned_c * mask
        
        # 将当前分类的结果累加到最终结果中
        z += demeaned_c_part
    
    return z

def cs_group_norm(x_tensor: torch.Tensor, y_tensor: torch.Tensor) -> torch.Tensor:
    import polars as pl
    import numpy as np

    def norm_by_sector(x: pl.DataFrame, y: pl.DataFrame) -> pl.DataFrame:
        """
        核心函数：按组别对 x 归一化到 [0, 1]
        
        参数:
            x: 包含特征值的 Polars DataFrame
            y: 包含分类标签的 Polars DataFrame
        
        返回:
            归一化后的 Polars DataFrame
        """
        # 1. 添加时间索引列
        x = x.with_row_count("time_idx")
        y = y.with_row_count("time_idx")
        
        # 2. 保存原始股票列顺序（排除时间索引列）
        original_columns = [col for col in x.columns if col != "time_idx"]
        
        # 3. 将数据转换为长格式（Long Format）
        x_long = x.melt(
            id_vars=["time_idx"],
            variable_name="stock",
            value_name="x_value"
        )
        y_long = y.melt(
            id_vars=["time_idx"],
            variable_name="stock",
            value_name="sector"
        )
        
        # 4. 合并特征数据和分类数据
        combined = x_long.join(y_long, on=["time_idx", "stock"])
        
        # 5. 计算每个时间点-板块组的最小值和最大值
        min_max = combined.group_by(["time_idx", "sector"]).agg([
            pl.col("x_value").min().alias("min_x"),
            pl.col("x_value").max().alias("max_x")
        ])
        
        # 6. 将最小值和最大值合并回原始数据
        combined = combined.join(min_max, on=["time_idx", "sector"])
        
        # 7. 计算归一化值：(x_value - min_x) / (max_x - min_x)
        # 添加极小值防止分母为零
        epsilon = 1e-8
        combined = combined.with_columns(
            normalized = (
                (pl.col("x_value") - pl.col("min_x")) / 
                (pl.col("max_x") - pl.col("min_x") + epsilon)
            )
        )
        
        # 8. 转换回宽格式（Wide Format）
        result = combined.pivot(
            index="time_idx",
            columns="stock",
            values="normalized",
            aggregate_function=None
        ).sort("time_idx").select(["time_idx"] + original_columns)
        
        return result.drop("time_idx")  # 移除临时时间索引列

    # 主流程：张量 → NumPy → Polars → 归一化 → 张量
    # 1. 确保张量在 CPU 上
    x_np = x_tensor.cpu().numpy()
    y_np = y_tensor.cpu().numpy()
    
    # 2. 创建列名列表（假设股票名称为 stock_0, stock_1, ..., stock_{num_stocks-1}）
    num_stocks = x_tensor.shape[1]
    stocks = [f"stock_{i}" for i in range(num_stocks)]
    
    # 3. 转换为 Polars DataFrame
    x_df = pl.DataFrame(x_np, stocks)
    y_df = pl.DataFrame(y_np, stocks)
    
    # 4. 执行归一化操作
    normalized_df = norm_by_sector(x_df, y_df)
    
    # 5. 转换回 NumPy 数组并确保形状正确
    normalized_np = normalized_df.to_numpy()
    assert normalized_np.shape == x_np.shape, "输出形状与输入不一致"
    
    # 6. 转换回张量（可选：返回到原始设备）
    normalized_tensor = torch.from_numpy(normalized_np).to(x_tensor.device)
    
    return normalized_tensor

def cs_mean(x: torch.tensor):
    '''
    求解横截面上有值数据的均值,并将所有有值部分赋值为该均值
    算子试运行中20250418
    '''
    cs_mean = x.nanmean(dim=-1)
    mask=torch.isnan(x)
    x_csmean=cs_mean.unsqueeze(-1).expand((x.shape[0],x.shape[1]))
    x_csmean=x_csmean*(~mask)
    mask2=x_csmean==0
    x_csmean[mask2]=float('nan')
    return x_csmean

def cs_stddev(x: torch.Tensor)->torch.Tensor:
    '''
    求解横截面上有值数据的标准差,并将所有有值部分赋值为该标准差
    算子试运行中20250418
    '''
    cs_demean=x-x.nanmean(dim=-1,keepdim=True)
    x_std = torch.sqrt(torch.nanmean(torch.pow(cs_demean, 2), dim=-1))
    mask=torch.isnan(x)
    x_csstd=x_std.unsqueeze(0).expand((x.shape[1],x.shape[0])).t()
    x_csstd=x_csstd*(~mask)
    mask2=x_csstd==0
    x_csstd[mask2]=float('nan')
    return x_csstd

def ts_std_ratio(x: torch.Tensor, d: int):
    """
    计算大于零的部分和小于零的部分波动率之比
    """
    res = get_nan_tensor(shape=x.shape, device=x.device, dtype=x.dtype)
    x_unfold = x.unfold(0, d, 1)
    
    pos_x = x_unfold[x_unfold >= 0]
    neg_x = x_unfold[x_unfold <  0]
    
    up_volatility = torch.std(pos_x, dim=-1) if len(pos_x) > 0 else torch.tensor(0.0)
    down_volatility = torch.std(neg_x, dim=-1) if len(neg_x) > 0 else torch.tensor(0.0)
    
    res[d - 1:] = up_volatility / (down_volatility + 1e-6)
    
    return res



def ts_median_selq(x: torch.Tensor, cond_var: torch.Tensor, d: int, low_q: float, high_q: float) -> torch.Tensor:
    x_unfold = _unfold(x, d)
    cond_unfold = _unfold(cond_var, d)
    low_value = torch.nanquantile(cond_unfold, low_q, dim=-1, keepdim=True)
    high_value = torch.nanquantile(cond_unfold, high_q, dim=-1, keepdim=True)
    cond = (cond_unfold >= low_value) & (cond_unfold <= high_value)
    x_unfold[~cond] = torch.nan
    return _median(x_unfold).values

def cs_rank_mad(x: torch.Tensor, valid_mask: torch.Tensor = None, pct: bool = True,n_mad: float = 3.0) -> torch.Tensor:
    ranks = x.clone()
    if valid_mask is None:
        valid_mask = torch.isfinite(x)
    ranks[~valid_mask] = torch.finfo(x.dtype).max
    ranks = torch.argsort(torch.argsort(ranks, dim=-1), dim=-1)
    ranks = ranks.to(dtype=x.dtype) + 1
    if pct:
        valid_counts = valid_mask.sum(dim=-1, keepdim=True)
        ranks = ranks / valid_counts
        
        # 中值去极值截断
        median = torch.nanmedian(ranks, dim=-1, keepdim=True)[0]
        abs_dev = torch.abs(ranks - median)
        mad = torch.nanmedian(abs_dev, dim=-1, keepdim=True)[0]
        
        scale = 1.4826  # MAD到标准差的转换系数
        upper_bound = median + n_mad * scale * mad
        lower_bound = median - n_mad * scale * mad
        
        # 确保在[0,1]范围内
        upper_bound = torch.clamp(upper_bound, max=1.0)
        lower_bound = torch.clamp(lower_bound, min=0.0)
        
        ranks = torch.clamp(ranks, min=lower_bound, max=upper_bound)
    ranks[~valid_mask] = torch.nan
    return ranks

def cs_rank_quantile(x: torch.Tensor, valid_mask: torch.Tensor = None, pct: bool = True,trunc_range: tuple = (0.05, 0.95)) -> torch.Tensor:
    ranks = x.clone()
    if valid_mask is None:
        valid_mask = torch.isfinite(x)
    ranks[~valid_mask] = torch.finfo(x.dtype).max
    ranks = torch.argsort(torch.argsort(ranks, dim=-1), dim=-1)
    ranks = ranks.to(dtype=x.dtype) + 1
    if pct:
        valid_counts = valid_mask.sum(dim=-1, keepdim=True)
        ranks = ranks / valid_counts
        
        # 分位数截断
        lower, upper = trunc_range
        q_lower = torch.nanquantile(ranks, lower, dim=-1, keepdim=True)
        q_upper = torch.nanquantile(ranks, upper, dim=-1, keepdim=True)
        ranks = torch.clamp(ranks, min=q_lower, max=q_upper)
    ranks[~valid_mask] = torch.nan
    return ranks


def cs_min_max_norm(x: torch.Tensor, valid_mask: torch.Tensor = None) -> torch.Tensor:
    """
    对每个时间点的横截面数据进行minmax归一化，将数据映射到[0,1]区间
    
    参数:
    x: torch.Tensor - 输入数据
    valid_mask: torch.Tensor - 有效值掩码，默认为None
    
    返回:
    torch.Tensor - 归一化后的数据
    """
    if valid_mask is None:
        valid_mask = torch.isfinite(x)
    
    res = torch.full_like(x, torch.nan)
    
    x_masked = x.clone()
    x_masked[~valid_mask] = torch.nan
    
    temp_min = x.clone()
    temp_min[~valid_mask] = float('inf')
    x_min = temp_min.min(dim=1, keepdim=True).values
    
    temp_max = x.clone()
    temp_max[~valid_mask] = float('-inf')
    x_max = temp_max.max(dim=1, keepdim=True).values
    
    no_valid_rows = valid_mask.sum(dim=1) == 0
    x_min[no_valid_rows] = torch.nan
    x_max[no_valid_rows] = torch.nan
    
    x_range = x_max - x_min
    
    equal_mask = (x_range <= 0)
    
    norm_mask = valid_mask & (~equal_mask)
    res[norm_mask] = (x[norm_mask] - x_min.expand_as(x)[norm_mask]) / x_range.expand_as(x)[norm_mask]
    
    equal_valid_mask = valid_mask & equal_mask
    res[equal_valid_mask] = 0.5
    
    return res

from typing import List
def cs_PCA(x: List[torch.Tensor], n: int, d: int) -> torch.Tensor:
    """
    使用张量运算，在时间截面上对每个时间点的所有股票数据做 PCA，以股票为样本，因子为变量。
    返回指定主成分的时间序列。

    参数:
        x: List[Tensor]，长度为变量数 N, 每个 Tensor 为 [T, S], S 为股票数；
        n: int, 提取前 n 个主成分；
        d: int, 返回第 d 个主成分 (0 ≤ d < n);

    返回:
        Tensor: [T, S]，第 d 个主成分的值
    """
    N = len(x)
    T, S = x[0].shape
    data = torch.stack(x, dim=2)  # [T, S, N]

    # ----------- 标准化处理（每个时间点，按股票维度） -----------
    nan_mask = ~torch.isnan(data)  # [T, S, N]
    count = nan_mask.sum(dim=1, keepdim=True).clamp(min=1)
    mean = torch.where(nan_mask, data, 0).sum(dim=1, keepdim=True) / count
    std = torch.sqrt(torch.where(nan_mask, (data - mean) ** 2, 0).sum(dim=1, keepdim=True) / count)
    standardized = (data - mean) / (std + 1e-8)
    standardized = torch.nan_to_num(standardized, nan=0.0)  # [T, S, N]

    # ----------- 计算每个时间点的协方差矩阵 -----------
    # [T, N, S] @ [T, S, N] → [T, N, N]
    cov = torch.matmul(standardized.transpose(1, 2), standardized) / (S - 1)

    # ----------- 计算主成分（批量特征分解） -----------
    eigvals, eigvecs = torch.linalg.eigh(cov)  # [T, N], [T, N, N]

    # 获取前 n 个主成分向量的索引（按特征值从大到小排序）
    top_indices = torch.argsort(eigvals, descending=True, dim=1)[:, :n]  # [T, n]

    # 用高级索引取第 d 个主成分向量：[T, N]
    batch_indices = torch.arange(T).unsqueeze(1).expand(-1, N)  # [T, N]
    eigvec_d = eigvecs[torch.arange(T), :, top_indices[:, d]]  # [T, N]

    # ----------- 投影到第 d 个主成分方向 -----------
    # [T, S] ← batched matmul: [T, S, N] @ [T, N, 1]
    projected = torch.matmul(standardized, eigvec_d.unsqueeze(2)).squeeze(2)  # [T, S]

    return projected  # [T, S]


def cs_zscore_cdf_norm(x: torch.Tensor, valid_mask: torch.Tensor = None) -> torch.Tensor:
    """
    对截面数据进行zscore标准化，然后映射到正态分布累积概率密度函数的分位数上
    
    参数:
        x: 输入张量，包含因子数据
        valid_mask: 可选，标记有效数据的掩码
        
    返回:
        映射到正态分布分位数的张量
    """
    if valid_mask is None:
        valid_mask = ~torch.isnan(x)
        
    result = torch.full_like(x, float('nan'))

    def custom_normal_cdf(x):
        # 标准正态分布的累积分布函数
        # F(x) = 1/2 * [1 + erf(x/sqrt(2))]
        return 0.5 * (1.0 + torch.erf(x / torch.sqrt(torch.tensor(2.0))))
    
    
    # 对每个截面进行处理
    for i in range(x.shape[0]):
        valid_idx = valid_mask[i]
        if not torch.any(valid_idx):
            continue
            
        curr_x = x[i, valid_idx]
        
        # zscore
        mean = torch.mean(curr_x)
        std = torch.std(curr_x)
        z_scores = (curr_x - mean) / (std + 1e-8)
        
        quantiles = custom_normal_cdf(z_scores)
        
        result[i, valid_idx] = quantiles
        
    return result


def cs_marketcorrw(x:torch.Tensor,y:torch.Tensor)->torch.Tensor:
    '''
    截面相关性加权
    计算在横截面上个股的y特征与市场中所有股票的(皮尔逊)相关性情况(日内数据),得到相关性矩阵
    并与x进行矩阵相乘,即得到横截面上的y相关性加权后的x
    算子试运行中20250517
    '''
    def cs_corr_matrix(x:torch.Tensor):
        mask = ~torch.isnan(x).all(dim=1)
        # 使用掩码过滤张量
        x_final= x[mask,:]
        corr_matrix = torch.corrcoef(x_final.t())
        return corr_matrix
    
    def is_zero(x: torch.Tensor) -> torch.Tensor:
        mask=x==0.0
        x[mask]=float('nan')
        return x

    y_corr_matrix=cs_corr_matrix(y)

    mask_col_nan = ~torch.all(torch.isnan(y_corr_matrix), dim=0)
    mask_col=torch.isnan(y_corr_matrix) & mask_col_nan.unsqueeze(0)
    y_corr_matrix[mask_col] = 0

    mask_row_nan = ~torch.all(torch.isnan(x), dim=1)  # 按行判断

    nan_mask = torch.isnan(x) & mask_row_nan.unsqueeze(1)
    x[nan_mask] = 0
    x_corrw=torch.mm(x,y_corr_matrix)
    x_corrw=is_zero(x_corrw)
    return x_corrw

def cs_orth_add(x:torch.Tensor, y:torch.Tensor):
    return torch.add(y, cs_orth(x,y))

def cs_orth_sub(x:torch.Tensor, y:torch.Tensor):
    return torch.sub(y, cs_orth(x,y))

def cs_orth_mul(x:torch.Tensor, y:torch.Tensor):
    return torch.mul(y, cs_orth(x,y))

def cs_orth_div(x:torch.Tensor, y:torch.Tensor):
    return div(y, cs_orth(x,y))

def cs_schmidt(x: torch.Tensor, y: List[torch.Tensor]) -> torch.Tensor:
    def clean(t: torch.Tensor) -> torch.Tensor:
        mask = torch.isnan(t) | torch.isinf(t)
        if mask.any():
            t[mask] = torch.mean(t[~mask])
        return t
    x = clean(x.clone())
    y = [clean(ys) for ys in y]
    res = x.clone()
    for ys in y:
        res -= (torch.sum(x * ys, dim=1, keepdim=True) / torch.sum(ys**2, dim=1, keepdim=True).clamp(min=1e-12)) * ys
    return res

def cs_schmidt_last(x: List[torch.Tensor]) -> torch.Tensor:
    def clean(t: torch.Tensor) -> torch.Tensor:
        mask = torch.isnan(t) | torch.isinf(t)
        if mask.any():
            t[mask] = torch.mean(t[~mask])
        return t
    x = [clean(xs) for xs in x]
    for i in range(1,len(x)):
        res = x[i].clone()
        for j in range(i):
            res -= (torch.sum(x[i] * x[j], dim=1, keepdim=True) / torch.sum(x[j]**2, dim=1, keepdim=True).clamp(min=1e-12)) * x[j]
        x[i] = res
    return x[-1]

def cs_schmidt_sum(x: List[torch.Tensor]) -> torch.Tensor:
    def clean(t: torch.Tensor) -> torch.Tensor:
        mask = torch.isnan(t) | torch.isinf(t)
        if mask.any():
            t[mask] = torch.mean(t[~mask])
        return t
    x = [clean(xs) for xs in x]
    for i in range(1,len(x)):
        res = x[i].clone()
        for j in range(i):
            res -= (torch.sum(x[i] * x[j], dim=1, keepdim=True) / torch.sum(x[j]**2, dim=1, keepdim=True).clamp(min=1e-12)) * x[j]
        x[i] = res
    return sum(x)

# ------------------------------以下为回测算子，请勿修改------------------------------ #


def cs_certain_value_proportion(x: torch.Tensor, valid_mask: torch.Tensor, certain_value: any) -> torch.Tensor:
    valid_count = ((x - certain_value).abs() < EPS).sum(dim=-1)
    tot_count = valid_mask.sum(dim=-1)
    return valid_count / tot_count


def cs_nan_proportion(x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    valid_count = torch.isfinite(x).sum(dim=-1)
    tot_count = valid_mask.sum(dim=-1)
    return 1 - valid_count / tot_count


def cs_corr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x_demean = x - x.nanmean(dim=-1, keepdim=True)
    y_demean = y - y.nanmean(dim=-1, keepdim=True)

    x_std = torch.sqrt(torch.nansum(torch.pow(x_demean, 2), dim=-1))
    y_std = torch.sqrt(torch.nansum(torch.pow(y_demean, 2), dim=-1))

    numerator = (x_demean * y_demean).nansum(dim=-1)
    denominator = x_std * y_std
    res = numerator / denominator

    res[(x_std < EPS) | (y_std < EPS)] = torch.nan
    return res


def adj_cs_corr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    res = cs_corr(x, y)

    mean_res = torch.nanmean(res)
    std_res = torch.nanstd(res)

    if torch.isclose(std_res, torch.tensor(0.0)):
        return torch.nan  

    return mean_res / std_res

def cs_mean(x: torch.Tensor) -> torch.Tensor:
    return torch.where(torch.isfinite(x), x.nanmean(dim=-1, keepdim=True), torch.tensor(torch.nan, device=x.device))


def get_group_rank_cutpoint(tot_num: int, group_num: int, start_rank: int = 1,
                            device: torch.device = torch.device('cpu')):
    step = tot_num // group_num
    low_rank = torch.arange(start=0, end=group_num * step, step=step)
    high_rank = torch.arange(start=step, end=tot_num + 1, step=step)
    assert len(low_rank) == len(high_rank) == group_num
    high_rank[-1] = tot_num
    low_rank += (start_rank - 1)
    high_rank += (start_rank - 1)
    return low_rank.to(device), high_rank.to(device)


def get_group_quantile_cutpoint(group_num: int, device: torch.device = torch.device('cpu')):
    step = 1 / group_num
    low_rank = torch.tensor([step * x for x in range(group_num)])
    high_rank = torch.tensor([step * (x + 1) for x in range(group_num)])
    assert len(low_rank) == len(high_rank) == group_num
    low_rank[0] = -1
    high_rank[-1] = 1
    return low_rank.to(device), high_rank.to(device)


def cs_fillna_mean(x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    assert x.shape == valid_mask.shape
    res = x.clone()
    finite_mask = torch.isfinite(res) & valid_mask
    res[~finite_mask] = torch.nan
    res_mean = torch.nanmean(res, dim=-1, keepdim=True)
    res[~finite_mask] = res_mean.expand_as(res)[~finite_mask]
    res[~valid_mask] = torch.nan
    return res


def cs_group_mean(x: torch.Tensor, y: torch.Tensor, valid_mask: torch.Tensor, group_num: int) -> torch.Tensor:
    if valid_mask is not None:
        assert x.shape == y.shape == valid_mask.shape
    else:
        assert x.shape == y.shape
    res = x.clone()
    res_rank = cs_rank(res, valid_mask=valid_mask, pct=True)
    group_mean = get_nan_tensor(shape=(res.shape[0], group_num), device=res.device, dtype=res.dtype)
    for group_idx, (low_q, high_q) in enumerate(zip(*get_group_quantile_cutpoint(group_num, device=res.device))):
        cond = (res_rank > low_q) & (res_rank <= high_q)
        y_cond = torch.where(cond, y, torch.tensor(torch.nan, device=res.device))
        group_mean[:, group_idx] = y_cond.nanmean(dim=-1)
    return group_mean


def cs_head_mean(x: torch.Tensor, y: torch.Tensor, valid_mask: torch.Tensor, head_num_list: List[int],
                 tail_num_list: List[int]) -> torch.Tensor:
    if valid_mask is not None:
        assert x.shape == y.shape == valid_mask.shape
    else:
        assert x.shape == y.shape
    res = x.clone()
    res_rank = cs_rank(res, valid_mask=valid_mask, pct=False)
    num_list = head_num_list + tail_num_list
    group_mean = get_nan_tensor(shape=(res.shape[0], len(num_list)), device=res.device, dtype=res.dtype)
    for group_idx, count_num in enumerate(head_num_list, start=0):
        cond = res_rank <= count_num
        y_cond = torch.where(cond, y, torch.tensor(torch.nan, device=res.device))
        group_mean[:, group_idx] = y_cond.nanmean(dim=-1)
    dtype_min = torch.finfo(res_rank.dtype).min
    for group_idx, count_num in enumerate(tail_num_list, start=len(head_num_list)):
        res_rank_max = torch.nan_to_num(res_rank, nan=dtype_min, neginf=dtype_min, posinf=dtype_min).max(dim=-1,
                                                                                                         keepdim=True).values
        cond = res_rank_max - res_rank + 1 <= count_num
        y_cond = torch.where(cond, y, torch.tensor(torch.nan, device=res.device))
        group_mean[:, group_idx] = y_cond.nanmean(dim=-1)
    return group_mean

def cs_demean(x:torch.Tensor):
    '''
    去除截面因子
    '''
    return x - x.nanmean(dim=1, keepdim=True)

def cs_distance(x: torch.Tensor) -> torch.Tensor:
    """获取各股票与截面上其他股票绝对距离的均值

    
    """
    _, N = x.shape
    x = torch.where(x.isnan() | x.isinf(), torch.nanmean(x, dim=1, keepdim=True), x)
    mean_distance = torch.nansum(torch.abs(x.unsqueeze(-1) - x.unsqueeze(1)), dim=1) / (N-1)  # (T,N)
    
    return mean_distance

def cs_decompose(x: torch.Tensor, y: torch.Tensor, n: int =4, act = 'sigmoid') -> torch.Tensor:
    """
    参数：
    n：用n倍的MAD去极值

    功能：
    截面分域算子可以构建为：
    multiply(X, sigmoid(standardize(winsorize(Y))))
    该算子表示，使用因子Y对因子X进行连续分域：
    通过 winsorize 函数对因子Y进行 MAD 去极值、standardize 函数对因子 Y 标准化、
    sigmoid 激活函数对因子Y进行非线性变换，将其映射到区间（0,1）内，最后与因子X相乘。
    """
    res = cs_mad_winsorize(y,n)
    
    valid_mask = res.isfinite()
    max = torch.max(res.masked_fill(~valid_mask, float('-inf')), dim=1, keepdim = True)[0]
    min = torch.min(res.masked_fill(~valid_mask, float('inf')), dim=1, keepdim = True)[0]
    res = (res - min) / (max - min +EPS)
    if act == 'sigmoid':
        res = F.sigmoid(res)
    elif act == 'tanh':
        res = F.tanh(res)
    res = torch.mul(x, res)

    return res


def VOI_B(x,y):
   res=torch.zeros_like(x)
   mask1=(y[:-1]==y[1:])
   mask2=(y[:-1]>y[1:])
   t=x[1:]-x[:-1]
   res[1:][mask1]=t[mask1]
   res[1:][mask2]=x[1:][mask2]
   return res

def VOI_S(x,y):
   res=torch.zeros_like(x)
   mask1=(y[:-1]==y[1:])
   mask2=(y[:-1]<y[1:])
   t=x[1:]-x[:-1]
   res[1:][mask1]=t[mask1]
   res[1:][mask2]=x[1:][mask2]
   return res


    