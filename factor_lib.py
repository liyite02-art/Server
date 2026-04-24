# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 10:45:07 2023

@author: clementine
"""
import numpy as np
import pandas as pd
import heapq
import pickle
from scipy.stats import median_abs_deviation
from statsmodels.regression.rolling import RollingOLS
import statsmodels.api as sm
import os

# 此处std函数中引用的index_dict未更新，不过不影响函数计算的正确性
index_dict = {2010:0,2011:242,2012:486,2013:729,2014:967,2015:1212,2016:1456,
            2017:1700,2018:1944,2019:2187,2020:2431,2021:2674,2022:2917,2023:3159,2024:3401,2025:3423}
start_list = [2010,2011,2012,2013,2014,2015,2016,2017, 2018]
end_list = [2017,2018,2019,2020,2021,2022,2023,2024]
# end_list = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
test_dict = {2010:(2017,2018),2011:(2018,2019),2012:(2019,2020),2013:(2020,2021),2014:(2021,2022),2015:(2022,2023),2016:(2023,2024),2017:(2024,2025)}


# def standardize_linear(data, factorname, window, year_idx, lower_percentile=1, upper_percentile=99):  ### 待更改
#     """
#     进行线性标准化的函数，并应用截断以减少极值的影响。
#
#     Parameters
#     ----------
#     data : 包含因子值的dataframe;
#     factorname : 不包含窗口期的因子名字;
#     window : 一个窗口期;
#     lower_percentile : 下截断百分位数 (默认为1);
#     upper_percentile : 上截断百分位数 (默认为99);
#
#     Returns
#     -------
#     归一化后的dataframe.
#     """
#     for n in window:
#         thefactor = factorname.lower() + str(n)
#         thedata = data.loc[:, data.columns.get_level_values(1).isin([thefactor])].copy()  # 取出因子值的dataframe
#
#
#         thefactor = factorname + str(n)
#         for column in thedata.columns:
#             to_stand = thedata[column].copy()  # 取出因子值的一列
#
#             #选取数据
#             start_index = index_dict[start_list[0]]
#             end_index = index_dict[end_list[year_idx]]
#             stand_1 = to_stand.iloc[start_index:end_index]
#             stand_2 = to_stand.iloc[end_index:]
#
#             if len(stand_1) == sum(stand_1.isnull()):
#                 stand_1.fillna(0)
#             else:
#                 # 应用截断
#                 lower_bound = np.percentile(stand_1.dropna(), lower_percentile)
#                 upper_bound = np.percentile(stand_1.dropna(), upper_percentile)
#                 stand_1 = np.clip(stand_1, lower_bound, upper_bound)
#
#             _stand = pd.concat([stand_1,stand_2])
#
#             # 归一化的过程
#             index = _stand.first_valid_index()
#             if index is not None:
#                 first_non_nan_pos = _stand.index.get_loc(index)
#                 numdelay = 500
#                 n_plus_500_pos = first_non_nan_pos + numdelay
#                 if n_plus_500_pos < len(_stand):
#                     _stand[first_non_nan_pos:].replace([np.inf, -np.inf], np.nan, inplace=True)
#                     _stand[first_non_nan_pos:].fillna(method='ffill', inplace=True)
#                     cummax = _stand[first_non_nan_pos:].cummax()
#                     cummin = _stand[first_non_nan_pos:].cummin()
#                     numerator = 2 * (_stand[first_non_nan_pos:] - cummin)
#                     denominator = cummax - cummin
#                     denominator.replace(to_replace=0, method='ffill', inplace=True)
#                     _stand[first_non_nan_pos:] = -1 + numerator / denominator
#                     _stand[:n_plus_500_pos] = 0
#                     _stand = _stand.fillna(0)
#
#             data[column] = _stand
#
#     return data
#
#
# def standardize_mad(data, factorname, window='no'):
#     """
#     使用5倍MAD进行缩尾处理。
#
#     Parameters
#     ----------
#     data : 包含因子值的dataframe;
#     factorname : 不包含窗口期的因子名字;
#     window : 一个窗口期;
#
#     Returns
#     -------
#     处理后的dataframe.
#     """
#     if window == 'no':
#         thefactor = factorname
#     else:
#         thefactor = factorname + str(window)
#
#     thedata = data.loc[:, data.columns.get_level_values(1).isin([thefactor])].copy()  # 取出因子值的dataframe
#
#     for column in thedata.columns:
#         to_stand = thedata[column].copy()  # 取出因子值的一列
#         for i in range(7):
#         # for i in range(2):
#             #选取数据
#             start_index = index_dict[start_list[0]]
#             end_index = index_dict[end_list[i]]
#             stand_1 = to_stand.iloc[start_index:end_index]
#             stand_2 = to_stand.iloc[end_index+1:]
#
#             #计算MAD
#             mad = median_abs_deviation(stand_1.dropna())
#             #计算中位数
#             median = np.median(stand_1.dropna())
#
#             # 应用5倍MAD缩尾处理
#             lower_bound = median - 5 * mad
#             upper_bound = median + 5 * mad
#             # 缩尾
#             stand_1 = np.clip(stand_1, lower_bound, upper_bound)
#
#             _stand = pd.concat([stand_1,stand_2])
#
#             #原有的归一化过程，首次运行时，按需要修改。
#             index = _stand.first_valid_index()
#             if index is not None:
#                 first_non_nan_pos = _stand.index.get_loc(index)
#                 # numdelay = 500
#                 numdelay = 500
#                 n_plus_500_pos = first_non_nan_pos + numdelay
#                 if n_plus_500_pos < len(_stand):
#                     _stand[first_non_nan_pos:].replace([np.inf, -np.inf], np.nan, inplace=True)
#                     _stand[first_non_nan_pos:].fillna(method='ffill', inplace=True)
#                     cummax = _stand[first_non_nan_pos:].cummax()
#                     cummin = _stand[first_non_nan_pos:].cummin()
#                     numerator = 2 * (_stand[first_non_nan_pos:] - cummin)
#                     denominator = cummax - cummin
#                     denominator.replace(to_replace=0, method='ffill', inplace=True)
#                     _stand[first_non_nan_pos:] = -1 + numerator / denominator
#                     _stand[:n_plus_500_pos] = 0
#                     _stand = _stand.fillna(0)
#
#             a, b = column
#             data[(a, b+"_"+str(end_list[i]))] = _stand
#
#     return data
#
#
# def ranking_standardize(data,factorname,window='no'):
#     """
#     进行排名标准化的函数
#
#     Parameters
#     ----------
#     data : 包含因子值的dataframe;
#     factorname : 不包含窗口期的因子名字;
#     window : 一个窗口期;
#
#     Returns
#     -------
#     None.
#
#     """
#     if window=='no':
#         thefactor=factorname
#     else:
#         thefactor=factorname+str(window)
#
#     thedata=data.loc[:,data.columns.get_level_values(1).isin([thefactor])].copy()
#
#
#     for column in thedata.columns:
#         to_stand = thedata[column].copy()  # 取出因子值的一列
#         index = to_stand.first_valid_index()
#         if index is not None:
#             first_non_nan_pos = to_stand.index.get_loc(index)
#             numdelay = 500
#             n_plus_500_pos = first_non_nan_pos + numdelay
#         if n_plus_500_pos < len(to_stand):
#             for i in range(n_plus_500_pos,len(data)):
#                 to_stand_1 = thedata[column].iloc[:i+1].copy()  # 取出因子值的前 i+1 行
#                 # print(to_stand_1)
#                 # 使用前 i+1 行的数据进行排名
#                 ranks = to_stand_1.rank(method='min', na_option='keep')
#                 ranks = (ranks - ranks.min()) / (ranks.max() - ranks.min())  # 归一化到 0 到 1
#                 ranks = ranks * 2 - 1  # 映射到 -1 到 1
#
#                 # 更新 data 中的值
#                 to_stand.iloc[i] = ranks.iloc[-1]  # 更新第 i 行的值
#             to_stand[:n_plus_500_pos] = 0
#             to_stand = to_stand.fillna(0)
#         data[column] = to_stand
#
#     return data

kd_time_lst = [5,7,8,9]
def KD(data, window, futures_universe, time_freq='day'):
    """
    计算期货的K值和D值，每遇到新的n参，计算后同时保存到原始数据表中。滑动平均权重系数为3/5, 2/5.

    Parameters
    ----------
    data : pd.DataFrame
        数据的完整大表，需包含需要计算的期货数据的loww，highw和closew值以及已有的一些K,D值.
    n : int or list
        rsv的窗口.
    save_path : string
        新K,D值保存的数据路径，一般为原data的路径以覆盖原data数据表

    Returns
    -------
    None.

    """
    ret = pd.DataFrame()
    data = data.loc[futures_universe].copy()
    open = data['open'].unstack(level=0)
    openw = data['openw'].unstack(level=0)

    for a in window:
        k_low_list = data.groupby('future').apply(lambda group: group.loww.rolling(a, min_periods=a).min()).reset_index(
            level=0, drop=True)
        k_low_list.fillna(value=data.groupby('future').loww.transform(lambda group: group.expanding().min()),
                          inplace=True)
        k_high_list = data.groupby('future').apply(
            lambda group: group.highw.rolling(a, min_periods=a).max()).reset_index(level=0, drop=True)
        k_high_list.fillna(value=data.groupby('future').highw.transform(lambda group: group.expanding().max()),
                           inplace=True)
        data['k_rsv'] = (data['closew'] - k_low_list) / (k_high_list - k_low_list) * 100
        data['K_value'] = data.groupby('future').k_rsv.apply(lambda x: x.ewm(com=2).mean()).reset_index(
            level=0, drop=True)
        data['D_value'] = data.groupby('future').K_value.apply(lambda x: x.ewm(com=2).mean()).reset_index(
            level=0, drop=True)

        temp_k = data['K_value'].unstack(level=0)
        new_columns1 = pd.MultiIndex.from_tuples([(col, f'k{a}') for col in temp_k.columns],names=['future', 'price'])
        temp_k.columns = new_columns1

        temp_d = data['D_value'].unstack(level=0)
        new_columns2 = pd.MultiIndex.from_tuples([(col, f'd{a}') for col in temp_d.columns],names=['future', 'price'])
        temp_d.columns = new_columns2

        ret = pd.concat([ret,temp_k, temp_d], axis=1)

    open.columns = pd.MultiIndex.from_product([open.columns, ["open"]], names=["future", "price"])
    openw.columns = pd.MultiIndex.from_product([openw.columns, ["openw"]], names=["future", "price"])
    ret = pd.concat([ret, open, openw], axis=1)
    return ret

# =============================================================================
# 
# """
# KD指标计算
# """
# def KD(data, n, save_path=None):
#     """
#     计算期货的K值和D值，每遇到新的n参，计算后同时保存到原始数据表中。滑动平均权重系数为3/5, 2/5.
# 
#     Parameters
#     ----------
#     data : pd.DataFrame
#         数据的完整大表，需包含需要计算的期货数据的loww，highw和closew值以及已有的一些K,D值.
#     n : int or list
#         rsv的窗口.
#     save_path : string
#         新K,D值保存的数据路径，一般为原data的路径以覆盖原data数据表
# 
#     Returns
#     -------
#     None.
# 
#     """
#     if type(n) == list:
#         for a in n:
#             data=cal_kd(data, a, save_path=save_path)
#     elif type(n) == int: 
#         data=cal_kd(data, n, save_path=save_path) 
#     return data
# 
# def cal_kd(data, n, save_path):
#     for (future, _) in data.columns:
#         if (future, 'K_'+str(n)) in data.columns and (future, 'D_'+str(n)) in data.columns:
#             print('K and D value are already prepared!')
#             return
#         break
#         
#     for (future, _) in data.columns:
#         if (future, 'K_'+str(n)) in data.columns and (future, 'D_'+str(n)) in data.columns:
#             continue
#         close = data[(future, 'closew')]
#         high = data[(future, 'highw')]
#         low = data[(future, 'loww')]
#         k_low_list = low.rolling(n, min_periods=n).min()
#         k_low_list.fillna(value=low.expanding().min(), inplace=True)
#         k_high_list = high.rolling(n, min_periods=n).max()
#         k_high_list.fillna(value=high.expanding().max(), inplace=True)
#         k_rsv = (close - k_low_list) / (k_high_list - k_low_list) * 100
#         data[(future, 'K_'+str(n))] = k_rsv.ewm(com=2).mean()
#         data[(future, 'D_'+str(n))] = data[(future, 'K_'+str(n))].ewm(com=2).mean()
#     
#     # if save_path == None:
#     #     save_path = input('New K and D are calculated.\n YOU MUST INPUT DATA SAVE PATH FOR THEM! :')
#     # import pickle
#     # f = open(save_path, 'wb')
#     # pickle.dump(data, f)
#     # f.close()
#     
#     # print('K and D value are calculated and saved successfully!')
# 
#     return data
# =============================================================================

"""
aroon指标计算
"""

def Aroon(data, window, futures_universe, time_freq='day'):
    ret = pd.DataFrame()
    data = data.loc[futures_universe].copy()
    open = data['open'].unstack(level=0)
    openw = data['openw'].unstack(level=0)

    for a in window:
        rolling_high = data.groupby('future').apply(lambda group: group['closew'].rolling(a,min_periods=a).apply(np.argmax)).reset_index(level=0, drop=True)
        rolling_low = data.groupby('future').apply(lambda group: group['closew'].rolling(a,min_periods=a).apply(np.argmin)).reset_index(level=0, drop=True)

        data['up'] = 100 * (a - (rolling_high + 1)) / a
        data['down'] = 100 * (a - (rolling_low + 1)) / a

        temp_up = data['up'].unstack(level=0)
        new_columns_up = pd.MultiIndex.from_tuples([(col, f'up{a}') for col in temp_up.columns],names=['future', 'price'])
        temp_down = data['down'].unstack(level=0)
        new_columns_down = pd.MultiIndex.from_tuples([(col, f'down{a}') for col in temp_down.columns],names=['future', 'price'])

        temp_up.columns = new_columns_up
        temp_down.columns = new_columns_down

        ret = pd.concat([ret, temp_up, temp_down], axis=1)

    open.columns = pd.MultiIndex.from_product([open.columns, ["open"]], names=["future", "price"])
    openw.columns = pd.MultiIndex.from_product([openw.columns, ["openw"]], names=["future", "price"])
    ret = pd.concat([ret, open, openw], axis=1)

    return ret

"""
QIML1028(BIAS)指标计算
"""
bias_time_lst = [6, 9, 14,21,30,43]
def BIAS(data, window, futures_universe, time_freq='day'):
    """
    计算期货的BIAS值，每遇到新的n参，计算后同时加入到原始数据表中。

    Parameters
    ----------
    data : pd.DataFrame
        数据的完整大表，需包含需要计算的期货数据的closew。
    n : int or list
        BIAS的窗口。

    Returns
    -------
    DataFrame.
    """
    ret = pd.DataFrame()
    data = data.loc[futures_universe].copy()
    open = data['open'].unstack(level=0)
    openw = data['openw'].unstack(level=0)

    for a in window:
        data['BIAS'] = data.groupby('future').apply(lambda group: -100 * (group['closew'] - group['closew'].rolling(a).mean()) / group['close'].rolling(a).mean()).reset_index(level=0,drop=True)

        temp_bias = data['BIAS'].unstack(level=0)
        new_columns = pd.MultiIndex.from_tuples([(col, f'bias{a}') for col in temp_bias.columns],names=['future', 'price'])
        temp_bias.columns = new_columns
        ret = pd.concat([ret, temp_bias], axis=1)
        
    open.columns = pd.MultiIndex.from_product([open.columns, ["open"]], names=["future", "price"])
    openw.columns = pd.MultiIndex.from_product([openw.columns, ["openw"]], names=["future", "price"])
    ret = pd.concat([ret, open, openw], axis=1)

    return ret

"""
QIML0218指标计算
"""
br_time_lst = [6, 9, 14, 21, 30, 43]
def BR(data, window, futures_universe, time_freq='day'):
    """
    计算期货的BR值，每遇到新的n参，计算后同时加入到原始数据表中。

    Parameters
    ----------
    data : pd.DataFrame
        数据的完整大表，需包含需要计算的期货数据的loww，highw和closew。
    n : int or list
        br的窗口。
    save_path : string
        新BR值保存的数据路径，一般为原data的路径以覆盖原data数据表

    Returns
    -------
    None.
    """
    ret = pd.DataFrame()
    data = data.loc[futures_universe].copy()
    open = data['open'].unstack(level=0)
    openw = data['openw'].unstack(level=0)

    for a in window:
        data['inner1'] = data.groupby('future').apply(lambda group:(group['highw'] - group['closew'].shift(1)).clip(lower=0)).reset_index(level=0,drop=True)
        data['inner2'] = data.groupby('future').apply(lambda group:(group['closew'].shift(1) - group['loww']).clip(lower=0)).reset_index(level=0,drop=True)
        data['BR'] = data.groupby('future').inner1.rolling(a).sum().reset_index(level=0,drop=True) / data.groupby('future').inner2.rolling(a).sum().reset_index(level=0,drop=True)

        temp_br = data['BR'].unstack(level=0)
        new_columns = pd.MultiIndex.from_tuples([(col, f'br{a}') for col in temp_br.columns],names=['future', 'price'])
        temp_br.columns = new_columns

        ret = pd.concat([ret, temp_br], axis=1)

    open.columns = pd.MultiIndex.from_product([open.columns, ["open"]], names=["future", "price"])
    openw.columns = pd.MultiIndex.from_product([openw.columns, ["openw"]], names=["future", "price"])
    ret = pd.concat([ret, open, openw], axis=1)
    return ret

"""
WQAlpha006指标计算
"""

def WQAlpha006(data, window, futures_universe, time_freq='day'):
    ret = pd.DataFrame()
    data = data.loc[futures_universe].copy()
    open = data['open'].unstack(level=0)
    openw = data['openw'].unstack(level=0)

    for a in window:
        data['wqa006'] = data.groupby('future').apply(lambda group: -1 * group['openw'].rolling(a,min_periods=a).corr(group['volume99'])).reset_index(level=0, drop=True)
        temp_wqa = data['wqa006'].unstack(level=0)
        new_columns = pd.MultiIndex.from_tuples([(col, f'wqa006{a}') for col in temp_wqa.columns],names=['future', 'price'])
        temp_wqa.columns = new_columns
        ret = pd.concat([ret, temp_wqa], axis=1)

    open.columns = pd.MultiIndex.from_product([open.columns, ["open"]], names=["future", "price"])
    openw.columns = pd.MultiIndex.from_product([openw.columns, ["openw"]], names=["future", "price"])
    ret = pd.concat([ret, open, openw], axis=1)

    return ret


"""
WQAlpha005指标计算
"""

def WQAlpha005(data, window, futures_universe, time_freq='day'):
    ret = pd.DataFrame()
    data = data.loc[futures_universe].copy()
    open = data['open'].unstack(level=0)
    openw = data['openw'].unstack(level=0)

    for a in window:
        data['wqa005'] = data.groupby('future').apply(lambda group: -1 * (group['openw'] - group['settlementw'].rolling(a).mean()) * (group['closew'] - group['settlementw'])).reset_index(level=0, drop=True)
        temp_wqa = data['wqa005'].unstack(level=0)
        new_columns = pd.MultiIndex.from_tuples([(col, f'wqa005{a}') for col in temp_wqa.columns],names=['future', 'price'])
        temp_wqa.columns = new_columns
        ret = pd.concat([ret, temp_wqa], axis=1)

    open.columns = pd.MultiIndex.from_product([open.columns, ["open"]], names=["future", "price"])
    openw.columns = pd.MultiIndex.from_product([openw.columns, ["openw"]], names=["future", "price"])
    ret = pd.concat([ret, open, openw], axis=1)

    return ret


"""
CVI指标计算
"""

def CVI(data, window, futures_universe, time_freq='day'):
    ret = pd.DataFrame()
    data = data.loc[futures_universe].copy()
    open = data['open'].unstack(level=0)
    openw = data['openw'].unstack(level=0)

    for a in window:
        data['cvi'] = data.groupby('future').apply(lambda group: (group['highw'] - group['loww']).ewm(span=a).mean().diff(a) / (group['highw'] - group['loww']).ewm(span=a).mean() * 100).reset_index(level=0, drop=True)
        temp_cvi = data['cvi'].unstack(level=0)
        new_columns = pd.MultiIndex.from_tuples([(col, f'cvi{a}') for col in temp_cvi.columns],names=['future', 'price'])
        temp_cvi.columns = new_columns
        ret = pd.concat([ret, temp_cvi], axis=1)

    open.columns = pd.MultiIndex.from_product([open.columns, ["open"]], names=["future", "price"])
    openw.columns = pd.MultiIndex.from_product([openw.columns, ["openw"]], names=["future", "price"])
    ret = pd.concat([ret, open, openw], axis=1)

    return ret


"""
ATRC指标计算
"""

def ATRC(data, window, futures_universe, time_freq='day'):
    ret = pd.DataFrame()
    data = data.loc[futures_universe].copy()
    open = data['open'].unstack(level=0)
    openw = data['openw'].unstack(level=0)

    for a in window:
        data['atrc'] = data.groupby('future').apply(lambda group:
                        (
                            ((group['highw'] - group['loww']) > abs(group['closew'].shift(1) - group['highw'])) & ((group['highw'] - group['loww']) > abs(group['closew'].shift(1) - group['loww'])) * (group['highw'] - group['loww'])
                            +
                            ((abs(group['closew'].shift(1) - group['highw']) > (group['highw'] - group['loww'])) & (abs(group['closew'].shift(1) - group['highw']) > abs(group['closew'].shift(1) - group['loww']))) * abs(group['closew'].shift(1) - group['highw'])
                            +
                            ((abs(group['closew'].shift(1) - group['loww']) > (group['highw'] - group['loww'])) & (abs(group['closew'].shift(1) - group['loww']) > abs(group['closew'].shift(1) - group['highw']))) * abs(group['closew'].shift(1) - group['loww'])
                        ).rolling(a).mean() / group['close']).reset_index(level=0, drop=True)
        temp_atrc = data['atrc'].unstack(level=0)
        new_columns = pd.MultiIndex.from_tuples([(col, f'atrc{a}') for col in temp_atrc.columns],names=['future', 'price'])
        temp_atrc.columns = new_columns
        ret = pd.concat([ret, temp_atrc], axis=1)

    open.columns = pd.MultiIndex.from_product([open.columns, ["open"]], names=["future", "price"])
    openw.columns = pd.MultiIndex.from_product([openw.columns, ["openw"]], names=["future", "price"])
    ret = pd.concat([ret, open, openw], axis=1)

    return ret


"""
ATR指标计算
"""

def ATR(data, window, futures_universe, time_freq='day'):
    ret = pd.DataFrame()
    data = data.loc[futures_universe].copy()
    open = data['open'].unstack(level=0)
    openw = data['openw'].unstack(level=0)

    for a in window:
        data['atr'] = data.groupby('future').apply(lambda group:
                        (
                            ((group['highw'] - group['loww']) > abs(group['closew'].shift(1) - group['highw'])) & ((group['highw'] - group['loww']) > abs(group['closew'].shift(1) - group['loww'])) * (group['highw'] - group['loww'])
                            +
                            ((abs(group['closew'].shift(1) - group['highw']) > (group['highw'] - group['loww'])) & (abs(group['closew'].shift(1) - group['highw']) > abs(group['closew'].shift(1) - group['loww']))) * abs(group['closew'].shift(1) - group['highw'])
                            +
                            ((abs(group['closew'].shift(1) - group['loww']) > (group['highw'] - group['loww'])) & (abs(group['closew'].shift(1) - group['loww']) > abs(group['closew'].shift(1) - group['highw']))) * abs(group['closew'].shift(1) - group['loww'])
                        ).rolling(a).mean()).reset_index(level=0, drop=True)
        temp_atr = data['atr'].unstack(level=0)
        new_columns = pd.MultiIndex.from_tuples([(col, f'atr{a}') for col in temp_atr.columns],names=['future', 'price'])
        temp_atr.columns = new_columns
        ret = pd.concat([ret, temp_atr], axis=1)

    open.columns = pd.MultiIndex.from_product([open.columns, ["open"]], names=["future", "price"])
    openw.columns = pd.MultiIndex.from_product([openw.columns, ["openw"]], names=["future", "price"])
    ret = pd.concat([ret, open, openw], axis=1)

    return ret


"""
BBIC指标计算
"""

def BBIC(data, window, futures_universe, time_freq='day'):
    ret = pd.DataFrame()
    data = data.loc[futures_universe]
    open = data['open'].unstack(level=0)
    openw = data['openw'].unstack(level=0)

    for M1, M2, M3, M4 in window:
        data['bbic'] = data.groupby('future').apply(lambda group:
                        (
                            group['closew'].rolling(window=M1).mean()
                            + group['closew'].rolling(window=M2).mean()
                            + group['closew'].rolling(window=M3).mean()
                            + group['closew'].rolling(window=M4).mean()
                        ) / group['closew']).reset_index(level=0, drop=True)
        temp_bbic = data['bbic'].unstack(level=0)
        new_columns = pd.MultiIndex.from_tuples([(col, 'bbic' + str([M1, M2, M3, M4])) for col in temp_bbic.columns],names=['future', 'price'])
        temp_bbic.columns = new_columns
        ret = pd.concat([ret, temp_bbic], axis=1)

    open.columns = pd.MultiIndex.from_product([open.columns, ["open"]], names=["future", "price"])
    openw.columns = pd.MultiIndex.from_product([openw.columns, ["openw"]], names=["future", "price"])
    ret = pd.concat([ret, open, openw], axis=1)

    return ret

"""
CV指标计算
"""

def CV(data, window, futures_universe, time_freq='day'):
    ret = pd.DataFrame()
    data = data.loc[futures_universe].copy()
    open = data['open'].unstack(level=0)
    openw = data['openw'].unstack(level=0)

    for N, M in window:
        data['cv'] = data.groupby('future').apply(lambda group:
                        (
                            (group['highw'] - group['loww']).rolling(N).mean() -
                            (group['highw'] - group['loww']).rolling(N).mean().shift(M)
                        ) / (group['highw'] - group['loww']).rolling(N).mean().shift(M)).reset_index(level=0, drop=True)
        temp_cv = data['cv'].unstack(level=0)
        new_columns = pd.MultiIndex.from_tuples([(col, 'cv' + str([N, M])) for col in temp_cv.columns],names=['future', 'price'])
        temp_cv.columns = new_columns
        ret = pd.concat([ret, temp_cv], axis=1)

    open.columns = pd.MultiIndex.from_product([open.columns, ["open"]], names=["future", "price"])
    openw.columns = pd.MultiIndex.from_product([openw.columns, ["openw"]], names=["future", "price"])
    ret = pd.concat([ret, open, openw], axis=1)

    return ret

"""
CLOSEW指标计算
"""
closew_time_lst = [3, 5, 8, 12, 16, 20]
def CLOSEW(data, window, futures_universe, time_freq='day'):
    """
    计算期货的CLOSEW向下移动n个交易日的值，计算后同时加入到原始数据表中。

    Parameters
    ----------
    data : pd.DataFrame
        数据的完整大表，需包含需要计算的期货数据的closew，openw，highw和loww。

    Returns
    -------
    ret : DataFrame.
    """
    ret = pd.DataFrame()
    data = data.loc[futures_universe].copy()
    open = data['open'].unstack(level=0)
    openw = data['openw'].unstack(level=0)
    closew = data['closew'].unstack(level=0)


    for a in window:
        temp = closew.copy()
        temp = temp.shift(a)
        new_columns = pd.MultiIndex.from_tuples([(col, f'closew{a}') for col in temp.columns],names=['future', 'price'])
        temp.columns = new_columns
        ret = pd.concat([ret, temp], axis=1)
        
    open.columns = pd.MultiIndex.from_product([open.columns, ["open"]], names=["future", "price"])
    openw.columns = pd.MultiIndex.from_product([openw.columns, ["openw"]], names=["future", "price"])
    ret = pd.concat([ret, open, openw], axis=1)

    return ret

"""
VMACD指标计算
"""

def VMACD(data, window, futures_universe, time_freq='day'):
    ret = pd.DataFrame()
    data = data.loc[futures_universe].copy()
    open = data['open'].unstack(level=0)
    openw = data['openw'].unstack(level=0)

    for n1, n2, m in window:
        data['vmacd'] = data.groupby('future').apply(lambda group:
                        (
                            group['volume99'].rolling(n1).mean() - group['volume99'].rolling(n2).mean()
                        ) - (group['volume99'].rolling(n1).mean() - group['volume99'].rolling(n2).mean()).rolling(m).mean()).reset_index(level=0, drop=True)
        temp_vmacd = data['vmacd'].unstack(level=0)
        new_columns = pd.MultiIndex.from_tuples([(col, 'vmacd' + str([n1, n2, m])) for col in temp_vmacd.columns],names=['future', 'price'])
        temp_vmacd.columns = new_columns
        ret = pd.concat([ret, temp_vmacd], axis=1)

    open.columns = pd.MultiIndex.from_product([open.columns, ["open"]], names=["future", "price"])
    openw.columns = pd.MultiIndex.from_product([openw.columns, ["openw"]], names=["future", "price"])
    ret = pd.concat([ret, open, openw], axis=1)

    return ret

"""
TEMA指标计算
"""
# 使用正态归一方法
def TEMA(data, window, futures_universe, time_freq='day'):
    ret = pd.DataFrame()
    data = data.loc[futures_universe].copy()
    open = data['open'].unstack(level=0)
    openw = data['openw'].unstack(level=0)

    for n in window:
        data['tema'] = data.groupby('future').apply(lambda group:
                        3 * group['closew'].rolling(n).mean()
                        - 3 * group['closew'].rolling(n).mean().rolling(n).mean()
                        + group['closew'].rolling(n).mean().rolling(n).mean().rolling(n).mean()).reset_index(level=0, drop=True)
        temp_tema = data['tema'].unstack(level=0)
        new_columns = pd.MultiIndex.from_tuples([(col, f'tema{n}') for col in temp_tema.columns],names=['future', 'price'])
        temp_tema.columns = new_columns
        ret = pd.concat([ret, temp_tema], axis=1)

    open.columns = pd.MultiIndex.from_product([open.columns, ["open"]], names=["future", "price"])
    openw.columns = pd.MultiIndex.from_product([openw.columns, ["openw"]], names=["future", "price"])
    ret = pd.concat([ret, open, openw], axis=1)

    return ret

"""
EMV指标计算
"""
emv_time_lst = [1, 2, 3, 4, 5, 6]
def EMV(data, window, futures_universe, time_freq='day'):
    ret = pd.DataFrame()
    data = data.loc[futures_universe].copy()
    open = data['open'].unstack(level=0)
    openw = data['openw'].unstack(level=0)

    for a in window:
        data['dm'] = ((data['highw'] + data['loww']) - (data['highw'].shift(1) + data['loww'].shift(1))) / 2
        data['br'] = data['volume99'] / (data['highw'] - data['loww'])
        data['emv'] = data.groupby('future').apply(lambda group: (group['dm'] / group['br']).rolling(a).sum()).reset_index(level=0,drop=True)

        temp_emv = data['emv'].unstack(level=0)
        new_columns = pd.MultiIndex.from_tuples([(col, f'emv{a}') for col in temp_emv.columns],names=['future', 'price'])
        temp_emv.columns = new_columns
        ret = pd.concat([ret, temp_emv], axis=1)

    open.columns = pd.MultiIndex.from_product([open.columns, ["open"]], names=["future", "price"])
    openw.columns = pd.MultiIndex.from_product([openw.columns, ["openw"]], names=["future", "price"])
    ret = pd.concat([ret, open, openw], axis=1)
    return ret

"""
CMO指标计算
"""

def CMO(data, window, futures_universe, time_freq='day'):
    ret = pd.DataFrame()
    data = data.loc[futures_universe].copy()
    open = data['open'].unstack(level=0)
    openw = data['openw'].unstack(level=0)

    for n in window:
        data['cmo'] = data.groupby('future').apply(lambda group:
                        (
                            (
                                (group['closew'] > group['closew'].shift(1)) * (group['closew'] - group['closew'].shift(1)).rolling(n).sum()
                                - (group['closew'] < group['closew'].shift(1)) * (group['closew'].shift(1) - group['closew']).rolling(n).sum()
                            ) / (
                                (group['closew'] > group['closew'].shift(1)) * (group['closew'] - group['closew'].shift(1)).rolling(n).sum()
                                + (group['closew'] < group['closew'].shift(1)) * (group['closew'].shift(1) - group['closew']).rolling(n).sum()
                            )
                        )).reset_index(level=0, drop=True)
        temp_cmo = data['cmo'].unstack(level=0)
        new_columns = pd.MultiIndex.from_tuples([(col, f'cmo{n}') for col in temp_cmo.columns],names=['future', 'price'])
        temp_cmo.columns = new_columns
        ret = pd.concat([ret, temp_cmo], axis=1)

    open.columns = pd.MultiIndex.from_product([open.columns, ["open"]], names=["future", "price"])
    openw.columns = pd.MultiIndex.from_product([openw.columns, ["openw"]], names=["future", "price"])
    ret = pd.concat([ret, open, openw], axis=1)

    return ret


"""
PSY指标计算
"""

def PSY(data, window, futures_universe, time_freq='day'):
    ret = pd.DataFrame()
    data = data.loc[futures_universe]
    open = data['open'].unstack(level=0)
    openw = data['openw'].unstack(level=0)

    for n in window:
        data['psy'] = data.groupby('future').apply(lambda group:
                        group['closew'].rolling(n).apply(lambda x : (x > x.shift(1)).sum() / n, raw=False)
                        ).reset_index(level=0, drop=True)
        temp_psy = data['psy'].unstack(level=0)
        new_columns = pd.MultiIndex.from_tuples([(col, f'psy{n}') for col in temp_psy.columns],names=['future', 'price'])
        temp_psy.columns = new_columns
        ret = pd.concat([ret, temp_psy], axis=1)

    open.columns = pd.MultiIndex.from_product([open.columns, ["open"]], names=["future", "price"])
    openw.columns = pd.MultiIndex.from_product([openw.columns, ["openw"]], names=["future", "price"])
    ret = pd.concat([ret, open, openw], axis=1)

    return ret


"""
MAT指标计算
"""

def MAT(data, window_tuples, futures_universe):
    ret = pd.DataFrame()
    data = data.loc[futures_universe].copy()
    open = data['open'].unstack(level=0)
    openw = data['openw'].unstack(level=0)

    for n1, n2 in window_tuples:
        data['mat'] = data.groupby('future').apply(lambda group:
                        (group['volume99']/group['open_interest']).rolling(n1).mean()
                        / (group['volume99']/group['open_interest']).rolling(n2).mean()
                        ).reset_index(level=0, drop=True)
        temp_mat = data['mat'].unstack(level=0)
        new_columns = pd.MultiIndex.from_tuples([(col, 'mat' + str([n1, n2])) for col in temp_mat.columns],names=['future', 'price'])
        temp_mat.columns = new_columns
        ret = pd.concat([ret, temp_mat], axis=1)

    open.columns = pd.MultiIndex.from_product([open.columns, ["open"]], names=["future", "price"])
    openw.columns = pd.MultiIndex.from_product([openw.columns, ["openw"]], names=["future", "price"])
    ret = pd.concat([ret, open, openw], axis=1)

    return ret

"""
TF指标计算
"""
def TF(data, window, futures_universe, time_freq='day'):
    ret = pd.DataFrame()
    data = data.loc[futures_universe].copy()
    open = data['open'].unstack(level=0)
    openw = data['openw'].unstack(level=0)

    for n in window:
        data['tf'] = data.groupby('future').apply(lambda group:
                        group['total_turnover'].rolling(n).std()
                        ).reset_index(level=0, drop=True)
        temp_tf = data['tf'].unstack(level=0)
        new_columns = pd.MultiIndex.from_tuples([(col, f'tf{n}') for col in temp_tf.columns],names=['future', 'price'])
        temp_tf.columns = new_columns
        ret = pd.concat([ret, temp_tf], axis=1)

    open.columns = pd.MultiIndex.from_product([open.columns, ["open"]], names=["future", "price"])
    openw.columns = pd.MultiIndex.from_product([openw.columns, ["openw"]], names=["future", "price"])
    ret = pd.concat([ret, open, openw], axis=1)

    return ret

"""
ADTMMA指标计算
"""
def ADTMMA(data, window, futures_universe, time_freq='day'):
    ret = pd.DataFrame()
    data = data.loc[futures_universe].copy()
    open = data['open'].unstack(level=0)
    openw = data['openw'].unstack(level=0)

    for N, M in window:
        dtm = (data['openw'] < data['openw'].shift(1)) * 0 + (data['openw'] >= data['openw'].shift(1)) * ((data['highw'] - data['openw'] >= data['openw'] - data['openw'].shift(1)) * (data['highw'] - data['openw']) + (data['highw'] - data['openw'] < data['openw'] - data['openw'].shift(1)) * (data['openw'] - data['openw'].shift(1)))
        dbm = (data['openw'] >= data['openw'].shift(1)) * 0 + (data['openw'] < data['openw'].shift(1)) * ((data['openw'] - data['loww'] >= data['openw'] - data['openw'].shift(1)) * (data['openw'] - data['loww']) + (data['openw'] - data['loww'] < data['openw'] - data['openw'].shift(1)) * (data['openw'] - data['openw'].shift(1)))

        data['stm'] = dtm.rolling(N).sum()
        data['sbm'] = dbm.rolling(N).sum()

        adtmma_temp = (data['stm'] > data['sbm']) * (data['stm'] - data['sbm']) / data['stm'] + (data['stm'] == data['sbm']) * 0 + (data['stm'] < data['sbm']) * (data['stm'] - data['sbm']) / data['sbm']
        data['adtmma'] = adtmma_temp.rolling(M).mean()
        temp_adtmma = data['adtmma'].unstack(level=0)
        new_columns = pd.MultiIndex.from_tuples([(col, 'adtmma' + str([N, M])) for col in temp_adtmma.columns],names=['future', 'price'])
        temp_adtmma.columns = new_columns
        ret = pd.concat([ret, temp_adtmma], axis=1)

    open.columns = pd.MultiIndex.from_product([open.columns, ["open"]], names=["future", "price"])
    openw.columns = pd.MultiIndex.from_product([openw.columns, ["openw"]], names=["future", "price"])
    ret = pd.concat([ret, open, openw], axis=1)

    return ret

"""
TV指标计算
"""
def TV(data, window, futures_universe, time_freq='day'):
    ret = pd.DataFrame()
    data = data.loc[futures_universe].copy()
    open = data['open'].unstack(level=0)
    openw = data['openw'].unstack(level=0)

    for n in window:
        data['tv'] = data.groupby('future').apply(lambda group:
                        group['total_turnover'].rolling(n).mean() / group['total_turnover'].rolling(n).std()
                        ).reset_index(level=0, drop=True)
        temp_tv = data['tv'].unstack(level=0)
        new_columns = pd.MultiIndex.from_tuples([(col, f'tv{n}') for col in temp_tv.columns],names=['future', 'price'])
        temp_tv.columns = new_columns
        ret = pd.concat([ret, temp_tv], axis=1)

    open.columns = pd.MultiIndex.from_product([open.columns, ["open"]], names=["future", "price"])
    openw.columns = pd.MultiIndex.from_product([openw.columns, ["openw"]], names=["future", "price"])
    ret = pd.concat([ret, open, openw], axis=1)

    return ret

"""
MASSINDEX指标计算
"""
def MASSINDEX(data, window, futures_universe, time_freq='day'):
    ret = pd.DataFrame()
    data = data.loc[futures_universe].copy()
    open = data['open'].unstack(level=0)
    openw = data['openw'].unstack(level=0)

    for n in window:
        data['mi'] = data.groupby('future').apply(lambda group:
                        (group['highw'] - group['loww']).rolling(n).mean() / (group['highw'] - group['loww']).rolling(n).mean().rolling(n).mean()
                        ).reset_index(level=0, drop=True)
        temp_mi = data['mi'].unstack(level=0)
        new_columns = pd.MultiIndex.from_tuples([(col, f'massindex{n}') for col in temp_mi.columns],names=['future', 'price'])
        temp_mi.columns = new_columns
        ret = pd.concat([ret, temp_mi], axis=1)

    open.columns = pd.MultiIndex.from_product([open.columns, ["open"]], names=["future", "price"])
    openw.columns = pd.MultiIndex.from_product([openw.columns, ["openw"]], names=["future", "price"])
    ret = pd.concat([ret, open, openw], axis=1)

    return ret

"""
ADTM指标计算
"""
def ADTM(data, window, futures_universe, time_freq='day'):
    ret = pd.DataFrame()
    data = data.loc[futures_universe].copy()
    open = data['open'].unstack(level=0)
    openw = data['openw'].unstack(level=0)

    for n in window:
        dtm = (data['openw'] < data['openw'].shift(1)) * 0 + (data['openw'] >= data['openw'].shift(1)) * ((data['highw'] - data['openw'] >= data['openw'] - data['openw'].shift(1)) * (data['highw'] - data['openw']) + (data['highw'] - data['openw'] < data['openw'] - data['openw'].shift(1)) * (data['openw'] - data['openw'].shift(1)))
        dbm = (data['openw'] >= data['openw'].shift(1)) * 0 + (data['openw'] < data['openw'].shift(1)) * ((data['openw'] - data['loww'] >= data['openw'] - data['openw'].shift(1)) * (data['openw'] - data['loww']) + (data['openw'] - data['loww'] < data['openw'] - data['openw'].shift(1)) * (data['openw'] - data['openw'].shift(1)))

        data['stm'] = dtm.rolling(n).sum()
        data['sbm'] = dbm.rolling(n).sum()

        data['adtm'] = (data['stm'] > data['sbm']) * (data['stm'] - data['sbm']) / data['stm'] + (data['stm'] == data['sbm']) * 0 + (data['stm'] < data['sbm']) * (data['stm'] - data['sbm']) / data['sbm']
        temp_adtm = data['adtm'].unstack(level=0)
        new_columns = pd.MultiIndex.from_tuples([(col, f'adtm{n}') for col in temp_adtm.columns],names=['future', 'price'])
        temp_adtm.columns = new_columns
        ret = pd.concat([ret, temp_adtm], axis=1)

    open.columns = pd.MultiIndex.from_product([open.columns, ["open"]], names=["future", "price"])
    openw.columns = pd.MultiIndex.from_product([openw.columns, ["openw"]], names=["future", "price"])
    ret = pd.concat([ret, open, openw], axis=1)

    return ret

"""
TURNOVER指标计算
"""
def TURNOVER(data, window, futures_universe, time_freq='day'):
    ret = pd.DataFrame()
    data = data.loc[futures_universe].copy()
    open = data['open'].unstack(level=0)
    openw = data['openw'].unstack(level=0)

    for n in window:
        data['tvr'] = data.groupby('future').apply(lambda group:
                    group['total_turnover'].rolling(n).mean()
                    ).reset_index(level=0, drop=True)
        # Multiplying by the factor if year is 202X
        data.loc[data.index.get_level_values(level=0) >= '2020-01-01', 'tvr'] *= 2
        temp_tvr = data['tvr'].unstack(level=0)
        new_columns = pd.MultiIndex.from_tuples([(col, f'turnover{n}') for col in temp_tvr.columns],names=['future', 'price'])
        temp_tvr.columns = new_columns
        ret = pd.concat([ret, temp_tvr], axis=1)

    open.columns = pd.MultiIndex.from_product([open.columns, ["open"]], names=["future", "price"])
    openw.columns = pd.MultiIndex.from_product([openw.columns, ["openw"]], names=["future", "price"])
    ret = pd.concat([ret, open, openw], axis=1)

    return ret

"""
MA(WQAlpha041)/close指标计算
"""
wqalpha041ma_time_lst = [1, 2, 3, 4, 5, 6]
def WQALPHA041MA(data, window, futures_universe, time_freq='day'):
    ret = pd.DataFrame()
    data = data.loc[futures_universe].copy()
    open = data['open'].unstack(level=0)
    openw = data['openw'].unstack(level=0)

    for a in window:
        data['wqalpha041ma'] = (((data['highw'] * data['loww']).map(np.sqrt) - data['settlementw']) / data['closew'])
        data['wqalpha041ma'] = data.groupby('future').apply(lambda group: group['wqalpha041ma'].rolling(a).mean()).reset_index(level=0, drop=True)

        temp_wq = data['wqalpha041ma'].unstack(level=0)
        new_columns = pd.MultiIndex.from_tuples([(col, f'wqalpha041ma{a}') for col in temp_wq.columns],names=['future', 'price'])
        temp_wq.columns = new_columns
        ret = pd.concat([ret, temp_wq], axis=1)

    open.columns = pd.MultiIndex.from_product([open.columns, ["open"]], names=["future", "price"])
    openw.columns = pd.MultiIndex.from_product([openw.columns, ["openw"]], names=["future", "price"])
    ret = pd.concat([ret, open, openw], axis=1)
    return ret

"""
WQAlpha041指标计算
"""

def WQALPHA041(data, window,futures_universe):
    ret = pd.DataFrame()
    data = data.loc[futures_universe].copy()
    open = data['open'].unstack(level=0)
    openw = data['openw'].unstack(level=0)

    data['wqa'] = data.groupby('future').apply(lambda group:
                ((group['highw'] * group['loww']).map(np.sqrt) - group['settlement']) / group['closew']
                ).reset_index(level=0, drop=True)
    temp_wqa = data['wqa'].unstack(level=0)
    for a in window:
        temp = temp_wqa.copy()
        temp = temp.shift(a)
        new_columns = pd.MultiIndex.from_tuples([(col, f'wqalpha041{a}') for col in temp.columns],names=['future', 'price'])
        temp.columns = new_columns
        ret = pd.concat([ret, temp], axis=1)

    open.columns = pd.MultiIndex.from_product([open.columns, ["open"]], names=["future", "price"])
    openw.columns = pd.MultiIndex.from_product([openw.columns, ["openw"]], names=["future", "price"])
    ret = pd.concat([ret, open, openw], axis=1)

    return ret

"""
WQAlpha101指标计算
"""
wqalpha101_time_lst = [1, 2, 3, 4, 5,6]
def WQALPHA101(data, window, futures_universe, time_freq='day'):
    """
    计算期货的WQALPHA101值，计算后同时加入到原始数据表中。

    Parameters
    ----------
    data : pd.DataFrame
        数据的完整大表，需包含需要计算的期货数据的closew，openw，highw和loww。

    Returns
    -------
    ret : DataFrame.
    """
    ret = pd.DataFrame()
    data = data.loc[futures_universe].copy()
    open = data['open'].unstack(level=0)
    openw = data['openw'].unstack(level=0)

    data['wqalpha101'] = data.groupby('future').apply(lambda group: (group['closew'] - group['openw']) / (group['highw'] - group['loww'] + 0.001)).reset_index(level=0,drop=True)

    temp_wqalpha101 = data['wqalpha101'].unstack(level=0)
    for a in window:
        temp = temp_wqalpha101.copy()
        temp = temp.shift(a)
        new_columns = pd.MultiIndex.from_tuples([(col, f'wqalpha101{a}') for col in temp_wqalpha101.columns],names=['future', 'price'])
        temp.columns = new_columns
        ret = pd.concat([ret, temp], axis=1)
    open.columns = pd.MultiIndex.from_product([open.columns, ["open"]], names=["future", "price"])
    openw.columns = pd.MultiIndex.from_product([openw.columns, ["openw"]], names=["future", "price"])
    ret = pd.concat([ret, open, openw], axis=1)
    return ret


"""
MFI指标计算
"""

def MFI(data, window, futures_universe, time_freq='day'):
    ret = pd.DataFrame()
    data = data.loc[futures_universe].copy()
    open = data['open'].unstack(level=0)
    openw = data['openw'].unstack(level=0)

    for n in window:
        tp = (data['highw'] + data['loww'] + data['closew']) / 3

        data['mf'] = tp * data['volume99']
        data['pf'] = (tp > tp.shift(1)) * data['mf']
        data['nf'] = (tp <= tp.shift(1)) * data['mf']

        data['pf_rolling'] = data.groupby('future')['pf'].rolling(n).sum().reset_index(level=0, drop=True)
        data['nf_rolling'] = data.groupby('future')['nf'].rolling(n).sum().reset_index(level=0, drop=True)

        data['mf{}'.format(n)] = data['pf_rolling'] / data['nf_rolling']
        data['mr'] = data['mf{}'.format(n)] / (1 + data['mf{}'.format(n)])
        temp_mfi = data['mr'].unstack(level=0)
        new_columns = pd.MultiIndex.from_tuples([(col, f'mfi{n}') for col in temp_mfi.columns],names=['future', 'price'])
        temp_mfi.columns = new_columns
        ret = pd.concat([ret, temp_mfi], axis=1)

    open.columns = pd.MultiIndex.from_product([open.columns, ["open"]], names=["future", "price"])
    openw.columns = pd.MultiIndex.from_product([openw.columns, ["openw"]], names=["future", "price"])
    ret = pd.concat([ret, open, openw], axis=1)

    return ret



"""
RSI指标计算
"""
def RSI(data, window, futures_universe, time_freq='day'):
    ret = pd.DataFrame()
    data = data.loc[futures_universe].copy()
    open = data['open'].unstack(level=0)
    openw = data['openw'].unstack(level=0)

    for n in window:
        data['close_shift'] = data.groupby('future')['closew'].shift(1)
        data['pf'] = (data['closew'] > data['close_shift']) * (data['closew'] - data['close_shift'])
        data['nf'] = (data['closew'] <= data['close_shift']) * (data['close_shift'] - data['closew'])

        data['pf_rolling'] = data.groupby('future')['pf'].rolling(n).sum().reset_index(level=0, drop=True)
        data['nf_rolling'] = data.groupby('future')['nf'].rolling(n).sum().reset_index(level=0, drop=True)

        data['rsi{}'.format(n)] = data['pf_rolling'] / (data['pf_rolling'] + data['nf_rolling'])
        temp_rsi = data['rsi{}'.format(n)].unstack(level=0)
        new_columns = pd.MultiIndex.from_tuples([(col, f'rsi{n}') for col in temp_rsi.columns],names=['future', 'price'])
        temp_rsi.columns = new_columns
        ret = pd.concat([ret, temp_rsi], axis=1)

    open.columns = pd.MultiIndex.from_product([open.columns, ["open"]], names=["future", "price"])
    openw.columns = pd.MultiIndex.from_product([openw.columns, ["openw"]], names=["future", "price"])
    ret = pd.concat([ret, open, openw], axis=1)

    return ret


"""
AR指标计算
"""
def AR(data, window, futures_universe, time_freq='day'):
    ret = pd.DataFrame()
    data = data.loc[futures_universe].copy()
    open = data['open'].unstack(level=0)
    openw = data['openw'].unstack(level=0)

    for n in window:
        data['diff_high_open'] = data['highw'] - data['openw']
        data['diff_open_low'] = data['openw'] - data['loww']

        data['diff_high_open_sum'] = data.groupby('future')['diff_high_open'].rolling(n).sum().reset_index(level=0, drop=True)
        data['diff_open_low_sum'] = data.groupby('future')['diff_open_low'].rolling(n).sum().reset_index(level=0, drop=True)

        data['ar{}'.format(n)] = data['diff_high_open_sum'] / data['diff_open_low_sum']
        temp_ar = data['ar{}'.format(n)].unstack(level=0)
        new_columns = pd.MultiIndex.from_tuples([(col, f'ar{n}') for col in temp_ar.columns],names=['future', 'price'])
        temp_ar.columns = new_columns
        ret = pd.concat([ret, temp_ar], axis=1)

    open.columns = pd.MultiIndex.from_product([open.columns, ["open"]], names=["future", "price"])
    openw.columns = pd.MultiIndex.from_product([openw.columns, ["openw"]], names=["future", "price"])
    ret = pd.concat([ret, open, openw], axis=1)

    return ret

"""
momentum_attn指标计算
"""


# NO
def momentum_attn(data, window, futures_universe, time_freq='day'):
    ret = pd.DataFrame()
    data = data.loc[futures_universe].copy()
    open = data['open'].unstack(level=0)
    openw = data['openw'].unstack(level=0)

    for n in window:
        data['returns'] = data.groupby('future')['closew'].pct_change(1)

        data['returns_rolling_sum'] = data.groupby('future')['returns'].rolling(n[0]).sum().reset_index(level=0,
                                                                                                        drop=True)
        data['closew_polyfit'] = data.groupby('future')['closew'].rolling(n[1]).apply(
            lambda Y: np.polyfit(y=Y, x=np.arange(n[1]), deg=2)[0]).reset_index(level=0, drop=True)  # 使用二次多项式拟合

        data['momentum_attn{}'.format(n)] = data['returns_rolling_sum'] + data['closew_polyfit']
        temp_momentum_attn = data['momentum_attn{}'.format(n)].unstack(level=0)
        new_columns = pd.MultiIndex.from_tuples([(col, f'momentum_attn{n}') for col in temp_momentum_attn.columns],
                                                names=['future', 'price'])
        temp_momentum_attn.columns = new_columns
        ret = pd.concat([ret, temp_momentum_attn], axis=1)

    open.columns = pd.MultiIndex.from_product([open.columns, ["open"]], names=["future", "price"])
    openw.columns = pd.MultiIndex.from_product([openw.columns, ["openw"]], names=["future", "price"])
    ret = pd.concat([ret, open, openw], axis=1)

    return ret


# import pickle
# with open('.\\future_alldailydata_index_volumn_recovery2010-2022_rqdata.txt','rb') as f:
#     data = pickle.load(f)
# f.close()
# momentum_attn(data,[[20,10]], '.\\future_alldailydata_index_volumn_recovery2010-2022_rqdata.txt')

"""
k_divided_by_sigma指标计算
"""


def k_divided_by_sigma(data, window, futures_universe, time_freq='day'):
    ret = pd.DataFrame()
    data = data.loc[futures_universe].copy()
    open = data['open'].unstack(level=0)
    openw = data['openw'].unstack(level=0)

    for n in window:
        k1 = data.groupby('future')['closew'].rolling(n[0]).apply(
            lambda Y: np.polyfit(y=Y, x=np.arange(n[0]), deg=1)[0]).reset_index(level=0, drop=True)
        sigma1 = data.groupby('future')['closew'].rolling(n[0]).apply(
            lambda Y: np.std(Y - np.poly1d(np.polyfit(y=Y, x=np.arange(n[0]), deg=1))(np.arange(n[0])))).reset_index(
            level=0, drop=True)

        k2 = data.groupby('future')['closew'].rolling(n[0]).apply(
            lambda Y: np.poly1d(np.polyfit(y=Y, x=np.arange(n[0]), deg=2))(np.arange(n[0]))[-1] -
                      np.poly1d(np.polyfit(y=Y, x=np.arange(n[0]), deg=2))(np.arange(n[0]))[-2]).reset_index(level=0,
                                                                                                             drop=True)
        sigma2 = data.groupby('future')['closew'].rolling(n[0]).apply(
            lambda Y: np.std(Y - np.poly1d(np.polyfit(y=Y, x=np.arange(n[0]), deg=2))(np.arange(n[0])))).reset_index(
            level=0, drop=True)

        data['k_divided_by_sigma{}'.format(n)] = n[1] * k1 / sigma1 + k2 / sigma2
        temp_k_divided_by_sigma = data['k_divided_by_sigma{}'.format(n)].unstack(level=0)
        new_columns = pd.MultiIndex.from_tuples(
            [(col, f'k_divided_by_sigma{n}') for col in temp_k_divided_by_sigma.columns], names=['future', 'price'])
        temp_k_divided_by_sigma.columns = new_columns
        ret = pd.concat([ret, temp_k_divided_by_sigma], axis=1)

    open.columns = pd.MultiIndex.from_product([open.columns, ["open"]], names=["future", "price"])
    openw.columns = pd.MultiIndex.from_product([openw.columns, ["openw"]], names=["future", "price"])
    ret = pd.concat([ret, open, openw], axis=1)

    return ret


# import pickle
# with open('.\\future_alldailydata_index_volumn_recovery2010-2022_rqdata.txt','rb') as f:
#     data = pickle.load(f)
# f.close()
# k_divided_by_sigma(data,[[10,12.5]], '.\\future_alldailydata_index_volumn_recovery2010-2022_rqdata.txt')

"""
WQAlpha083指标计算
"""


def WQAlpha083(data, window, futures_universe, time_freq='day'):
    data = data.loc[futures_universe].copy()
    ret = pd.DataFrame()
    open = data['open'].unstack(level=0)
    openw = data['openw'].unstack(level=0)

    for n in window:
        if 'indexvol_recover' not in data.columns:
            continue

        inner = (data['highw'] - data['loww']) / data.groupby('future')['closew'].rolling(n[0]).mean().reset_index(
            level=0, drop=True)
        data['wqalpha083_{}'.format(n)] = (inner.shift(n[1]) * data['indexvol_recover']) / (
                    inner / (data['settlement'] - data['closew']))

        temp_wqalpha083 = data['wqalpha083_{}'.format(n)].unstack(level=0)
        new_columns = pd.MultiIndex.from_tuples([(col, f'wqalpha083{n}') for col in temp_wqalpha083.columns],
                                                names=['future', 'price'])
        temp_wqalpha083.columns = new_columns
        ret = pd.concat([ret, temp_wqalpha083], axis=1)

    open.columns = pd.MultiIndex.from_product([open.columns, ["open"]], names=["future", "price"])
    openw.columns = pd.MultiIndex.from_product([openw.columns, ["openw"]], names=["future", "price"])
    ret = pd.concat([ret, open, openw], axis=1)

    return ret


# import pickle
# with open('.\\future_alldailydata_2010-20230519_index_volumn_recovery_rqdata.txt','rb') as f:
#     data = pickle.load(f)
# f.close()
# WQAlpha083(data,[[5, 2]], '.\\future_alldailydata_2010-20230519_index_volumn_recovery_rqdata.txt')

"""
WQAlpha040指标计算
"""


def WQAlpha040(data, window, futures_universe, time_freq='day'):
    data = data.loc[futures_universe].copy()
    ret = pd.DataFrame()
    open = data['open'].unstack(level=0)
    openw = data['openw'].unstack(level=0)

    for n in window:
        if 'volume99' not in data.columns:
            continue
        data['wqalpha040_{}'.format(n)] = -1 * data.groupby('future')['highw'].rolling(n).std().reset_index(level=0,
                                                                                                            drop=True) * data.groupby(
            'future').apply(lambda group: group['highw'].rolling(n).corr(group['volume99'])).reset_index(level=0,
                                                                                                         drop=True)

        temp_wqalpha040 = data['wqalpha040_{}'.format(n)].unstack(level=0)
        new_columns = pd.MultiIndex.from_tuples([(col, f'wqalpha040{n}') for col in temp_wqalpha040.columns],
                                                names=['future', 'price'])
        temp_wqalpha040.columns = new_columns
        ret = pd.concat([ret, temp_wqalpha040], axis=1)

    open.columns = pd.MultiIndex.from_product([open.columns, ["open"]], names=["future", "price"])
    openw.columns = pd.MultiIndex.from_product([openw.columns, ["openw"]], names=["future", "price"])
    ret = pd.concat([ret, open, openw], axis=1)

    return ret


# import pickle
# with open('.\\future_alldailydata_index_volumn_recovery2010-2022_rqdata.txt','rb') as f:
#     data = pickle.load(f)
# f.close()
# WQAlpha040(data,10, '.\\future_alldailydata_index_volumn_recovery2010-2022_rqdata.txt')


# """
# VHF_times_MA指标计算
# """
# vhf_times_ma_time_lst = [1,2,5,8,11,20]
# def VHF_times_MA(data, window, futures_universe, time_freq='day'):
#     data = data.loc[futures_universe].copy()
#     ret = pd.DataFrame()
#     open = data['open'].unstack(level=0)
#     openw = data['openw'].unstack(level=0)
#
#     n = [20, 15]
#
#     for m in window:
#         vhf = data.groupby('future').apply(
#             lambda group: abs(group.highw.rolling(n[0]).max() - group.loww.rolling(n[0]).min()) / abs(
#                 group.closew.diff(1)).rolling(n[0]).sum()).reset_index(level=0, drop=True)
#         ma = data.groupby('future')['closew'].rolling(n[1]).mean().reset_index(level=0, drop=True) / data['closew']
#         data['vhf_ma_{}'.format(m)] = (vhf * ma).shift(m)
#
#         temp_vhf_ma = data['vhf_ma_{}'.format(m)].unstack(level=0)
#         new_columns = pd.MultiIndex.from_tuples([(col, f'VHF_MA{m}') for col in temp_vhf_ma.columns],
#                                                 names=['future', 'price'])
#         temp_vhf_ma.columns = new_columns
#         ret = pd.concat([ret, temp_vhf_ma], axis=1)
#
#     open.columns = pd.MultiIndex.from_product([open.columns, ["open"]], names=["future", "price"])
#     openw.columns = pd.MultiIndex.from_product([openw.columns, ["openw"]], names=["future", "price"])
#     ret = pd.concat([ret, open, openw], axis=1)
#
#     return ret

def MA(data, window, futures_universe, time_freq='day'):
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]

    ret = pd.DataFrame()
    for w in window:
        w1, w2 = map(int, w)

        _data = data.loc[futures_universe]
        _data['ori'] = _data['closew'].groupby('future').apply(
            lambda x: (((x.shift(1) > x.shift(1).rolling(w1).mean()) + 1 - 1)
                       - ((x.shift(1) <= x.shift(1).rolling(w1).mean()) + 1 - 1)) + 1 - 1).reset_index(level=0, drop=True)
        _data['over16'] = _data.groupby('future')['closew'].apply(
            lambda x: (
                              (((x.shift(1) > x.shift(1).rolling(w1).mean()) &
                                (x.shift(2) > x.shift(2).rolling(w1).mean()) &
                                (x.shift(3) > x.shift(3).rolling(w1).mean()) &
                                (x.shift(4) > x.shift(4).rolling(w1).mean()) &
                                (x.shift(5) > x.shift(5).rolling(w1).mean())) + 1 - 1)
                              - (((x.shift(1) < x.shift(1).rolling(w1).mean()) &
                                  (x.shift(2) < x.shift(2).rolling(w1).mean()) &
                                  (x.shift(3) < x.shift(3).rolling(w1).mean()) &
                                  (x.shift(4) < x.shift(4).rolling(w1).mean()) &
                                  (x.shift(5) < x.shift(5).rolling(w1).mean())) + 1 - 1)
                      ) + 1 - 1
        ).reset_index(level=0, drop=True)
        _data['over4'] = _data.groupby('future')['closew'].apply(
            lambda x: (
                    ((((x.shift(1) > x.shift(1).rolling(w2).mean()) |
                       (x.shift(2) > x.shift(2).rolling(w2).mean())) + 1 - 1)
                     - (((x.shift(1) < x.shift(1).rolling(w2).mean()) |
                         (x.shift(2) < x.shift(2).rolling(w2).mean())) + 1 - 1)) + 1 - 1
            )
        ).reset_index(level=0, drop=True)
        _data['formular'] = _data['over16'] + _data['over4']
        _data['formular'].replace(-1.0, 0.0, inplace=True)
        _data['formular'].replace(1.0, 0.0, inplace=True)
        _data['formular'].replace(2.0, 1.0, inplace=True)
        _data['formular'].replace(-2.0, -1.0, inplace=True)
        _data['formular'].replace(0.0, np.nan, inplace=True)

        def cal(series):
            index = series.index.get_loc(series.first_valid_index())
            series.iloc[:index] = 0
            return series

        _data.groupby('future')['formular'].apply(lambda x: cal(x))
        # 顺延操作
        _data['formular'] = _data.groupby('future')['formular'].apply(
                      lambda x:  x.fillna(method='ffill', axis=0)).reset_index(level=0, drop=True)
        _ret = _data['formular'].unstack().T
        _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["signal45edit" + str(w)]], names=["future", "price"])
        ret = pd.concat([ret, _ret], axis=1)

    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret


def MA1(data, window, futures_universe, time_freq='day'):
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]

    ret = pd.DataFrame()
    for w1 in window:

        _data = data.loc[futures_universe]

        _data['over16'] = _data.groupby('future')['closew'].apply(
            lambda x: (
                              (((x.shift(1) > x.shift(1).rolling(w1).mean()) &
                                (x.shift(2) > x.shift(2).rolling(w1).mean()) &
                                (x.shift(3) > x.shift(3).rolling(w1).mean()) &
                                (x.shift(4) > x.shift(4).rolling(w1).mean()) &
                                (x.shift(5) > x.shift(5).rolling(w1).mean())) + 1 - 1)
                              - (((x.shift(1) < x.shift(1).rolling(w1).mean()) &
                                  (x.shift(2) < x.shift(2).rolling(w1).mean()) &
                                  (x.shift(3) < x.shift(3).rolling(w1).mean()) &
                                  (x.shift(4) < x.shift(4).rolling(w1).mean()) &
                                  (x.shift(5) < x.shift(5).rolling(w1).mean())) + 1 - 1)
                      ) + 1 - 1
        ).reset_index(level=0, drop=True)

        _data['formular'] = _data['over16']

        _data['formular'].replace(0.0, np.nan, inplace=True)

        def cal(series):
            index = series.index.get_loc(series.first_valid_index())
            series.iloc[:index] = 0
            return series

        _data.groupby('future')['formular'].apply(lambda x: cal(x))
        # 顺延操作
        _data['formular'] = _data.groupby('future')['formular'].apply(
            lambda x: x.fillna(method='ffill', axis=0)).reset_index(level=0, drop=True)
        _ret = _data['formular'].unstack().T
        _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["signal45edit" + str(w1)]], names=["future", "price"])
        ret = pd.concat([ret, _ret], axis=1)

    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret


def MA1EDIT(data, window, futures_universe, time_freq='day'):
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]

    ret = pd.DataFrame()
    for w1 in window:


        _data = data.loc[futures_universe]
        _data['ori'] = _data['closew'].groupby('future').apply(
            lambda x: (((x.shift(1) > x.shift(1).rolling(w1).mean()) + 1 - 1)
                       - ((x.shift(1) <= x.shift(1).rolling(w1).mean()) + 1 - 1)) + 1 - 1).reset_index(level=0,
                                                                                                       drop=True)
        _data['over16'] = _data.groupby('future')['closew'].apply(
            lambda x: (
                              (((x.shift(1) - x.shift(1).rolling(w1).mean()) +
                                (x.shift(2) - x.shift(2).rolling(w1).mean()) +
                                (x.shift(3) - x.shift(3).rolling(w1).mean()) +
                                (x.shift(4) - x.shift(4).rolling(w1).mean()) +
                                (x.shift(5) - x.shift(5).rolling(w1).mean())) + 1 - 1)
                      ) + 1 - 1
        ).reset_index(level=0, drop=True)

        _data['formular'] = _data['over16']

        _data['formular'].replace(0.0, np.nan, inplace=True)

        def cal(series):
            index = series.index.get_loc(series.first_valid_index())
            series.iloc[:index] = 0
            return series

        _data.groupby('future')['formular'].apply(lambda x: cal(x))
        # 顺延操作
        _data['formular'].fillna(method='ffill', axis=0, inplace=True)
        _ret = _data['formular'].unstack().T
        _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["signal45edit" + str(w1)]], names=["future", "price"])
        ret = pd.concat([ret, _ret], axis=1)

    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret

def MA1EDIT2(data, window, futures_universe, time_freq='day'):
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]

    ret = pd.DataFrame()
    for w1 in window:

        _data = data.loc[futures_universe]

        _data['over16'] = _data.groupby('future')['closew'].apply(
            lambda x: (
                              (((x.shift(1) > x.shift(1).rolling(w1).mean()) &
                                (x.shift(2) > x.shift(2).rolling(w1).mean())
                                ) + 1 - 1)
                              - (((x.shift(1) < x.shift(1).rolling(w1).mean()) &
                                  (x.shift(2) < x.shift(2).rolling(w1).mean()) ) + 1 - 1)
                      ) + 1 - 1
        ).reset_index(level=0, drop=True)

        _data['formular'] = _data['over16']

        _data['formular'].replace(0.0, np.nan, inplace=True)

        def cal(series):
            index = series.index.get_loc(series.first_valid_index())
            series.iloc[:index] = 0
            return series

        _data.groupby('future')['formular'].apply(lambda x: cal(x))
        # 顺延操作
        _data['formular'].fillna(method='ffill', axis=0, inplace=True)
        _ret = _data['formular'].unstack().T
        _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["signal45edit" + str(w1)]], names=["future", "price"])
        ret = pd.concat([ret, _ret], axis=1)

    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret



def MA2(data, window, futures_universe, time_freq='day'):
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]

    ret = pd.DataFrame()
    for w in window:
        w1, w2 = map(int, w)

        _data = data.loc[futures_universe]
        _data['ori'] = _data['closew'].groupby('future').apply(
            lambda x: (((x.shift(1) > x.shift(1).rolling(w1).mean()) + 1 - 1)
                       - ((x.shift(1) <= x.shift(1).rolling(w1).mean()) + 1 - 1)) + 1 - 1).reset_index(level=0, drop=True)

        _data['over4'] = _data.groupby('future')['closew'].apply(
            lambda x: (
                    ((((x.shift(1) > x.shift(1).rolling(w2).mean()) |
                       (x.shift(2) > x.shift(2).rolling(w2).mean())) + 1 - 1)
                     - (((x.shift(1) < x.shift(1).rolling(w2).mean()) |
                         (x.shift(2) < x.shift(2).rolling(w2).mean())) + 1 - 1)) + 1 - 1
            )
        ).reset_index(level=0, drop=True)

        _data['formular'] = _data['over4']
        _data['formular'].replace(0.0, np.nan, inplace=True)
        # _data['formular'].iloc[:_data.get_loc(_data['formular'].first_valid_index())] = 0
        def cal(series):
            index = series.index.get_loc(series.first_valid_index())
            series.iloc[:index] = 0
            return series

        _data.groupby('future')['formular'].apply(lambda x: cal(x))
        # 顺延操作
        _data['formular'].fillna(method='ffill', axis=0, inplace=True)
        _ret = _data['formular'].unstack().T
        _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["signal45edit" + str(w)]], names=["future", "price"])
        ret = pd.concat([ret, _ret], axis=1)

    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret




# shift_time_lst = [1,2,5,8,11,20]
# vhf_ma_time_lst = [20 15 0 20 15 1 20 15 4 20 15 7 20 15 10 20 15 19]
# 使用-w tp模式
def VHF_MA(data, window, futures_universe, time_freq='day'):
    data = data.loc[futures_universe].copy()
    ret = pd.DataFrame()
    open = data['open'].unstack(level=0)
    openw = data['openw'].unstack(level=0)

    for n in window:
        vhf = data.groupby('future').apply(
            lambda group: abs(group.highw.rolling(n[0]).max() - group.loww.rolling(n[0]).min()) / abs(
                group.closew.diff(1)).rolling(n[0]).sum()).reset_index(level=0, drop=True)
        ma = data.groupby('future')['closew'].rolling(n[1]).mean().reset_index(level=0, drop=True) / data['closew']
        data['vhf_ma_{}'.format(n)] = vhf * ma

        temp_vhf_ma = data['vhf_ma_{}'.format(n)].unstack(level=0).shift(n[2])
        new_columns = pd.MultiIndex.from_tuples([(col, f'VHF_MA{n}') for col in temp_vhf_ma.columns],
                                                names=['future', 'price'])
        temp_vhf_ma.columns = new_columns
        ret = pd.concat([ret, temp_vhf_ma], axis=1)

    open.columns = pd.MultiIndex.from_product([open.columns, ["open"]], names=["future", "price"])
    openw.columns = pd.MultiIndex.from_product([openw.columns, ["openw"]], names=["future", "price"])
    ret = pd.concat([ret, open, openw], axis=1)

    return ret


"""
LIQU指标计算
"""


def LIQU(data, window, futures_universe, time_freq='day'):
    data = data.loc[futures_universe].copy()
    ret = pd.DataFrame()
    open = data['open'].unstack(level=0)
    openw = data['openw'].unstack(level=0)

    for n in window:
        returns = data.groupby('future')['closew'].pct_change(1)
        lose_day = -1 * (returns < 0)
        illiq = ((returns / data['total_turnover']) * lose_day).rolling(n[0]).sum() / lose_day.rolling(
            n[0]).sum() * 1e12
        data['liqu_{}'.format(n)] = illiq - illiq.rolling(n[1]).mean()

        temp_liqu = data['liqu_{}'.format(n)].unstack(level=0)
        new_columns = pd.MultiIndex.from_tuples([(col, f'LIQU{n}') for col in temp_liqu.columns],
                                                names=['future', 'price'])
        temp_liqu.columns = new_columns
        ret = pd.concat([ret, temp_liqu], axis=1)

    open.columns = pd.MultiIndex.from_product([open.columns, ["open"]], names=["future", "price"])
    openw.columns = pd.MultiIndex.from_product([openw.columns, ["openw"]], names=["future", "price"])
    ret = pd.concat([ret, open, openw], axis=1)

    return ret


# import pickle
# with open('.\\future_alldailydata_index_volumn_recovery2010-2022_rqdata.txt','rb') as f:
#     data = pickle.load(f)
# f.close()
# LIQU(data,[20,250], '.\\future_alldailydata_index_volumn_recovery2010-2022_rqdata.txt')

"""
COPPOCK指标计算
"""


def COPPOCK(data, window, futures_universe, time_freq='day'):
    data = data.loc[futures_universe].copy()
    ret = pd.DataFrame()
    open = data['open'].unstack(level=0)
    openw = data['openw'].unstack(level=0)

    for n in window:
        def wma(close, period):
            return close.rolling(window=period).apply(lambda x: x[::-1].cumsum().sum() * 2 / period / (period + 1))

        rn1 = (data.groupby('future')['closew'].apply(lambda x: x - x.shift(n[0])) / data.groupby('future')[
            'closew'].apply(lambda x: x.shift(n[0])))
        rn2 = (data.groupby('future')['closew'].apply(lambda x: x - x.shift(n[1])) / data.groupby('future')[
            'closew'].apply(lambda x: x.shift(n[1])))
        rc = rn1 + rn2
        data['coppock_{}'.format(n)] = wma(rc, n[2])

        temp_coppock = data['coppock_{}'.format(n)].unstack(level=0)
        new_columns = pd.MultiIndex.from_tuples([(col, f'COPPOCK{n}') for col in temp_coppock.columns],
                                                names=['future', 'price'])
        temp_coppock.columns = new_columns
        ret = pd.concat([ret, temp_coppock], axis=1)

    open.columns = pd.MultiIndex.from_product([open.columns, ["open"]], names=["future", "price"])
    openw.columns = pd.MultiIndex.from_product([openw.columns, ["openw"]], names=["future", "price"])
    ret = pd.concat([ret, open, openw], axis=1)

    return ret


# import pickle
# with open('.\\future_alldailydata_index_volumn_recovery2010-2022_rqdata.txt','rb') as f:
#     data = pickle.load(f)
# f.close()
# COPPOCK(data,[14,11,10], '.\\future_alldailydata_index_volumn_recovery2010-2022_rqdata.txt')

"""
A_ILLIQ指标计算
"""


def A_ILLIQ(data, window, futures_universe, time_freq='day'):
    data = data.loc[futures_universe].copy()
    ret = pd.DataFrame()

    open = data['open'].unstack(level=0)
    open.columns = pd.MultiIndex.from_product([open.columns, ["open"]], names=["future", "price"])
    openw = data['openw'].unstack(level=0)
    openw.columns = pd.MultiIndex.from_product([openw.columns, ["openw"]], names=["future", "price"])

    for n in window:
        rc = (data['closew'] - data['openw']) / data['open']
        illiq = (rc.abs() / ((rc / data['total_turnover']) * 10000000000000)).groupby('future').rolling(
            n).mean().reset_index(level=0, drop=True)

        data['aillq_{}'.format(n)] = illiq

        temp_aillq = data['aillq_{}'.format(n)].unstack(level=0)
        new_columns = pd.MultiIndex.from_tuples([(col, f'aillq{n}') for col in temp_aillq.columns],
                                                names=['future', 'price'])
        temp_aillq.columns = new_columns
        ret = pd.concat([ret, temp_aillq], axis=1)

    ret = pd.concat([ret, open, openw], axis=1)

    return ret


# import pickle
# with open('.\\future_alldailydata_index_volumn_recovery2010-2022_rqdata.txt','rb') as f:
#     data = pickle.load(f)
# f.close()
# A_ILLIQ(data,20, '.\\future_alldailydata_index_volumn_recovery2010-2022_rqdata.txt')

"""
MATR指标计算
"""


def MATR(data, window, futures_universe, time_freq='day'):
    data = data.loc[futures_universe].copy()
    ret = pd.DataFrame()
    open = data['open'].unstack(level=0)
    openw = data['openw'].unstack(level=0)

    for n in window:
        data['matr_{}'.format(n)] = (data.groupby('future').apply(
            lambda group: (group['volume99'] / group['open_interest']).rolling(n).mean())).reset_index(level=0,
                                                                                                       drop=True)

        temp_matr = data['matr_{}'.format(n)].unstack(level=0)
        new_columns = pd.MultiIndex.from_tuples([(col, f'matr{n}') for col in temp_matr.columns],
                                                names=['future', 'price'])
        temp_matr.columns = new_columns
        ret = pd.concat([ret, temp_matr], axis=1)

    open.columns = pd.MultiIndex.from_product([open.columns, ["open"]], names=["future", "price"])
    openw.columns = pd.MultiIndex.from_product([openw.columns, ["openw"]], names=["future", "price"])
    ret = pd.concat([ret, open, openw], axis=1)

    return ret


# import pickle
# with open('.\\future_alldailydata_index_volumn_recovery2010-2022_rqdata.txt','rb') as f:
#     data = pickle.load(f)
# f.close()
# MATR(data,20, '.\\future_alldailydata_index_volumn_recovery2010-2022_rqdata.txt')
"""
ILLIQ指标计算
"""


def ILLIQ(data, window, futures_universe, time_freq='day'):
    data = data.loc[futures_universe].copy()
    ret = pd.DataFrame()

    open = data['open'].unstack(level=0)
    open.columns = pd.MultiIndex.from_product([open.columns, ["open"]], names=["future", "price"])
    openw = data['openw'].unstack(level=0)
    openw.columns = pd.MultiIndex.from_product([openw.columns, ["openw"]], names=["future", "price"])

    for n in window:
        r = (data['closew'] - data['openw']) / data['open']
        up = -1 * (r < 0) * r
        down = (r < 0) * data['total_turnover'] / 10000000000000

        illiq = ((up / down).groupby('future').rolling(n).mean().reset_index(level=0, drop=True))
        data['illiq_{}'.format(n)] = illiq

        temp_illiq = data['illiq_{}'.format(n)].unstack(level=0)
        new_columns = pd.MultiIndex.from_tuples([(col, f'illiq{n}') for col in temp_illiq.columns],
                                                names=['future', 'price'])
        temp_illiq.columns = new_columns
        ret = pd.concat([ret, temp_illiq], axis=1)

    ret = pd.concat([ret, open, openw], axis=1)

    return ret


# =============================================================================
# def ILLIQ(data, n, save_path=None):
#     if type(n) == list:
#         for a in n:
#             cal_illiq(data, a, save_path=save_path)
#     elif type(n) == int:
#         cal_illiq(data, n, save_path=save_path)
#
# def cal_illiq(data, n, save_path):
# # =============================================================================
# #     for (future, _) in data.columns:
# #         if (future, 'illiq%s'%(n)) in data.columns:
# #             print('ILLIQ value are already prepared with period {}!'.format(n))
# #             return
# #         break
# # =============================================================================
#
#     for (future, _) in data.columns:
# # =============================================================================
# #         if (future, 'illiq%s'%(n)) in data.columns:
# #             continue
# # =============================================================================
#
#         rc = list(data[future].closew.rolling(n))
#         ro_w = list(data[future].openw.rolling(n))
#         ro = list(data[future].open.rolling(n))
#         rt = list(data[future].total_turnover.rolling(n))
#         illiq = []
#         for i in range(0,len(rc)):
#             r = (rc[i]-ro_w[i])/ro[i]
#             up = -1*(r<0)*r
#             down = (r<0)*rt[i]/10000000000000
#             illiqi = 0
#             if sum(r<0) != 0:
#                 illiqi = 1/sum(r<0) * np.nansum((down != 0) * (up/down))
#             illiq.append(illiqi)
#         data.loc[:,(future,'illiq%s'%(n))] = illiq
#
#     data=standardize(data,'illiq',n)
#
#     if save_path == None:
#         save_path = input('New ILLIQ are calculated.\n YOU MUST INPUT DATA SAVE PATH FOR THEM! :')
#     import pickle
#     f = open(save_path, 'wb')
#     pickle.dump(data, f)
#     f.close()
#
#     print('ILLIQ value are calculated and saved successfully!')
#     return
# =============================================================================

# import pickle
# with open('.\\future_alldailydata_index_volumn_recovery2010-2022_rqdata.txt','rb') as f:
#     data = pickle.load(f)
# f.close()
# ILLIQ(data,20, '.\\future_alldailydata_index_volumn_recovery2010-2022_rqdata.txt')

"""
HIGH指标计算
"""


def HIGH(data, window, futures_universe, time_freq='day'):
    data = data.loc[futures_universe].copy()
    ret = pd.DataFrame()

    for n in window:
        data['high_{}'.format(n)] = (
            data.groupby('future').apply(lambda group: group['closew'] / group['highw'].rolling(n).max())).reset_index(
            level=0, drop=True)

        temp_high = data['high_{}'.format(n)].unstack(level=0)
        new_columns = pd.MultiIndex.from_tuples([(col, f'HIGH{n}') for col in temp_high.columns],
                                                names=['future', 'price'])
        temp_high.columns = new_columns
        ret = pd.concat([ret, temp_high], axis=1)

    return ret


# import pickle
# with open('.\\future_alldailydata_index_volumn_recovery2010-2022_rqdata.txt','rb') as f:
#     data = pickle.load(f)
# f.close()
# HIGH(data,250, '.\\future_alldailydata_index_volumn_recovery2010-2022_rqdata.txt')

"""
RETURNS指标计算
"""


def RETURNS(data, window, futures_universe, time_freq='day'):
    data = data.loc[futures_universe].copy()
    ret = pd.DataFrame()
    open = data['open'].unstack(level=0)
    openw = data['openw'].unstack(level=0)

    for n in window:
        data['returns_{}'.format(n)] = (data.groupby('future').apply(
            lambda group: (group['closew'] - group['closew'].shift(n)) / group['closew'].shift(n))).reset_index(
            level=0, drop=True)

        temp_returns = data['returns_{}'.format(n)].unstack(level=0)
        new_columns = pd.MultiIndex.from_tuples([(col, f'RETURNS{n}') for col in temp_returns.columns],
                                                names=['future', 'price'])
        temp_returns.columns = new_columns
        ret = pd.concat([ret, temp_returns], axis=1)

    open.columns = pd.MultiIndex.from_product([open.columns, ["open"]], names=["future", "price"])
    openw.columns = pd.MultiIndex.from_product([openw.columns, ["openw"]], names=["future", "price"])
    ret = pd.concat([ret, open, openw], axis=1)

    return ret


# import pickle
# with open('.\\future_alldailydata_index_volumn_recovery2010-2022_rqdata.txt','rb') as f:
#     data = pickle.load(f)
# f.close()
# RETURNS(data,20, '.\\future_alldailydata_index_volumn_recovery2010-2022_rqdata.txt')

"""
IMI指标计算
"""


def IMI(data, window, futures_universe, time_freq='day'):
    data = data.loc[futures_universe].copy()
    ret = pd.DataFrame()
    open = data['open'].unstack(level=0)
    openw = data['openw'].unstack(level=0)

    for n in window:
        usum = (data.groupby('future').apply(
            lambda x: ((x['closew'] > x['openw']) * (x['closew'] - x['openw'])).rolling(n).sum())).reset_index(level=0,
                                                                                                               drop=True)
        dsum = (data.groupby('future').apply(
            lambda x: ((x['closew'] <= x['openw']) * (x['openw'] - x['closew'])).rolling(n).sum())).reset_index(level=0,
                                                                                                                drop=True)

        data['imi_{}'.format(n)] = (usum + dsum == 0) + (usum + dsum != 0) * (usum / (usum + dsum))

        temp_imi = data['imi_{}'.format(n)].unstack(level=0)
        new_columns = pd.MultiIndex.from_tuples([(col, f'imi{n}') for col in temp_imi.columns],
                                                names=['future', 'price'])
        temp_imi.columns = new_columns
        ret = pd.concat([ret, temp_imi], axis=1)

    open.columns = pd.MultiIndex.from_product([open.columns, ["open"]], names=["future", "price"])
    openw.columns = pd.MultiIndex.from_product([openw.columns, ["openw"]], names=["future", "price"])
    ret = pd.concat([ret, open, openw], axis=1)

    return ret


# import pickle
# with open('.\\future_alldailydata_index_volumn_recovery2010-2022_rqdata.txt','rb') as f:
#     data = pickle.load(f)
# f.close()
# IMI(data,20, '.\\future_alldailydata_index_volumn_recovery2010-2022_rqdata.txt')

# """
# RSI指标计算
# """

# def RSI(data, n, save_path=None):
#     if type(n) == list:
#         for a in n:
#             cal_rsi(data, a, save_path=save_path)
#     elif type(n) == int:
#         cal_rsi(data, n, save_path=save_path)

# def cal_rsi(data, n, save_path):
#     for (future, _) in data.columns:
#         if (future, 'rsi%s'%(n)) in data.columns:
#             print('RSI value are already prepared with period {}!'.format(n))
#             return
#         break

#     for (future, _) in data.columns:
#         if (future, 'rsi%s'%(n)) in data.columns:
#             continue

#         pf = ((data[future].closew > data[future].closew.shift(1)) * (data[future].closew - data[future].closew.shift(1))).rolling(n).sum()
#         nf = ((data[future].closew <= data[future].closew.shift(1)) * (data[future].closew.shift(1) - data[future].closew)).rolling(n).sum()
#         mr = pf / nf

#         data.loc[:,(future,'rsi%s'%(n))] = mr / (1 + mr)
#     if save_path == None:
#         save_path = input('New RSI are calculated.\n YOU MUST INPUT DATA SAVE PATH FOR THEM! :')
#     import pickle
#     f = open(save_path, 'wb')
#     pickle.dump(data, f)
#     f.close()

#     print('RSI value are calculated and saved successfully!')
#     return

# import pickle
# with open('.\\future_alldailydata_index_volumn_recovery2010-2022_rqdata.txt','rb') as f:
#     data = pickle.load(f)
# f.close()
# RSI(data,20, '.\\future_alldailydata_index_volumn_recovery2010-2022_rqdata.txt')

"""
VRSI指标计算
"""


def VRSI(data, window, futures_universe, time_freq='day'):
    data = data.loc[futures_universe].copy()
    ret = pd.DataFrame()
    open = data['open'].unstack(level=0)
    openw = data['openw'].unstack(level=0)

    for n in window:
        back = (data.groupby('future').apply(
            lambda x: (x['closew'] == x['closew'].shift(1)) * x['volume99'] / 2)).reset_index(level=0, drop=True)
        u = (data.groupby('future').apply(
            lambda x: (x['closew'] > x['closew'].shift(1)) * (x['volume99'] - back) + back)).reset_index(level=0,
                                                                                                         drop=True)
        d = (data.groupby('future').apply(
            lambda x: (x['closew'] < x['closew'].shift(1)) * (x['volume99'] - back) + back)).reset_index(level=0,
                                                                                                         drop=True)

        uu = ((n - 1) * u.shift(1) + u) / n
        dd = ((n - 1) * d.shift(1) + d) / n

        data['vrsi_{}'.format(n)] = 100 * uu / (uu + dd)

        temp_vrsi = data['vrsi_{}'.format(n)].unstack(level=0)
        new_columns = pd.MultiIndex.from_tuples([(col, f'vrsi{n}') for col in temp_vrsi.columns],
                                                names=['future', 'price'])
        temp_vrsi.columns = new_columns
        ret = pd.concat([ret, temp_vrsi], axis=1)

    open.columns = pd.MultiIndex.from_product([open.columns, ["open"]], names=["future", "price"])
    openw.columns = pd.MultiIndex.from_product([openw.columns, ["openw"]], names=["future", "price"])
    ret = pd.concat([ret, open, openw], axis=1)

    return ret


# import pickle
# with open('.\\future_alldailydata_index_volumn_recovery2010-2022_rqdata.txt','rb') as f:
#     data = pickle.load(f)
# f.close()
# VRSI(data,20, '.\\future_alldailydata_index_volumn_recovery2010-2022_rqdata.txt')

"""
ROCMA指标计算
"""


def ROCMA(data, window, futures_universe, time_freq='day'):
    data = data.loc[futures_universe].copy()
    ret = pd.DataFrame()
    open = data['open'].unstack(level=0)
    open.columns = pd.MultiIndex.from_product([open.columns, ["open"]], names=["future", "price"])
    openw = data['openw'].unstack(level=0)
    openw.columns = pd.MultiIndex.from_product([openw.columns, ["openw"]], names=["future", "price"])

    for n in window:
        roc = data.groupby('future').apply(
            lambda x: (x['closew'] - x['closew'].shift(n[0])) / x['close'].shift(n[0])).reset_index(level=0, drop=True)
        data['rocma_{}'.format(n)] = roc.rolling(n[1]).mean()

        temp_rocma = data['rocma_{}'.format(n)].unstack(level=0)
        new_columns = pd.MultiIndex.from_tuples([(col, f'rocma{n}') for col in temp_rocma.columns],
                                                names=['future', 'price'])
        temp_rocma.columns = new_columns
        ret = pd.concat([ret, temp_rocma], axis=1)

    ret = pd.concat([ret, open, openw], axis=1)

    return ret


# import pickle
# with open('.\\future_alldailydata_index_volumn_recovery2010-2022_rqdata.txt','rb') as f:
#     data = pickle.load(f)
# f.close()
# ROCMA(data,[12,6], '.\\future_alldailydata_index_volumn_recovery2010-2022_rqdata.txt')

"""
ACM指标计算
"""


def ACM(data, window, futures_universe, time_freq='day'):
    data = data.loc[futures_universe].copy()
    ret = pd.DataFrame()

    open = data['open'].unstack(level=0)
    open.columns = pd.MultiIndex.from_product([open.columns, ["open"]], names=["future", "price"])
    openw = data['openw'].unstack(level=0)
    openw.columns = pd.MultiIndex.from_product([openw.columns, ["openw"]], names=["future", "price"])

    for n in window:
        m = int(n[1] * n[0])
        amp = (data['highw'] - data['loww']).groupby('future').rolling(n[1]).apply(lambda x: sum(heapq.nsmallest(m, x)),
                                                                                   raw=True).reset_index(level=0,
                                                                                                         drop=True)

        data['acm_{}'.format(n[0] * 1000 + n[1])] = amp

        temp_acm = data['acm_{}'.format(n[0] * 1000 + n[1])].unstack(level=0)
        new_columns = pd.MultiIndex.from_tuples([(col, f'acm{n[0], n[1]}') for col in temp_acm.columns],
                                                names=['future', 'price'])
        temp_acm.columns = new_columns
        ret = pd.concat([ret, temp_acm], axis=1)

    ret = pd.concat([ret, open, openw], axis=1)

    return ret


# import pickle
# with open('.\\future_alldailydata_index_volumn_recovery2010-2022_rqdata.txt','rb') as f:
#     data = pickle.load(f)
# f.close()
# ACM(data,[0.5,160], '.\\future_alldailydata_index_volumn_recovery2010-2022_rqdata.txt')

"""
RI指标计算
"""


def RI(data, window, futures_universe, time_freq='day'):
    data = data.loc[futures_universe].copy()
    ret = pd.DataFrame()

    open = data['open'].unstack(level=0)
    open.columns = pd.MultiIndex.from_product([open.columns, ["open"]], names=["future", "price"])
    openw = data['openw'].unstack(level=0)
    openw.columns = pd.MultiIndex.from_product([openw.columns, ["openw"]], names=["future", "price"])

    for n in window:
        a = data['highw'] - data['loww']
        b = (data['closew'].groupby('future').shift(1) - data['highw']).abs()
        c = (data['closew'].groupby('future').shift(1) - data['loww']).abs()
        tr = a.where((a > b) & (a > c), b.where(b > c, c))
        condition = ((data['closew'] > data['closew'].groupby('future').shift(1)))
        w = condition * (tr / (data['closew'] - data['closew'].groupby('future').shift(1)) - tr) + tr
        minw = w.groupby('future').rolling(n[1]).min().reset_index(level=0, drop=True)
        maxw = w.groupby('future').rolling(n[0]).max().reset_index(level=0, drop=True)
        sr_condition = maxw > minw
        sr1 = (w - minw) / (maxw - minw) * 100
        sr2 = (w - minw) * 100
        sr = sr_condition * (sr1 - sr2) + sr2
        data['ri_{}'.format(n[0] * 1000 + n[1])] = sr.groupby('future').rolling(n[1]).mean().reset_index(level=0,
                                                                                                         drop=True)

        temp_ri = data['ri_{}'.format(n[0] * 1000 + n[1])].unstack(level=0)
        new_columns = pd.MultiIndex.from_tuples([(col, f'ri{n[0], n[1]}') for col in temp_ri.columns],
                                                names=['future', 'price'])
        temp_ri.columns = new_columns
        ret = pd.concat([ret, temp_ri], axis=1)

    ret = pd.concat([ret, open, openw], axis=1)

    return ret


# import pickle
# with open('.\\future_alldailydata_2010-20230519_index_volumn_recovery_rqdata.txt','rb') as f:
#     data = pickle.load(f)
# f.close()
# RI(data,[20,5], '.\\future_alldailydata_2010-20230519_index_volumn_recovery_rqdata.txt')

"""
MAEMV指标计算
"""


def MAEMV(data, window, futures_universe, time_freq='day'):
    data = data.loc[futures_universe].copy()
    ret = pd.DataFrame()

    open = data['open'].unstack(level=0)
    open.columns = pd.MultiIndex.from_product([open.columns, ["open"]], names=["future", "price"])
    openw = data['openw'].unstack(level=0)
    openw.columns = pd.MultiIndex.from_product([openw.columns, ["openw"]], names=["future", "price"])

    for n in window:
        dm = ((data['highw'] + data['loww']) - (
                    data['highw'].groupby('future').shift(1) + data['loww'].groupby('future').shift(1))) / 2
        br = data['indexvol_recover'] / (data['highw'] - data['loww'])
        emv = (dm / br).groupby('future').rolling(n[0]).sum().reset_index(level=0, drop=True)
        data['maemv_{}'.format(n[0] * 1000 + n[1])] = emv.groupby('future').rolling(n[1]).mean().reset_index(level=0,
                                                                                                             drop=True)

        temp_maemv = data['maemv_{}'.format(n[0] * 1000 + n[1])].unstack(level=0)
        new_columns = pd.MultiIndex.from_tuples([(col, f'maemv{n[0] * 1000 + n[1]}') for col in temp_maemv.columns],
                                                names=['future', 'price'])
        temp_maemv.columns = new_columns
        ret = pd.concat([ret, temp_maemv], axis=1)

    ret = pd.concat([ret, open, openw], axis=1)

    return ret


# import pickle
# with open('.\\future_alldailydata_2010-20230519_index_volumn_recovery_rqdata.txt','rb') as f:
#     data = pickle.load(f)
# f.close()
# MAEMV(data,[14,9], '.\\future_alldailydata_2010-20230519_index_volumn_recovery_rqdata.txt')

"""
DDI指标计算
"""


def DDI(data, window, futures_universe, time_freq='day'):
    data = data.loc[futures_universe].copy()
    ret = pd.DataFrame()

    open = data['open'].unstack(level=0)
    open.columns = pd.MultiIndex.from_product([open.columns, ["open"]], names=["future", "price"])
    openw = data['openw'].unstack(level=0)
    openw.columns = pd.MultiIndex.from_product([openw.columns, ["openw"]], names=["future", "price"])

    for n in window:
        condition1 = (data['highw'].groupby('future').shift(1) + data['loww'].groupby('future').shift(1)) < (
                    data['highw'] + data['loww'])
        condition2 = (data['highw'].groupby('future').shift(1).abs() - data['highw'].abs()) >= (
                    data['loww'].groupby('future').shift(1).abs() - data['loww'].abs())
        dmz = condition1 * condition2 * data['highw'].groupby('future').shift(1).abs() - data['highw'].abs()

        condition3 = (data['highw'].groupby('future').shift(1) + data['loww'].groupby('future').shift(1)) >= (
                    data['highw'] + data['loww'])
        condition4 = (data['highw'].groupby('future').shift(1).abs() - data['highw'].abs()) < (
                    data['loww'].groupby('future').shift(1).abs() - data['loww'].abs())
        dmf = condition3 * condition4 * data['loww'].groupby('future').shift(1).abs() - data['loww'].abs()

        dmz_sum = dmz.groupby('future').rolling(n).sum().reset_index(level=0, drop=True)
        dmf_sum = dmf.groupby('future').rolling(n).sum().reset_index(level=0, drop=True)

        diz = dmz_sum / (dmz_sum + dmf_sum)
        dif = dmf_sum / (dmz_sum + dmf_sum)

        data['ddi_{}'.format(n)] = ((dmz_sum != 0) | (dmf_sum != 0)) * (diz - dif)

        temp_ddi = data['ddi_{}'.format(n)].unstack(level=0)
        new_columns = pd.MultiIndex.from_tuples([(col, f'ddi{n}') for col in temp_ddi.columns],
                                                names=['future', 'price'])
        temp_ddi.columns = new_columns
        ret = pd.concat([ret, temp_ddi], axis=1)

    ret = pd.concat([ret, open, openw], axis=1)

    return ret


# import pickle
# with open('.\\future_alldailydata_index_volumn_recovery2010-2022_rqdata.txt','rb') as f:
#     data = pickle.load(f)
# f.close()
# DDI(data,20, '.\\future_alldailydata_index_volumn_recovery2010-2022_rqdata.txt')

"""
VR指标计算
"""


def VR(data, window, futures_universe, time_freq='day'):
    data = data.loc[futures_universe].copy()
    ret = pd.DataFrame()

    open = data['open'].unstack(level=0)
    open.columns = pd.MultiIndex.from_product([open.columns, ["open"]], names=["future", "price"])
    openw = data['openw'].unstack(level=0)
    openw.columns = pd.MultiIndex.from_product([openw.columns, ["openw"]], names=["future", "price"])

    for n in window:
        condition1 = data['closew'].groupby('future').shift(1) < data['closew']
        A = condition1 * data['volume99']
        condition2 = data['closew'].groupby('future').shift(1) >= data['closew']
        B = condition2 * data['volume99']

        A_sum = A.groupby('future').rolling(n).sum().reset_index(level=0, drop=True)
        B_sum = B.groupby('future').rolling(n).sum().reset_index(level=0, drop=True)

        data['vr_{}'.format(n)] = A_sum / B_sum * 100

        temp_vr = data['vr_{}'.format(n)].unstack(level=0)
        new_columns = pd.MultiIndex.from_tuples([(col, f'vr{n}') for col in temp_vr.columns], names=['future', 'price'])
        temp_vr.columns = new_columns
        ret = pd.concat([ret, temp_vr], axis=1)

    ret = pd.concat([ret, open, openw], axis=1)

    return ret


# import pickle
# with open('.\\future_alldailydata_2010-20230519_index_volumn_recovery_rqdata.txt','rb') as f:
#     data = pickle.load(f)
# f.close()
# VR(data,26, '.\\future_alldailydata_2010-20230519_index_volumn_recovery_rqdata.txt')

"""
ASI指标计算
"""


def ASI(data, window, futures_universe, time_freq='day'):
    data = data.loc[futures_universe].copy()
    open = data['open'].unstack(level=0)
    open.columns = pd.MultiIndex.from_product([open.columns, ["open"]], names=["future", "price"])
    openw = data['openw'].unstack(level=0)
    openw.columns = pd.MultiIndex.from_product([openw.columns, ["openw"]], names=["future", "price"])

    A = (data['highw'] - data['closew'].groupby("future").shift(1)).abs()
    B = (data['loww'] - data['closew'].groupby("future").shift(1)).abs()
    C = (data['highw'] - data['loww'].groupby("future").shift(1)).abs()
    D = (data['closew'].groupby("future").shift(1) - data['openw'].groupby("future").shift(1)).abs()
    E = data['closew'] - data['closew'].groupby("future").shift(1)
    F = data['closew'] - data['openw']
    G = data['closew'].groupby("future").shift(1) - data['openw'].groupby("future").shift(1)

    X = E + 0.5 * F + G
    K = B + (A > B) * (A - B)
    R2 = C + 0.25 * D + ((B > A) & (B > C)) * (B + 0.5 * A - C)
    R = R2 + ((A > B) & (A > C)) * (A + 0.5 * B + 0.25 * D - R2)

    SI = 16 * X / R * K

    ret = pd.DataFrame()
    for n in window:
        ASI_cumsum = SI.groupby("future").rolling(n).sum().reset_index(level=0, drop=True)
        data['asi_{}'.format(n)] = ASI_cumsum
        temp_asi = data['asi_{}'.format(n)].unstack(level=0)
        new_columns = pd.MultiIndex.from_tuples([(col, f'asi{n}') for col in temp_asi.columns],
                                                names=['future', 'price'])
        temp_asi.columns = new_columns
        ret = pd.concat([ret, temp_asi], axis=1)

    ret = pd.concat([ret, open, openw], axis=1)

    return ret


# import pickle
# with open('.\\future_alldailydata_2010-20230519_index_volumn_recovery_rqdata.txt','rb') as f:
#     data = pickle.load(f)
# f.close()
# ASI(data,26, '.\\future_alldailydata_2010-20230519_index_volumn_recovery_rqdata.txt')


"""
DBCD指标计算
"""


def DBCD(data, window, futures_universe, time_freq='day'):
    data = data.loc[futures_universe].copy()
    open = data['open'].unstack(level=0)
    open.columns = pd.MultiIndex.from_product([open.columns, ["open"]], names=["future", "price"])
    openw = data['openw'].unstack(level=0)
    openw.columns = pd.MultiIndex.from_product([openw.columns, ["openw"]], names=["future", "price"])

    ret = pd.DataFrame()
    for n in window:
        bias = data.groupby("future").apply(
            lambda group: (group.closew - group.closew.rolling(n[0]).mean()) / group.closew.rolling(
                n[0]).mean()).reset_index(level=0, drop=True)
        diff = bias - bias.groupby("future").shift(n[1])
        data['dbcd_{}'.format(n)] = diff.groupby("future").rolling(n[2]).mean().reset_index(level=0, drop=True)
        temp_dbcd = data['dbcd_{}'.format(n)].unstack(level=0)
        new_columns = pd.MultiIndex.from_tuples([(col, f'dbcd{n}') for col in temp_dbcd.columns],
                                                names=['future', 'price'])
        temp_dbcd.columns = new_columns
        ret = pd.concat([ret, temp_dbcd], axis=1)

    ret = pd.concat([ret, open, openw], axis=1)
    return ret


# import pickle
# with open('.\\future_alldailydata_2010-20230519_index_volumn_recovery_rqdata.txt','rb') as f:
#     data = pickle.load(f)
# f.close()
# DBCD(data,[5,16,17], '.\\future_alldailydata_2010-20230519_index_volumn_recovery_rqdata.txt')

"""
WQALPHA002指标计算
"""


def WQALPHA002(data, window, futures_universe, time_freq='day'):
    data = data.loc[futures_universe].copy()

    open = data['open'].unstack(level=0)
    open.columns = pd.MultiIndex.from_product([open.columns, ["open"]], names=["future", "price"])

    openw = data['openw'].unstack(level=0)
    openw.columns = pd.MultiIndex.from_product([openw.columns, ["openw"]], names=["future", "price"])

    # Check if 'indexvol_recover' is present in columns
    if "indexvol_recover" not in data.columns:
        raise ValueError("Column 'indexvol_recover' is not present in data.")

    ret = pd.DataFrame()
    for n in window:
        inner1 = np.log(data['indexvol_recover']).groupby('future').diff(n[0])
        inner2 = (data['closew'] - openw['openw']) / openw['openw']
        data['wqalpha002_{}'.format(n)] = inner1.groupby("future").rolling(n[1]).corr(inner2).reset_index(level=0,
                                                                                                          drop=True)
        temp_wqalpha002 = data['wqalpha002_{}'.format(n)].unstack(level=0)
        new_columns = pd.MultiIndex.from_tuples([(col, f'wqalpha002{n}') for col in temp_wqalpha002.columns],
                                                names=['future', 'price'])
        temp_wqalpha002.columns = new_columns
        ret = pd.concat([ret, temp_wqalpha002], axis=1)

    return pd.concat([ret, open, openw], axis=1)


# import pickle
# with open('.\\future_alldailydata_2010-20230519_index_volumn_recovery_rqdata.txt','rb') as f:
#     data = pickle.load(f)
# f.close()
# WQALPHA002(data,[2,6], '.\\future_alldailydata_2010-20230519_index_volumn_recovery_rqdata.txt')

"""
WQALPHA008指标计算
"""


def WQALPHA008(data, window, futures_universe, time_freq='day'):
    data = data.loc[futures_universe].copy()
    open = data['open'].unstack(level=0)
    open.columns = pd.MultiIndex.from_product([open.columns, ['open']], names=['future', 'price'])
    openw = data['openw'].unstack(level=0)
    openw.columns = pd.MultiIndex.from_product([openw.columns, ['openw']], names=['future', 'price'])
    ret = pd.DataFrame()

    for n in window:
        inner = data.groupby('future').apply(
            lambda group: group.openw.rolling(n[0]).sum() * group.closew.rolling(n[0]).sum()).reset_index(level=0,
                                                                                                          drop=True)
        data['wqalpha008_{}'.format(n)] = -1 * (inner - inner.groupby('future').shift(n[1]))
        temp_wqalpha008 = data['wqalpha008_{}'.format(n)].unstack(level=0)
        new_columns = pd.MultiIndex.from_tuples([(col, f'wqalpha008{n}') for col in temp_wqalpha008.columns],
                                                names=['future', 'price'])
        temp_wqalpha008.columns = new_columns
        ret = pd.concat([ret, temp_wqalpha008], axis=1)

    return pd.concat([ret, open, openw], axis=1)


# import pickle
# with open('.\\future_alldailydata_2010-20230519_index_volumn_recovery_rqdata.txt','rb') as f:
#     data = pickle.load(f)
# f.close()
# WQALPHA008(data,[5, 10], '.\\future_alldailydata_2010-20230519_index_volumn_recovery_rqdata.txt')

"""
WQALPHA026指标计算
"""


def WQALPHA026(data, window, futures_universe, time_freq='day'):
    """
    计算期货的WQALPHA026指标，每遇到新的窗口参数，计算后同时保存到原始数据表中。

    Parameters
    ----------
    data : pd.DataFrame
        数据的完整大表，需包含需要计算的期货数据的indexvol_recover和highw值以及已有的一些WQALPHA026值.
    window : list
        WQALPHA026的窗口参数, 是一个二元素列表.
    futures_universe : list
        需要计算的期货名称列表
    Returns
    -------
    result : pd.DataFrame
        计算得到的WQALPHA026值表.
    """
    ret = pd.DataFrame()
    data = data.loc[futures_universe].copy()
    open = data['open'].unstack(level=0)
    open.columns = pd.MultiIndex.from_product([open.columns, ['open']], names=['future', 'price'])
    openw = data['openw'].unstack(level=0)
    openw.columns = pd.MultiIndex.from_product([openw.columns, ['openw']], names=['future', 'price'])

    for n in window:
        inner1 = data.groupby('future').indexvol_recover.rolling(n[0], min_periods=n[0]).apply(
            lambda x: x.rank(pct=True)).reset_index(level=0, drop=True)
        inner2 = data.groupby('future').highw.rolling(n[0], min_periods=n[0]).apply(
            lambda x: pd.Series(x).rank(pct=True)).reset_index(level=0, drop=True)
        inner3 = data.groupby('future').apply(
            lambda df: pd.Series(inner1[df.index]).rolling(n[0], min_periods=n[0]).corr(
                pd.Series(inner2[df.index])))  # grouped correlation needs to be computed in this way
        data['wqalpha026'] = -1 * inner3.rolling(n[1], min_periods=n[1]).max().reset_index(level=0, drop=True)

        temp = data['wqalpha026'].unstack(level=0)
        new_columns = pd.MultiIndex.from_tuples([(col, f'wqalpha026_{n[0]}_{n[1]}') for col in temp.columns],
                                                names=['future', 'price'])
        temp.columns = new_columns
        ret = pd.concat([ret, temp], axis=1)

    ret = pd.concat([ret, open, openw], axis=1)
    # ret = standardize(ret,'wqalpha026_',n)

    return ret


# import pickle
# with open('.\\future_alldailydata_2010-20230519_index_volumn_recovery_rqdata.txt','rb') as f:
#     data = pickle.load(f)
# f.close()
# WQALPHA026(data,[5, 3], '.\\future_alldailydata_2010-20230519_index_volumn_recovery_rqdata.txt')

"""
WQALPHA035指标计算
"""


def WQALPHA035(data, window, futures_universe, time_freq='day'):
    """
    计算期货的WQALPHA035指标，每遇到新的窗口参数，计算后同时保存到原始数据表中。

    Parameters
    ----------
    data : pd.DataFrame
        数据的完整大表，需包含需要计算的期货数据的indexvol_recover,closew,highw和loww值以及已有的一些WQALPHA035值.
    windows : list
        WQALPHA035的窗口参数, 是一个三元素列表.
    futures_universe : list
        需要计算的期货名称列表
    Returns
    -------
    result : pd.DataFrame
        计算得到的WQALPHA035值表以及目标期货的open和openw值.
    """
    ret = pd.DataFrame()
    data = data.loc[futures_universe].copy()

    open = data['open'].unstack(level=0)
    openw = data['openw'].unstack(level=0)

    for n in window:
        returns = data.groupby('future').closew.apply(lambda df: df.pct_change())
        inner1 = data.groupby('future').indexvol_recover.rolling(n[0], min_periods=n[0]).apply(
            lambda x: pd.Series(x).rank(pct=True)).reset_index(level=0, drop=True)
        inner2 = 1 - data.groupby('future').apply(
            lambda df: (df.closew + df.highw - df.loww).rolling(n[1], min_periods=n[1]).apply(
                lambda x: pd.Series(x).rank(pct=True))).reset_index(level=0, drop=True)
        inner3 = 1 - data.groupby('future').apply(
            lambda df: pd.Series(returns[df.index]).rolling(n[2], min_periods=n[2]).apply(
                lambda x: pd.Series(x).rank(pct=True))).reset_index(level=0, drop=True)
        data['wqalpha035'] = inner1 * inner2 * inner3

        temp = data['wqalpha035'].unstack(level=0)
        new_columns = pd.MultiIndex.from_tuples([(col, f'wqalpha035_{n[0]}_{n[1]}_{n[2]}') for col in temp.columns],
                                                names=['future', 'price'])
        temp.columns = new_columns
        ret = pd.concat([ret, temp], axis=1)

    open.columns = pd.MultiIndex.from_product([open.columns, ["open"]], names=["future", "price"])
    openw.columns = pd.MultiIndex.from_product([openw.columns, ["openw"]], names=["future", "price"])
    ret = pd.concat([ret, open, openw], axis=1)

    # ret = standardize(ret,'wqalpha035_',n)

    return ret


# import pickle
# with open('.\\future_alldailydata_2010-20230519_index_volumn_recovery_rqdata.txt','rb') as f:
#     data = pickle.load(f)
# f.close()
# WQALPHA035(data,[32,16,32], '.\\future_alldailydata_2010-20230519_index_volumn_recovery_rqdata.txt')

"""
WQALPHA043指标计算
"""


def WQALPHA043(data, window, futures_universe, time_freq='day'):
    """
    计算期货的WQALPHA043指标，每遇到新的窗口参数，计算后同时保存到原始数据表中。

    Parameters
    ----------
    data : pd.DataFrame
        数据的完整大表，需包含需要计算的期货数据的indexvol_recover和closew值以及已有的一些WQALPHA043值.
    windows : list
        WQALPHA043的窗口参数, 是一个四元素列表.
    futures_universe : list
        需要计算的期货名称列表
    Returns
    -------
    result : pd.DataFrame
        计算得到的WQALPHA043值表以及目标期货的open和openw值.
    """
    ret = pd.DataFrame()
    data = data.loc[futures_universe].copy()

    open = data['open'].unstack(level=0)
    openw = data['openw'].unstack(level=0)

    for n in window:
        inner1 = data.groupby('future').apply(
            lambda df: (df.indexvol_recover / df.indexvol_recover.rolling(n[0], min_periods=n[0]).mean()).rolling(n[1],
                                                                                                                  min_periods=
                                                                                                                  n[
                                                                                                                      1]).apply(
                lambda x: x.rank(pct=True))).reset_index(level=0, drop=True)
        inner2 = data.groupby('future').closew.apply(
            lambda df: (-1 * df.shift(n[2])).rolling(n[3], min_periods=n[3]).apply(
                lambda x: x.rank(pct=True))).reset_index(level=0, drop=True)
        data['wqalpha043'] = inner1 * inner2

        temp = data['wqalpha043'].unstack(level=0)
        new_columns = pd.MultiIndex.from_tuples(
            [(col, f'wqalpha043{n[0]}_{n[1]}_{n[2]}_{n[3]}') for col in temp.columns], names=['future', 'price'])
        temp.columns = new_columns
        ret = pd.concat([ret, temp], axis=1)

    open.columns = pd.MultiIndex.from_product([open.columns, ["open"]], names=["future", "price"])
    openw.columns = pd.MultiIndex.from_product([openw.columns, ["openw"]], names=["future", "price"])
    ret = pd.concat([ret, open, openw], axis=1)

    # ret = standardize(ret,'wqalpha043_',n)

    return ret


# import pickle
# with open('.\\future_alldailydata_2010-20230519_index_volumn_recovery_rqdata.txt','rb') as f:
#     data = pickle.load(f)
# f.close()
# WQALPHA043(data,[20, 20, 7, 8], '.\\future_alldailydata_2010-20230519_index_volumn_recovery_rqdata.txt')

"""
WQALPHA049指标计算
"""


def WQALPHA049(data, window, futures_universe, time_freq='day'):
    """
    计算期货的WQALPHA049指标，每遇到新的窗口参数，计算后同时保存到原始数据表中。

    Parameters
    ----------
    data : pd.DataFrame
        数据的完整大表，需包含需要计算的期货数据的closew值以及已有的一些WQALPHA049值.
    windows : list
        WQALPHA049的窗口参数, 是一个二元素列表.
    futures_universe : list
        需要计算的期货名称列表
    Returns
    -------
    result : pd.DataFrame
        计算得到的WQALPHA049值表以及目标期货的open和openw值.
    """
    ret = pd.DataFrame()
    data = data.loc[futures_universe].copy()

    open = data['open'].unstack(level=0)
    openw = data['openw'].unstack(level=0)

    for n in window:
        inner = data.groupby('future').apply(lambda df: (df.closew.shift(n[0]) - df.closew.shift(n[1]) - (
                    df.closew.shift(n[1]) - df.closew)) / 10).reset_index(level=0, drop=True)
        data['wqalpha049'] = (inner < -1) * 10 + (inner >= -1) * (
            data.groupby('future').apply(lambda df: df.closew.shift(1) - df.closew)).reset_index(level=0, drop=True)

        temp = data['wqalpha049'].unstack(level=0)
        new_columns = pd.MultiIndex.from_tuples([(col, f'wqalpha049{n[0]}_{n[1]}') for col in temp.columns],
                                                names=['future', 'price'])
        temp.columns = new_columns
        ret = pd.concat([ret, temp], axis=1)

    open.columns = pd.MultiIndex.from_product([open.columns, ["open"]], names=["future", "price"])
    openw.columns = pd.MultiIndex.from_product([openw.columns, ["openw"]], names=["future", "price"])
    ret = pd.concat([ret, open, openw], axis=1)

    # ret = standardize(ret,'wqalpha049_',n)

    return ret


# import pickle
# with open('.\\future_alldailydata_2010-20230519_index_volumn_recovery_rqdata.txt','rb') as f:
#     data = pickle.load(f)
# f.close()
# WQALPHA049(data,[20, 10], '.\\future_alldailydata_2010-20230519_index_volumn_recovery_rqdata.txt')

"""
WQALPHA053指标计算
"""


def WQALPHA053(data, window, futures_universe, time_freq='day'):
    """
    计算期货的WQALPHA053指标，每遇到新的窗口参数，计算后同时保存到原始数据表中。

    Parameters
    ----------
    data : pd.DataFrame
        数据的完整大表，需包含需要计算的期货数据的closew，loww，highw值以及已有的一些WQALPHA053值.
    windows : list
        WQALPHA053的窗口参数, 是一个单元素列表.
    futures_universe : list
        需要计算的期货名称列表
    Returns
    -------
    result : pd.DataFrame
        计算得到的WQALPHA053值表以及目标期货的open和openw值.
    """
    ret = pd.DataFrame()
    data = data.loc[futures_universe].copy()

    open = data['open'].unstack(level=0)
    openw = data['openw'].unstack(level=0)

    for n in window:
        inner = ((data.closew - data.loww) - (data.highw - data.closew)) / (data.closew - data.loww)
        data['wqalpha053'] = -1 * inner.groupby('future').apply(lambda x: x.diff(n))

        temp = data['wqalpha053'].unstack(level=0)
        new_columns = pd.MultiIndex.from_tuples([(col, f'wqalpha053{n}') for col in temp.columns],
                                                names=['future', 'price'])
        temp.columns = new_columns
        ret = pd.concat([ret, temp], axis=1)

    open.columns = pd.MultiIndex.from_product([open.columns, ["open"]], names=["future", "price"])
    openw.columns = pd.MultiIndex.from_product([openw.columns, ["openw"]], names=["future", "price"])
    ret = pd.concat([ret, open, openw], axis=1)

    # ret = standardize(ret,'wqalpha053_',n)

    return ret


# import pickle
# with open('.\\future_alldailydata_2010-20230519_index_volumn_recovery_rqdata.txt','rb') as f:
#     data = pickle.load(f)
# f.close()
# WQALPHA053(data,9, '.\\future_alldailydata_2010-20230519_index_volumn_recovery_rqdata.txt')

"""
WQALPHA054指标计算
"""


def WQALPHA054(data, futures_universe):
    ret = pd.DataFrame()
    data = data.loc[futures_universe].copy()

    open = data['open'].unstack(level=0)
    openw = data['openw'].unstack(level=0)

    data['wqalpha054'] = -1 * (data.loww - data.closew) * np.power(data.openw, 5) / (data.loww - data.highw) / np.power(
        data.closew, 5)
    temp = data['wqalpha054'].unstack(level=0)
    new_columns = pd.MultiIndex.from_tuples([(col, 'wqalpha054') for col in temp.columns], names=['future', 'price'])
    temp.columns = new_columns
    ret = pd.concat([ret, temp], axis=1)

    open.columns = pd.MultiIndex.from_product([open.columns, ["open"]], names=["future", "price"])
    openw.columns = pd.MultiIndex.from_product([openw.columns, ["openw"]], names=["future", "price"])
    ret = pd.concat([ret, open, openw], axis=1)

    # ret = standardize(ret,'wqalpha054_','no')

    return ret


# import pickle
# with open('.\\future_alldailydata_2010-20230519_index_volumn_recovery_rqdata.txt','rb') as f:
#     data = pickle.load(f)
# f.close()
# WQALPHA054(data, '.\\future_alldailydata_2010-20230519_index_volumn_recovery_rqdata.txt')

"""
CYE指标计算
"""


def CYE(data, window, futures_universe, time_freq='day'):
    ret = pd.DataFrame()
    data = data.loc[futures_universe].copy()

    open = data['open'].unstack(level=0)
    openw = data['openw'].unstack(level=0)

    for n in window:
        cyel = data.groupby('future').apply(lambda group: (group.closew.rolling(n[0]).mean() - group.closew.rolling(
            n[0]).mean().shift(1)) / group.closew.rolling(n[0]).mean().shift(1)).reset_index(level=0, drop=True)
        cyes = data.groupby('future').apply(lambda group: (group.closew.rolling(n[1]).mean().rolling(
            n[0]).mean() - group.closew.rolling(n[1]).mean().rolling(n[0]).mean()) / group.closew.rolling(
            n[1]).mean().rolling(n[0]).mean()).reset_index(level=0, drop=True)
        data['cye'] = cyel - cyes

        temp = data['cye'].unstack(level=0)
        new_columns = pd.MultiIndex.from_tuples([(col, f'cye{n[0]}_{n[1]}') for col in temp.columns],
                                                names=['future', 'price'])
        temp.columns = new_columns
        ret = pd.concat([ret, temp], axis=1)

    open.columns = pd.MultiIndex.from_product([open.columns, ["open"]], names=["future", "price"])
    openw.columns = pd.MultiIndex.from_product([openw.columns, ["openw"]], names=["future", "price"])
    ret = pd.concat([ret, open, openw], axis=1)

    # ret = standardize(ret,'cye',n)

    return ret


# import pickle
# with open('.\\future_alldailydata_2010-20230519_index_volumn_recovery_rqdata.txt','rb') as f:
#     data = pickle.load(f)
# f.close()
# CYE(data,[5, 20], '.\\future_alldailydata_2010-20230519_index_volumn_recovery_rqdata.txt')

"""
DKX指标计算
"""


def DKX(data, window, futures_universe, time_freq='day'):
    ret = pd.DataFrame()
    data = data.loc[futures_universe].copy()

    open = data['open'].unstack(level=0)
    openw = data['openw'].unstack(level=0)

    def wma(df, N):
        return df.rolling(N).apply(lambda x: x[::-1].cumsum().sum() * 2 / N / (N + 1))

    for n in window:
        mid = (3 * data.closew + data.loww + data.openw + data.highw) / 6
        data['dkx'] = mid.groupby('future').apply(lambda group: (wma(group, n[0]) / group).rolling(n[1]).mean())

        temp = data['dkx'].unstack(level=0)
        new_columns = pd.MultiIndex.from_tuples([(col, f'dkx{n[0]}_{n[1]}') for col in temp.columns],
                                                names=['future', 'price'])
        temp.columns = new_columns
        ret = pd.concat([ret, temp], axis=1)

    open.columns = pd.MultiIndex.from_product([open.columns, ["open"]], names=["future", "price"])
    openw.columns = pd.MultiIndex.from_product([openw.columns, ["openw"]], names=["future", "price"])
    ret = pd.concat([ret, open, openw], axis=1)

    # ret = standardize(ret,'dkx',n)

    return ret


# import pickle
# with open('.\\future_alldailydata_2010-20230519_index_volumn_recovery_rqdata.txt','rb') as f:
#     data = pickle.load(f)
# f.close()
# DKX(data,[20, 10], '.\\future_alldailydata_2010-20230519_index_volumn_recovery_rqdata.txt')

"""
WQALPHA057指标计算
"""


def WQALPHA057(data, window, futures_universe, time_freq='day'):
    ret = pd.DataFrame()
    data = data.loc[futures_universe].copy()

    open = data['open'].unstack(level=0)
    openw = data['openw'].unstack(level=0)

    def wma(df, N):
        return df.rolling(N).apply(lambda x: x[::-1].cumsum().sum() * 2 / N / (N + 1))

    for n in window:
        N1, N2 = n[0], n[1]
        to_subtract = data.groupby('future').apply(lambda df: df.closew.rolling(N1).apply(np.argmax))
        data['wqalpha057'] = -1 * (data.closew - data.settlement) / wma(N1 - to_subtract, N2).reset_index(level=0,
                                                                                                          drop=True)

        temp = data['wqalpha057'].unstack(level=0)
        new_columns = pd.MultiIndex.from_tuples([(col, f'wqalpha057{n[0]}_{n[1]}') for col in temp.columns],
                                                names=['future', 'price'])
        temp.columns = new_columns
        ret = pd.concat([ret, temp], axis=1)

    open.columns = pd.MultiIndex.from_product([open.columns, ["open"]], names=["future", "price"])
    openw.columns = pd.MultiIndex.from_product([openw.columns, ["openw"]], names=["future", "price"])
    ret = pd.concat([ret, open, openw], axis=1)

    # ret = standardize(ret, 'wqalpha057_', n)

    return ret


# import pickle
# with open('.\\future_alldailydata_2010-20230519_index_volumn_recovery_rqdata.txt','rb') as f:
#     data = pickle.load(f)
# f.close()
# WQALPHA057(data,[30, 2], '.\\future_alldailydata_2010-20230519_index_volumn_recovery_rqdata.txt')
"""
WQALPHA061指标计算
"""


def WQALPHA061(data, window, futures_universe, time_freq='day'):
    '''
    计算 WQALPHA061 (带 rank) 并直接返回
    :param data 默认旋转后
    :param window 是一个二维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return 待确定
    '''

    ret = pd.DataFrame()
    data = data.loc[futures_universe].copy()
    open = data['open'].unstack(level=0)
    openw = data['openw'].unstack(level=0)

    for w in window:
        [N1, N2, N3] = w
        data['inner1'] = data.groupby('future', group_keys=False).settlement.apply(
            lambda s: s - s.rolling(N1, min_periods=N1).min())
        data['rank1'] = data['inner1'].unstack(level=0).rank(axis=1, pct=True).stack().swaplevel()

        data['volume_mean'] = data.groupby('future', group_keys=False).volume.apply(
            lambda v: v.rolling(N2, min_periods=N2).mean())
        data['inner2'] = data.groupby('future', group_keys=False).apply(
            lambda df: df.settlement.rolling(N3, min_periods=N3).corr(df.volume_mean)).reset_index(level=0, drop=True)
        data['rank2'] = data['inner2'].unstack(level=0).rank(axis=1, pct=True).stack().swaplevel()

        temp = (data['rank1'] - data['rank2']).unstack(level=0)
        new_columns = pd.MultiIndex.from_tuples([(col, f'wqalpha061{N1}_{N2}_{N3}') for col in temp.columns],
                                                names=['future', 'price'])
        temp.columns = new_columns
        ret = pd.concat([ret, temp], axis=1)

    open.columns = pd.MultiIndex.from_product([open.columns, ["open"]], names=["future", "price"])
    openw.columns = pd.MultiIndex.from_product([openw.columns, ["openw"]], names=["future", "price"])
    ret = pd.concat([ret, open, openw], axis=1)

    return ret


"""
WQALPHA088_close指标计算, 窗口
"""


# wqalpha088_time_lst = [
#     [4, 4, 10, 10, 4, 4, 3],
#     [6, 6, 20, 20, 6, 6, 4],
#     [8, 8, 40, 20, 8, 7, 3],
#     [6, 6, 10, 10, 4, 4, 4],
#     [16, 16, 40, 10, 16, 16, 8],
#     [20, 20, 20, 20, 20, 20, 20]
# ]
def WQALPHA088_close(data, window, futures_universe, time_freq='day'):
    '''
    计算 WQALPHA088_close (带 rank) 并直接返回
    :param data 默认旋转后，即 data_rotated
    :param window 是一个二维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return ret 待确定
    '''

    ret = pd.DataFrame()
    data = data.loc[futures_universe].copy()
    open = data['open'].unstack(level=0)
    openw = data['openw'].unstack(level=0)

    def wma(ts, N):
        return ts.rolling(N).apply(lambda x: x[::-1].cumsum().sum() * 2 / N / (N + 1))

    for w in window:
        [N1, N2, N3, N4, N5, N6, N7] = w

        data['adv40'] = data.groupby('future', group_keys=False).volume.apply(
            lambda v: v.rolling(N3, min_periods=N3).mean())
        data['inner1'] = data.groupby('future', group_keys=False).apply(
            lambda df: wma(df.openw + df.loww - df.highw - df.closew, N1) / df.closew)
        rank1 = data['inner1'].unstack(level=0).rank(axis=1, pct=True)
        data['rank_adv'] = data['adv40'].groupby('future').apply(lambda df: df.rolling(N4).rank(pct=True))
        data['inner2_corr'] = data.groupby('future', group_keys=False).apply(
            lambda df: df.close.rolling(N2, min_periods=N2).rank(pct=True).rolling(N5, min_periods=N5).corr(
                df['rank_adv']))
        data['inner2'] = wma(data['inner2_corr'], N6).groupby('future').apply(
            lambda df: df.rolling(N7, min_periods=N7).rank(pct=True))
        rank2 = data['inner2'].unstack(level=0)
        _ret = (rank1 <= rank2) * rank1 + (rank1 > rank2) * rank2

        new_columns = pd.MultiIndex.from_tuples(
            [(col, f'wqalpha088_close{N1}_{N2}_{N3}_{N4}_{N5}_{N6}_{N7}') for col in _ret.columns],
            names=['future', 'price'])
        _ret.columns = new_columns
        ret = pd.concat([ret, _ret], axis=1)

    open.columns = pd.MultiIndex.from_product([open.columns, ["open"]], names=["future", "price"])
    openw.columns = pd.MultiIndex.from_product([openw.columns, ["openw"]], names=["future", "price"])
    ret = pd.concat([ret, open, openw], axis=1)

    return ret


# ret_test = WQALPHA088_close(data=data_rotated, window=wqalpha088_time_lst, futures_universe=gu24)
# ret_test
"""
WQALPHA086指标计算, 窗口
"""


# wqalpha086_time_lst = [
#     [10, 10, 10, 10],
#     [10, 10, 6, 15],
#     [20, 15, 6, 20],
#     [20, 10, 10, 15],
#     [5, 5, 20, 20],
#     [5, 10, 10, 5]
# ]

def WQALPHA086(data, window, futures_universe, time_freq='day'):
    '''
    计算 WQALPHA086 (带 rank) 并直接返回
    :param data 默认旋转后，即 data_rotated
    :param window 是一个二维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return 待确定
    '''
    ret = pd.DataFrame()
    data = data.loc[futures_universe].copy()
    open = data['open'].unstack(level=0)
    openw = data['openw'].unstack(level=0)

    for w in window:
        [N1, N2, N3, N4] = w
        data['adv20'] = data.groupby('future', group_keys=False).volume99.apply(
            lambda v: v.rolling(N1, min_periods=N1).mean())
        data['adv20'] = data.groupby('future', group_keys=False).apply(
            lambda v: v.adv20.rolling(N2, min_periods=N2).sum())
        data['inner1'] = data.groupby('future', group_keys=False).apply(
            lambda df: df.close.rolling(N3, min_periods=N3).corr(df.adv20))
        data['rank1'] = data['inner1'].rolling(N4, min_periods=N4).rank(pct=True)
        data['rank2'] = (data['close'] - data['settlementw']).unstack(level=0).rank(axis=1,
                                                                                    pct=True).stack().swaplevel()

        temp = (data['rank1'] - data['rank2']).unstack(level=0)
        new_columns = pd.MultiIndex.from_tuples([(col, f'wqalpha086{N1}_{N2}_{N3}_{N4}') for col in temp.columns],
                                                names=['future', 'price'])
        temp.columns = new_columns
        ret = pd.concat([ret, temp], axis=1)

    open.columns = pd.MultiIndex.from_product([open.columns, ["open"]], names=["future", "price"])
    openw.columns = pd.MultiIndex.from_product([openw.columns, ["openw"]], names=["future", "price"])
    ret = pd.concat([ret, open, openw], axis=1)

    return ret


# ret_test = WQALPHA086(data=data_rotated, window=wqalpha086_time_lst, futures_universe=gu24)
# ret_test
"""
WQAlpha083指标计算
"""


# wqalpha083_time_lst = [
#     [3,1],
#     [6,1],
#     [11, 1],
#     [18, 1],
#     [27, 1],
#     [38, 1]
# ]
def WQALPHA083(data, window, futures_universe, time_freq='day'):
    '''
    计算 WQALPHA083 并直接返回
    :param data 默认旋转后，即 data_rotated
    :param window 是一个二维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return ret
    '''

    ret = pd.DataFrame()

    data = data.loc[futures_universe].copy()

    open = data['open'].unstack(level=0)
    openw = data['openw'].unstack(level=0)

    for w in window:
        [N1, N2] = w
        data['inner'] = (data['highw'] - data['loww'] + 1e-6) / data['closew'].rolling(N1, min_periods=N1).mean()
        data['inner1'] = data['inner'].shift(N2) * data['volume99']
        data['inner2'] = data['inner'] / (data['settlementw'] - data['closew'] + 1e-6)

        output = (data['inner1'] / data['inner2']).unstack(level=0)

        new_columns = pd.MultiIndex.from_tuples([(col, f'wqalpha083{N1}_{N2}') for col in output.columns],
                                                names=['future', 'price'])
        output.columns = new_columns
        ret = pd.concat([ret, output], axis=1)

    open.columns = pd.MultiIndex.from_product([open.columns, ["open"]], names=["future", "price"])
    openw.columns = pd.MultiIndex.from_product([openw.columns, ["openw"]], names=["future", "price"])
    ret = pd.concat([ret, open, openw], axis=1)

    return ret


def WQALPHA080(data, window, futures_universe, time_freq='day'):
    '''
    计算 WQALPHA080 (带 rank) 并直接返回
    :param data 默认旋转后，即 data_rotated
    :param window 是一个二维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return ret 待确定
    '''

    ret = pd.DataFrame()
    data = data.loc[futures_universe].copy()
    open = data['open'].unstack(level=0)
    openw = data['openw'].unstack(level=0)

    for w in window:
        [N1, N2, N3, N4, N5] = w

        data['adv10'] = data.groupby('future', group_keys=False).volume99.apply(
            lambda v: v.rolling(N3, min_periods=N3).mean())

        data['inner1'] = ((data['openw'] * N1 + (1 - N1)) * data['highw'].groupby('future').diff(N2)) / data['closew']
        rank1 = data['inner1'].unstack(level=0).rank(axis=1, pct=True)
        data['inner2'] = (data['highw'].rolling(N4, min_periods=N4).corr(data['adv10'])).groupby('future').apply(
            lambda df: df.rolling(N5, min_periods=N5).rank(pct=True))
        rank2 = data['inner2'].unstack(level=0)
        _ret = np.power(rank1, rank2)

        new_columns = pd.MultiIndex.from_tuples([(col, f'wqalpha080{N1}_{N2}_{N3}_{N4}_{N5}') for col in _ret.columns],
                                                names=['future', 'price'])
        _ret.columns = new_columns
        ret = pd.concat([ret, _ret], axis=1)

    open.columns = pd.MultiIndex.from_product([open.columns, ["open"]], names=["future", "price"])
    openw.columns = pd.MultiIndex.from_product([openw.columns, ["openw"]], names=["future", "price"])
    ret = pd.concat([ret, open, openw], axis=1)

    return ret


# TODO delay
def WQALPHA081(data, window, futures_universe, time_freq='day'):
    '''
    计算 WQALPHA081 (带 rank) 并直接返回
    :param data 默认旋转后，即 data_rotated
    :param window 是一个二维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return ret 待确定
    '''

    ret = pd.DataFrame()

    data = data.loc[futures_universe].copy()

    open = data['open'].unstack(level=0)
    openw = data['openw'].unstack(level=0)

    for w in window:
        [N1, N2, N3, N4, N5, N6] = w
        data['adv10'] = data.groupby('future', group_keys=False).volume99.apply(
            lambda v: v.rolling(N1, min_periods=N1).mean())
        data['adv10'] = data.groupby('future', group_keys=False).apply(lambda group: group.adv10.rolling(N2).sum())
        data['corr1'] = data.groupby('future', group_keys=False).apply(
            lambda df: df.settlementw.rolling(N3, min_periods=N3).corr(df['adv10']))
        data['inner1'] = data.groupby('future', group_keys=False).apply(
            lambda df: (df['corr1'] ** N4).rolling(N5, min_periods=N5).apply(np.prod))
        rank1 = data['inner1'].unstack(level=0).rank(axis=1, pct=True)
        data['inner2'] = data.groupby('future', group_keys=False).apply(
            lambda df: df.settlementw.rolling(N6, min_periods=N6).corr(df.volume99))
        rank2 = data['inner2'].unstack(level=0).rank(axis=1, pct=True)
        _ret = rank1 - rank2

        new_columns = pd.MultiIndex.from_tuples(
            [(col, f'wqalpha081{N1}_{N2}_{N3}_{N4}_{N5}_{N6}') for col in _ret.columns], names=['future', 'price'])
        _ret.columns = new_columns
        ret = pd.concat([ret, _ret], axis=1)

    open.columns = pd.MultiIndex.from_product([open.columns, ["open"]], names=["future", "price"])
    openw.columns = pd.MultiIndex.from_product([openw.columns, ["openw"]], names=["future", "price"])
    ret = pd.concat([ret, open, openw], axis=1)

    return ret


def WQALPHA074(data, window, futures_universe, time_freq='day'):
    '''
    计算 WQALPHA074 (带 rank) 并直接返回
    :param data 默认旋转后，即 data_rotated
    :param window 是一个二维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return ret 待确定
    '''
    ret = pd.DataFrame()
    data = data.loc[futures_universe].copy()
    open = data['open'].unstack(level=0)
    openw = data['openw'].unstack(level=0)

    for w in window:
        [N1, N2, N3, N4, N5] = w
        data['adv30'] = data.groupby('future', group_keys=False).volume99.apply(
            lambda v: v.rolling(N1, min_periods=N1).mean())
        data['adv30'] = data.groupby('future', group_keys=False).apply(lambda group: group.adv30.rolling(N2).sum())
        data['inner1'] = data.groupby('future', group_keys=False).apply(
            lambda df: df.closew.rolling(N3, min_periods=N3).corr(df['adv30']))
        rank1 = data['inner1'].unstack(level=0).rank(axis=1, pct=True)
        data['inner2'] = data.groupby('future', group_keys=False).apply(
            lambda df: (df.highw * N4 + df.settlementw * (1 - N4)).rolling(N5, min_periods=N5).corr(df.volume99))
        rank2 = data['inner2'].unstack(level=0).rank(axis=1, pct=True)
        _ret = rank1 - rank2

        new_columns = pd.MultiIndex.from_tuples([(col, f'wqalpha074{N1}_{N2}_{N3}_{N4}_{N5}') for col in _ret.columns],
                                                names=['future', 'price'])
        _ret.columns = new_columns
        ret = pd.concat([ret, _ret], axis=1)

    open.columns = pd.MultiIndex.from_product([open.columns, ["open"]], names=["future", "price"])
    openw.columns = pd.MultiIndex.from_product([openw.columns, ["openw"]], names=["future", "price"])
    ret = pd.concat([ret, open, openw], axis=1)

    return ret


def WQALPHA079(data, window, futures_universe, time_freq='day'):
    '''
    计算 WQALPHA079 (带 rank) 并直接返回
    :param data 默认旋转后，即 data_rotated
    :param window 是一个二维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return ret 待确定
    '''
    ret = pd.DataFrame()
    data = data.loc[futures_universe].copy()
    open = data['open'].unstack(level=0)
    openw = data['openw'].unstack(level=0)

    for w in window:
        [N1, N2, N3, N4, N5, N6] = w
        data['adv60'] = data.groupby('future', group_keys=False).volume99.apply(
            lambda v: v.rolling(N4, min_periods=N4).mean())

        data['inner1'] = data.groupby('future').apply(
            lambda group: (group['closew'] * N1 + group['openw'] * (1 - N1)).diff(N2)).reset_index(level=0, drop=True)
        rank1 = data['inner1'].unstack(level=0).rank(axis=1, pct=True)
        data['ranked_setmt'] = data.groupby('future', group_keys=False).settlementw.apply(
            lambda df: df.rolling(N3, min_periods=N3).rank(pct=True))
        data['ranked_adv60'] = data.groupby('future', group_keys=False).apply(
            lambda group: group.adv60.rolling(N5).rank(pct=True))
        data['inner2'] = data.groupby('future', group_keys=False).apply(
            lambda df: df.ranked_setmt.rolling(N6, min_periods=N6).corr(df['ranked_adv60']))
        rank2 = data['inner2'].unstack(level=0).rank(axis=1, pct=True)

        _ret = rank1 - rank2
        new_columns = pd.MultiIndex.from_tuples(
            [(col, f'wqalpha079{N1}_{N2}_{N3}_{N4}_{N5}_{N6}') for col in _ret.columns], names=['future', 'price'])
        _ret.columns = new_columns
        ret = pd.concat([ret, _ret], axis=1)

    open.columns = pd.MultiIndex.from_product([open.columns, ["open"]], names=["future", "price"])
    openw.columns = pd.MultiIndex.from_product([openw.columns, ["openw"]], names=["future", "price"])
    ret = pd.concat([ret, open, openw], axis=1)

    return ret


def WQALPHA071(data, window, futures_universe, time_freq='day'):
    '''
    计算 WQALPHA071 (带 rank) 并直接返回
    :param data 默认旋转后，即 data_rotated
    :param window 是一个二维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return ret 待确定
    '''

    def wma(ts, N):
        return ts.rolling(N).apply(lambda x: x[::-1].cumsum().sum() * 2 / N / (N + 1))

    ret = pd.DataFrame()
    data = data.loc[futures_universe].copy()

    open = data['open'].unstack(level=0)
    openw = data['openw'].unstack(level=0)

    for w in window:
        [N1, N3, N5] = w
        N2 = 60
        N7 = N6 = N4 = N3
        N8 = N5

        data['adv60'] = data.groupby('future', group_keys=False).volume99.apply(
            lambda v: v.rolling(N2, min_periods=N2).mean())

        data['ranked_close'] = data.groupby('future', group_keys=False).closew.apply(
            lambda df: df.rolling(N1, min_periods=N1).rank(pct=True))
        data['ranked_adv60'] = data.groupby('future', group_keys=False).apply(
            lambda group: group.adv60.rolling(N3).rank(pct=True))
        data['corr'] = data.groupby('future', group_keys=False).apply(
            lambda df: df.ranked_close.rolling(N4, min_periods=N4).corr(df['ranked_adv60']))
        data['inner1'] = data.groupby('future', group_keys=False).apply(lambda df: wma(df['corr'], N5))
        inner1 = data.groupby('future', group_keys=False).apply(lambda group: group.inner1.rolling(N6).rank(pct=True))
        rank1 = inner1.unstack(level=0)

        data['inner2'] = data.groupby('future', group_keys=False).apply(
            lambda df: (wma(df.loww + df.openw - 2 * df.settlementw, N7)))
        inner2 = data.groupby('future', group_keys=False).apply(lambda group: group.inner2.rolling(N8).rank(pct=True))
        rank2 = inner2.unstack(level=0)

        _ret = (rank1 >= rank2) * rank1 + (rank1 < rank2) * rank2
        new_columns = pd.MultiIndex.from_tuples([(col, f'wqalpha071{N1}_{N3}_{N5}') for col in _ret.columns],
                                                names=['future', 'price'])
        _ret.columns = new_columns
        ret = pd.concat([ret, _ret], axis=1)

    open.columns = pd.MultiIndex.from_product([open.columns, ["open"]], names=["future", "price"])
    openw.columns = pd.MultiIndex.from_product([openw.columns, ["openw"]], names=["future", "price"])
    ret = pd.concat([ret, open, openw], axis=1)

    return ret


def WQALPHA071_t(data, window, futures_universe, time_freq='day'):
    '''
    计算 WQALPHA071 (带 rank) 并直接返回
    :param data 默认旋转后，即 data_rotated
    :param window 是一个二维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return ret 待确定
    '''

    def wma(ts, N):
        return ts.rolling(N).apply(lambda x: x[::-1].cumsum().sum() * 2 / N / (N + 1))

    ret = pd.DataFrame()
    data = data.loc[futures_universe].copy()
    open = data['open'].unstack(level=0)
    openw = data['openw'].unstack(level=0)

    for w in window:
        [N1, N3, N5] = w
        N2 = 60
        N7 = N6 = N4 = N3
        N8 = N5

        data['adv60'] = data.groupby('future', group_keys=False).volume99.apply(
            lambda v: v.rolling(N2, min_periods=N2).mean())

        data['rank_close'] = data.groupby('future', group_keys=False).closew.apply(
            lambda df: df.rolling(N1, min_periods=N1).rank(pct=True))
        data['rank_adv60'] = data.groupby('future', group_keys=False).apply(
            lambda group: group.groupby('future').adv60.rolling(N3, min_periods=N3 - 5).rank(pct=True)).reset_index(
            level=0, drop=True)
        data['corr'] = data.groupby('future', group_keys=False).apply(
            lambda df: df.rank_close.rolling(N4, min_periods=N4).corr(df['rank_adv60']))
        inner1 = data.groupby('future', group_keys=False).apply(
            lambda df: wma(df['corr'], N5).rolling(N6, min_periods=N6).rank(pct=True))
        rank1 = inner1.unstack(level=0)

        data['inner2'] = data.groupby('future', group_keys=False).apply(
            lambda df: wma(df.loww + df.openw - 2 * df.settlementw, N7))
        inner2 = data.groupby('future', group_keys=False).apply(lambda group: group.inner2.rolling(N8).rank(pct=True))
        rank2 = inner2.unstack(level=0).rolling(N8, min_periods=N8).rank(pct=True)

        _ret = rank1 - rank2
        new_columns = pd.MultiIndex.from_tuples([(col, f'wqalpha071_t{N1}_{N3}_{N5}') for col in _ret.columns],
                                                names=['future', 'price'])
        _ret.columns = new_columns
        ret = pd.concat([ret, _ret], axis=1)

    open.columns = pd.MultiIndex.from_product([open.columns, ["open"]], names=["future", "price"])
    openw.columns = pd.MultiIndex.from_product([openw.columns, ["openw"]], names=["future", "price"])
    ret = pd.concat([ret, open, openw], axis=1)

    return ret


def WQALPHA064(data, window, futures_universe, time_freq='day'):
    '''
    计算 WQALPHA064 (带 rank) 并直接返回
    :param data 默认旋转后, 即 data_rotated
    :param window 是一个二维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return 待确定
    '''
    ret = pd.DataFrame()
    data = data.loc[futures_universe].copy()
    open_price = data['open'].unstack(level=0)
    openw = data['openw'].unstack(level=0)

    for w in window:
        [N1, N2, N3, N4, N5] = w
        data['adv30'] = data.groupby('future', group_keys=False).volume99.apply(
            lambda v: v.rolling(N3, min_periods=N3).mean())

        data['inner1'] = data.groupby('future', group_keys=False).apply(
            lambda group: (group['openw'] * N1 + group['loww'] * (1 - N1)).rolling(N2, min_periods=N2).mean())
        data['ranked_adv30'] = data.groupby('future', group_keys=False).apply(
            lambda group: group.adv30.rolling(N2).sum())
        inner1 = data.groupby('future', group_keys=False).apply(
            lambda group: group['ranked_adv30'].rolling(N4, min_periods=N4).corr(group.adv30))
        rank1 = inner1.unstack(level=0).rank(axis=1, pct=True)

        data['inner2'] = data.groupby('future', group_keys=False).apply(
            lambda group: ((group['highw'] + group['loww']) / 2 * N1 + group['settlementw'] * (1 - N1)).diff(N5))
        rank2 = data['inner2'].unstack(level=0).rank(axis=1, pct=True)

        _ret = rank1 - rank2
        new_columns = pd.MultiIndex.from_tuples([(col, f'wqalpha064{N1}_{N2}_{N3}_{N4}_{N5}') for col in _ret.columns],
                                                names=['future', 'price'])
        _ret.columns = new_columns
        pd.concat([ret, _ret], axis=1)

    open_price.columns = pd.MultiIndex.from_product([open_price.columns, ["open"]], names=["future", "price"])
    openw.columns = pd.MultiIndex.from_product([openw.columns, ["openw"]], names=["future", "price"])
    ret = pd.concat([ret, open_price, openw], axis=1)

    return ret


"""
WQALPHA095指标计算, 窗口
"""


# wqalpha095_time_lst = [
#     [6, 8, 15, 8, 8],
#     [10, 15, 30, 10, 10],
#     [12, 19, 40, 13, 12],
#     [15, 15, 30, 20, 20],
#     [20, 25, 30, 20, 20],
#     [30, 30, 40, 30, 30]
# ]

def WQALPHA095(data, window, futures_universe, time_freq='day'):
    '''
    计算 WQALPHA095 (带 rank) 并直接返回
    :param data 默认旋转后，即 data_rotated
    :param window 是一个二维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return ret
    '''

    ret = pd.DataFrame()
    data = data.loc[futures_universe].copy()
    open_price = data['open'].unstack(level=0)
    openw = data['openw'].unstack(level=0)

    for w in window:
        [N1, N2, N3, N4, N5] = w

        data['adv40'] = data.groupby('future', group_keys=False).volume99.apply(
            lambda v: v.rolling(N3, min_periods=N3).mean())

        data['inner1'] = data.groupby('future', group_keys=False).openw.apply(
            lambda v: v.rolling(N1, min_periods=N1).min() / v * (-1))
        rank1 = data['inner1'].unstack(level=0).rank(axis=1, pct=True)

        data['high_low'] = data.groupby('future', group_keys=False).apply(
            lambda df: (df.highw + df.loww).rolling(N2, min_periods=N2).sum())
        data['rolled_adv40'] = data.groupby('future', group_keys=False).apply(
            lambda df: df.adv40.rolling(N2, min_periods=N2).sum())
        data['corr'] = data.groupby('future', group_keys=False).apply(
            lambda df: df.high_low.rolling(N4, min_periods=N4).corr(df.rolled_adv40))
        inner2 = data.groupby('future', group_keys=False).apply(
            lambda df: df['corr'].rolling(N5, min_periods=N5).rank(pct=True))
        rank2 = data['corr'].unstack(level=0)

        _ret = rank1 - rank2
        new_columns = pd.MultiIndex.from_tuples([(col, f'wqalpha095{N1}_{N2}_{N3}_{N4}_{N5}') for col in _ret.columns],
                                                names=['future', 'price'])
        _ret.columns = new_columns
        ret = pd.concat([ret, _ret], axis=1)

    open_price.columns = pd.MultiIndex.from_product([open_price.columns, ["open"]], names=["future", "price"])
    openw.columns = pd.MultiIndex.from_product([openw.columns, ["openw"]], names=["future", "price"])
    ret = pd.concat([ret, open_price, openw], axis=1)

    return ret


# ret_test = WQALPHA095(data=data_rotated, window=wqalpha095_time_lst, futures_universe=gu24)
# ret_test
"""
WQALPHA094指标计算, shift
"""


def WQALPHA094_shift(data, window, futures_universe, time_freq='day'):
    '''
    计算 WQALPHA094 (带 rank) 并直接返回
    :param data 默认旋转后，即 data_rotated
    :param window 是一个二维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return ret
    '''

    ret = pd.DataFrame()
    data = data.loc[futures_universe].copy()
    open_price = data['open'].unstack(level=0)
    openw = data['openw'].unstack(level=0)
    for w in window:
        [N1, N2, N3, N4, N5, N6] = w

        data['adv60'] = data.groupby('future', group_keys=False).volume99.apply(
            lambda v: v.rolling(N3, min_periods=N3).mean())

        data['inner1'] = data.groupby('future', group_keys=False).settlementw.apply(
            lambda v: v.rolling(N1, min_periods=N1).min() / v * (-1))
        rank1 = data['inner1'].unstack(level=0).rank(axis=1, pct=True)

        data['rank_settlementw'] = data.groupby('future', group_keys=False).settlementw.apply(
            lambda v: v.rolling(N2, min_periods=N2).rank(pct=True))
        data['rank_adv60'] = data.groupby('future', group_keys=False).apply(
            lambda v: v.adv60.rolling(N4, min_periods=N4).rank(pct=True))
        data['inner2'] = data.groupby('future', group_keys=False).apply(
            lambda v: v.rank_settlementw.rolling(N5, min_periods=N5).corr(v['rank_adv60']))
        inner2 = data.groupby('future', group_keys=False).apply(
            lambda v: v.inner2.rolling(N6, min_periods=N6).rank(pct=True))
        rank2 = inner2.unstack(level=0)

        _ret = np.power(rank1, rank2)
        new_columns = pd.MultiIndex.from_tuples(
            [(col, f'wqalpha094_shift{N1}_{N2}_{N3}_{N4}_{N5}_{N6}') for col in _ret.columns],
            names=['future', 'price'])
        _ret.columns = new_columns
        ret = pd.concat([ret, _ret], axis=1)

    open_price.columns = pd.MultiIndex.from_product([open_price.columns, ["open"]], names=["future", "price"])
    openw.columns = pd.MultiIndex.from_product([openw.columns, ["openw"]], names=["future", "price"])
    ret = pd.concat([ret, open_price, openw], axis=1)

    return ret


def WQALPHA095_shift(data, window, futures_universe, time_freq='day'):
    '''
    计算 WQALPHA095_shift (带 rank) 并直接返回
    :param data 默认旋转后，即 data_rotated
    :param window 是一个二维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return ret
    '''

    ret = pd.DataFrame()
    data = data.loc[futures_universe].copy()
    open_price = data['open'].unstack(level=0)
    openw = data['openw'].unstack(level=0)

    def wma(ts, N):
        return ts.rolling(N).apply(lambda x: x[::-1].cumsum().sum() * 2 / N / (N + 1))

    for w in window:
        [N1, N2, N3, N4, N5] = w
        data['adv40'] = data.groupby('future', group_keys=False).volume99.apply(
            lambda v: v.rolling(N3, min_periods=N3).mean())

        data['inner1'] = data.groupby('future', group_keys=False).openw.apply(
            lambda v: v.rolling(N1, min_periods=N1).min() / v * (-1))
        rank1 = data['inner1'].unstack(level=0).rank(pct=True)

        data['high_low'] = data.groupby('future', group_keys=False).apply(
            lambda df: (df.highw + df.low).rolling(N2, min_periods=N2).sum())
        data['ranked_adv40'] = data.groupby('future', group_keys=False).apply(
            lambda df: df.adv40.rolling(N2, min_periods=N2).sum())
        data['corr'] = data.groupby('future', group_keys=False).apply(
            lambda df: df.high_low.rolling(N4, min_periods=N4).corr(df.ranked_adv40))
        inner2 = data.groupby('future', group_keys=False).apply(
            lambda df: df['corr'].rolling(N5, min_periods=N5).rank(pct=True))
        rank2 = inner2.unstack(level=0)

        _ret = rank1 - rank2
        new_columns = pd.MultiIndex.from_tuples(
            [(col, f'wqalpha095_shift{N1}_{N2}_{N3}_{N4}_{N5}') for col in _ret.columns], names=['future', 'price'])
        _ret.columns = new_columns
        ret = pd.concat([ret, _ret], axis=1)

    open_price.columns = pd.MultiIndex.from_product([open_price.columns, ["open"]], names=["future", "price"])
    openw.columns = pd.MultiIndex.from_product([openw.columns, ["openw"]], names=["future", "price"])
    ret = pd.concat([ret, open_price, openw], axis=1)

    return ret


# 以下为国泰君安 191

gtja001_time_lst = [3, 4, 6, 10, 15, 20]

def GTJA001(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA001 
    corr(rank(delta(log(volume), 1)), rank((close - open) / open), 6)

    :param data 默认旋转后，即 data_rotated
    :param window 是一个一维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return ret
    '''

    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]
    low = data.low[futures_universe]

    ret = pd.DataFrame()
    for w in window:
        N = w

        _data = data.loc[futures_universe]

        # _data['inner1'] = np.log(_data.volume).diff(1)
        _data['inner1'] = np.log(_data.volume99).unstack().diff(1,axis = 1).stack()
        _data['rank1'] = _data.inner1.unstack().rank(pct=True).stack()

        _data['inner2'] = (_data.closew - _data.openw) / _data.openw
        _data['rank2'] = _data.inner2.unstack().rank(pct=True).stack()

        _ret = _data.rank1.rolling(N).corr(_data.rank2)
        _ret = _ret.unstack().T
        _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja001" + str(w)]], names=["futures", "price"])
        ret = pd.concat([ret, _ret], axis=1)

    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret

gtja002_time_lst = [1, 2, 5, 8, 12, 15]

def GTJA002(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA002
    delta(((close - low) - (high - close)) / (high - low), 1)

    :param data 默认旋转后，即 data_rotated
    :param window 是一个一维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return ret
    '''

    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]
    low = data.low[futures_universe]

    ret = pd.DataFrame()
    # FACTOR_PATH = './factor_daily/GTJA002.csv'
    # if not os.path.exists(FACTOR_PATH):
    #     _data = data.loc[futures_universe]
    #     _data['factor'] = ((_data.closew - _data.loww) - (_data.highw - _data.closew)) / (_data.highw - _data.loww)
    #     facotr_df = _data['factor'].reset_index()
    #     facotr_df.to_csv(FACTOR_PATH,index=False)
        
    for w in window:
        N = w

        _data = data.loc[futures_universe]

        _data['inner1'] = ((_data.closew - _data.loww) - (_data.highw - _data.closew)) / (_data.highw - _data.loww)
        _data['inner'] = _data['inner1'].unstack().diff(N,axis = 1).stack()

        _ret = _data.inner
        # _ret = _data.inner.diff(N)
        _ret = _ret.unstack().T
        _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja002" + str(w)]], names=["futures", "price"])
        ret = pd.concat([ret, _ret], axis=1)


    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret

gtja003_time_lst = [2, 3, 6, 10, 15, 20]

def GTJA003(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA003
    sum(close == delay(close, 1) ? 0 : close - (close > delay(close, 1) ? min(low, delay(close, 1)) : max(high, delay(close, 1))), 6)

    :param data 默认旋转后，即 data_rotated
    :param window 是一个一维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return ret
    '''

    # 
    # volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]
    low = data.low[futures_universe]

    ret = pd.DataFrame()
    for w in window:
        N = w

        _data = data.loc[futures_universe]

        _data['close_1'] = _data.closew.shift(1)
        _data['close_1'] = _data.closew.unstack().shift(1).stack()
        _data['inner'] = np.nan
        _data.loc[_data.closew == _data.close_1, 'inner'] = 0
        _data.loc[_data.closew > _data.close_1, 'inner'] = _data[['loww', 'close_1']].min(axis=1)
        _data.loc[_data.closew < _data.close_1, 'inner'] = _data[['highw', 'close_1']].max(axis=1)
        _data['inner'] = _data['inner'].unstack().rolling(N,axis = 1).sum().stack()

        _ret = _data.inner
        # _ret = _data.inner.rolling(N).sum()
        _ret = _ret.unstack().T
        _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja003" + str(w)]], names=["futures", "price"])
        ret = pd.concat([ret, _ret], axis=1)

    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret

gtja004_time_lst = [
    [1, 2, 8],
    [2, 3, 10],
    [2, 8, 15],
    [4, 12, 20],
    [5, 15, 25],
    [6, 20, 30]
]

def GTJA004(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA004
    mean(close, 8) + std(close, 8) < mean(close, 2) ? -1 : mean(close, 2) < mean(close, 8) - std(close, 8) ? 1 : (1 <= volume / mean(volume, 20)) ? 1 : -1

    :param data 默认旋转后，即 data_rotated
    :param window 是一个一维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return ret
    '''

    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]
    low = data.low[futures_universe]

    ret = pd.DataFrame()
    for w in window:
        N1, N2, N3 = w

        _data = data.loc[futures_universe]

        _data['g004'] = np.nan
        _data[_data.closew.rolling(N2).mean() + _data.closew.rolling(N2).std() < _data.closew.rolling(N1).mean()] = -1
        _data[_data.closew.rolling(N2).mean() - _data.closew.rolling(N2).std() > _data.closew.rolling(N1).mean()] = 1
        _data[(_data.g004 == np.nan) & (_data.volume99 >= _data.volume99.rolling(N3).mean())] = 1
        _data[(_data.g004 == np.nan) & (_data.volume99 < _data.volume99.rolling(N3).mean())] = -1
        
        _ret = _data.g004
        _ret = _ret.unstack().T
        _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja004" + str(w)]], names=["futures", "price"])
        ret = pd.concat([ret, _ret], axis=1)


    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret

# =============================================================================
# gtja005_time_list = [
#     [3, 2],
#     [4, 3],
#     [5, 3],
#     [7, 5],
#     [10, 6],
#     [15, 12]
# ]
# 
# =============================================================================

gtja005_time_lst = [
    [5, 3],
    [7, 5],
    [10, 7],
    [14, 9],
    [19, 11],
    [25, 13]
]

def GTJA005(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA005
    tsmax(corr(tsrank(volume, 5), tsrank(high, 5), 5), 3)

    :param data 默认旋转后，即 data_rotated
    :param window 是一个一维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return ret
    '''

    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]
    low = data.low[futures_universe]

    def tsrank(array):
        s = pd.Series(array)
        return np.array(s.rank(method="min", ascending=False))[-1]

    ret = pd.DataFrame()
    for w in window:
        N1, N2 = w

        _data = data.loc[futures_universe]

        # _data['inner1'] = _data.volume.unstack().rolling(N1,axis = 1).mean().stack()
        # _data['inner1'] = _data.volume.unstack().rolling(N1,axis = 1).apply(tsrank).stack()
        _data['inner1'] = _data.volume99.rolling(N1).apply(tsrank)
        # _data['inner2'] = _data.highw.unstack().rolling(N1,axis = 1).apply(tsrank).stack()
        _data['inner2'] = _data.highw.rolling(N1).apply(tsrank)
        # _data['inner'] = _data.inner1.unstack().rolling(N1,axis=1).corr(_data.inner2).stack()
        _data['inner'] = _data.inner1.rolling(N1).corr(_data.inner2)
        # _data['g005'] = _data.inner.unstack().rolling(N2,axis = 1).max().stack()
        _data['g005'] = _data.inner.rolling(N2).max()
        # print('_datainner1')
        # print(_data['inner1'])
# =============================================================================
#         print('_datainner2')
#         print(_data['inner2'])
#         print('_datainner)
#         print(_data['inner'])
# =============================================================================
        
        # _ret = _data.inner1.unstack().T
        _ret = _data.g005.unstack().T
        _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja005" + str(w)]], names=["futures", "price"])
        ret = pd.concat([ret, _ret], axis=1)

    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret

gtja006edit_time_lst = [
    [.15, 4],
    [.5, 4],
    [.85, 4],
    [.7, 10],
    [.3, 10],
    [.5, 20]
]

def GTJA006EDIT(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA006edit
    -1 * delta(open * .85 + high * .15, 4)

    :param data 默认旋转后，即 data_rotated
    :param window 是一个一维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return ret
    '''

    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]
    low = data.low[futures_universe]

    ret = pd.DataFrame()
# =============================================================================
#     for w in window:
#         N1, N2 = w
# 
#         _data = data.loc[futures_universe]
# 
#         _data['g006'] = -1 * (_data.openw * N1 + _data.highw * (1 - N1)).diff(N2)
# 
#         _ret = _data.g006.unstack().T
#         _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja006" + str(w)]], names=["futures", "price"])
#         ret = pd.concat([ret, _ret], axis=1)
# =============================================================================

    for w in window:
        for name in futures_universe:
            N1, N2 = w
    
            _data = data.loc[name]
    
            _data['g006'] = -1 * (_data.openw * N1 + _data.highw * (1 - N1)).diff(N2)
    
            new_columns = pd.MultiIndex.from_product([[str(name)], ["gtja006edit" + str(w)]])  
            temp = _data['g006'].to_frame(name='g006').set_index(_data.index)  
            temp.columns = new_columns  
            ret = pd.concat([ret, temp], axis=1) 


    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret


gtja007_time_lst = [1, 2, 3, 6, 10, 15]

def GTJA007(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA007
    (rank(max(vwap - close, 3)) + rank(min(vwap - close, 3))) * rank(delta(volume, 3))

    :param data 默认旋转后，即 data_rotated
    :param window 是一个一维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return ret
    '''

    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]
    low = data.low[futures_universe]

    ret = pd.DataFrame()
# =============================================================================
#     for w in window:
#         N = w
# 
#         _data = data.loc[futures_universe]
#         _data['vwapw'] = (_data.openw + _data.closew + _data.highw + _data.loww) / 4.
# 
#         _data['const'] = 3.
#         _data['inner_price'] = _data.vwapw - _data.closew
#         _data['inner1'] = _data[['inner_price', 'const']].max(axis=1)
#         _data['inner2'] = _data[['inner_price', 'const']].min(axis=1)
#         _data['inner3'] = _data.volume.diff(N)
# 
#         _data['rank1'] = _data.inner1.unstack().rank(pct=True).stack()
#         _data['rank2'] = _data.inner2.unstack().rank(pct=True).stack()
#         _data['rank3'] = _data.inner3.unstack().rank(pct=True).stack()
# 
#         _data['g007'] = (_data.rank1 + _data.rank2) * _data.rank3
# 
#         _ret = _data.g007.unstack().T
#         _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja007" + str(w)]], names=["futures", "price"])
#         ret = pd.concat([ret, _ret], axis=1)
# =============================================================================
        
    for w in window:
        N = w

        _data = data.loc[futures_universe]
        _data['vwapw'] = (_data.openw + _data.closew + _data.highw + _data.loww) / 4.

        _data['const'] = 3.
        _data['inner_price'] = _data.vwapw - _data.closew
        _data['inner1'] = _data[['inner_price', 'const']].max(axis=1)
        _data['inner2'] = _data[['inner_price', 'const']].min(axis=1)
        # _data['inner3'] = _data.volume.diff(N)
        _data['inner3'] = _data.volume99.unstack().diff(N,axis=1).stack()

        _data['rank1'] = _data.inner1.unstack().rank(pct=True).stack()
        _data['rank2'] = _data.inner2.unstack().rank(pct=True).stack()
        _data['rank3'] = _data.inner3.unstack().rank(pct=True).stack()

        _data['g007'] = (_data.rank1 + _data.rank2) * _data.rank3

        _ret = _data.g007.unstack().T
        _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja007" + str(w)]], names=["futures", "price"])
        ret = pd.concat([ret, _ret], axis=1)


    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret

gtja007edit_time_lst = [1, 2, 3, 6, 10, 15]

def GTJA007EDIT(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA007edit
    把 max, min 换成 tsmax, tsmin
    (rank(tsmax(vwap - close, 3)) + rank(tsmin(vwap - close, 3))) * rank(delta(volume, 3))

    :param data 默认旋转后，即 data_rotated
    :param window 是一个一维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return ret
    '''

    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]
    low = data.low[futures_universe]

    ret = pd.DataFrame()
# =============================================================================
#     for w in window:
#         N = w
# 
#         _data = data.loc[futures_universe]
#         _data['vwapw'] = (_data.openw + _data.closew + _data.highw + _data.loww) / 4.
# 
#         _data['const'] = 3.
#         _data['inner_price'] = _data.vwapw - _data.closew
#         _data['inner1'] = _data.inner_price.rolling(N).max()
#         _data['inner2'] = _data.inner_price.rolling(N).min()
#         _data['inner3'] = _data.volume.diff(N)
# 
#         _data['rank1'] = _data.inner1.unstack().rank(pct=True).stack()
#         _data['rank2'] = _data.inner2.unstack().rank(pct=True).stack()
#         _data['rank3'] = _data.inner3.unstack().rank(pct=True).stack()
# 
#         _data['g007'] = (_data.rank1 + _data.rank2) * _data.rank3
# 
#         _ret = _data.g007.unstack().T
#         _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja007" + str(w)]], names=["futures", "price"])
#         ret = pd.concat([ret, _ret], axis=1)
# =============================================================================
        
    for w in window:
        N = w

        _data = data.loc[futures_universe]
        _data['vwapw'] = (_data.openw + _data.closew + _data.highw + _data.loww) / 4.

        _data['const'] = 3.
        _data['inner_price'] = _data.vwapw - _data.closew
        _data['inner1'] = _data.inner_price.rolling(N).max()
        _data['inner2'] = _data.inner_price.rolling(N).min()
        _data['inner3'] = _data.volume99.unstack().diff(N,axis=1).stack()

        _data['rank1'] = _data.inner1.unstack().rank(pct=True).stack()
        _data['rank2'] = _data.inner2.unstack().rank(pct=True).stack()
        _data['rank3'] = _data.inner3.unstack().rank(pct=True).stack()

        _data['g007'] = (_data.rank1 + _data.rank2) * _data.rank3

        _ret = _data.g007.unstack().T
        _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja007edit" + str(w)]], names=["futures", "price"])
        ret = pd.concat([ret, _ret], axis=1)

    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret

gtja008_time_lst = [
    [.5, 2],
    [.8, 4],
    [.2, 4],
    [.2, 10],
    [.2, 15],
    [.5, 15]
]



# =============================================================================
# def GTJA008(data, n, save_path=None):
#     if type(n) == list:
#         for a in n:
#             data=cal_gtja008(data, a, save_path=save_path)
#     elif type(n) == int:
#         data=cal_gtja008(data, n, save_path=save_path)
#     return data
#         
# def cal_gtja008(data, n, save_path):
#     """
#     rank(delta((high + low) / 2 * .2 + vwap * .8, 4))
#     """
#     # for (future, _) in data.columns:
#         # if (future, 'wqa006_%s'%(n)) in data.columns:
#             # print('WQAlpha006 value are already prepared with period {}!'.format(n))
#             # return
#         # break
#     
#     for (future, _) in data.columns:
#         if (future, 'gtja008%s'%(n)) in data.columns:
#             continue
#         try:
#             data.loc[:,(future, 'wqa006_%s'%(n))] = -1 * data[future].openw.rolling(n).corr(data[future].volume99)
#         except:
#             import pdb; pdb.set_trace()
# 
#         
#     if save_path == None:
#         save_path = input('New WQAlpha006 are calculated.\n YOU MUST INPUT DATA SAVE PATH FOR THEM! :')
#     import pickle
#     f = open(save_path, 'wb')
#     pickle.dump(data, f)
#     f.close()
#     
#     print('WQAlpha006 value are calculated and saved successfully!')
#     return data
# =============================================================================

def GTJA008(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA008
    rank(delta((high + low) / 2 * .2 + vwap * .8, 4))

    :param data 默认旋转后，即 data_rotated
    :param window 是一个一维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return ret
    '''

    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]
    low = data.low[futures_universe]

    ret = pd.DataFrame()
# =============================================================================
#     for w in window:
#         N1, N2 = w
# 
#         _data = data.loc[futures_universe]
#         _data['vwapw'] = (_data.openw + _data.closew + _data.highw + _data.loww) / 4.
# 
#         print(_data['vwapw'])
#         _data['g008'] = ((_data.highw + _data.loww) / 2 * N1 + _data.vwapw * (1 - N1)).diff(N2)
# 
#         _ret = _data.g008.unstack().T
#         _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja008" + str(w)]], names=["futures", "price"])
#         ret = pd.concat([ret, _ret], axis=1)
# =============================================================================
    for w in window:
        N1, N2 = w

        _data = data.loc[futures_universe]
        _data['vwapw'] = (_data.openw + _data.closew + _data.highw + _data.loww) / 4.

        _data['g008'] = ((_data.highw + _data.loww) / 2 * N1 + _data.vwapw * (1 - N1)).unstack().diff(N2,axis=1).stack()

        _ret = _data.g008.unstack().T
        _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja008" + str(w)]], names=["futures", "price"])
        ret = pd.concat([ret, _ret], axis=1)

# =============================================================================
#     for w in window:
#         for name in futures_universe:
#             N1, N2 = w
#     
#             _data = data.loc[name]
#             _data['vwapw'] = (_data.openw + _data.closew + _data.highw + _data.loww) / 4.
#             _data['g008'] = ((_data.highw + _data.loww) / 2 * N1 + _data.vwapw * (1 - N1)).diff(N2)
#             
#             new_columns = pd.MultiIndex.from_product([[str(name)], ["gtja008" + str(w)]])  
#             temp = _data['g008'].to_frame(name='g008').set_index(_data.index)  
#             temp.columns = new_columns  
#             ret = pd.concat([ret, temp], axis=1)            
# =============================================================================
            
    
# =============================================================================
#             _ret = _data.g008.unstack().T
#             _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja008" + str(w)]], names=["futures", "price"])
#             ret = pd.concat([ret, _ret], axis=1)
# =============================================================================


    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret

gtja009_time_lst = [1. / 7, 1.5 / 7, 2. / 7, 3. / 7, 4. / 7, 5. / 7]

def GTJA009(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA009
    SMA(((high + low) / 2 - (delay(high, 1) + delay(low, 1)) / 2) * (high - low) / volume, 7, 2)
    其中 SMA(A, n, m) 表示 Yi+1 = (Ai * m + Yi * (n - m)) / n 

    :param data 默认旋转后，即 data_rotated
    :param window 是一个一维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return ret
    '''

    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]
    low = data.low[futures_universe]

    ret = pd.DataFrame()
# =============================================================================
#     for w in window:
#         sma_alpha = w
# 
#         _data = data.loc[futures_universe]
#         _data['inner'] = (_data.highw + _data.loww - _data.highw.shift(1) - _data.loww.shift(1)) / 2 * (_data.highw - _data.loww) / _data.volume
# 
#         _data['g009'] = _data.inner.ewm(adjust=False, alpha=sma_alpha).mean()
# 
#         _ret = _data.g009.unstack().T
#         _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja009" + str(w)]], names=["futures", "price"])
#         ret = pd.concat([ret, _ret], axis=1)
# =============================================================================
    for w in window:
        for name in futures_universe:
            sma_alpha = w
    
            _data = data.loc[name]
            _data['inner'] = (_data.highw + _data.loww - _data.highw.shift(1) - _data.loww.shift(1)) / 2 * (_data.highw - _data.loww) / _data.volume99
    
            _data['g009'] = _data.inner.ewm(adjust=False, alpha=sma_alpha).mean()
            
            new_columns = pd.MultiIndex.from_product([[str(name)], ["gtja009" + str(w)]])  
            temp = _data['g009'].to_frame(name='g009').set_index(_data.index)  
            temp.columns = new_columns  
            ret = pd.concat([ret, temp], axis=1)      

    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret

gtja010_time_lst = [
    [3, 10],
    [4, 15],
    [5, 20],
    [8, 20],
    [8, 30],
    [10, 35]
]

def GTJA010(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA010
    rank(max((ret < 0 ? std(ret, 20) : close) ^ 2, 5))

    :param data 默认旋转后，即 data_rotated
    :param window 是一个一维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return ret
    '''

    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]
    low = data.low[futures_universe]

    ret = pd.DataFrame()
    for w in window:
        N1, N2 = w

        _data = data.loc[futures_universe]

        _data['const'] = N1
        # _data['returns'] = _data.closew.pct_change()
        _data['returns'] = _data.closew.unstack().pct_change(axis=1).stack()
        _data['inner'] = _data.closew
        # _data.loc[_data.returns < 0, "inner"] = _data.returns.rolling(N2).std()
        _data.loc[_data.returns < 0, "inner"] = _data.returns.unstack().rolling(N2).std().stack()
        _data['inner'] = np.power(_data.inner, 2)
        _data['g010'] = _data[['inner', 'const']].max(axis=1)

        _ret = _data.g010.unstack().T
        _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja010" + str(w)]], names=["futures", "price"])
        ret = pd.concat([ret, _ret], axis=1)

# =============================================================================
#     for w in window:
#         N1, N2 = w
# 
#         _data = data.loc[futures_universe]
# 
#         _data['const'] = N1
#         _data['returns'] = _data.closew.pct_change()
#         _data['inner'] = _data.closew
#         _data.loc[_data.returns < 0, "inner"] = _data.returns.rolling(N2).std()
#         _data['inner'] = np.power(_data.inner, 2)
#         _data['g010'] = _data[['inner', 'const']].max(axis=1)
# 
#         _ret = _data.g010.unstack().T
#         _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja010" + str(w)]], names=["futures", "price"])
#         ret = pd.concat([ret, _ret], axis=1)
# =============================================================================
# =============================================================================
#     for w in window:
#         for name in futures_universe:
#             N1, N2 = w
#     
#             _data = data.loc[name]
#     
#             _data['const'] = N1
#             _data['returns'] = _data.closew.pct_change()
#             _data['inner'] = _data.closew
#             _data.loc[_data.returns < 0, "inner"] = _data.returns.rolling(N2).std()
#             _data['inner'] = np.power(_data.inner, 2)
#             _data['g010'] = _data[['inner', 'const']].max(axis=1)
#     
#             new_columns = pd.MultiIndex.from_product([[str(name)], ["gtja010" + str(w)]])  
#             temp = _data['g010'].to_frame(name='g010').set_index(_data.index)  
#             temp.columns = new_columns  
#             ret = pd.concat([ret, temp], axis=1)
# =============================================================================

    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret


# gtja011_time_lst = [2, 3, 6, 10, 15, 20]
gtja011_time_lst = [2, 3, 6, 8, 12, 15]

def GTJA011(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA011
    sum(((close - low) - (high - close)) / (high - low) * volume, 6)

    :param data 默认旋转后，即 data_rotated
    :param window 是一个一维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return ret
    '''

    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]
    low = data.low[futures_universe]

    ret = pd.DataFrame()
# =============================================================================
#     for w in window:
#         N = w
# 
#         _data = data.loc[futures_universe]
# 
#         _data['inner'] = ((_data.closew - _data.loww) - (_data.highw - _data.closew)) / (_data.highw - _data.loww) * _data.volume
# 
#         _data['g011'] = _data.inner.rolling(N).sum()
# 
#         _ret = _data.g011.unstack().T
#         _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja011" + str(w)]], names=["futures", "price"])
#         ret = pd.concat([ret, _ret], axis=1)
# =============================================================================

    for w in window:
        N = w

        _data = data.loc[futures_universe]

        _data['inner'] = ((_data.closew - _data.loww) - (_data.highw - _data.closew)) / (_data.highw - _data.loww) * _data.volume99

        _data['g011'] = _data.inner.unstack().rolling(N,axis=1).sum().stack()

        _ret = _data.g011.unstack().T
        _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja011" + str(w)]], names=["futures", "price"])
        ret = pd.concat([ret, _ret], axis=1)
        
# =============================================================================
#     for w in window:
#         for name in futures_universe:
#             N = w
#     
#             _data = data.loc[name]
#     
#             _data['inner'] = ((_data.closew - _data.loww) - (_data.highw - _data.closew)) / (_data.highw - _data.loww) * _data.volume
#     
#             _data['g011'] = _data.inner.rolling(N).sum()
#     
#             new_columns = pd.MultiIndex.from_product([[str(name)], ["gtja011" + str(w)]])  
#             temp = _data['g011'].to_frame(name='g011').set_index(_data.index)  
#             temp.columns = new_columns  
#             ret = pd.concat([ret, temp], axis=1)
# =============================================================================

    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret

gtja012_time_lst = [3, 5, 10, 15, 20, 25]

def GTJA012(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA012
    rank(open - mean(vwap, 10)) * rank(abs(close - vwap))

    :param data 默认旋转后，即 data_rotated
    :param window 是一个一维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return ret
    '''

    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]
    low = data.low[futures_universe]

    ret = pd.DataFrame()
    for w in window:
        N = w

        _data = data.loc[futures_universe]
        _data['vwapw'] = (_data.openw + _data.closew + _data.highw + _data.loww) / 4.

        _data['inner1'] = _data.vwapw.unstack().rolling(N,axis=1).mean().stack()
        _data['inner2'] = (_data.closew - _data.vwapw).abs()

        _data['rank1'] = _data.inner1.unstack().rank(pct=True,axis=1).stack()
        _data['rank2'] = _data.inner2.unstack().rank(pct=True,axis=1).stack()

        _data['g012'] = _data.rank1 * _data.rank2

        _ret = _data.g012.unstack().T
        _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja012" + str(w)]], names=["futures", "price"])
        ret = pd.concat([ret, _ret], axis=1)

# =============================================================================
#     for w in window:
#         N = w
# 
#         _data = data.loc[futures_universe]
#         _data['vwapw'] = (_data.openw + _data.closew + _data.highw + _data.loww) / 4.
# 
#         _data['inner1'] = _data.vwapw.rolling(N).mean()
#         _data['inner2'] = (_data.closew - _data.vwapw).abs()
# 
#         _data['rank1'] = _data.inner1.unstack().rank(pct=True).stack()
#         _data['rank2'] = _data.inner2.unstack().rank(pct=True).stack()
# 
#         _data['g012'] = _data.rank1 * _data.rank2
# 
#         _ret = _data.g012.unstack().T
#         _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja012" + str(w)]], names=["futures", "price"])
#         ret = pd.concat([ret, _ret], axis=1)
# =============================================================================

# =============================================================================
#     for w in window:
#         for name in futures_universe:
#             N = w
#     
#             _data = data.loc[name]
#             _data['vwapw'] = (_data.openw + _data.closew + _data.highw + _data.loww) / 4.
#     
#             _data['inner1'] = _data.vwapw.rolling(N).mean()
#             _data['inner2'] = (_data.closew - _data.vwapw).abs()
#     
#             _data['rank1'] = _data.inner1.rank(pct=True)
#             _data['rank2'] = _data.inner2.rank(pct=True)
#     
#             _data['g012'] = _data.rank1 * _data.rank2
#     
#             new_columns = pd.MultiIndex.from_product([[str(name)], ["gtja012" + str(w)]])  
#             temp = _data['g012'].to_frame(name='g012').set_index(_data.index)  
#             temp.columns = new_columns  
#             ret = pd.concat([ret, temp], axis=1)
# =============================================================================


    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret

gtja014_time_lst = [1, 3, 5, 8, 12, 15]

def GTJA014(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA014
    close - close.shift(5)

    :param data 默认旋转后，即 data_rotated
    :param window 是一个一维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return ret
    '''

    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]
    low = data.low[futures_universe]

    ret = pd.DataFrame()


    for w in window:
        N = w

        _data = data.loc[futures_universe]
        _data['g014'] = _data.closew.unstack().diff(N,axis=1).stack()

        _ret = _data.g014.unstack().T
        _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja014" + str(w)]], names=["futures", "price"])
        ret = pd.concat([ret, _ret], axis=1)


    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret

gtja015_time_lst = [1, 2, 3, 5, 7, 10]

def GTJA015(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA015
    open / close.shift(1) - 1

    :param data 默认旋转后，即 data_rotated
    :param window 是一个一维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return ret
    '''

    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]
    low = data.low[futures_universe]

    ret = pd.DataFrame()
    for w in window:
        N = w

        _data = data.loc[futures_universe]
        _data['inner'] = _data.closew.unstack().shift(N,axis=1).stack()
        _data['g015'] = _data.openw / _data.inner - 1

        _ret = _data.g015.unstack().T
        _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja015" + str(w)]], names=["futures", "price"])
        ret = pd.concat([ret, _ret], axis=1)
        


    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret

gtja016_time_lst = [ 3,4, 5, 8, 12, 15]

def GTJA016(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA016
    -1 * tsmax(rank(corr(rank(volume), rank(vwap), 5)), 5)

    :param data 默认旋转后，即 data_rotated
    :param window 是一个一维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return ret
    '''

    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]
    low = data.low[futures_universe]

    ret = pd.DataFrame()
    for w in window:
        N = w

        _data = data.loc[futures_universe]
        _data['vwapw'] = (_data.openw + _data.closew + _data.highw + _data.loww) / 4.

        _data['rank1'] = _data.volume99.unstack().rank(pct=True).stack()
        _data['rank2'] = _data.vwapw.unstack().rank(pct=True).stack()

        _data['inner'] = _data.rank1.rolling(N).corr(_data.rank2)
        _data['rank3'] = _data.inner.unstack().rank(pct=True).stack()

        _data['g016'] = -1 * _data.rank3.rolling(N).max()

        _ret = _data.g016.unstack().T
        _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja016" + str(w)]], names=["futures", "price"])
        ret = pd.concat([ret, _ret], axis=1)



    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret

gtja017_time_lst = [
    [5, 3],
    [10, 5],
    [15, 5],
    [15, 8],
    [20, 8],
    [30, 10]
]

def GTJA017(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA017
    -1 * tsmax(rank(corr(rank(volume), rank(vwap), 5)), 5)

    :param data 默认旋转后，即 data_rotated
    :param window 是一个一维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return ret
    '''

    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]
    low = data.low[futures_universe]

    ret = pd.DataFrame()
# =============================================================================
#     for w in window:
#         N1, N2 = w
# 
#         _data = data.loc[futures_universe]
#         _data['vwapw'] = (_data.openw + _data.closew + _data.highw + _data.loww) / 4.
# 
#         _data['inner'] = _data.vwapw - _data.vwapw.rolling(N1).max()
#         
#         # print(_data.inner.unstack())
#         _data['rank1'] = _data.inner.unstack().rank(pct=True).stack()
#         _data['inner2'] = _data.closew.diff(N2)
# 
#         _data['g017'] = np.power(_data.rank1, _data.inner2)
# 
#         _ret = _data.g017.unstack().T
#         _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja017" + str(w)]], names=["futures", "price"])
#         ret = pd.concat([ret, _ret], axis=1)
# =============================================================================

    for w in window:
        N1, N2 = w

        _data = data.loc[futures_universe]
        _data['vwapw'] = (_data.openw + _data.closew + _data.highw + _data.loww) / 4.

        _data['inner'] = _data.vwapw - _data.vwapw.unstack().rolling(N1,axis=1).max().stack()
        
        # print(_data.inner.unstack())
        _data['rank1'] = _data.inner.unstack().rank(pct=True).stack()
        _data['inner2'] = _data.closew.unstack().diff(N2,axis=1).stack()

        _data['g017'] = np.power(_data.rank1, _data.inner2)

        _ret = _data.g017.unstack().T
        _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja017" + str(w)]], names=["futures", "price"])
        ret = pd.concat([ret, _ret], axis=1)
    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret

# gtja018_time_lst = [1, 3, 5, 8, 12, 15]
gtja018_time_lst = [2, 4, 6, 12, 9, 18]

def GTJA018(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA018
    close / delay(close, 5)

    :param data 默认旋转后，即 data_rotated
    :param window 是一个一维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return ret
    '''

    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]
    low = data.low[futures_universe]

    ret = pd.DataFrame()


# =============================================================================
#     for w in window:
#         N = w
# 
#         _data = data.loc[futures_universe]
#         # _data['vwapw'] = (_data.openw + _data.closew + _data.highw + _data.loww) / 4.
# 
#         _data['g018'] = _data.closew / _data.closew.shift(N)
# 
#         _ret = _data.g018.unstack().T
#         _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja018" + str(w)]], names=["futures", "price"])
#         ret = pd.concat([ret, _ret], axis=1)
# =============================================================================
        
    for w in window:
        for name in futures_universe:
            N = w
    
            _data = data.loc[name]
            # _data['vwapw'] = (_data.openw + _data.closew + _data.highw + _data.loww) / 4.
    
            _data['g018'] = (_data.closew -_data.closew.shift(N) )/ _data.close + 1
    
            new_columns = pd.MultiIndex.from_product([[str(name)], ["gtja018" + str(w)]])  
            temp = _data['g018'].to_frame(name='g018').set_index(_data.index)  
            temp.columns = new_columns  
            ret = pd.concat([ret, temp], axis=1) 

    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret

# gtja019_time_lst = [1, 3, 5, 8, 12, 15]
gtja019_time_lst = [3, 4, 5, 8, 12, 15]

def GTJA019(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA019
    close < delay(close, 5) ? (close - delay(close, 5)) / delay(close, 5) : (close == delay(close, 5) ? 0 : (close - delay(close, 5)) / close)

    :param data 默认旋转后，即 data_rotated
    :param window 是一个一维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return ret
    '''

    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]
    low = data.low[futures_universe]

    ret = pd.DataFrame()
# =============================================================================
#     for w in window:
#         N = w
# 
#         _data = data.loc[futures_universe]
#         # _data['vwapw'] = (_data.openw + _data.closew + _data.highw + _data.loww) / 4.
# 
#         _data['inner'] = _data.closew.shift(N)
# 
#         _data['g019'] = (_data.closew - _data.inner) / _data[['closew', 'inner']].max(axis=1)
# 
#         _ret = _data.g019.unstack().T
#         _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja019" + str(w)]], names=["futures", "price"])
#         ret = pd.concat([ret, _ret], axis=1)
# =============================================================================

    for w in window:
        for name in futures_universe:
            N = w
    
            _data = data.loc[name]
            # _data['vwapw'] = (_data.openw + _data.closew + _data.highw + _data.loww) / 4.
    
            _data['inner'] = _data.closew.shift(N)
            _data['inner1'] = _data.close.shift(N)
            # 将['close', 'inner1'] 更改为 closew  inner zwh 6.12
            _data['g019'] = (_data.closew - _data.inner) / _data[['closew', 'inner']].max(axis=1)
    
            new_columns = pd.MultiIndex.from_product([[str(name)], ["gtja019" + str(w)]])  
            temp = _data['g019'].to_frame(name='g019').set_index(_data.index)  
            temp.columns = new_columns  
            ret = pd.concat([ret, temp], axis=1)

    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret


gtja020_time_lst = [2, 3, 6, 8, 12, 15]

def GTJA020(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA020
    (close - delay(close, 6)) / delay(close, 6)

    :param data 默认旋转后，即 data_rotated
    :param window 是一个一维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return ret
    '''

    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]
    low = data.low[futures_universe]

    ret = pd.DataFrame()
# =============================================================================
#     for w in window:
#         N = w
# 
#         _data = data.loc[futures_universe]
#         # _data['vwapw'] = (_data.openw + _data.closew + _data.highw + _data.loww) / 4.
# 
#         _data['g020'] = (_data.closew - _data.closew.shift(N)) / _data.closew.shift(N)
# 
#         _ret = _data.g020.unstack().T
#         _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja020" + str(w)]], names=["futures", "price"])
#         ret = pd.concat([ret, _ret], axis=1)
# =============================================================================

    for w in window:
        for name in futures_universe:
            N = w
    
            _data = data.loc[name]
            # _data['vwapw'] = (_data.openw + _data.closew + _data.highw + _data.loww) / 4.
    
            _data['g020'] = (_data.closew - _data.closew.shift(N)) / _data.close.shift(N)
    
            new_columns = pd.MultiIndex.from_product([[str(name)], ["gtja020" + str(w)]])  
            temp = _data['g020'].to_frame(name='g020').set_index(_data.index)  
            temp.columns = new_columns  
            ret = pd.concat([ret, temp], axis=1)

    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret

gtja021_time_lst = [2, 3, 6, 8, 12, 15]

def GTJA021(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA021
    regbeta(mean(close, 6), sequence(6))
    其中 regbeta(A, B, n) 是前 n 期 A 对 B 做回归得到的回归系数
    
    :param data 默认旋转后，即 data_rotated
    :param window 是一个一维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return ret
    '''

    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]
    low = data.low[futures_universe]

    ret = pd.DataFrame()
# =============================================================================
#     for w in window:
#         N = w
# 
#         _data = data.loc[futures_universe]
#         # _data['vwapw'] = (_data.openw + _data.closew + _data.highw + _data.loww) / 4.
# 
#         _data['inner'] = _data.closew.rolling(N).mean()
# 
#         _data['g021'] = _data.inner.rolling(N).apply(lambda y: np.polyfit(y=y, x=np.arange(N), deg=1)[0])
#         
#         _ret = _data.g021.unstack().T
#         _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja021" + str(w)]], names=["futures", "price"])
#         ret = pd.concat([ret, _ret], axis=1)
# =============================================================================

    for w in window:
        for name in futures_universe:
            N = w
    
            _data = data.loc[name]
            # _data['vwapw'] = (_data.openw + _data.closew + _data.highw + _data.loww) / 4.
    
            _data['inner'] = _data.closew.rolling(N).mean()
    
            _data['g021'] = _data.inner.rolling(N).apply(lambda y: np.polyfit(y=y, x=np.arange(N), deg=1)[0])
            
            new_columns = pd.MultiIndex.from_product([[str(name)], ["gtja021" + str(w)]])  
            temp = _data['g021'].to_frame(name='g021').set_index(_data.index)  
            temp.columns = new_columns  
            ret = pd.concat([ret, temp], axis=1) 

    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret

gtja022_time_lst = [
    [4, 2, 1 / 4],
    [6, 3, 1 / 6],
    [6, 3, 1 / 12],
    [8, 4, 1 / 12],
    [10, 5, 1 / 10],
    [12, 6, 1 / 8]
]

def GTJA022(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA022
    sma(delta((close - mean(close, 6)) / mean(close, 6), 3), 12, 1)
    
    :param data 默认旋转后，即 data_rotated
    :param window 是一个一维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return ret
    '''

    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]
    low = data.low[futures_universe]

    ret = pd.DataFrame()
# =============================================================================
#     for w in window:
#         N1, N2, sma_alpha = w
# 
#         _data = data.loc[futures_universe]
#         # _data['vwapw'] = (_data.openw + _data.closew + _data.highw + _data.loww) / 4.
# 
#         _data['inner'] = (_data.closew - _data.closew.rolling(N1).mean()).diff(N2)
# 
#         _data['g022'] = _data.inner.ewm(adjust=False, alpha=sma_alpha).mean()
# 
#         _ret = _data.g022.unstack().T
#         _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja022" + str(w)]], names=["futures", "price"])
#         ret = pd.concat([ret, _ret], axis=1)
# =============================================================================

    for w in window:
        for name in futures_universe:
            N1, N2, sma_alpha = w
    
            _data = data.loc[name]
            # _data['vwapw'] = (_data.openw + _data.closew + _data.highw + _data.loww) / 4.
    
            _data['inner'] = (_data.closew - _data.closew.rolling(N1).mean()).diff(N2)
    
            _data['g022'] = _data.inner.ewm(adjust=False, alpha=sma_alpha).mean()
    
            new_columns = pd.MultiIndex.from_product([[str(name)], ["gtja022" + str(w)]])  
            temp = _data['g022'].to_frame(name='g022').set_index(_data.index)  
            temp.columns = new_columns  
            ret = pd.concat([ret, temp], axis=1)

    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret

gtja023_time_lst = [
    [10, 1 / 12],
    [15, 1 / 15],
    [20, 1 / 20],
    [25, 1 / 20],
    [30, 1 / 20],
    [40, 1 / 30]
]

def GTJA023(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA023
    sma(close > delay(close, 1) ? std(close, 20) : 0, 20, 1) / (sma(close > delay(close, 1) ? std(close, 20) : 0, 20, 1) + sma(close <= delay(close, 1) ? std(close, 20) : 0, 20, 1))
    
    :param data 默认旋转后，即 data_rotated
    :param window 是一个一维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return ret
    '''

    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]
    low = data.low[futures_universe]

    ret = pd.DataFrame()
# =============================================================================
#     for w in window:
#         N, sma_alpha = w
# 
#         _data = data.loc[futures_universe]
#         # _data['vwapw'] = (_data.openw + _data.closew + _data.highw + _data.loww) / 4.
# 
#         _data['inner1'] = 0
#         _data.loc[_data.closew > _data.closew.shift(1), "inner1"] = _data.closew.rolling(N).std()
#         _data['inner2'] = 0
#         _data.loc[_data.closew <= _data.closew.shift(1), "inner2"] = _data.closew.rolling(N).std()
# 
#         _data['inner3'] = _data.inner1.ewm(adjust=False, alpha=sma_alpha).mean()
#         _data['inner4'] = _data.inner2.ewm(adjust=False, alpha=sma_alpha).mean()
# 
#         _data['g023'] = _data.inner3 / (_data.inner3 + _data.inner4)
# 
#         _ret = _data.g023.unstack().T
#         _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja023" + str(w)]], names=["futures", "price"])
#         ret = pd.concat([ret, _ret], axis=1)
# =============================================================================

    for w in window:
        for name in futures_universe:
            N, sma_alpha = w
    
            _data = data.loc[name]
            # _data['vwapw'] = (_data.openw + _data.closew + _data.highw + _data.loww) / 4.
    
            _data['inner1'] = 0
            _data.loc[_data.closew > _data.closew.shift(1), "inner1"] = _data.closew.rolling(N).std()
            _data['inner2'] = 0
            _data.loc[_data.closew <= _data.closew.shift(1), "inner2"] = _data.closew.rolling(N).std()
    
            _data['inner3'] = _data.inner1.ewm(adjust=False, alpha=sma_alpha).mean()
            _data['inner4'] = _data.inner2.ewm(adjust=False, alpha=sma_alpha).mean()
    
            _data['g023'] = _data.inner3 / (_data.inner3 + _data.inner4)
    
            new_columns = pd.MultiIndex.from_product([[str(name)], ["gtja023" + str(w)]])  
            temp = _data['g023'].to_frame(name='g023').set_index(_data.index)  
            temp.columns = new_columns  
            ret = pd.concat([ret, temp], axis=1)  

    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret

gtja024_time_lst = [
    [3, 1 / 3],
    [3, 1 / 5],
    [5, 1 / 5],
    [10, 1 / 5],
    [20, 1 / 5],
    [20, 1 / 10]
]
# =============================================================================
# gtja024_time_lst = [
#     [1, 1 / 3],
#     [2, 1 / 5],
#     [6, 1 / 5],
#     [11, 1 / 5],
#     [21, 1 / 5],
#     [31, 1 / 10]
# ]
# =============================================================================
def GTJA024(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA024
    sma(close - delay(close, 5), 5, 1)
    
    :param data 默认旋转后，即 data_rotated
    :param window 是一个一维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return ret
    '''

    # 
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]
    low = data.low[futures_universe]

    ret = pd.DataFrame()
# =============================================================================
#     for w in window:
#         N, sma_alpha = w
# 
#         _data = data.loc[futures_universe]
#         # _data['vwapw'] = (_data.openw + _data.closew + _data.highw + _data.loww) / 4.
# 
#         _data['inner'] = _data.closew.diff(N)
# 
#         _data['g024'] = _data.inner.ewm(adjust=False, alpha=sma_alpha).mean()
# 
#         _ret = _data.g024.unstack().T
#         _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja024" + str(w)]], names=["futures", "price"])
#         ret = pd.concat([ret, _ret], axis=1)
# =============================================================================

    for w in window:
        for name in futures_universe:
            N, sma_alpha = w
    
            _data = data.loc[name]
            # _data['vwapw'] = (_data.openw + _data.closew + _data.highw + _data.loww) / 4.
    
            _data['inner'] = _data.closew.diff(N)
    
            _data['g024'] = _data.inner.ewm(adjust=False, alpha=sma_alpha).mean()
    
            new_columns = pd.MultiIndex.from_product([[str(name)], ["gtja024" + str(w)]])  
            temp = _data['g024'].to_frame(name='g024').set_index(_data.index)  
            temp.columns = new_columns  
            ret = pd.concat([ret, temp], axis=1)


    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret

gtja025_time_lst = [
    [4, 12, 3, 100],
    [5, 15, 6, 200],
    [7, 20, 9, 250],
    [10, 30, 12, 250],
    [15, 40, 15, 150],
    [20, 45, 20, 150],
]

def GTJA025(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA025
    rank(delta(close, 7) * (1 - rank(decaylinear(volume / mean(volume, 20), 9)))) * (1 + rank(sum(ret, 250)))
    
    :param data 默认旋转后，即 data_rotated
    :param window 是一个一维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return ret
    '''

    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]
    low = data.low[futures_universe]

    def wma(series, N):
        return series.rolling(N).apply(lambda x: x[:: -1].cumsum().sum() * 2 / N / (N + 1))

    ret = pd.DataFrame()
    for w in window:
        N1, N2, N3, N4 = w

        _data = data.loc[futures_universe]
        _data['returns'] = _data.closew.unstack().pct_change(axis=1).stack()

        _data['inner1'] = wma(_data.volume99 / _data.volume99.unstack().rolling(N2,axis=1).mean().stack(), N3)
        _data['rank1'] = _data.inner1.unstack().rank(pct=True).stack()

        _data['inner2'] = _data.closew.unstack().diff(N1,axis=1).stack() * (1 - _data.rank1)
        _data['rank2'] = _data.inner2.unstack().rank(pct=True).stack()

        _data['inner3'] = _data.returns.unstack().rolling(N4,axis=1).sum().stack()
        _data['rank3'] = _data.inner3.unstack().rank(pct=True).stack()

        _data['g025'] = _data.rank2 * (1 + _data.rank3)

        _ret = _data.g025.unstack().T
        _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja025" + str(w)]], names=["futures", "price"])
        ret = pd.concat([ret, _ret], axis=1)

# =============================================================================
#     for w in window:
#         N1, N2, N3, N4 = w
# 
#         _data = data.loc[futures_universe]
#         _data['returns'] = _data.closew.pct_change()
# 
#         _data['inner1'] = wma(_data.volume / _data.volume.rolling(N2).mean(), N3)
#         _data['rank1'] = _data.inner1.unstack().rank(pct=True).stack()
# 
#         _data['inner2'] = _data.closew.diff(N1) * (1 - _data.rank1)
#         _data['rank2'] = _data.inner2.unstack().rank(pct=True).stack()
# 
#         _data['inner3'] = _data.returns.rolling(N4).sum()
#         _data['rank3'] = _data.inner3.unstack().rank(pct=True).stack()
# 
#         _data['g025'] = _data.rank2 * (1 + _data.rank3)
# 
#         _ret = _data.g025.unstack().T
#         _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja025" + str(w)]], names=["futures", "price"])
#         ret = pd.concat([ret, _ret], axis=1)
# =============================================================================


    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret

gtja026_time_lst = [
    [4, 3, 60],
    [5, 4, 120],
    [7, 5, 230],
    [8, 6, 240],
    [10, 10, 240],
    [15, 15, 240]
]

def GTJA026(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA026
    mean(close, 7) - close + corr(vwap, delay(close, 5), 230)
    
    :param data 默认旋转后，即 data_rotated
    :param window 是一个一维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return ret
    '''

    # 
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]
    low = data.low[futures_universe]

    ret = pd.DataFrame()
# =============================================================================
#     for w in window:
#         N1, N2, N3 = w
# 
#         _data = data.loc[futures_universe]
#         _data['vwapw'] = (_data.openw + _data.closew + _data.highw + _data.loww) / 4
#         _data['g026'] = _data.closew.rolling(N1).mean() - _data.closew + _data.vwapw.rolling(N3).corr(_data.closew.shift(N2))
# 
#         _ret = _data.g026.unstack().T
#         _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja026" + str(w)]], names=["futures", "price"])
#         ret = pd.concat([ret, _ret], axis=1)
# =============================================================================

    for w in window:
        for name in futures_universe:
            N1, N2, N3 = w
    
            _data = data.loc[name]
            _data['vwapw'] = (_data.openw + _data.closew + _data.highw + _data.loww) / 4
            _data['g026'] = _data.closew.rolling(N1).mean() - _data.closew + _data.vwapw.rolling(N3).corr(_data.closew.shift(N2))
    
            new_columns = pd.MultiIndex.from_product([[str(name)], ["gtja026" + str(w)]])  
            temp = _data['g026'].to_frame(name='g026').set_index(_data.index)  
            temp.columns = new_columns  
            ret = pd.concat([ret, temp], axis=1) 


    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret

gtja027_time_lst = [
    [1, 2, 10],
    [2, 4, 11],
    [3, 6, 12],
    [4, 8, 15],
    [6, 12, 18],
    [8, 16, 20]
]

def GTJA027(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA027
    wma(delta(close, 3) / delay(close, 3) + delta(close, 6) / delay(close, 6), 12)
    
    :param data 默认旋转后，即 data_rotated
    :param window 是一个一维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return ret
    '''

    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]
    low = data.low[futures_universe]

    def wma(series, N):
        return series.rolling(N).apply(lambda x: x[:: -1].cumsum().sum() * 2 / N / (N + 1))

    ret = pd.DataFrame()
# =============================================================================
#     for w in window:
#         N1, N2, N3 = w
# 
#         _data = data.loc[futures_universe]
#         _data['inner'] = _data.closew.diff(N1) / _data.closew.shift(N1) + _data.closew.diff(N2) / _data.closew.shift(N2)
# 
#         _data['g027'] = wma(_data.inner, N3)
#         _ret = _data.g027.unstack().T
#         _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja027" + str(w)]], names=["futures", "price"])
#         ret = pd.concat([ret, _ret], axis=1)
# =============================================================================

    for w in window:
        for name in futures_universe:
            N1, N2, N3 = w
    
            _data = data.loc[name]
            _data['inner'] = _data.closew.diff(N1) / _data.closew.shift(N1) + _data.closew.diff(N2) / _data.closew.shift(N2)
    
            _data['g027'] = wma(_data.inner, N3)
            new_columns = pd.MultiIndex.from_product([[str(name)], ["gtja027" + str(w)]])  
            temp = _data['g027'].to_frame(name='g027').set_index(_data.index)  
            temp.columns = new_columns  
            ret = pd.concat([ret, temp], axis=1) 


    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret

gtja028_time_lst = [
    [1, 1, 4, 1 / 4],
    [2, 1, 6, 1 / 3],
    [3, 2, 9, 1 / 3],
    [3, 1, 12, 1 / 2],
    [4, 1, 12, 1 / 5],
    [5, 1, 15, 1 / 6]
]

def GTJA028(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA028
    3 * sma((close - tsmin(low, 9)) / (tsmax(high, 9) - tsmin(low, 9)), 3, 1) - 2 * sma(sma((close - tsmin(low, 9)) / (tsmax(high, 9) - tsmin(low, 9)), 3, 1), 3, 1)
    
    :param data 默认旋转后，即 data_rotated
    :param window 是一个一维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return ret
    '''

    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]
    low = data.low[futures_universe]

    def wma(series, N):
        return series.rolling(N).apply(lambda x: x[:: -1].cumsum().sum() * 2 / N / (N + 1))

    ret = pd.DataFrame()
# =============================================================================
#     for w in window:
#         N1, N2, N3, sma_alpha = w
# 
#         _data = data.loc[futures_universe]
#         _data['inner1'] = (_data.closew - _data.loww.rolling(N3).min()) / (_data.highw.rolling(N3).max() - _data.loww.rolling(N3).min())
#         _data['inner2'] = _data.inner1.ewm(adjust=False, alpha=sma_alpha).mean()
#         _data['inner3'] = _data.inner2.ewm(adjust=False, alpha=sma_alpha).mean()
# 
#         _data['g028'] = N1 * _data.inner2 - N2 * _data.inner3
#         _ret = _data.g028.unstack().T
#         _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja028" + str(w)]], names=["futures", "price"])
#         ret = pd.concat([ret, _ret], axis=1)
# =============================================================================

    for w in window:
        for name in futures_universe:
            N1, N2, N3, sma_alpha = w
    
            _data = data.loc[name]
            _data['inner1'] = (_data.closew - _data.loww.rolling(N3).min()) / (_data.highw.rolling(N3).max() - _data.loww.rolling(N3).min())
            _data['inner2'] = _data.inner1.ewm(adjust=False, alpha=sma_alpha).mean()
            _data['inner3'] = _data.inner2.ewm(adjust=False, alpha=sma_alpha).mean()
    
            _data['g028'] = N1 * _data.inner2 - N2 * _data.inner3
            new_columns = pd.MultiIndex.from_product([[str(name)], ["gtja028" + str(w)]])  
            temp = _data['g028'].to_frame(name='g028').set_index(_data.index)  
            temp.columns = new_columns  
            ret = pd.concat([ret, temp], axis=1)  


    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret

gtja029_time_lst = [2, 4, 6, 9, 12, 18]

def GTJA029(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA029
    3 * sma((close - tsmin(low, 9)) / (tsmax(high, 9) - tsmin(low, 9)), 3, 1) - 2 * sma(sma((close - tsmin(low, 9)) / (tsmax(high, 9) - tsmin(low, 9)), 3, 1), 3, 1)
    
    :param data 默认旋转后，即 data_rotated
    :param window 是一个一维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return ret
    '''

    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]
    low = data.low[futures_universe]

    ret = pd.DataFrame()
# =============================================================================
#     for w in window:
#         N = w
# 
#         _data = data.loc[futures_universe]
#         _data['g029'] = _data.closew.diff(N) / _data.closew.shift(N) * _data.volume
#         _ret = _data.g029.unstack().T
#         _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja029" + str(w)]], names=["futures", "price"])
#         ret = pd.concat([ret, _ret], axis=1)
# =============================================================================

    for w in window:
        for name in futures_universe:
            N = w
    
            _data = data.loc[name]
            _data['g029'] = _data.closew.diff(N) / _data.closew.shift(N) * _data.volume99
            new_columns = pd.MultiIndex.from_product([[str(name)], ["gtja029" + str(w)]])  
            temp = _data['g029'].to_frame(name='g029').set_index(_data.index)  
            temp.columns = new_columns  
            ret = pd.concat([ret, temp], axis=1)  


    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret

gtja031_time_lst = [6, 9, 12, 16, 20, 30]

def GTJA031(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA031
    (close - mean(close, 12)) / mean(close, 12)
    
    :param data 默认旋转后，即 data_rotated
    :param window 是一个一维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return ret
    '''

    # 
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]
    low = data.low[futures_universe]

    ret = pd.DataFrame()
# =============================================================================
#     for w in window:
#         N = w
# 
#         _data = data.loc[futures_universe]
#         _data['g031'] = (_data.closew - _data.closew.rolling(N).mean()) / _data.closew.rolling(N).mean()
#         _ret = _data.g031.unstack().T
#         _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja031" + str(w)]], names=["futures", "price"])
#         ret = pd.concat([ret, _ret], axis=1)
# =============================================================================

    for w in window:
        for name in futures_universe:
            N = w
    
            _data = data.loc[name]
            _data['g031'] = (_data.closew - _data.closew.rolling(N).mean()) / _data.close.rolling(N).mean()
            new_columns = pd.MultiIndex.from_product([[str(name)], ["gtja031" + str(w)]])  
            temp = _data['g031'].to_frame(name='g031').set_index(_data.index)  
            temp.columns = new_columns  
            ret = pd.concat([ret, temp], axis=1) 


    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret

gtja032_time_lst = [3, 5, 8, 12, 16, 20]

def GTJA032(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA032
    sum(rank(corr(rank(high), rank(volume), 3)), 3)
    
    :param data 默认旋转后，即 data_rotated
    :param window 是一个一维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return ret
    '''
    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]
    low = data.low[futures_universe]

    ret = pd.DataFrame()
# =============================================================================
#     for w in window:
#         N = w
# 
#         _data = data.loc[futures_universe]
# 
#         _data['rank1'] = _data.high.unstack().rank(pct=True).stack()
#         _data['rank2'] = _data.volume.unstack().rank(pct=True).stack()
# 
#         _data['inner'] = _data.rank1.rolling(N).corr(_data.rank2)
#         _data['rank3'] = _data.inner.unstack().rank(pct=True).stack()
# 
#         _data['g032'] = _data.rank3.rolling(N).sum()
#         _ret = _data.g032.unstack().T
#         _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja032" + str(w)]], names=["futures", "price"])
#         ret = pd.concat([ret, _ret], axis=1)
# =============================================================================
    for w in window:
        N = w

        _data = data.loc[futures_universe]

        _data['rank1'] = _data.high.unstack().rank(pct=True).stack()
        _data['rank2'] = _data.volume99.unstack().rank(pct=True).stack()

        _data['inner'] = _data.rank1.unstack().rolling(N, axis=1).corr(_data.rank2.unstack(), pairwise=True).stack()
        # _data['inner'] = _data.rank1.unstack().rolling(N,axis=1).corr(_data.rank2.unstack()).stack()
        print('0000000000000000000000')
        # print(_data.rank1.unstack().rolling(N,axis=1))
        print(_data.rank1.unstack())
        print('-----------------------------------------')
        print(_data.rank2.unstack())
        print('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
        print(_data.inner.unstack())
        print('bbbbbbbbbbbbbbbbbbbbbbbbb')
        print(_data.inner['TA'])
# =============================================================================
#         _data['rank3'] = _data.inner.unstack().rank(pct=True).stack()
# 
#         _data['g032'] = _data.rank3.unstack().rolling(N,axis=1).sum().stack()
#         _ret = _data.g032.unstack().T
#         _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja032" + str(w)]], names=["futures", "price"])
#         ret = pd.concat([ret, _ret], axis=1)
# =============================================================================


    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret

gtja033_time_lst = [
    [3, 50, 10],
    [3, 100, 15], 
    [5, 240, 20],
    [6, 240, 40],
    [8, 240, 60],
    [10, 240, 120]
]

def GTJA033(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA033
    delta(tsmin(low, 5), 5) * rank(sum(ret, 240) - sum(ret, 20)) * tsrank(volume, 5)
    
    :param data 默认旋转后，即 data_rotated
    :param window 是一个一维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return ret
    '''

    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]
    low = data.low[futures_universe]

    def tsrank(array):
        s = pd.Series(array)
        return np.array(s.rank(method="min", ascending=False))[-1]

    ret = pd.DataFrame()
# =============================================================================
#     for w in window:
#         N1, N2, N3 = w
# 
#         _data = data.loc[futures_universe]
# 
#         _data['returns'] = _data.closew.pct_change()
#         _data['inner1'] = (_data.loww.rolling(N1).min()).diff(N1)
#         
#         _data['inner2'] = _data.returns.rolling(N2).sum() - _data.returns.rolling(N3).sum()
#         _data['rank2'] = _data.inner2.unstack().rank(pct=True).stack()
# 
#         _data['inner3'] = _data.volume.rolling(N1).apply(tsrank)
# 
#         _data['g033'] = _data.inner1 * _data.rank2 * _data.inner3
#         _ret = _data.g033.unstack().T
#         _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja033" + str(w)]], names=["futures", "price"])
#         ret = pd.concat([ret, _ret], axis=1)
# =============================================================================
    for w in window:
        N1, N2, N3 = w

        _data = data.loc[futures_universe]

        _data['returns'] = _data.closew.unstack().pct_change(axis=1).stack()
        _data['inner1'] = (_data.loww.unstack().rolling(N1,axis=1).min()).diff(N1,axis=1).stack()
        
        _data['inner2'] = _data.returns.unstack().rolling(N2,axis=1).sum().stack() - _data.returns.unstack().rolling(N3,axis=1).sum().stack()
        _data['rank2'] = _data.inner2.unstack().rank(pct=True).stack()

        _data['inner3'] = _data.volume99.unstack().rolling(N1,axis=1).apply(tsrank).stack()

        _data['g033'] = _data.inner1 * _data.rank2 * _data.inner3
        _ret = _data.g033.unstack().T
        _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja033" + str(w)]], names=["futures", "price"])
        ret = pd.concat([ret, _ret], axis=1)
        

    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret

gtja034_time_lst = [3, 6, 12, 24, 36, 48]

def GTJA034(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA034
    mean(close, 12) / close
    
    :param data 默认旋转后，即 data_rotated
    :param window 是一个一维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return ret
    '''

    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]
    low = data.low[futures_universe]

    ret = pd.DataFrame()
    for w in window:
        N = w

        _data = data.loc[futures_universe]

        _data['g034'] = 1+ (_data.closew.unstack().rolling(N,axis=1).mean().stack()-_data.closew) / _data.close

        _ret = _data.g034.unstack().T
        _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja034" + str(w)]], names=["futures", "price"])
        ret = pd.concat([ret, _ret], axis=1)



    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret

gtja035_time_lst = [
    [1, 5, 10, 4],
    [1, 10, 12, 5],
    [1, 15, 17, 7],
    [1, 25, 25, 10],
    [3, 25, 25, 10],
    [5, 20, 20, 10]
]

def GTJA035(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA035
    min(rank(decaylinear(delta(open, 1), 15)), rank(decaylinear(corr(volume, open, 17), 7)))
    
    :param data 默认旋转后，即 data_rotated
    :param window 是一个一维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return ret
    '''

    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]
    low = data.low[futures_universe]

    def wma(series, N):
        return series.rolling(N).apply(lambda x: x[:: -1].cumsum().sum() * 2 / N / (N + 1))

    ret = pd.DataFrame()
# =============================================================================
#     for w in window:
#         N1, N2, N3, N4 = 1, 15, 17, 7
# 
#         _data = data.loc[futures_universe]
# 
#         _data['inner1'] = wma(_data.openw.diff(N1), N2)
#         _data['rank1'] = _data.inner1.unstack().rank(pct=True).stack()
# 
#         _data['inner2'] = wma(_data.volume.rolling(N3).corr(_data.open), N4)
#         _data['rank2'] = _data.inner2.unstack().rank(pct=True).stack()
# 
#         _data['g035'] = _data[['rank1', 'rank2']].min(axis=1)
# 
#         _ret = _data.g035.unstack().T
#         _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja035" + str(w)]], names=["futures", "price"])
#         ret = pd.concat([ret, _ret], axis=1)
# =============================================================================

    for w in window:
        N1, N2, N3, N4 = 1, 15, 17, 7

        _data = data.loc[futures_universe]

        _data['inner1'] = wma(_data.openw.unstack().diff(N1,axis=1), N2,axis=1).stack()
        _data['rank1'] = _data.inner1.unstack().rank(pct=True).stack()

        _data['inner2'] = wma(_data.volume99.unstack().rolling(N3,axis=1).corr(_data.open.unstack()), N4,axis=1).stack()
        _data['rank2'] = _data.inner2.unstack().rank(pct=True).stack()

        _data['g035'] = _data[['rank1', 'rank2']].min(axis=1)

        _ret = _data.g035.unstack().T
        _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja035" + str(w)]], names=["futures", "price"])
        ret = pd.concat([ret, _ret], axis=1)


    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret

gtja036_time_lst = [
    [3, 2],
    [4, 2],
    [6, 2],
    [12, 6],
    [12, 12],
    [12, 20]
]

def GTJA036(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA036
    rank(sum(corr(rank(volume), rank(vwap), 6), 2))
    
    :param data 默认旋转后，即 data_rotated
    :param window 是一个一维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return ret
    '''

    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]
    low = data.low[futures_universe]

    ret = pd.DataFrame()
    for w in window:
        N1, N2 = w

        _data = data.loc[futures_universe]
        _data['vwapw'] = (_data.openw + _data.closew + _data.highw + _data.loww) / 4

        _data['rank1'] = _data.volume99.unstack().rank(pct=True).stack()
        _data['rank2'] = _data.vwapw.unstack().rank(pct=True).stack()

        _data['inner'] = _data.rank1.rolling(N1).corr(_data.rank2)  # ???为什么对rank1进行rollilng呢

        _data['g036'] = _data.inner.rolling(N2).sum()

        _ret = _data.g036.unstack().T
        _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja036" + str(w)]], names=["futures", "price"])
        ret = pd.concat([ret, _ret], axis=1)

    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret

gtja037_time_lst = [
    [3, 2],
    [3, 5],
    [5, 10],
    [8, 10],
    [15, 10],
    [15, 15]
]

def GTJA037(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA037
    rank(delta(sum(open, 5) * sum(ret, 5), 10))
    
    :param data 默认旋转后，即 data_rotated
    :param window 是一个一维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return ret
    '''

    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]
    low = data.low[futures_universe]

    ret = pd.DataFrame()
# =============================================================================
#     for w in window:
#         N1, N2 = w
# 
#         _data = data.loc[futures_universe]
#         _data['returns'] = _data.closew.pct_change()
#         _data['inner'] = _data.openw.rolling(N1).sum() * _data.returns.rolling(N1).sum()
# 
#         _data['g037'] = _data.inner.diff(N2)
#         _ret = _data.g037.unstack().T
#         _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja037" + str(w)]], names=["futures", "price"])
#         ret = pd.concat([ret, _ret], axis=1)
# =============================================================================

    for w in window:
        for name in futures_universe:
            N1, N2 = w
    
            _data = data.loc[name]
            _data['returns'] = _data.closew.pct_change()
            _data['inner'] = _data.openw.rolling(N1).sum() * _data.returns.rolling(N1).sum()
    
            _data['g037'] = _data.inner.diff(N2)
            new_columns = pd.MultiIndex.from_product([[str(name)], ["gtja037" + str(w)]])  
            temp = _data['g037'].to_frame(name='g037').set_index(_data.index)  
            temp.columns = new_columns  
            ret = pd.concat([ret, temp], axis=1)  

    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret

gtja038_time_lst = [
    [10, 4],
    [15, 3],
    [20, 2],
    [20, 5],
    [25, 5],
    [40, 3]
]

def GTJA038(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA038
    mean(high, 20) < high ? delta(high, 2) : 0
    
    :param data 默认旋转后，即 data_rotated
    :param window 是一个一维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return ret
    '''

    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]
    low = data.low[futures_universe]

    ret = pd.DataFrame()

    for w in window:
        for name in futures_universe:
            N1, N2 = w
    
            _data = data.loc[name]
            _data['inner'] = _data.highw.rolling(N1).mean()
    
            _data['g038'] = 0
            _data.loc[_data.inner < _data.highw, 'g038'] = _data.highw.diff(N2)
            new_columns = pd.MultiIndex.from_product([[str(name)], ["gtja038" + str(w)]])  
            temp = _data['g038'].to_frame(name='g038').set_index(_data.index)  
            temp.columns = new_columns  
            ret = pd.concat([ret, temp], axis=1)
    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret

gtja039_time_lst = [
    [1, 3, .3, 50, 7, 7, 7],
    [2, 5, .5, 100, 10, 12, 10],
    [2, 8, .3, 180, 17, 14, 12],
    [2, 8, .8, 180, 20, 20, 20],
    [4, 10, .5, 200, 25, 25, 25],
    [2, 8, .5, 200, 40, 40, 40]
]

def GTJA039(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA039
    rank(decaylinear(delta(close, 2), 8)) - rank(decaylinear(corr(vwap * .3 + open * .7, sum(mean(volume, 180), 17), 14), 12))
    
    :param data 默认旋转后，即 data_rotated
    :param window 是一个一维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return ret
    '''

    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]
    low = data.low[futures_universe]

    ret = pd.DataFrame()
    for w in window:
        N1, N2, N3, N4, N5, N6, N7 = w

        _data = data.loc[futures_universe]
        
        def wma(series, N):
            return series.rolling(N).apply(lambda x: x[:: -1].cumsum().sum() * 2 / N / (N + 1))

        _data['inner1'] = wma(_data.closew.unstack().diff(N1), N2,axis=1).stack()
        _data['inner21'] = _data.vwapw * N3 + _data.openw * (1 - N3)
        _data['inner22'] = (_data.volume99.unstack().rolling(N4,axis=1).mean()).rolling(N5,axis=1).sum().stack()
        _data['inner23'] = _data.inner21.unstack().rolling(N6,axis=1).corr(_data.inner22).stack()
        _data['inner2'] = wma(_data.inner23.unstack(), N7,axis=1).stack()

        _data['rank1'] = _data.inner1.unstack().rank(pct=True).stack()
        _data['rank2'] = _data.inner2.unstack().rank(pct=True).stack()

        _data['g039'] = _data.rank1 - _data.rank2
        _ret = _data.g039.unstack().T
        _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja039" + str(w)]], names=["futures", "price"])
        ret = pd.concat([ret, _ret], axis=1)
# =============================================================================
#     for w in window:
#         N1, N2, N3, N4, N5, N6, N7 = w
# 
#         _data = data.loc[futures_universe]
#         
#         def wma(series, N):
#             return series.rolling(N).apply(lambda x: x[:: -1].cumsum().sum() * 2 / N / (N + 1))
# 
#         _data['inner1'] = wma(_data.closew.diff(N1), N2)
#         _data['inner21'] = _data.vwapw * N3 + _data.openw * (1 - N3)
#         _data['inner22'] = (_data.volume.rolling(N4).mean()).rolling(N5).sum()
#         _data['inner23'] = _data.inner21.rolling(N6).corr(_data.inner22)
#         _data['inner2'] = wma(_data.inner23, N7)
# 
#         _data['rank1'] = _data.inner1.unstack().rank(pct=True).stack()
#         _data['rank2'] = _data.inner2.unstack().rank(pct=True).stack()
# 
#         _data['g039'] = _data.rank1 - _data.rank2
#         _ret = _data.g039.unstack().T
#         _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja039" + str(w)]], names=["futures", "price"])
#         ret = pd.concat([ret, _ret], axis=1)
# =============================================================================
    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret

# gtja040_time_lst = [10, 16, 26, 40, 50, 60]
gtja040_time_lst = [12, 24, 36, 48, 60, 72]

def GTJA040(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA040
    sum(close > delay(close, 1) ? volume : 0, 26) / sum(close <= delay(close, 1) ? volume : 0, 26)
    
    :param data 默认旋转后，即 data_rotated
    :param window 是一个一维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return ret
    '''

    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]
    low = data.low[futures_universe]

    ret = pd.DataFrame()
# =============================================================================
#     for w in window:
#         N = w
# 
#         _data = data.loc[futures_universe]
#         
#         _data['inner1'] = 0
#         _data.loc[_data.closew > _data.closew.shift(1), "inner1"] = _data.volume
# 
#         _data['inner2'] = 0
#         _data.loc[_data.closew <= _data.closew.shift(1), "inner2"] = _data.volume
# 
#         _data['g040'] = _data.inner1.rolling(N).sum() / _data.inner2.rolling(N).sum()
# 
#         _ret = _data.g040.unstack().T
#         _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja040" + str(w)]], names=["futures", "price"])
#         ret = pd.concat([ret, _ret], axis=1)
# =============================================================================

    for w in window:
        for name in futures_universe:
            N = w
    
            _data = data.loc[name]
            
            _data['inner1'] = 0
            _data.loc[_data.closew > _data.closew.shift(1), "inner1"] = _data.volume99
    
            _data['inner2'] = 0
            _data.loc[_data.closew <= _data.closew.shift(1), "inner2"] = _data.volume99
    
            _data['g040'] = _data.inner1.rolling(N).sum() / _data.inner2.rolling(N).sum()
    
            new_columns = pd.MultiIndex.from_product([[str(name)], ["gtja040" + str(w)]])  
            temp = _data['g040'].to_frame(name='g040').set_index(_data.index)  
            temp.columns = new_columns  
            ret = pd.concat([ret, temp], axis=1) 


    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret

gtja041_time_lst = [
    [2, 1],
    [2, 3],
    [3, 5],
    [3, 8],
    [5, 6],
    [6, 8]
]

def GTJA041(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA041
    rank(max(delta(vwap, 3), 5))
    
    :param data 默认旋转后，即 data_rotated
    :param window 是一个一维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return ret
    '''

    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]
    low = data.low[futures_universe]

    ret = pd.DataFrame()
# =============================================================================
#     for w in window:
#         N1, N2 = w
# 
#         _data = data.loc[futures_universe]
#         _data['vwapw'] = (_data.openw + _data.closew + _data.highw + _data.loww) / 4
#         
#         _data['const'] = N2
#         _data['inner'] = _data.vwapw.diff(N1)
# 
#         _data['g041'] = _data[['const', 'inner']].max(axis=1)
# 
#         _ret = _data.g041.unstack().T
#         _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja041" + str(w)]], names=["futures", "price"])
#         ret = pd.concat([ret, _ret], axis=1)
# =============================================================================

    for w in window:
        for name in futures_universe:
            N1, N2 = w
    
            _data = data.loc[name]
            _data['vwapw'] = (_data.openw + _data.closew + _data.highw + _data.loww) / 4
            
            _data['const'] = N2
            _data['inner'] = _data.vwapw.diff(N1)
    
            _data['g041'] = _data[['const', 'inner']].max(axis=1)
    
            new_columns = pd.MultiIndex.from_product([[str(name)], ["gtja041" + str(w)]])  
            temp = _data['g041'].to_frame(name='g041').set_index(_data.index)  
            temp.columns = new_columns  
            ret = pd.concat([ret, temp], axis=1)


    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret

gtja041edit_time_lst = [
    [1, 3],
    [2, 4],
    [3, 5],
    [5, 8],
    [7, 10],
    [10, 15]
]
# =============================================================================
# gtja041edit_time_lst = [
#     [2, 1],
#     [2, 3],
#     [3, 5],
#     [3, 8],
#     [5, 6],
#     [6, 8]
# ]
# =============================================================================


def GTJA041EDIT(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA041edit
    max 改成 tsmax
    rank(tsmax(delta(vwap, 3), 5))
    
    :param data 默认旋转后，即 data_rotated
    :param window 是一个一维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return ret
    '''

    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]

    ret = pd.DataFrame()
# =============================================================================
#     for w in window:
#         N1, N2 = w
# 
#         _data = data.loc[futures_universe]
#         _data['vwapw'] = (_data.openw + _data.closew + _data.highw + _data.loww) / 4
# 
#         _data['g041'] = (_data.vwap.diff(N1)).rolling(N2).max()
# 
#         _ret = _data.g041.unstack().T
#         _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja041" + str(w)]], names=["futures", "price"])
#         ret = pd.concat([ret, _ret], axis=1)
# =============================================================================

    for w in window:
        for name in futures_universe:
            N1, N2 = w
    
            _data = data.loc[name]
            _data['vwapw'] = (_data.openw + _data.closew + _data.highw + _data.loww) / 4
    
            _data['g041'] = (_data.vwapw.diff(N1)).rolling(N2).max()
    
            new_columns = pd.MultiIndex.from_product([[str(name)], ["gtja041edit" + str(w)]])  
            temp = _data['g041'].to_frame(name='g041').set_index(_data.index)  
            temp.columns = new_columns  
            ret = pd.concat([ret, temp], axis=1) 


    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret

gtja042_time_lst = [4, 6, 10, 15, 24, 36]

def GTJA042(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA042
    rank(std(high, 10)) * corr(high, volume, 10)
    
    :param data 默认旋转后，即 data_rotated
    :param window 是一个一维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return ret
    '''

    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]
    low = data.low[futures_universe]

    ret = pd.DataFrame()
# =============================================================================
#     for w in window:
#         N = w
# 
#         _data = data.loc[futures_universe]
#         _data['inner1'] = _data.highw.rolling(N).std()
#         _data['rank1'] = _data.inner1.unstack().rank(pct=True).stack()
#         _data['inner2'] = _data.highw.rolling(N).corr(_data.volume)
# 
#         _data['g042'] = _data.rank1 * _data.inner2
# 
#         _ret = _data.g042.unstack().T
#         _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja042" + str(w)]], names=["futures", "price"])
#         ret = pd.concat([ret, _ret], axis=1)
# =============================================================================

    for w in window:
        N = w

        _data = data.loc[futures_universe]
        _data['inner1'] = _data.highw.unstack().rolling(N,axis=1).std().stack()
        _data['rank1'] = _data.inner1.unstack().rank(pct=True).stack()
        _data['inner2'] = _data.highw.unstack().rolling(N,axis=1).corr(_data.volume99.unstack(),pairwise=True).stack()

        _data['g042'] = _data.rank1 * _data.inner2

        _ret = _data.g042.unstack().T
        _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja042" + str(w)]], names=["futures", "price"])
        ret = pd.concat([ret, _ret], axis=1)
        
    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret

# gtja043_time_lst = [2, 3, 6, 10, 15, 25]
gtja043_time_lst = [3, 6, 12, 24, 36, 48]

def GTJA043(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA043
    sum(close > delay(close, 1) ? volume : (close < delay(close, 1) ? -volume : 0), 6)
    
    :param data 默认旋转后，即 data_rotated
    :param window 是一个一维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return ret
    '''

    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]
    low = data.low[futures_universe]

    ret = pd.DataFrame()
# =============================================================================
#     for w in window:
#         N = w
# 
#         _data = data.loc[futures_universe]
#         _data['inner'] = 0
#         _data.loc[_data.closew > _data.closew.shift(1), "inner"] = _data.volume
#         _data.loc[_data.closew < _data.closew.shift(1), "inner"] = -_data.volume
# 
#         _data['g043'] = _data.inner.rolling(N).sum()
# 
#         _ret = _data.g043.unstack().T
#         _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja043" + str(w)]], names=["futures", "price"])
#         ret = pd.concat([ret, _ret], axis=1)
# =============================================================================

    for w in window:
        for name in futures_universe:
            N = w
    
            _data = data.loc[name]
            _data['inner'] = 0
            _data.loc[_data.closew > _data.closew.shift(1), "inner"] = _data.volume99
            _data.loc[_data.closew < _data.closew.shift(1), "inner"] = -_data.volume99
    
            _data['g043'] = _data.inner.rolling(N).sum()
    
            new_columns = pd.MultiIndex.from_product([[str(name)], ["gtja043" + str(w)]])  
            temp = _data['g043'].to_frame(name='g043').set_index(_data.index)  
            temp.columns = new_columns  
            ret = pd.concat([ret, temp], axis=1)  


    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret


def GTJA043EDIT(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA043
    sum(close > delay(close, 1) ? volume : (close < delay(close, 1) ? -volume : 0), 6)
    
    :param data 默认旋转后，即 data_rotated
    :param window 是一个一维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return ret
    '''

    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]
    low = data.low[futures_universe]

    ret = pd.DataFrame()
    # FACTOR_PATH = './factor_daily/GTJA043EDIT.csv'
    # if not os.path.exists(FACTOR_PATH):
    #     factor_df = pd.DataFrame()
    #     for w in window:
    #         for name in futures_universe:
    #             N = w
    #             _data = data.loc[name]
    #             _data['inner'] = (_data.closew > _data.closew.shift(1)) * _data.volume99 - (_data.closew < _data.closew.shift(1)) * _data.volume99
    #             _data['GTJA043EDIT'] = _data.inner.rolling(N).sum()
        
    #             new_columns = pd.MultiIndex.from_product([[str(name)], ["GTJA043EDIT_" + str(w)]])  
    #             temp = _data['GTJA043EDIT'].to_frame(name='GTJA043EDIT').set_index(_data.index)  
    #             temp.columns = new_columns  
    #             factor_df = pd.concat([factor_df, temp], axis=1)
    #     factor_df.to_csv(FACTOR_PATH)

    for w in window:
        for name in futures_universe:
            N = w
    
            _data = data.loc[name]
            _data['inner'] = (_data.closew > _data.closew.shift(1)) * _data.volume99 - (_data.closew < _data.closew.shift(1)) * _data.volume99
    
            _data['g043'] = _data.inner.rolling(N).sum()
    
            new_columns = pd.MultiIndex.from_product([[str(name)], ["gtja043" + str(w)]])  
            temp = _data['g043'].to_frame(name='g043').set_index(_data.index)  
            temp.columns = new_columns  
            ret = pd.concat([ret, temp], axis=1)  


    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret

gtja044_time_lst = [
    [3, 4, 4, 4, 1, 6, 6],
    [5, 6, 6, 6, 2, 8, 8],
    [10, 7, 6, 4, 3, 10, 15],
    [15, 10, 10, 10, 4, 15, 18],
    [20, 15, 15, 15, 5, 20, 20],
    [25, 20, 20, 20, 8, 25, 25]
]

def GTJA044(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA044
    tsrank(decaylinear(corr(low, mean(volume, 10), 7), 6), 4) + tsrank(decaylinear(delta(vwap, 3), 10), 15)
    
    :param data 默认旋转后，即 data_rotated
    :param window 是一个一维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return ret
    '''

    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]
    low = data.low[futures_universe]

    def tsrank(array):
        s = pd.Series(array)
        return np.array(s.rank(method="min", ascending=False))[-1]
    
    def wma(series, N):
        return series.rolling(N).apply(lambda x: x[:: -1].cumsum().sum() * 2 / N / (N + 1))

    ret = pd.DataFrame()
    for w in window:
        N1, N2, N3, N4, N5, N6, N7 = w

        _data = data.loc[futures_universe]
        _data['vwapw'] = (_data.openw + _data.closew + _data.highw + _data.loww) / 4

        _data['inner1'] = wma(_data.loww.rolling(N2).corr(_data.volume99.rolling(N1).mean()), N3)
        _data['rank1'] = _data.inner1.rolling(N4).apply(tsrank)

        _data['inner2'] = wma(_data.vwapw.shift(N5), N6)
        _data['rank2'] = _data.inner2.rolling(N7).apply(tsrank)

        _data['g044'] = _data.rank1 + _data.rank2

        _ret = _data.g044.unstack().T
        _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja044" + str(w)]], names=["futures", "price"])
        ret = pd.concat([ret, _ret], axis=1)


    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret

gtja045_time_lst = [
    [.2, 2, 50, 8],
    [.5, 2, 100, 10],
    [.6, 1, 150, 15],
    [.8, 1, 120, 20],
    [.5, 3, 100, 30],
    [.5, 5, 150, 40]
]

def GTJA045(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA045
    rank(delta(close * .6 + open * .4, 1)) * rank(corr(vwap, mean(volume, 150), 15))
    
    :param data 默认旋转后，即 data_rotated
    :param window 是一个一维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return ret
    '''

    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]
    low = data.low[futures_universe]

    ret = pd.DataFrame()
    for w in window:
        N1, N2, N3, N4 = w

        _data = data.loc[futures_universe]
        _data['vwapw'] = (_data.openw + _data.closew + _data.highw + _data.loww) / 4

        _data['inner1'] = (_data.closew * N1 + _data.openw * (1 - N1)).shift(N2)
        _data['rank1'] = _data.inner1.unstack().rank(pct=True).stack()

        _data['inner2'] = _data.vwapw.rolling(N4).corr(_data.volume99.rolling(N3).mean())
        _data['rank2'] = _data.inner2.unstack().rank(pct=True).stack()

        _data['g045'] = _data.rank1 * _data.rank2

        _ret = _data.g045.unstack().T
        _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja045" + str(w)]], names=["futures", "price"])
        ret = pd.concat([ret, _ret], axis=1)


    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret

gtja046_time_lst = [
    [1, 2, 4, 8],
    [2, 4, 8, 16],
    [3, 6, 12, 24],
    [4, 8, 16, 32],
    [6, 12, 24, 48],
    [8, 16, 32, 64]
]

def GTJA046(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA046
    (mean(close, 3) + mean(close, 6) + mean(close, 12) + mean(close, 24)) / close
    
    :param data 默认旋转后，即 data_rotated
    :param window 是一个一维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return ret
    '''

    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]
    low = data.low[futures_universe]

    ret = pd.DataFrame()
# =============================================================================
#     for w in window:
#         N1, N2, N3, N4 = w
# 
#         _data = data.loc[futures_universe]
#         _data['g046'] = (_data.closew.rolling(N1).mean() + _data.closew.rolling(N2).mean() + _data.closew.rolling(N3).mean() + _data.closew.rolling(N4).mean()) / _data.closew
# 
#         _ret = _data.g046.unstack().T
#         _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja046" + str(w)]], names=["futures", "price"])
#         ret = pd.concat([ret, _ret], axis=1)
# =============================================================================

    for w in window:
        for name in futures_universe:
            N1, N2, N3, N4 = w
    
            _data = data.loc[name]
            _data['g046'] = (_data.closew.rolling(N1).mean() + _data.closew.rolling(N2).mean() + _data.closew.rolling(N3).mean() + _data.closew.rolling(N4).mean()) / _data.closew
    
            new_columns = pd.MultiIndex.from_product([[str(name)], ["gtja046" + str(w)]])  
            temp = _data['g046'].to_frame(name='g046').set_index(_data.index)  
            temp.columns = new_columns  
            ret = pd.concat([ret, temp], axis=1) 


    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret

gtja047_time_lst = [
    [2, 1 / 4],
    [4, 1 / 6],
    [6, 1 / 9],
    [10, 1 / 12],
    [15, 1 / 6],
    [20, 1 / 6]
]

def GTJA047(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA047
    sma((tsmax(high, 6) - close) / (tsmax(high, 6) - tsmin(low, 6)), 9, 1)
    
    :param data 默认旋转后，即 data_rotated
    :param window 是一个一维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return ret
    '''

    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]
    low = data.low[futures_universe]

    ret = pd.DataFrame()
# =============================================================================
#     for w in window:
#         N, sma_alpha = w
# 
#         _data = data.loc[futures_universe]
#         _data['inner'] = (_data.highw.rolling(N).max() - _data.closew) / (_data.highw.rolling(N).max() - _data.loww.rolling(N).min())
# 
#         _data['g047'] = _data.inner.ewm(adjust=False, alpha=sma_alpha).mean()
#         _ret = _data.g047.unstack().T
#         _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja047" + str(w)]], names=["futures", "price"])
#         ret = pd.concat([ret, _ret], axis=1)
# =============================================================================
 
    for w in window:
        for name in futures_universe:
            N, sma_alpha = w
    
            _data = data.loc[name]
            _data['inner'] = (_data.highw.rolling(N).max() - _data.closew) / (_data.highw.rolling(N).max() - _data.loww.rolling(N).min())
    
            _data['g047'] = _data.inner.ewm(adjust=False, alpha=sma_alpha).mean()
            
            new_columns = pd.MultiIndex.from_product([[str(name)], ["gtja047" + str(w)]])  
            temp = _data['g047'].to_frame(name='g047').set_index(_data.index)  
            temp.columns = new_columns  
            ret = pd.concat([ret, temp], axis=1)     
 

    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret


# =============================================================================
# gtja047_time_lst = [
#     [2, 1 / 4],
#     [4, 1 / 6],
#     [6, 1 / 9],
#     [10, 1 / 12],
#     [15, 1 / 6],
#     [20, 1 / 6]
# ]
# =============================================================================

# =============================================================================
# def GTJA047(data, window, futures_universe, time_freq='day'):
#     '''
#     计算 GTJA047
#     sma((tsmax(high, 6) - close) / (tsmax(high, 6) - tsmin(low, 6)), 9, 1)
#     
#     :param data 默认旋转后，即 data_rotated
#     :param window 是一个一维数组
#     :param futures_universe 是计算 rank 的范围。默认是 gu24
#     :return ret
#     '''
# 
#     
#     volume = data.volume99[futures_universe]
#     open_price = data.open[futures_universe]
#     openw = data.openw[futures_universe]
#     high = data.high[futures_universe]
#     highw = data.highw[futures_universe]
#     low = data.low[futures_universe]
#     loww = data.loww[futures_universe]
#     close = data.close[futures_universe]
#     closew = data.closew[futures_universe]
#     low = data.low[futures_universe]
# 
#     ret = pd.DataFrame()
# # =============================================================================
# #     for w in window:
# #         N, sma_alpha = w
# # 
# #         _data = data.loc[futures_universe]
# #         _data['inner'] = (_data.highw.rolling(N).max() - _data.closew) / (_data.highw.rolling(N).max() - _data.loww.rolling(N).min())
# # 
# #         _data['g047'] = _data.inner.ewm(adjust=False, alpha=sma_alpha).mean()
# #         _ret = _data.g047.unstack().T
# #         _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja047" + str(w)]], names=["futures", "price"])
# #         ret = pd.concat([ret, _ret], axis=1)
# # =============================================================================
# 
#     for w in window:
#         for name in futures_universe:
#             N, sma_alpha = w
#     
#             _data = data.loc[name]
#             _data['inner'] = (_data.highw.rolling(N).max() - _data.closew) / (_data.highw.rolling(N).max() - _data.loww.rolling(N).min())
#     
#             _data['g047'] = _data.inner.ewm(adjust=False, alpha=sma_alpha).mean()
#             new_columns = pd.MultiIndex.from_product([[str(name)], ["gtja047" + str(w)]])  
#             temp = _data['g047'].to_frame(name='g047').set_index(_data.index)  
#             temp.columns = new_columns  
#             ret = pd.concat([ret, temp], axis=1) 
#         
#     # 把 open 和 openw 贴进去
#     _open = open_price.unstack().T
#     _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
#     _openw = openw.unstack().T
#     _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
#     ret = pd.concat([ret, _open, _openw], axis=1)
#     return ret
# =============================================================================
gtja048_time_lst = [
    [2,8],
    [4, 16],
    [5, 20],
    [6, 24],
    [8, 32],
    [10, 40]
]
def GTJA048(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA048
    rank((sign(delta(close, 1)) + sign(delay(close, 1) - delay(close, 2)) + sign(delay(close, 2) - delay(close, 3))) * sum(volume, 5)) / sum(volume, 20)
    :param df 是带有原始数据的 DataFrame
    :param universe 需要计算因子值的股票, string 的 list
    :param start_date 开始计算日期
    :param end_date 结束计算日期
    :return: 因子值, index 为日期, column 为股票代码
    '''
    
    
    
    
    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]
    low = data.low[futures_universe]

    ret = pd.DataFrame()
    for w in window:
        for name in futures_universe:
            N1, N2 = w
    
            _data = data.loc[name]
            _data['inner'] = (np.sign(_data.close - _data.close.shift(1)) + np.sign(_data.close.shift(1) - _data.close.shift(2)) + np.sign(_data.close.shift(2) - _data.close.shift(3))) * _data.volume99.rolling(N1).sum()
    
            _data['g048'] = _data.inner / _data.volume99.rolling(N2).sum()
            
            new_columns = pd.MultiIndex.from_product([[str(name)], ["gtja048" + str(w)]])  
            temp = _data['g048'].to_frame(name='g048').set_index(_data.index)  
            temp.columns = new_columns  
            ret = pd.concat([ret, temp], axis=1) 

# =============================================================================
#     for w in window:
#         N1, N2 = w
# 
#         _data = data.loc[futures_universe]
#         _data['inner'] = (np.sign(_data.close - _data.close.shift(1)) + np.sign(_data.close.shift(1) - _data.close.shift(2)) + np.sign(_data.close.shift(2) - _data.close.shift(3))) * _data.volume.rolling(N1).sum()
#         _data['rank1'] = _data.inner.unstack().rank(pct=True).stack()
# 
#         _data['g048'] = _data.rank1 / _data.volume.rolling(N2).sum()
#         
#         _ret = _data.g048.unstack().T
#         _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja048" + str(w)]], names=["futures", "price"])
#         ret = pd.concat([ret, _ret], axis=1)
# =============================================================================
        
    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    print('gtja048 value are calculated and saved successfully!')
    return ret
       
    
    
gtja049_time_lst = [4, 6, 12, 20, 30, 50]
def GTJA049(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA049
    sum(high + low >= delay(high, 1) + delay(low, 1) ? 0 : max(abs(delta(high, 1)), abs(delta(low, 1))), 12) / (sum(high + low >= delay(high, 1) + delay(low, 1) ? 0 : max(abs(delta(high, 1)), abs(delta(low, 1))), 12) + sum(high + low <= delay(high, 1) + delay(low, 1) ? 0 : max(abs(delta(high, 1)), abs(delta(low, 1))), 12))
    :param df 是带有原始数据的 DataFrame
    :param universe 需要计算因子值的股票, string 的 list
    :param start_date 开始计算日期
    :param end_date 结束计算日期
    :return: 因子值, index 为日期, column 为股票代码
    '''
    
    
    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]
    low = data.low[futures_universe]

    ret = pd.DataFrame()
    for w in window:
        for name in futures_universe:
            N = w
    
            _data = data.loc[name]
            _data['high_diff'] = np.abs(_data.high.diff(1))
            _data['low_diff'] = np.abs(_data.low.diff(1))
            _data['inner_price'] = _data[['high_diff', 'low_diff']].max(axis=1)
            
            _data['cond1'] = _data.high + _data.low >= _data.high.shift(1) + _data.low.shift(1)
            _data['cond2'] = _data.high + _data.low <= _data.high.shift(1) + _data.low.shift(1)
    
            _data['inner1'] = _data.inner_price
            _data.loc[_data.cond1, "inner1"] = 0
    
            _data['inner2'] = _data.inner_price
            _data.loc[_data.cond2, "inner2"] = 0
    
            _data['g049'] = _data.inner1.rolling(N).sum() / (_data.inner1.rolling(N).sum() + _data.inner2.rolling(N).sum())
            
            new_columns = pd.MultiIndex.from_product([[str(name)], ["gtja049" + str(w)]])  
            temp = _data['g049'].to_frame(name='g049').set_index(_data.index)  
            temp.columns = new_columns  
            ret = pd.concat([ret, temp], axis=1)

# =============================================================================
#     for w in window:
#         N = w
# 
#         _data = data.loc[futures_universe]
#         _data['high_diff'] = np.abs(_data.high.diff(1))
#         _data['low_diff'] = np.abs(_data.low.diff(1))
#         _data['inner_price'] = _data[['high_diff', 'low_diff']].max(axis=1)
#         
#         _data['cond1'] = _data.high + _data.low >= _data.high.shift(1) + _data.low.shift(1)
#         _data['cond2'] = _data.high + _data.low <= _data.high.shift(1) + _data.low.shift(1)
# 
#         _data['inner1'] = _data.inner_price
#         _data.loc[_data.cond1, "inner1"] = 0
# 
#         _data['inner2'] = _data.inner_price
#         _data.loc[_data.cond2, "inner2"] = 0
# 
#         _data['g049'] = _data.inner1.rolling(N).sum() / (_data.inner1.rolling(N).sum() + _data.inner2.rolling(N).sum())
#         _ret = _data.g049.unstack().T
#         _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja049" + str(w)]], names=["futures", "price"])
#         ret = pd.concat([ret, _ret], axis=1)
# =============================================================================
    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret
    
    

# gtja052_time_lst = [7, 12, 26, 44, 68, 100]
gtja052_time_lst = [3, 5, 10, 18, 27, 44]
   
def GTJA052(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA052
    sum(max(0, high - delay((high + low + close) / 3, 1)), 26) / sum(max(0, delay((high + low + close) / 3, 1) - low), 26)
    :param df 是带有原始数据的 DataFrame
    :param universe 需要计算因子值的股票, string 的 list
    :param start_date 开始计算日期
    :param end_date 结束计算日期
    :return: 因子值, index 为日期, column 为股票代码
    '''

    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]

    ret = pd.DataFrame()
    for w in window:
        for name in futures_universe:
            N = w
    
            _data = data.loc[name]
            _data['const'] = 0
            # 更改为含贴水，zwh 2024.6.4
            _data['inner11'] = _data.highw - ((_data.highw + _data.loww +_data.closew) / 3).shift(1)
            _data['inner21'] = ((_data.highw + _data.loww + _data.closew) / 3).shift(1) - _data.loww
    
            _data['inner1'] = _data[['const', 'inner11']].max(axis=1)
            _data['inner2'] = _data[['const', 'inner21']].max(axis=1)
    
            _data['g052'] = _data.inner1.rolling(N).mean() / _data.inner2.rolling(N).mean()
            
            new_columns = pd.MultiIndex.from_product([[str(name)], ["gtja052" + str(w)]])  
            temp = _data['g052'].to_frame(name='g052').set_index(_data.index)  
            temp.columns = new_columns  
            ret = pd.concat([ret, temp], axis=1)

    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret



gtja053_time_lst = [4, 6, 12, 20, 30, 50] 
def GTJA053(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA053
    count(close > delay(close, 1), 12) / 12
    :param df 是带有原始数据的 DataFrame
    :param universe 需要计算因子值的股票, string 的 list
    :param start_date 开始计算日期
    :param end_date 结束计算日期
    :return: 因子值, index 为日期, column 为股票代码
    '''
    
    
    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]
    low = data.low[futures_universe]

    ret = pd.DataFrame()
    for w in window:
        for name in futures_universe:
            N = w
    
            _data = data.loc[name]
            _data ['inner'] = _data.closew > _data.closew.shift(1)
    
            _data ['g053'] = _data.inner.rolling(N).sum()
            
            new_columns = pd.MultiIndex.from_product([[str(name)], ["gtja053" + str(w)]])  
            temp = _data['g053'].to_frame(name='g053').set_index(_data.index)  
            temp.columns = new_columns  
            ret = pd.concat([ret, temp], axis=1)

# =============================================================================
#     for w in window:
#         N = w
# 
#         _data = data.loc[futures_universe]
#         _data ['inner'] = _data .close > _data .close.shift(1)
# 
#         _data ['g053'] = _data .inner.rolling(N).sum()
#         _ret = _data.g053.unstack().T
#         _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja053" + str(w)]], names=["futures", "price"])
#         ret = pd.concat([ret, _ret], axis=1)
# =============================================================================
    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret


# gtja054_time_lst = [3, 5, 10, 18, 27, 44] 
gtja054_time_lst = [5, 10, 15, 20, 25, 30]
def GTJA054(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA054
    rank(std(abs(close - open)) + (close - open) + corr(close, open, 10))
    :param df 是带有原始数据的 DataFrame
    :param universe 需要计算因子值的股票, string 的 list
    :param start_date 开始计算日期
    :param end_date 结束计算日期
    :return: 因子值, index 为日期, column 为股票代码
    '''

    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]


    ret = pd.DataFrame()
    for w in window:
        for name in futures_universe:
            N= w
    
            _data = data.loc[name]
           
            # 更改为贴水修正后的数据  zwh 6.12
            _data['inner'] = (_data.closew - _data.openw)
    
            _data['g054'] = _data.inner.rolling(N).sum()
            
            new_columns = pd.MultiIndex.from_product([[str(name)], ["gtja054" + str(w)]])  
            temp = _data['g054'].to_frame(name='g054').set_index(_data.index)  
            temp.columns = new_columns  
            ret = pd.concat([ret, temp], axis=1) 

    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret

    
gtja056_time_lst = [[6, 19, 40, 13],
                     [12, 14, 40, 13], 
                     [12, 19, 40, 13], 
                     [12, 26, 40, 13], 
                     [12, 19, 60, 13], 
                     [12, 19, 40, 17]] 
def GTJA056(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA056
    rank(open - tsmin(open, 12)) < rank(rank(corr(sum((high + low) / 2, 19), sum(mean(volume, 40), 19), 13) ^ 5))
    
    :param df 是带有原始数据的 DataFrame
    :param universe 需要计算因子值的股票, string 的 list
    :param start_date 开始计算日期
    :param end_date 结束计算日期
    :return: 因子值, index 为日期, column 为股票代码
    '''
    
    
    
    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]
    low = data.low[futures_universe]

    ret = pd.DataFrame()
    for w in window:
        for name in futures_universe:
            N1, N2, N3, N4 = w
    
            _data = data.loc[name]
           
    
            _data['inner1'] = _data.openw - _data.openw.shift(N1).min()
            # _data['rank1'] = _data.inner1.unstack().rank(pct=True).stack()
    
            _data['inner21'] = ((_data.high + _data.low) / 2).rolling(N2).sum()
            _data['inner22'] = (_data.volume99.rolling(N3).mean()).rolling(N2).sum()
            _data['inner2'] = _data.inner21.rolling(N4).corr(_data.inner22)*(_data.openw - _data.openw.shift(N1))
            # _data['rank2'] = _data.inner2.unstack().rank(pct=True).stack()
    
            _data['g056'] = _data.inner1 - _data.inner2
            new_columns = pd.MultiIndex.from_product([[str(name)], ["gtja056" + str(w)]])  
            temp = _data['g056'].to_frame(name='g056').set_index(_data.index)  
            temp.columns = new_columns  
            ret = pd.concat([ret, temp], axis=1)  

# =============================================================================
#     for w in window:
#         N1, N2, N3, N4 = w
# 
#         _data = data.loc[futures_universe]
#        
# 
#         _data['inner1'] = _data.open - _data.open.shift(N1).min()
#         _data['rank1'] = _data.inner1.unstack().rank(pct=True).stack()
# 
#         _data['inner21'] = ((_data.high + _data.low) / 2).rolling(N2).sum()
#         _data['inner22'] = (_data.volume.rolling(N3).mean()).rolling(N2).sum()
#         _data['inner2'] = _data.inner21.rolling(N4).corr(_data.inner22)
#         _data['rank2'] = _data.inner2.unstack().rank(pct=True).stack()
# 
#         _data['g056'] = _data.rank1 - _data.rank2
#         _ret = _data.g056.unstack().T
#         _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja056" + str(w)]], names=["futures", "price"])
#         ret = pd.concat([ret, _ret], axis=1)
# =============================================================================
    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret


gtja057_time_lst = [
    [4, 1 / 2],
    [6, 1 / 3],
    [9, 1 / 3],
    [15, 1 / 3],
    [15, 1 / 6],
    [20, 1 / 6]
]
def GTJA057(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA057
    sma((close - tsmin(low, 9)) / (tsmax(high, 9) - tsmin(low, 9)), 3, 1)
    
    :param df 是带有原始数据的 DataFrame
    :param universe 需要计算因子值的股票, string 的 list
    :param start_date 开始计算日期
    :param end_date 结束计算日期
    :return: 因子值, index 为日期, column 为股票代码
    '''
    
    
    
    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]
    low = data.low[futures_universe]

    ret = pd.DataFrame()
    for w in window:
        for name in futures_universe:
            N,sma_alpha= w
    
            _data = data.loc[name]
           
    
            _data['inner'] = (_data.close - _data.low.rolling(N).min()) / (_data.high.rolling(N).max() - _data.low.rolling(N).min())
    
            _data['g057'] = _data.inner.ewm(adjust=False, alpha=sma_alpha).mean()
            new_columns = pd.MultiIndex.from_product([[str(name)], ["gtja057" + str(w)]])  
            temp = _data['g057'].to_frame(name='g057').set_index(_data.index)  
            temp.columns = new_columns  
            ret = pd.concat([ret, temp], axis=1)
# =============================================================================
# 
#     for w in window:
#         N,sma_alpha= w
# 
#         _data = data.loc[futures_universe]
#        
# 
#         _data['inner'] = (_data.close - _data.low.rolling(N).min()) / (_data.high.rolling(N).max() - _data.low.rolling(N).min())
# 
#         _data['g057'] = _data.inner.ewm(adjust=False, alpha=sma_alpha).mean()
#         _ret = _data.g057.unstack().T
#         _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja057" + str(w)]], names=["futures", "price"])
#         ret = pd.concat([ret, _ret], axis=1)
# =============================================================================
    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret


# gtja058_time_lst = [4, 12, 20, 32, 48, 60]
gtja058_time_lst = [3, 4, 5, 8, 12, 15]

def GTJA058(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA058
    count(close > delay(close, 1), 20)
    
    :param df 是带有原始数据的 DataFrame
    :param universe 需要计算因子值的股票, string 的 list
    :param start_date 开始计算日期
    :param end_date 结束计算日期
    :return: 因子值, index 为日期, column 为股票代码
    '''
    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]
    low = data.low[futures_universe]

    ret = pd.DataFrame()
    # FACTOR_PATH = './factor_daily/GTJA058.csv'
    # if not os.path.exists(FACTOR_PATH):
    #     factor_df = pd.DataFrame()
    #     for w in window:
    #         for name in futures_universe:
    #             N = w
    #             _data = data.loc[name]
    #             _data['inner'] =  _data.closew > _data.closew.shift(1)
    #             _data['GTJA058'] = _data.inner.rolling(N).sum()
        
    #             new_columns = pd.MultiIndex.from_product([[str(name)], ["GTJA058_" + str(w)]])  
    #             temp = _data['GTJA058'].to_frame(name='GTJA058').set_index(_data.index)  
    #             temp.columns = new_columns  
    #             factor_df = pd.concat([factor_df, temp], axis=1)
    #     factor_df.to_csv(FACTOR_PATH)

    for w in window:
        for name in futures_universe:
            N= w
    
            _data = data.loc[name]
           
            invalid_range = _data.closew.isna()

            # 更改为贴水后的价格  zwh 6.5
            _data['inner'] = (_data.closew > _data.closew.shift(1)).astype(int)
            _data[invalid_range] = np.nan

            _data['g058'] = _data.inner.rolling(N).sum()
            new_columns = pd.MultiIndex.from_product([[str(name)], ["gtja058" + str(w)]])  
            temp = _data['g058'].to_frame(name='g058').set_index(_data.index)  
            temp.columns = new_columns  
            ret = pd.concat([ret, temp], axis=1) 
         
# =============================================================================
#     for w in window:
#         N= w
# 
#         _data = data.loc[futures_universe]
#        
# 
#         _data['inner'] = _data.close > _data.close.shift(1)
#         
#         _data['g058'] = _data.inner.rolling(N).sum()
#         _ret = _data.g058.unstack().T
#         _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja058" + str(w)]], names=["futures", "price"])
#         ret = pd.concat([ret, _ret], axis=1)
# =============================================================================
    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret


gtja059_time_lst = [4, 12, 20, 32, 48, 60]
def GTJA059(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA059
    sum(close == delay(close, 1) ? 0 : close - (close > delay(close, 1) ? min(low, delay(close, 1)) : max(high, delay(close, 1))), 20)

    :param df 是带有原始数据的 DataFrame
    :param universe 需要计算因子值的股票, string 的 list
    :param start_date 开始计算日期
    :param end_date 结束计算日期
    :return: 因子值, index 为日期, column 为股票代码
    '''
    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]
    low = data.low[futures_universe]

    ret = pd.DataFrame()
    for w in window:
        for name in futures_universe:
            N= w
    
            _data = data.loc[name]
           
    
            _data['inner'] = 0.
            _data['close_1'] = _data.closew.shift(1)
            _data.loc[_data.closew > _data.close_1, "inner"] = _data[['loww', 'close_1']].min(axis=1)
            _data.loc[_data.closew < _data.close_1, "inner"] = _data[['highw', 'close_1']].max(axis=1)
            # _data.loc[_data.close < _data.close_1, "inner"] = _data[['highw', 'close_1']].min(axis=1)
            
            _data['g059'] = _data.inner.rolling(N).sum()
            new_columns = pd.MultiIndex.from_product([[str(name)], ["gtja059" + str(w)]])  
            temp = _data['g059'].to_frame(name='g059').set_index(_data.index)  
            temp.columns = new_columns  
            ret = pd.concat([ret, temp], axis=1) 
# =============================================================================
#     for w in window:
#         N= w
# 
#         _data = data.loc[futures_universe]
#        
# 
#         _data['inner'] = 0.
#         _data['close_1'] = _data.close.shift(1)
#         _data.loc[_data.close > _data.close_1, "inner"] = _data[['low', 'close_1']].min(axis=1)
#         _data.loc[_data.close < _data.close_1, "inner"] = _data[['high', 'close_1']].min(axis=1)
#         
#         _data['g059'] = _data.inner.unstack().rolling(N,axis=1).sum().stack()
#         _ret = _data.g059.unstack().T
#         _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja059" + str(w)]], names=["futures", "price"])
#         ret = pd.concat([ret, _ret], axis=1)
# =============================================================================
# =============================================================================
#     for w in window:
#         N= w
# 
#         _data = data.loc[futures_universe]
#        
# 
#         _data['inner'] = 0.
#         _data['close_1'] = _data.close.shift(1)
#         _data.loc[_data.close > _data.close_1, "inner"] = _data[['low', 'close_1']].min(axis=1)
#         _data.loc[_data.close < _data.close_1, "inner"] = _data[['high', 'close_1']].min(axis=1)
#         
#         _data['g059'] = _data.inner.rolling(N).sum()
#         _ret = _data.g059.unstack().T
#         _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja059" + str(w)]], names=["futures", "price"])
#         ret = pd.concat([ret, _ret], axis=1)
# =============================================================================
    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret
    

# gtja060_time_lst = [4, 12, 25, 32, 48, 60]  
# gtja060_time_lst = [2, 4, 6, 9, 12, 18]
gtja060_time_lst = [3, 4, 5, 8, 12, 15]
# gtja060_time_lst = [2, 3, 4, 6, 12, 9]
    
def GTJA060(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA060
    sum(((close - low) / (high - close)) / (high - low) * volume, 20)

    :param df 是带有原始数据的 DataFrame
    :param universe 需要计算因子值的股票, string 的 list
    :param start_date 开始计算日期
    :param end_date 结束计算日期
    :return: 因子值, index 为日期, column 为股票代码
    '''
    
    
    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]
    low = data.low[futures_universe]

    ret = pd.DataFrame()
    for w in window:
        for name in futures_universe:
            N= w
    
            _data = data.loc[name]
            def custom_operation(value):
                # CHANGE
                # result = (((value['closew'] - value['loww']) / (value['highw'] - value['closew'] + 1e-6)) / (
                #             value['highw'] - value['loww'] + 1e-6) * value['volume99'])
                result = ((value['closew'] - value['loww']) * value['volume99'])

                return result
            _data['inner']=_data.apply(custom_operation, axis=1)  
            _data['g060'] = (_data.inner.rolling(N).sum())
            new_columns = pd.MultiIndex.from_product([[str(name)], ["gtja060" + str(w)]])  
            temp = _data['g060'].to_frame(name='g060').set_index(_data.index)  
            temp.columns = new_columns  
            ret = pd.concat([ret, temp], axis=1) 

    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret


# gtja060edit_time_lst = [4, 12, 25, 32, 48, 60]  
gtja060edit_time_lst = [3, 5, 10, 18, 27, 44]

def GTJA060EDIT(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA060
    sum(((close - low) / (high - close)) / (high - low) /(open-low)* volume, 20)

    :param df 是带有原始数据的 DataFrame
    :param universe 需要计算因子值的股票, string 的 list
    :param start_date 开始计算日期
    :param end_date 结束计算日期
    :return: 因子值, index 为日期, column 为股票代码
    '''
    
    
    settlement = data.settlementw[futures_universe]
    volume = data.volume99[futures_universe]
    _price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]
    low = data.low[futures_universe]

    ret = pd.DataFrame()
    for w in window:
        N= w

        _data = data.loc[futures_universe]
        def custom_operation(value):  
            result = (((value['closew'] - value['loww']) / (value['highw'] - value['closew'] + 1e-6)) / (value['highw'] - value['loww'] + 1e-6)/(value['openw'] - value['loww'] + 1e-6) * value['volume99'])
            if abs(result) < 1e8:  
                return result  
            # if abs(value['high'] - value['close']) > 1e-8 and abs(value['high'] - value['low']) > 1e-8:  
            #     return result  
            else:  
                return 0  
        _data['inner']=_data.apply(custom_operation, axis=1)  
        # _data['inner'] = ((((_data.close - _data.low) / (_data.high - _data.close+1e-6)) / (_data.high - _data.low+1e-6) * _data.volume))
        # _data['inner'] = ((((_data.close - _data.low) / (_data.high - _data.close)) / (_data.high - _data.low) * _data.volume))
        da=_data.unstack(level='future')
        names=da.columns.get_level_values(1)
        for name in names:  
            da[('g060',name)] = (da['inner'][name].rolling(N).sum())
        da=da.stack(["future"]).swaplevel(axis=0).sort_index()
        _data['g060']=da['g060']
        # _data['g060'] = (_data.inner.rolling(N).sum())
        _ret = _data.g060.unstack().T
        _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja060edit" + str(w)]], names=["futures", "price"])
        ret = pd.concat([ret, _ret], axis=1)

    # 把 open 和 openw 贴进去
    _open = _price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret


gtja060edit2_time_lst = [2, 4, 6, 9, 12, 18]
    
def GTJA060EDIT2(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA060 除去high-low
    sum(((close - low) / (high - close)) / (high - low) , 20)

    :param df 是带有原始数据的 DataFrame
    :param universe 需要计算因子值的股票, string 的 list
    :param start_date 开始计算日期
    :param end_date 结束计算日期
    :return: 因子值, index 为日期, column 为股票代码
    '''
    
    
    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]
    low = data.low[futures_universe]

    ret = pd.DataFrame()
    for w in window:
        for name in futures_universe:
            N= w
    
            _data = data.loc[name]
            def custom_operation(value):  
                result = (((value['closew'] - value['loww']) / (value['highw'] - value['closew'] + 1e-6)) * value['volume99'])
                if abs(result) < 1e8:  
                    return result  
                # if abs(value['high'] - value['close']) > 1e-8 and abs(value['high'] - value['low']) > 1e-8:  
                #     return result  
                else:  
                    return 0  
            _data['inner']=_data.apply(custom_operation, axis=1)  
            # _data['inner'] = ((((_data.close - _data.low) / (_data.high - _data.close+1e-6)) / (_data.high - _data.low+1e-6) * _data.volume))
            # _data['inner'] = ((((_data.close - _data.low) / (_data.high - _data.close)) / (_data.high - _data.low) * _data.volume))
            _data['g060'] = (_data.inner.rolling(N).sum())
            new_columns = pd.MultiIndex.from_product([[str(name)], ["gtja060edit2" + str(w)]])  
            temp = _data['g060'].to_frame(name='g060').set_index(_data.index)  
            temp.columns = new_columns  
            ret = pd.concat([ret, temp], axis=1) 

    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret

# =============================================================================
# gtja061_time_lst = [
#     [12, 40, 8, 17],
#     [6, 80, 16, 17],
#     [12, 80, 8, 17],
#     [18, 80, 8, 17],
#     [12, 80, 8, 34],
#     [24, 80, 16, 34]
# ]  
# =============================================================================
gtja061_time_lst = [
    [6, 40, 8],
    [6, 80, 8],
    [12, 80, 8],
    [18, 80, 8],
    [12, 80, 16],
    [24, 80, 16]
] 
def GTJA061(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA061
    max(rank(decaylinear(delta(vwap, 1), 12)), rank(decaylinear(rank(corr(low, mean(volume, 80), 8)), 17)))

    :param df 是带有原始数据的 DataFrame
    :param universe 需要计算因子值的股票, string 的 list
    :param start_date 开始计算日期
    :param end_date 结束计算日期
    :return: 因子值, index 为日期, column 为股票代码
    '''
    
    
    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]
    low = data.low[futures_universe]

    def wma(series, N):
        return series.rolling(N).apply(lambda x: x[:: -1].cumsum().sum() * 2 / N / (N + 1))

    ret = pd.DataFrame()
    for w in window:
        for name in futures_universe:
            N1, N2, N3 = w
    
            _data = data.loc[name]
            _data['vwapw'] = (_data.openw + _data.closew + _data.highw + _data.loww) / 4
            _data['inner1'] = wma(_data.vwapw.diff(1), N1)
            # _data['rank1'] = _data.inner1.unstack().rank(pct=True).stack()
    
            _data['inner21'] = _data.low.rolling(N3).corr(_data.volume99.rolling(N2).mean())* _data['inner1']
            # _data['rank21'] = _data.inner21.unstack().rank(pct=True).stack()
            # _data['inner2'] = wma(_data.rank21, N4)
            # _data['rank2'] = _data.inner2.unstack().rank(pct=True).stack()
            
            _data['g061'] = _data.inner1 - _data.inner21
    
            new_columns = pd.MultiIndex.from_product([[str(name)], ["gtja061" + str(w)]])  
            temp = _data['g061'].to_frame(name='g061').set_index(_data.index)  
            temp.columns = new_columns  
            ret = pd.concat([ret, temp], axis=1) 
# =============================================================================
#     for w in window:
#         N1, N2, N3, N4 = w
# 
#         _data = data.loc[futures_universe]
#         _data['vwapw'] = (_data.openw + _data.closew + _data.highw + _data.loww) / 4
#         _data['inner1'] = wma(_data.vwapw.diff(1), N1)
#         _data['rank1'] = _data.inner1.unstack().rank(pct=True).stack()
# 
#         _data['inner21'] = _data.low.rolling(N3).corr(_data.volume.rolling(N2).mean())
#         _data['rank21'] = _data.inner21.unstack().rank(pct=True).stack()
#         _data['inner2'] = wma(_data.rank21, N4)
#         _data['rank2'] = _data.inner2.unstack().rank(pct=True).stack()
#         
#         _data['g061'] = _data[['rank1', 'rank2']].max(axis=1)
# 
#         _ret = _data.g061.unstack().T
#         _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja061" + str(w)]], names=["futures", "price"])
#         ret = pd.concat([ret, _ret], axis=1)
# =============================================================================
    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret


gtja062_time_lst = [2, 3, 5, 8, 12, 15]
def GTJA062(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA062
    corr(high, rank(volume), 5)

    :param df 是带有原始数据的 DataFrame
    :param universe 需要计算因子值的股票, string 的 list
    :param start_date 开始计算日期
    :param end_date 结束计算日期
    :return: 因子值, index 为日期, column 为股票代码
    '''
    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]
    low = data.low[futures_universe]



    ret = pd.DataFrame()
    for w in window:
        for name in futures_universe:
            N = w
    
            _data = data.loc[name]
            _data['volume'] = _data.volume99
            # _data['rank1'] = _data.volume.unstack().rank(pct=True).stack()
            
            _data['g062'] = _data.highw.rolling(N).corr(_data.volume99)
    
            new_columns = pd.MultiIndex.from_product([[str(name)], ["gtja062" + str(w)]])  
            temp = _data['g062'].to_frame(name='g062').set_index(_data.index)  
            temp.columns = new_columns  
            ret = pd.concat([ret, temp], axis=1) 
# =============================================================================
#     for w in window:
#         N = w
# 
#         _data = data.loc[futures_universe]
#         _data['rank1'] = _data.volume.unstack().rank(pct=True).stack()
#         
#         _data['g062'] = _data.high.rolling(N).corr(_data.rank1)
# 
#         _ret = _data.g062.unstack().T
#         _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja062" + str(w)]], names=["futures", "price"])
#         ret = pd.concat([ret, _ret], axis=1)
# =============================================================================

    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret


gtja063_time_lst = [
    [1, 1 / 6],
    [2, 1 / 6],
    [2, 1 / 12],
    [3, 1 / 6],
    [3, 1 / 12],
    [3, 1 / 24]
]

def GTJA063(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA063
    sma(max(delta(close, 1), 0), 6, 1) / sma(abs(delta(close, 1)), 6, 1)

    :param df 是带有原始数据的 DataFrame
    :param universe 需要计算因子值的股票, string 的 list
    :param start_date 开始计算日期
    :param end_date 结束计算日期
    :return: 因子值, index 为日期, column 为股票代码
    '''
    
    # 
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]
    low = data.low[futures_universe]


    ret = pd.DataFrame()
    for w in window:
        for name in futures_universe:
            N,sma_alpha = w
    
            _data = data.loc[name]
            _data['close_delta'] = _data.closew.diff(N)
            _data['const'] = 0.
            
            _data['inner1'] = _data[['const', 'close_delta']].max(axis=1)
            _data['inner2'] = np.abs(_data.close_delta)            
   
            _data['g063'] = _data.inner1.ewm(adjust=False, alpha=sma_alpha).mean() / (_data.inner2.ewm(adjust=False, alpha=sma_alpha).mean()+0.0000000001)
    
            new_columns = pd.MultiIndex.from_product([[str(name)], ["gtja063" + str(w)]])  
            temp = _data['g063'].to_frame(name='g063').set_index(_data.index)  
            temp.columns = new_columns  
            ret = pd.concat([ret, temp], axis=1)

# =============================================================================
#     for w in window:
#         N,sma_alpha = w
# 
#         _data = data.loc[futures_universe]
#     
#         _data['close_delta'] = _data.closew.unstack().diff(N,axis=1).stack()
#         _data['const'] = 0.
#         
#         _data['inner1'] = _data[['const', 'close_delta']].max(axis=1)
#         _data['inner2'] = np.abs(_data.close_delta)
#         
#         _data['g063'] = _data.inner1.ewm(adjust=False, alpha=sma_alpha).mean() / (_data.inner2.ewm(adjust=False, alpha=sma_alpha).mean()+0.0000000001)
# 
#         _ret = _data.g063.unstack().T
#         _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja063" + str(w)]], names=["futures", "price"])
#         ret = pd.concat([ret, _ret], axis=1)
# =============================================================================
        

    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret
    # sma_alpha = 1. / 6
    # data = df.loc[universe]

    # data['close_delta'] = data.close.diff(1)
    # data['const'] = 0.
    
    # data['inner1'] = data[['const', 'close_delta']].max(axis=1)
    # data['inner2'] = np.abs(data.close_delta)
    
    # data['g063'] = data.inner1.ewm(adjust=False, alpha=sma_alpha).mean() / data.inner2.ewm(adjust=False, alpha=sma_alpha).mean()
    # return data.g063.unstack().T[start_date: end_date]

gtja064_time_lst = [2, 3, 4, 6, 12, 9]
# gtja064_time_lst = [1, 2, 6, 11, 21, 31]

def GTJA064(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA064 原公式后项有问题，已修改
    decaylinear(corr(rank(vwap), rank(volume), 4), 4)

    :param df 是带有原始数据的 DataFrame
    :param universe 需要计算因子值的股票, string 的 list
    :param start_date 开始计算日期
    :param end_date 结束计算日期
    :return: 因子值, index 为日期, column 为股票代码
    '''
    
    
    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]
    low = data.low[futures_universe]



    ret = pd.DataFrame()
    for w in window:
        N = w

        _data = data.loc[futures_universe]
        

        def wma(series, N):
            return series.rolling(N).apply(lambda x: x[:: -1].cumsum().sum() * 2 / N / (N + 1))
        _data['vwapw'] = (_data.openw + _data.closew + _data.highw + _data.loww) / 4
        _data['rank1'] = _data.vwapw.unstack().rank(pct=True).stack()
        _data['rank2'] = _data.volume99.unstack().rank(pct=True).stack()

        _data['inner'] = _data.rank1.rolling(N).corr(_data.rank2)
        
        _data['g064'] = wma(_data.inner, N)

        _ret = _data.g064.unstack().T
        _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja064" + str(w)]], names=["futures", "price"])
        ret = pd.concat([ret, _ret], axis=1)
    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret


gtja065_time_lst = [2,  4, 6, 12, 9,18]   
def GTJA065(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA065
    mean(close, 6) / close

    :param df 是带有原始数据的 DataFrame
    :param universe 需要计算因子值的股票, string 的 list
    :param start_date 开始计算日期
    :param end_date 结束计算日期
    :return: 因子值, index 为日期, column 为股票代码
    '''
    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]
    low = data.low[futures_universe]



    ret = pd.DataFrame()
    for w in window:
        for name in futures_universe:
            N = w
            _data = data.loc[name]
            
            _data['g065'] = ((_data.closew.rolling(N).mean()- _data.closew)/ _data.close)+1
    
            new_columns = pd.MultiIndex.from_product([[str(name)], ["gtja065" + str(w)]])  
            temp = _data['g065'].to_frame(name='g065').set_index(_data.index)  
            temp.columns = new_columns  
            ret = pd.concat([ret, temp], axis=1)
# =============================================================================
#     for w in window:
#         N = w
# 
#         _data = data.loc[futures_universe]
#         
# 
#         _data['g065'] = _data.close.rolling(N).mean() / data.close
# 
#         _ret = _data.g065.unstack().T
#         _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja065" + str(w)]], names=["futures", "price"])
#         ret = pd.concat([ret, _ret], axis=1)
# =============================================================================
    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret
    

gtja066_time_lst = [2,  4, 6, 12, 9,18]
def GTJA066(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA066
    (close - mean(close, 6)) / mean(close, 6)

    :param df 是带有原始数据的 DataFrame
    :param universe 需要计算因子值的股票, string 的 list
    :param start_date 开始计算日期
    :param end_date 结束计算日期
    :return: 因子值, index 为日期, column 为股票代码
    '''
    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]
    low = data.low[futures_universe]



    ret = pd.DataFrame()
    # FACTOR_PATH = './factor_daily/GTJA066.csv'
    # if not os.path.exists(FACTOR_PATH):
    #     factor_df = pd.DataFrame()
    #     for w in window:
    #         for name in futures_universe:
    #             N = w
    #             _data = data.loc[name]
    #             _data['GTJA066'] = (_data.closew - _data.closew.rolling(N).mean()) / _data.closew.rolling(N).mean()
        
    #             new_columns = pd.MultiIndex.from_product([[str(name)], ["GTJA066_" + str(w)]])  
    #             temp = _data['GTJA066'].to_frame(name='GTJA066').set_index(_data.index)  
    #             temp.columns = new_columns  
    #             factor_df = pd.concat([factor_df, temp], axis=1)
    #     factor_df.to_csv(FACTOR_PATH)

    for w in window:
        for name in futures_universe:
            N = w
    
            _data = data.loc[name]
            
            # 分母更改为贴水后 closew   zwh  6.5
            _data['g066'] = (_data.closew - _data.closew.rolling(N).mean()) / _data.closew.rolling(N).mean()
    
            new_columns = pd.MultiIndex.from_product([[str(name)], ["gtja066" + str(w)]])  
            temp = _data['g066'].to_frame(name='g066').set_index(_data.index)  
            temp.columns = new_columns  
            ret = pd.concat([ret, temp], axis=1)

# =============================================================================
#     for w in window:
#         N = w
# 
#         _data = data.loc[futures_universe]
#         
# 
#         _data['g066'] = (_data.close - _data.close.rolling(N).mean()) / _data.close.rolling(N).mean()
# 
#         _ret = _data.g066.unstack().T
#         _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja066" + str(w)]], names=["futures", "price"])
#         ret = pd.concat([ret, _ret], axis=1)
# =============================================================================

    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret



gtja067_time_lst = [
    [1, 1 / 6],
    [1, 1 / 12],
    [1, 1 / 24],
    [3, 1 / 6],
    [3, 1 / 12],
    [3, 1 / 24]
]
    
def GTJA067(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA067
    sma(max(delta(close, 1), 0), 24, 1) / sma(abs(delta(close, 1)), 24, 1)

    :param df 是带有原始数据的 DataFrame
    :param universe 需要计算因子值的股票, string 的 list
    :param start_date 开始计算日期
    :param end_date 结束计算日期
    :return: 因子值, index 为日期, column 为股票代码
    '''
    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]
    low = data.low[futures_universe]



    ret = pd.DataFrame()
    for w in window:
        for name in futures_universe:
            N ,sma_alpha= w
    
            _data = data.loc[name]
            
    
            _data['close_delta'] = _data.closew.diff(N)  # Change: former-diff(1); now-diff(N)
            _data['const'] = 0.
            
            _data['inner1'] = _data[['const', 'close_delta']].max(axis=1)
            _data['inner2'] = np.abs(_data.close_delta)
    
            _data['g067'] = _data.inner1.ewm(adjust=False, alpha=sma_alpha).mean() /_data.inner2.ewm(adjust=False, alpha=sma_alpha).mean()
    
            new_columns = pd.MultiIndex.from_product([[str(name)], ["gtja067" + str(w)]])  
            temp = _data['g067'].to_frame(name='g067').set_index(_data.index)  
            temp.columns = new_columns  
            ret = pd.concat([ret, temp], axis=1) 
# =============================================================================
#     for w in window:
#         N ,sma_alpha= w
# 
#         _data = data.loc[futures_universe]
#         
# 
#         _data['close_delta'] = _data.close.diff(1)
#         _data['const'] = 0.
#         
#         _data['inner1'] = _data[['const', 'close_delta']].max(axis=1)
#         _data['inner2'] = np.abs(_data.close_delta)
# 
#         _data['g067'] = _data.inner1.ewm(adjust=False, alpha=sma_alpha).mean() /_data.inner2.ewm(adjust=False, alpha=sma_alpha).mean()
# 
#         _ret = _data.g067.unstack().T
#         _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja067" + str(w)]], names=["futures", "price"])
#         ret = pd.concat([ret, _ret], axis=1)
# =============================================================================
    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret


gtja068_time_lst = [1/ 30,  1 / 15, 2 / 15, 3/ 15,5/ 15,6/ 15]  
def GTJA068(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA068
    sma(((high + low) / 2 - (delay(high, 1) + delay(low, 1)) / 2) * (high - low) / volume, 15, 2)

    :param df 是带有原始数据的 DataFrame
    :param universe 需要计算因子值的股票, string 的 list
    :param start_date 开始计算日期
    :param end_date 结束计算日期
    :return: 因子值, index 为日期, column 为股票代码
    '''
    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]
    low = data.low[futures_universe]



    ret = pd.DataFrame()
    for w in window:
        for name in futures_universe:
            sma_alpha= w
    
            _data = data.loc[name]
            
    
            _data['inner'] = ((_data.highw +_data.loww) / 2 - (_data.highw.shift(1) + _data.loww.shift(1)) / 2) * (_data.highw - _data.loww) / _data.volume99
    
            _data['g068'] =_data.inner.ewm(adjust=False, alpha=sma_alpha).mean()
            new_columns = pd.MultiIndex.from_product([[str(name)], ["gtja068" + str(w)]])  
            temp = _data['g068'].to_frame(name='g068').set_index(_data.index)  
            temp.columns = new_columns  
            ret = pd.concat([ret, temp], axis=1) 
# =============================================================================
#     for w in window:
#         sma_alpha= w
# 
#         _data = data.loc[futures_universe]
#         
# 
#         _data['inner'] = ((_data.high +_data.low) / 2 - (_data.high.shift(1) + _data.low.shift(1)) / 2) * (_data.high - _data.low) / _data.volume
# 
#         _data['g068'] =_data.inner.ewm(adjust=False, alpha=sma_alpha).mean()
#         _ret = _data.g068.unstack().T
#         _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja068" + str(w)]], names=["futures", "price"])
#         ret = pd.concat([ret, _ret], axis=1)
# =============================================================================

    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret
  
  
# gtja069_time_lst = [4, 12, 20, 32, 48, 60]
gtja069_time_lst = [1, 2, 6, 11, 21, 31]
def GTJA069(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA069
    sum(dtm, 20) > sum(dbm, 20) ? (sum(dtm, 20) - sum(dbm, 20)) / sum(dtm, 20) : sum(dtm, 20) == sum(dbm, 20) ? 0 : (sum(dtm, 20) - sum(dbm, 20)) / sum(dbm, 20)

    # sum(dtm, 20) > sum(dbm, 20)  => (sum(dtm, 20) - sum(dbm, 20)) / sum(dtm, 20)
    # sum(dtm, 20) == sum(dbm, 20) => 0
    # sum(dtm, 20) < sum(dbm, 20)  => (sum(dtm, 20) - sum(dbm, 20)) / sum(dbm, 20)

    :param df 是带有原始数据的 DataFrame
    :param universe 需要计算因子值的股票, string 的 list
    :param start_date 开始计算日期
    :param end_date 结束计算日期
    :return: 因子值, index 为日期, column 为股票代码
    '''
    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]
    low = data.low[futures_universe]



    ret = pd.DataFrame()
    for w in window:
        for name in futures_universe:
            N= w
    
            _data = data.loc[name]
            
    
            _data['dtm'] = (_data.openw > _data.openw.shift(1)) * ((_data.highw - _data.openw >= _data.openw - _data.openw.shift(1)) * (_data.highw - _data.openw) + (_data.highw - _data.openw < _data.openw - _data.openw.shift(1)) * (_data.openw - _data.openw.shift(1)))
            _data['dbm'] = (_data.openw < _data.openw.shift(1)) * ((_data.openw - _data.loww >= _data.openw - _data.openw.shift(1)) * (_data.openw - _data.loww) + (_data.openw - _data.loww < _data.openw - _data.openw.shift(1)) * (_data.openw - _data.openw.shift(1)))
    
            _data['inner1'] = _data.dtm.rolling(N).sum()
            _data['inner2'] = _data.dbm.rolling(N).sum()
    
            _data['inner'] = (_data.inner1 -_data.inner2) / _data[['inner1', 'inner2']].max(axis=1)
            _data.loc[(_data.dtm == 0) & (_data.dbm == 0), "inner"] = 0
    
            _data['g069'] = _data.inner
            new_columns = pd.MultiIndex.from_product([[str(name)], ["gtja069" + str(w)]])  
            temp = _data['g069'].to_frame(name='g069').set_index(_data.index)  
            temp.columns = new_columns  
            ret = pd.concat([ret, temp], axis=1) 
# =============================================================================
#     for w in window:
#         N= w
# 
#         _data = data.loc[futures_universe]
#         
# 
#         _data['dtm'] = (_data.open > _data.open.shift(1)) * ((_data.high - _data.open >= _data.open - _data.open.shift(1)) * (_data.high - _data.open) + (_data.high - _data.open < _data.open - _data.open.shift(1)) * (_data.open - _data.open.shift(1)))
#         _data['dbm'] = (_data.open < _data.open.shift(1)) * ((_data.open - _data.low >= _data.open - _data.open.shift(1)) * (_data.open - _data.low) + (_data.open - _data.low < _data.open - _data.open.shift(1)) * (_data.open - _data.open.shift(1)))
# 
#         _data['inner1'] = _data.dtm.rolling(N).sum()
#         _data['inner2'] = _data.dbm.rolling(N).sum()
# 
#         _data['inner'] = (_data.inner1 -_data.inner2) / _data[['inner1', 'inner2']].max(axis=1)
#         _data.loc[(data.dtm == 0) & (data.dbm == 0), "inner"] = 0
# 
#         _data['g069'] = _data.inner
#         _ret = _data.g069.unstack().T
#         _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja069" + str(w)]], names=["futures", "price"])
#         ret = pd.concat([ret, _ret], axis=1)
# =============================================================================

    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret


gtja070_time_lst = [3, 6, 12, 24, 36, 48]   
def GTJA070(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA070
    std(amount, 6)

    :param df 是带有原始数据的 DataFrame
    :param universe 需要计算因子值的股票, string 的 list
    :param start_date 开始计算日期
    :param end_date 结束计算日期
    :return: 因子值, index 为日期, column 为股票代码
    '''
    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]
    low = data.low[futures_universe]



    ret = pd.DataFrame()
# =============================================================================
#     for w in window:
#         N= w
# 
#         _data = data.loc[futures_universe]
#         
# 
#         _data['g070'] = data.total_turnover.rolling(N).std()
#         _ret = _data.g070.unstack().T
#         _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja070" + str(w)]], names=["futures", "price"])
#         ret = pd.concat([ret, _ret], axis=1)
# =============================================================================

    for w in window:
        for name in futures_universe:
            N= w
    
            _data = data.loc[name]
            
            _data['g070'] = _data.total_turnover.rolling(N).std()
            new_columns = pd.MultiIndex.from_product([[str(name)], ["gtja070" + str(w)]])  
            temp = _data['g070'].to_frame(name='g070').set_index(_data.index)  
            temp.columns = new_columns  
            ret = pd.concat([ret, temp], axis=1) 
    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret


gtja071_time_lst = [3, 6, 12, 24, 36, 48] 
def GTJA071(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA071
    (close - mean(close, 24)) / mean(close, 24)

    :param df 是带有原始数据的 DataFrame
    :param universe 需要计算因子值的股票, string 的 list
    :param start_date 开始计算日期
    :param end_date 结束计算日期
    :return: 因子值, index 为日期, column 为股票代码
    '''
    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]
    low = data.low[futures_universe]



    ret = pd.DataFrame()
    for w in window:
        for name in futures_universe:
            N= w
    
            _data = data.loc[name]
            # 分母更改为closew  zwh 6.5
            _data['g071'] = (_data.closew - _data.closew.rolling(N).mean()) / _data.closew.rolling(N).mean()
            new_columns = pd.MultiIndex.from_product([[str(name)], ["gtja071" + str(w)]])  
            temp = _data['g071'].to_frame(name='g071').set_index(_data.index)  
            temp.columns = new_columns  
            ret = pd.concat([ret, temp], axis=1) 
    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret


gtja072_time_lst = [
    [1, 1 / 3],
    [2, 1 / 9],
    [6, 1 / 15],
    [11, 1 / 21],
    [21, 1 / 27],
    [31, 1 / 33]
]
def GTJA072(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA072
    sma((tsmax(high, 6) - close) / (tsmax(high, 6) - tsmin(low, 6)), 15, 1)

    :param df 是带有原始数据的 DataFrame
    :param universe 需要计算因子值的股票, string 的 list
    :param start_date 开始计算日期
    :param end_date 结束计算日期
    :return: 因子值, index 为日期, column 为股票代码
    '''
    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]
    low = data.low[futures_universe]



    ret = pd.DataFrame()
    for w in window:
        for name in futures_universe:
            N ,sma_alpha= w
    
            _data = data.loc[name]
            _data['inner'] = (_data.highw.rolling(N).max() - _data.closew) / (_data.highw.rolling(N).max() - _data.loww.rolling(N).min())   # change: former-close; now-closew
           
            _data['g072'] = _data.inner.ewm(adjust=False, alpha=sma_alpha).mean()
            
            new_columns = pd.MultiIndex.from_product([[str(name)], ["gtja072" + str(w)]])  
            temp = _data['g072'].to_frame(name='g072').set_index(_data.index)  
            temp.columns = new_columns  
            ret = pd.concat([ret, temp], axis=1) 
        
    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret

# =============================================================================
# def GTJA073(df, universe, start_date, end_date):
#     '''
#     计算 GTJA073
#     tsrank(decaylinear(decaylinear(corr(close, volume, 10), 16), 4), 5) - rank(decaylinear(corr(vwap, mean(volume, 30), 4), 3))
# 
#     :param df 是带有原始数据的 DataFrame
#     :param universe 需要计算因子值的股票, string 的 list
#     :param start_date 开始计算日期
#     :param end_date 结束计算日期
#     :return: 因子值, index 为日期, column 为股票代码
#     '''
#     N1, N2, N3, N4 = 10, 16, 4, 5
#     N5, N6, N7 = 30, 4, 3
#     data = df.loc[universe]
# 
#     def wma(series, N):
#         return series.rolling(N).apply(lambda x: x[:: -1].cumsum().sum() * 2 / N / (N + 1))
#     
#     def tsrank(array):
#         s = pd.Series(array)
#         return np.array(s.rank(method="min", ascending=False))[-1]
# 
#     data['inner1'] = wma(wma(data.close.rolling(N1).corr(data.volume), N2), N3)
#     data['rank1'] = data.inner1.rolling(N4).apply(tsrank)
# 
#     data['inner2'] = wma(data.vwap.rolling(N6).corr(data.volume.rolling(N5).mean()), N7)
#     data['rank2'] = data.inner2.unstack().rank(pct=True).stack()
#    
#     data['g073'] = data.rank1 - data.rank2
#     return data.g073.unstack().T[start_date: end_date]
# =============================================================================

gtja073_time_lst = [
    [(10, 16, 4, 5), (30, 4, 3)],
    [(11, 17, 5, 6), (31, 5, 4)],
    [(12, 18, 6, 7), (32, 6, 5)],
    [(13, 19, 7, 8), (33, 7, 6)],
    [(14, 20, 8, 9), (34, 8, 7)],
    [(15, 21, 9, 10), (35, 9, 8)]
]

def GTJA073(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA073
    tsrank(decaylinear(decaylinear(corr(close, volume, 10), 16), 4), 5) - rank(decaylinear(corr(vwap, mean(volume, 30), 4), 3))

    :param df 是带有原始数据的 DataFrame
    :param universe 需要计算因子值的股票, string 的 list
    :param start_date 开始计算日期
    :param end_date 结束计算日期
    :return: 因子值, index 为日期, column 为股票代码
    '''
    close = data.close[futures_universe]
    volume = data.volume99[futures_universe]
    vwap = data.vwap[futures_universe]

    ret = pd.DataFrame()
    for w in window:
        for name in futures_universe:
            N1, N2, N3, N4 = w[0]
            N5, N6, N7 = w[1]
    
            _data = data.loc[name]
            _data['inner1'] = wma(wma(_data.close.rolling(N1).corr(_data.volume99), N2), N3)
            _data['rank1'] = _data.inner1.rolling(N4).apply(tsrank)

            _data['inner2'] = wma(_data.vwap.rolling(N6).corr(_data.volume99.rolling(N5).mean()), N7)
            _data['rank2'] = _data.inner2.unstack().rank(pct=True).stack()
   
            _data['g073'] = _data.rank1 - _data.rank2
            
            new_columns = pd.MultiIndex.from_product([[str(name)], ["gtja073" + str(w)]])  
            temp = _data['g073'].to_frame(name='g073').set_index(_data.index)  
            temp.columns = new_columns  
            ret = pd.concat([ret, temp], axis=1) 
    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret


    
def GTJA074(df, universe, start_date, end_date):
    '''
    计算 GTJA074
    rank(corr(sum(low * .35 + vwap * .65, 20), sum(mean(volume, 40), 20), 7)) + rank(corr(rank(vwap), rank(volume), 6))

    :param df 是带有原始数据的 DataFrame
    :param universe 需要计算因子值的股票, string 的 list
    :param start_date 开始计算日期
    :param end_date 结束计算日期
    :return: 因子值, index 为日期, column 为股票代码
    '''
    N1, N2 = .35, 20
    N3, N4, N5 = 40, 7, 6
    data = df.loc[universe]

    data['inner11'] = (data.low * N1 + data.vwap * (1 - N1)).rolling(N2).sum()
    data['inner12'] = (data.volume99.rolling(N3).mean()).rolling(N2).sum()
    data['inner1'] = data.inner11.rolling(N4).corr(data.inner12)
    data['rank1'] = data.inner1.unstack().rank(pct=True).stack()

    data['rank21'] = data.vwap.unstack().rank(pct=True).stack()
    data['rank22'] = data.volume99.unstack().rank(pct=True).stack()
    data['inner2'] = data.rank21.rolling(N5).corr(data.rank22)
    data['rank2'] = data.inner2.unstack().rank(pct=True).stack()
   
    data['g074'] = data.rank1 + data.rank2
    return data.g074.unstack().T[start_date: end_date]

gtja076_time_lst = [5, 10, 20, 40, 60, 120]
# gtja076_time_list = [3, 6, 12, 20, 36, 48]
def GTJA076(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA076
    std(abs(close / delay(close, 1) - 1) / volume, 20) / mean(abs(close / delay(close, 1) - 1) / volume, 20)

    :param df 是带有指数价格数据的 DataFrame
    :param universe 需要计算因子值的股票, string 的 list
    :param start_date 开始计算日期
    :param end_date 结束计算日期
    :return: 因子值, index 为日期, column 为股票代码
    '''
    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]
    low = data.low[futures_universe]



    ret = pd.DataFrame()
    for w in window:
        for name in futures_universe:
            N = w
    
            _data = data.loc[name]
            _data['inner'] = np.abs(_data.closew / _data.closew.shift(1) - 1) / _data.volume99
   
            _data['g076'] = _data.inner.rolling(N).std() / _data.inner.rolling(N).mean()
            
            new_columns = pd.MultiIndex.from_product([[str(name)], ["gtja076" + str(w)]])  
            temp = _data['g076'].to_frame(name='g076').set_index(_data.index)  
            temp.columns = new_columns  
            ret = pd.concat([ret, temp], axis=1) 
    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)    
    return ret
# =============================================================================
# def GTJA076(df, universe, start_date, end_date):
#     '''
#     计算 GTJA076
#     std(abs(close / delay(close, 1) - 1) / volume, 20) / mean(abs(close / delay(close, 1) - 1) / volume, 20)
# 
#     :param df 是带有指数价格数据的 DataFrame
#     :param universe 需要计算因子值的股票, string 的 list
#     :param start_date 开始计算日期
#     :param end_date 结束计算日期
#     :return: 因子值, index 为日期, column 为股票代码
#     '''
#     N = 20
#     data = df.loc[universe]
# 
#     data['inner'] = np.abs(data.close / data.close.shift(1) - 1) / data.volume
#    
#     data['g076'] = data.inner.rolling(N).std() / data.inner.rolling(N).mean()
#     return data.g076.unstack().T[start_date: end_date]
# =============================================================================
def GTJA077(df, universe, start_date, end_date):
    '''
    计算 GTJA077
    min(rank(decaylinear((high + low) / 2 - vwap, 20)), rank(decaylinear(corr((high + low) / 2, mean(volume, 40), 3), 6)))

    :param df 是带有原始数据的 DataFrame
    :param universe 需要计算因子值的股票, string 的 list
    :param start_date 开始计算日期
    :param end_date 结束计算日期
    :return: 因子值, index 为日期, column 为股票代码
    '''
    N1, N2, N3, N4 = 20, 40, 3, 6
    data = df.loc[universe]

    def wma(series, N):
        return series.rolling(N).apply(lambda x: x[:: -1].cumsum().sum() * 2 / N / (N + 1))

    data['inner1'] = wma((data.high + data.low) / 2 - data.vwap, N1)
    data['rank1'] = data.inner1.unstack().rank(pct=True).stack()
    data['inner2'] = wma(((data.high + data.low) / 2).rolling(N3).corr(data.volume99.rolling(N2).mean()), N4)
    data['rank2'] = data.inner2.unstack().rank(pct=True).stack()
   
    data['g077'] = data[['rank1', 'rank2']].min(axis=1)
    return data.g077.unstack().T[start_date: end_date]

gtja078_time_lst = [12, 24, 36, 48, 60, 72]

def GTJA078(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA078
    ((high + low + close) / 3 - mean((high + low + close) / 3, 12)) / mean(abs(close - mean((high + low + close) / 3, 12)), 12)

    :param df 是带有原始数据的 DataFrame
    :param universe 需要计算因子值的股票, string 的 list
    :param start_date 开始计算日期
    :param end_date 结束计算日期
    :return: 因子值, index 为日期, column 为股票代码
    '''
    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]
    low = data.low[futures_universe]



    ret = pd.DataFrame()
    for w in window:
        for name in futures_universe:
            N = w
    
            _data = data.loc[name]
            _data['inner'] = (_data.highw + _data.loww + _data.closew) / 3
   
            _data['g078'] = ((_data.inner - _data.inner.rolling(N).mean()) / (np.abs(_data.closew - _data.inner.rolling(N).mean()))).rolling(N).mean()
            
            new_columns = pd.MultiIndex.from_product([[str(name)], ["gtja078" + str(w)]])  
            temp = _data['g078'].to_frame(name='g078').set_index(_data.index)  
            temp.columns = new_columns  
            ret = pd.concat([ret, temp], axis=1) 
    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)    
    return ret
# =============================================================================
# def GTJA078(df, universe, start_date, end_date):
#     '''
#     计算 GTJA078
#     ((high + low + close) / 3 - mean((high + low + close) / 3, 12)) / mean(abs(close - mean((high + low + close) / 3, 12)), 12)
# 
#     :param df 是带有原始数据的 DataFrame
#     :param universe 需要计算因子值的股票, string 的 list
#     :param start_date 开始计算日期
#     :param end_date 结束计算日期
#     :return: 因子值, index 为日期, column 为股票代码
#     '''
#     N = 12
#     data = df.loc[universe]
# 
#     data['inner'] = (data.high + data.low + data.close) / 3
#    
#     data['g078'] = (data.inner - data.inner.rolling(N).mean()) / (np.abs(data.close - data.inner.rolling(N).mean())).rolling(N).mean()
#     return data.g078.unstack().T[start_date: end_date]
# =============================================================================

# gtja079_time_lst = [12, 24, 36, 48, 60, 72]
gtja079_time_lst = [10, 16, 26, 40, 50, 60]

def GTJA079(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA079
    sma(max(delta(close, 1), 0), 12, 1) / sma(abs(delta(close, 1)), 12, 1)

    :param df 是带有原始数据的 DataFrame
    :param universe 需要计算因子值的股票, string 的 list
    :param start_date 开始计算日期
    :param end_date 结束计算日期
    :return: 因子值, index 为日期, column 为股票代码
    '''
    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]
    low = data.low[futures_universe]



    ret = pd.DataFrame()
    for w in window:
        for name in futures_universe:
            sma_alpha = 1. / w
    
            _data = data.loc[name]
            _data['const'] = 0.
            _data['inner'] = _data.closew.diff(1)

            _data['inner1'] = _data[['const', 'inner']].max(axis=1)
            _data['inner2'] = np.abs(_data.inner)

            _data['g079'] = _data.inner1.ewm(adjust=False, alpha=sma_alpha).mean() / _data.inner2.ewm(adjust=False, alpha=sma_alpha).mean()
            
            new_columns = pd.MultiIndex.from_product([[str(name)], ["gtja079" + str(w)]])  
            temp = _data['g079'].to_frame(name='g079').set_index(_data.index)  
            temp.columns = new_columns  
            ret = pd.concat([ret, temp], axis=1) 
    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret

# =============================================================================
# def GTJA079(df, universe, start_date, end_date):
#     '''
#     计算 GTJA079
#     sma(max(delta(close, 1), 0), 12, 1) / sma(abs(delta(close, 1)), 12, 1)
# 
#     :param df 是带有原始数据的 DataFrame
#     :param universe 需要计算因子值的股票, string 的 list
#     :param start_date 开始计算日期
#     :param end_date 结束计算日期
#     :return: 因子值, index 为日期, column 为股票代码
#     '''
#     sma_alpha = 1. / 12
#     data = df.loc[universe]
# 
#     data['const'] = 0.
#     data['inner'] = data.close.diff(1)
# 
#     data['inner1'] = data[['const', 'inner']].max(axis=1)
#     data['inner2'] = np.abs(data.inner)
# 
#     data['g079'] = data.inner1.ewm(adjust=False, alpha=sma_alpha).mean() / data.inner2.ewm(adjust=False, alpha=sma_alpha).mean()
#     return data.g079.unstack().T[start_date: end_date]
# =============================================================================

gtja080_time_lst = [5, 10, 15, 20, 25, 30]

def GTJA080(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA080
    -1 * delta(volume, 5) / delay(volume, 5)

    :param df 是带有原始数据的 DataFrame
    :param universe 需要计算因子值的股票, string 的 list
    :param start_date 开始计算日期
    :param end_date 结束计算日期
    :return: 因子值, index 为日期, column 为股票代码
    '''
    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]
    low = data.low[futures_universe]



    ret = pd.DataFrame()
    for w in window:
        for name in futures_universe:
            N = w
    
            _data = data.loc[name]
            _data['g080'] = _data.volume99.diff(N) / _data.volume99.shift(N)
            
            new_columns = pd.MultiIndex.from_product([[str(name)], ["gtja080" + str(w)]])  
            temp = _data['g080'].to_frame(name='g080').set_index(_data.index)  
            temp.columns = new_columns  
            ret = pd.concat([ret, temp], axis=1) 
    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret

gtja081_time_lst = [5, 10, 15, 20, 25, 30]
def GTJA081(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA081
    sma(volume, 21, 2)

    :param data 默认旋转后，即 data_rotated
    :param window 是一个一维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return ret
    '''

    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]

    
    ret = pd.DataFrame()
    for w in window:
        N = w
        sma_alpha = 2. / N

        _data = data.loc[futures_universe]

        _data['g081'] = _data.volume99.unstack().ewm(adjust=False, alpha=sma_alpha,axis=1).mean().stack()
        _ret = _data.g081.unstack().T
        _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja081" + str(w)]], names=["futures", "price"])
        ret = pd.concat([ret, _ret], axis=1)
    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)

    return ret

gtja082_time_lst = [
    [3, 1 / 3],
    [6, 1 / 6],
    [9, 1 / 9],
    [15, 1 / 15],
    [21, 1 / 21],
    [33, 1 / 33]
]
def GTJA082(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA082
    sma((tsmax(high, 6) - close) / (tsmax(high, 6) - tsmin(low, 6)), 20, 1)

    :param data 默认旋转后，即 data_rotated
    :param window 是一个一维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return ret
    '''

    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]

    ret = pd.DataFrame()
    for w in window:
        N,sma_alpha = w

        _data = data.loc[futures_universe]

        _data['inner'] = (_data.highw.rolling(N).max() - _data.closew) / (_data.highw.rolling(N).max() - _data.loww.rolling(N).min())  # change: former-close; now-closew

        _data['g082'] = _data.inner.unstack().ewm(adjust=False, alpha=sma_alpha,axis=1).mean().stack()
        _ret = _data.g082.unstack().T
        _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja082" + str(w)]], names=["futures", "price"])
        ret = pd.concat([ret, _ret], axis=1)
    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)

    return ret

gtja083_time_lst = [5, 10, 15, 20, 25, 30]

# =============================================================================
# gtja083_time_lst = [
#     2,
#     3,
#     6,
#     11,
#     21,
#     31
# ]
# =============================================================================
def GTJA083(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA083
    corr(rank(high), rank(volume), 5)

    :param data 默认旋转后，即 data_rotated
    :param window 是一个一维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return ret
    '''

    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]

    ret = pd.DataFrame()
    for w in window:
        N = w

        _data = data.loc[futures_universe]

        _data['rank1'] = _data.highw.unstack().rank(pct=True).stack()
        _data['rank2'] = _data.volume99.unstack().rank(pct=True).stack()

        _data['g083'] = _data.rank1.rolling(N).corr(_data.rank2)
        _ret = _data.g083.unstack().T
        _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja083" + str(w)]], names=["futures", "price"])
        ret = pd.concat([ret, _ret], axis=1)
    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)

    return ret

# gtja084_time_lst = [5, 10, 15, 20, 25, 30]
gtja084_time_lst = [2, 4, 6, 9, 12, 18]

def GTJA084(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA084
    sum(close > delay(close, 1) ? volume : close < delay(close, 1) ? -volume : 0, 20)

    :param data 默认旋转后，即 data_rotated
    :param window 是一个一维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return ret
    '''

    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]

    ret = pd.DataFrame()
    for w in window:
        N = w

        _data = data.loc[futures_universe]
        _data['inner'] = (_data.closew > _data.closew.shift(1)) * _data.volume99 - (_data.closew < _data.closew.shift(1)) * _data.volume99

        _data['g084'] = _data.inner.unstack().rolling(N,axis=1).sum().stack()
        _ret = _data.g084.unstack().T
        _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja084" + str(w)]], names=["futures", "price"])
        ret = pd.concat([ret, _ret], axis=1)
    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)

    return ret

# 待定
def GTJA085(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA085
    tsrank(volume / mean(volume, 20), 20) * tsrank(-delta(close, 7), 8)

    :param data 默认旋转后，即 data_rotated
    :param window 是一个一维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return ret
    '''

    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]

    def tsrank(array):
        s = pd.Series(array)
        return np.array(s.rank(method="min", ascending=False))[-1]

    ret = pd.DataFrame()
    for w in window:
        N1, N2, N3 = w

        _data = data.loc[futures_universe]

        _data['inner1'] = _data.volume99 / _data.volume99.rolling(N1).mean()
        _data['rank1'] = _data.inner1.rolling(N1).apply(tsrank)
        _data['inner2'] = -_data.close.diff(N2)
        _data['rank2'] = _data.inner2.rolling(N3).apply(tsrank)

        _data['g085'] = _data.rank1 * _data.rank2
        _ret = _data.g085.unstack().T
        _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja085" + str(w)]], names=["futures", "price"])
        ret = pd.concat([ret, _ret], axis=1)
    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)

    return ret

gtja086_time_lst = [ [6, 3], [10, 5], [14, 7], [18, 9], [24, 12], [30, 15]]
def GTJA086(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA086
    inner = (delay(close, 20) - delay(close, 10)) / 10 - (delay(close, 10) - close) / 10
    inner > 0.25 ? -1 : inner < 0 ? 1 : delay(close, 1) - close

    :param data 默认旋转后，即 data_rotated
    :param window 是一个一维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return ret
    '''

    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]

    _const1, _const2 = 0.25, 0

    ret = pd.DataFrame()
# =============================================================================
#     for w in window:
#         N1, N2 = w
# 
#         _data = data.loc[futures_universe]
# 
#         # _data['inner'] = ((_data.closew.unstack().shift(N1,axis=1) - _data.closew.unstack().shift(N2,axis=1)) / (N1 - N2) - (_data.closew.unstack().shift(N2,axis=1) - _data.close.unstack()) / N2).stack()
#         _data['inner'] = ((_data.closew.unstack().shift(N1,axis=1) - _data.closew.unstack().shift(N2,axis=1)) / (N1 - N2) - (_data.closew.unstack().shift(N2,axis=1) - _data.closew.unstack()) / N2).stack()
#         _data['const1'] = _const1
#         _data['const2'] = _const2
# 
#         # _data['g086'] = (_data.inner > _data.const1) * -1 + (_data.inner < _data.const2) * 1 + ((_data.const2 <= _data.inner) & (_data.inner <= _data.const1)) * -_data.close.diff(1)
#         _data['g086'] = (_data.inner > _data.const1) * -1 + (_data.inner < _data.const2) * 1 + ((_data.const2 <= _data.inner) & (_data.inner <= _data.const1)) * -_data.closew.unstack().diff(1,axis=1).stack()
#         _ret = _data.g086.unstack().T
#         _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja086" + str(w)]], names=["futures", "price"])
#         ret = pd.concat([ret, _ret], axis=1)
# =============================================================================
    for w in window:
        for name in futures_universe:
            N1, N2 = w
    
            _data = data.loc[name]
    
            _data['inner'] = (_data.closew.shift(N1) - _data.closew.shift(N2)) / (N1 - N2) - (_data.closew.shift(N2) - _data.closew) / N2
            _data['const1'] = _const1
            _data['const2'] = _const2
    
            _data['g086'] = (_data.inner > _data.const1) * -1 + (_data.inner < _data.const2) * 1 + ((_data.const2 <= _data.inner) & (_data.inner <= _data.const1)) * -_data.closew.diff(1)
    
            new_columns = pd.MultiIndex.from_product([[str(name)], ["gtja086" + str(w)]])  
            temp = _data['g086'].to_frame(name='g086').set_index(_data.index)  
            temp.columns = new_columns  
            ret = pd.concat([ret, temp], axis=1)  
    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)

    return ret

gtja086edit_time_lst = [ [6, 3], [10, 5], [14, 7], [18, 9], [24, 12], [30, 15]]
def GTJA086EDIT(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA086Edit
    只保留 inner 项，并把减号改为除号
    (delay(close, 20) - delay(close, 10)) / (delay(close, 10) - close)

    :param data 默认旋转后，即 data_rotated
    :param window 是一个一维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return ret
    '''

    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]

    ret = pd.DataFrame()
    for w in window:
        N1, N2 = w

        _data = data.loc[futures_universe]

        _data['g086'] = (_data.closew.unstack().shift(N1,axis=1).stack() - _data.closew.unstack().shift(N2,axis=1).stack()) / (_data.closew.unstack().shift(N2,axis=1).stack() - _data.closew+1e-6)
        _ret = _data.g086.unstack().T
        _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja086edit" + str(w)]], names=["futures", "price"])
        ret = pd.concat([ret, _ret], axis=1)
    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)

    return ret


gtja087_time_lst = [ [6, 3], [10, 5], [14, 7], [18, 9], [24, 12], [30, 15]]
def GTJA087(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA087
    rank(decaylinear(delta(vwap, 4), 7)) + tsrank(decaylinear(low - vwap))
    
    :param data 默认旋转后，即 data_rotated
    :param window 是一个一维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return ret
    '''
    
    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]
    
    def wma(series, N):
        return series.unstack().rolling(N,axis=1).apply(lambda x: x[:: -1].cumsum().sum() * 2 / N / (N + 1)).stack()
        # return series.rolling(N).apply(lambda x: x[:: -1].cumsum().sum() * 2 / N / (N + 1))
    
    def tsrank(array):
        s = pd.Series(array)
        return np.array(s.rank(method="min", ascending=False))[-1]
    
    ret = pd.DataFrame()
    for w in window:
        N1, N2 = w
    
        _data = data.loc[futures_universe]
        _data['vwapw'] = (_data.openw + _data.closew + _data.highw + _data.loww) / 4
        _data['inner1'] = wma(_data.vwapw.unstack().diff(N1,axis=1).stack(), N2)
        _data['inner21'] = wma((_data.loww - _data.vwapw) / (_data.openw - (_data.highw + _data.loww) / 2), N1)
        _data['inner2'] = _data.inner21.unstack().rolling(N2,axis=1).apply(tsrank).stack()
    
        _data['g087'] = _data.inner1 + _data.inner2
        _ret = _data.g087.unstack().T
        _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja087" + str(w)]], names=["futures", "price"])
        ret = pd.concat([ret, _ret], axis=1)
    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret



gtja088_time_lst = [12,24,36,48,60,72]
def GTJA088(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA088
    (close - delay(close, 20)) / delay(close, 20)

    :param data 默认旋转后，即 data_rotated
    :param window 是一个一维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return ret
    '''

    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]

    ret = pd.DataFrame()
    for w in window:
        N = w

        _data = data.loc[futures_universe]
        # 分母修改为closew  zwh  6.5
        _data['g088'] = (_data.closew - _data.closew.unstack().shift(N,axis=1).stack()) / _data.closew.unstack().shift(N,axis=1).stack()
        _ret = _data.g088.unstack().T
        _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja088" + str(w)]], names=["futures", "price"])
        ret = pd.concat([ret, _ret], axis=1)
    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret

gtja089_time_lst = [
    [3,6,6],
    [6,12,10],
    [9,18,8],
    [13,27,10],
    [17,35,15],
    [25,51,20],
    ]
def GTJA089(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA089
    sma(close, 13, 2) - sma(close, 27, 2) - sma(sma(close, 13, 2) - sma(close, 27, 2), 10, 2)

    :param data 默认旋转后，即 data_rotated
    :param window 是一个一维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return ret
    '''

    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]

    ret = pd.DataFrame()
    for w in window:
        sma_alpha1 = 2. / w[0]
        sma_alpha2 = 2. / w[1]
        sma_alpha3 = 2. / w[2]

        _data = data.loc[futures_universe]

        _data['inner'] = _data.closew.unstack().ewm(adjust=False, alpha=sma_alpha1,axis=1).mean().stack() - _data.closew.unstack().ewm(adjust=False, alpha=sma_alpha2,axis=1).mean().stack()

        _data['g089'] = _data.inner - _data.inner.unstack().ewm(adjust=False, alpha=sma_alpha3,axis=1).mean().stack()
        _ret = _data.g089.unstack().T
        _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja089" + str(w)]], names=["futures", "price"])
        ret = pd.concat([ret, _ret], axis=1)
    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret

gtja090_time_lst = [12, 24, 36, 48, 60, 72]
def GTJA090(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA090
    rank(corr(rank(vwap), rank(volume), 5), -1)

    :param data 默认旋转后，即 data_rotated
    :param window 是一个一维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return ret
    '''

    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]

    ret = pd.DataFrame()
    for w in window:
        N = w

        _data = data.loc[futures_universe]
        _data['vwapw'] = (_data.openw + _data.closew + _data.highw + _data.loww) / 4
        _data['rank1'] = _data.vwapw.unstack().rank(pct=True).stack()
        _data['rank2'] = _data.volume99.unstack().rank(pct=True).stack()

        _data['g090'] = _data.rank1.rolling(N).corr(_data.rank2)
        _ret = _data.g090.unstack().T
        _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja090" + str(w)]], names=["futures", "price"])
        ret = pd.concat([ret, _ret], axis=1)
    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret

gtja091_time_lst = [
    [3,12],
    [6,18],
    [9,27],
    [13,40],
    [17,51],
    [25,75],
    ]
def GTJA091(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA091
    原公式中的 max 改成了 tsmax
    rank(close - tsmax(close, 5)) * rank(corr(mean(volume, 40), low, 5))

    :param data 默认旋转后，即 data_rotated
    :param window 是一个一维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return ret
    '''

    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]

    def tsmax(array):
        s = pd.Series(array)
        return np.array(s.rank(method="max", ascending=False))[-1]

    ret = pd.DataFrame()
    for w in window:
        N1, N2 = w

        _data = data.loc[futures_universe]

        _data['inner1'] = _data.closew - _data.closew.unstack().rolling(N1,axis=1).apply(tsmax).stack()
        _data['rank1'] = _data.inner1.unstack().rank(pct=True).stack()
        _data['inner2'] = _data.loww.unstack().rolling(N1,axis=1).corr(_data.volume99.unstack().rolling(N2,axis=1).mean().stack()).stack()
        _data['rank2'] = _data.inner2.unstack().rank(pct=True).stack()

        _data['g091'] = _data.rank1 * _data.rank2
        _ret = _data.g091.unstack().T
        _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja091" + str(w)]], names=["futures", "price"])
        ret = pd.concat([ret, _ret], axis=1)

# =============================================================================
#     for w in window:
#         for name in futures_universe:
#             N1, N2 = w
#     
#             _data = data.loc[name]
#     
#             _data['inner1'] = _data.close - _data.close.rolling(N1).apply(tsmax)
#             _data['rank1'] = _data.inner1.unstack().rank(pct=True).stack()
#             _data['inner2'] = _data.low.rolling(N1).corr(_data.volume.rolling(N2).mean())
#             _data['rank2'] = _data.inner2.unstack().rank(pct=True).stack()
#     
#             _data['g091'] = _data.rank1 * _data.rank2
#             new_columns = pd.MultiIndex.from_product([[str(name)], ["gtja091" + str(w)]])  
#             temp = _data['g091'].to_frame(name='g091').set_index(_data.index)  
#             temp.columns = new_columns  
#             ret = pd.concat([ret, temp], axis=1)
# =============================================================================
    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret

gtja093_time_lst = [12, 24, 36, 48, 60, 72]
def GTJA093(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA093
    sum(open >= delay(open, 1) ? 0 : max(open - low, open - delay(open, 1)), 20)

    :param data 默认旋转后，即 data_rotated
    :param window 是一个一维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return ret
    '''

    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]

    ret = pd.DataFrame()
    for w in window:
        N = w

        _data = data.loc[futures_universe]

        _data['inner1'] = _data.openw - _data.loww
        _data['inner2'] = _data.openw.unstack().diff(1,axis=1).stack()

        _data['inner'] = 0
        _data.loc[_data.inner2 < 0, "inner"] = _data[['inner1', 'inner2']].max(axis=1)

        _data['g093'] = _data.inner.unstack().rolling(N,axis=1).sum().stack()
        _ret = _data.g093.unstack().T
        _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja093" + str(w)]], names=["futures", "price"])
        ret = pd.concat([ret, _ret], axis=1)
    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret

gtja094_time_lst = [12, 24, 36, 48, 60, 72]
def GTJA094(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA094
    sum(close > delay(close, 1) ? volume : (close < delay(close, 1) ? -volume : 0), 30)

    :param data 默认旋转后，即 data_rotated
    :param window 是一个一维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return ret
    '''

    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]

    ret = pd.DataFrame()
    for w in window:
        N = w

        _data = data.loc[futures_universe]

        _data['inner'] = 0
        _data.loc[_data.closew > _data.closew.shift(1), "inner"] = _data.volume99
        _data.loc[_data.closew < _data.closew.shift(1), "inner"] = -_data.volume99

        _data['g094'] = _data.inner.unstack().rolling(N,axis=1).sum().stack()
        _ret = _data.g094.unstack().T
        _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja094" + str(w)]], names=["futures", "price"])
        ret = pd.concat([ret, _ret], axis=1)
    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret


gtja095_time_lst = [ 6, 12, 24, 36, 54, 72]
def GTJA095(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA095
    -1 * std(money, 20)

    :param data 默认旋转后，即 data_rotated
    :param window 是一个一维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return ret
    '''

    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]

    ret = pd.DataFrame()
    for w in window:
        N = w

        _data = data.loc[futures_universe]

        _data['g095'] = _data.total_turnover.unstack().rolling(N,axis=1).std().stack()
        _ret = _data.g095.unstack().T
        _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja095" + str(w)]], names=["futures", "price"])
        ret = pd.concat([ret, _ret], axis=1)
    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret


gtja096_time_lst = [
    [6,1/3],
    [9,1/3],
    [12,1/6],
    [18,1/6],
    [30,1/12],
    [48,1/12],
]
def GTJA096(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA096
    sma(sma((close - tsmin(low, 9)) / (tsmax(high, 9) - tsmin(low, 9)), 3, 1), 3, 1)

    :param data 默认旋转后，即 data_rotated
    :param window 是一个一维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return ret
    '''

    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]

    def tsmin(array):
        s = pd.Series(array)
        return np.array(s.rank(method="min", ascending=False))[-1]

    def tsmax(array):
        s = pd.Series(array)
        return np.array(s.rank(method="max", ascending=False))[-1]

    ret = pd.DataFrame()
    for w in window:
        N,sma_alpha = w

        _data = data.loc[futures_universe]

        _data['inner'] = (_data.closew - _data.loww.rolling(N).apply(tsmin)) / (_data.highw.rolling(N).apply(tsmax) - _data.loww.rolling(N).apply(tsmin))

        _data['g096'] = _data.inner.unstack().ewm(adjust=False, alpha=sma_alpha,axis=1).mean().stack()
        _ret = _data.g096.unstack().T
        _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja096" + str(w)]], names=["futures", "price"])
        ret = pd.concat([ret, _ret], axis=1)
    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret


gtja097_time_lst = [6, 12, 24, 36, 54, 72]
def GTJA097(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA097
    std(volume, 10)

    :param data 默认旋转后，即 data_rotated
    :param window 是一个一维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return ret
    '''

    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]

    ret = pd.DataFrame()
    for w in window:
        N = w

        _data = data.loc[futures_universe]

        _data['g097'] = _data.volume99.unstack().rolling(N,axis=1).std().stack()
        _ret = _data.g097.unstack().T
        _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja097" + str(w)]], names=["futures", "price"])
        ret = pd.concat([ret, _ret], axis=1)
    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret


gtja098_time_lst = [
    [6,2,0.05],
    [12,4,0.05],
    [24,8,0.05],
    [48,16,0.05],
    [96,32,0.05],
    [192,64,0.05],
]
def GTJA098(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA098
    (delta(sum(close, 100) / 100, 100) / delay(close, 100) <= 0.05) ? (-1 * (close - tsmin(close, 100))) : (-1 * delta(close, 3))

    :param data 默认旋转后，即 data_rotated
    :param window 是一个一维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return ret
    '''

    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]

    def tsmin(array):
        s = pd.Series(array)
        return np.array(s.rank(method="min", ascending=False))[-1]

    ret = pd.DataFrame()
    for w in window:
        N1, N2,const = w

        _data = data.loc[futures_universe]

        _data['inner1'] = _data.closew.unstack().rolling(N1,axis=1).mean().stack()
        _data['inner'] = _data.inner1.unstack().diff(N1,axis=1).stack() / _data.closew.unstack().shift(N1,axis=1).stack()

        _data['g098'] = -1 * _data.closew.unstack().diff(N2,axis=1).stack()
        _data.loc[_data.inner <= const, "g098"] = -1 * (_data.closew - _data.closew.unstack().rolling(N1,axis=1).apply(tsmin).stack())

        _ret = _data.g098.unstack().T
        _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja098" + str(w)]], names=["futures", "price"])
        ret = pd.concat([ret, _ret], axis=1)
    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret


gtja099_time_lst = [6, 12, 24, 36, 54, 72]
def GTJA099(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA099
    - covariance(rank(close), rank(volume), 5)

    :param data 默认旋转后，即 data_rotated
    :param window 是一个一维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return ret
    '''

    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]

    ret = pd.DataFrame()
    for w in window:
        N = w

        _data = data.loc[futures_universe]

        _data['rank1'] = _data.closew.unstack().rank(pct=True).stack()
        _data['rank2'] = _data.volume99.unstack().rank(pct=True).stack()

        _data['g099'] = -1 * _data.rank1.rolling(N).cov(_data.rank2)
        _ret = _data.g099.unstack().T
        _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja099" + str(w)]], names=["futures", "price"])
        ret = pd.concat([ret, _ret], axis=1)
    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret


gtja100_time_lst = [ 6, 12, 24, 36, 54, 72]
def GTJA100(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA100
    std(volume, 20)

    :param data 默认旋转后，即 data_rotated
    :param window 是一个一维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return ret
    '''

    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]

    ret = pd.DataFrame()
    for w in window:
        N = w

        _data = data.loc[futures_universe]

        _data['g100'] = _data.volume99.unstack().rolling(N,axis=1).std().stack()
        _ret = _data.g100.unstack().T
        _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja100" + str(w)]], names=["futures", "price"])
        ret = pd.concat([ret, _ret], axis=1)
    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret


gtja102_time_lst = [2, 3, 6, 12, 18, 24]
def GTJA102(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA102
    - sma(max(delta(volume, 1), 0), 6, 1) / sma(abs(delta(volume, 1)), 6, 1)

    :param data 默认旋转后，即 data_rotated
    :param window 是一个一维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return ret
    '''

    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]

    ret = pd.DataFrame()
    for w in window:
        sma_alpha = 1. / w

        _data = data.loc[futures_universe]

        _data['delta_volume'] = _data.volume99.unstack().diff(1,axis=1).stack()
        _data['const'] = 0.

        _data['inner1'] = (_data[['delta_volume', 'const']].max(axis=1)).unstack().ewm(adjust=False, alpha=sma_alpha,axis=1).mean().stack()
        _data['inner2'] = (np.abs(_data.volume99.diff(1))).unstack().ewm(adjust=False, alpha=sma_alpha,axis=1).mean().stack()

        _data['g102'] = - _data.inner1 / _data.inner2
        _ret = _data.g102.unstack().T
        _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja102" + str(w)]], names=["futures", "price"])
        ret = pd.concat([ret, _ret], axis=1)
    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret


gtja103_time_lst = [4, 12, 20, 32, 48, 60]

def GTJA103(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA103
    20 - lowday(low, 20)

    :param data 默认旋转后，即 data_rotated
    :param window 是一个一维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return ret
    '''

    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]

    ret = pd.DataFrame()
    for w in window:
        N = w

        _data = data.loc[futures_universe]

        _data['g103'] = _data.loww.unstack().rolling(N,axis=1).apply(np.argmin).stack()
        _ret = _data.g103.unstack().T
        _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja103" + str(w)]], names=["futures", "price"])
        ret = pd.concat([ret, _ret], axis=1)
    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret


gtja104_time_lst = [[4,2,10], 
                    [6,2,20],
                    [9,2,20],
                    [15,2,40],
                    [25,2,40],
                    [40,2,40],
                    ]

def GTJA104(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA104
    - delta(corr(high, volume, 5), 5) * rank(std(close, 20))

    :param data 默认旋转后，即 data_rotated
    :param window 是一个一维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return ret
    '''

    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]

    ret = pd.DataFrame()
    for w in window:
        N1, N2, N3 = w

        _data = data.loc[futures_universe]

        _data['inner1'] = (_data.highw.rolling(N1).corr(_data.volume99)).diff(N2)
        _data['inner2'] = _data.closew.unstack().rolling(N3,axis=1).std().stack()
        _data['rank2'] = _data.inner2.unstack().rank(pct=True).stack()

        _data['g104'] = - _data.inner1 * _data.rank2
        _ret = _data.g104.unstack().T
        _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja104" + str(w)]], names=["futures", "price"])
        ret = pd.concat([ret, _ret], axis=1)
    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret


gtja105_time_lst = [4, 12, 20, 32, 48, 60]
def GTJA105(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA105
    -1 * corr(rank(open), rank(volume), 10)

    :param data 默认旋转后，即 data_rotated
    :param window 是一个一维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return ret
    '''

    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]

    ret = pd.DataFrame()
    for w in window:
        N = w

        _data = data.loc[futures_universe]

        _data['rank1'] = _data.openw.unstack().rank(pct=True).stack()
        _data['rank2'] = _data.volume99.unstack().rank(pct=True).stack()

        _data['g105'] = -1 * _data.rank1.rolling(N).corr(_data.rank2)
        _ret = _data.g105.unstack().T
        _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja105" + str(w)]], names=["futures", "price"])
        ret = pd.concat([ret, _ret], axis=1)


    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret


gtja106_time_lst = [4, 12, 20, 32, 48, 60]
def GTJA106(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA106
    close - delay(close, 20)

    :param data 默认旋转后，即 data_rotated
    :param window 是一个一维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return ret
    '''

    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]

    ret = pd.DataFrame()
    for w in window:
        N = w

        _data = data.loc[futures_universe]

        _data['g106'] = _data.closew.unstack().diff(N,axis=1).stack()
        _ret = _data.g106.unstack().T
        _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja106" + str(w)]], names=["futures", "price"])
        ret = pd.concat([ret, _ret], axis=1)


    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret


gtja107_time_lst = [1, 2, 6, 11, 21, 31]
def GTJA107(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA107
    rank(open - delay(high, 1)) * rank(open - delay(close, 1)) * rank(open - delay(low, 1))

    :param data 默认旋转后，即 data_rotated
    :param window 是一个一维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return ret
    '''

    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]

    ret = pd.DataFrame()
    for w in window:
        N = w

        _data = data.loc[futures_universe]

        _data['inner1'] = _data.openw - _data.highw.unstack().shift(N,axis=1).stack()
        _data['rank1'] = _data.inner1.unstack().rank(pct=True).stack()
        _data['inner2'] = _data.openw - _data.closew.unstack().shift(N,axis=1).stack()
        _data['rank2'] = _data.inner2.unstack().rank(pct=True).stack()
        _data['inner3'] = _data.openw - _data.loww.unstack().shift(N,axis=1).stack()
        _data['rank3'] = _data.inner3.unstack().rank(pct=True).stack()

        _data['g107'] = _data.rank1 * _data.rank2 * _data.rank3
        _ret = _data.g107.unstack().T
        _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja107" + str(w)]], names=["futures", "price"])
        ret = pd.concat([ret, _ret], axis=1)


    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret


gtja108_time_lst =[
    [2,20,6],
    [2,40,6],
    [2,60,6],
    [2,80,6],
    [2,100,6],
    [2,120,6],
    ]
def GTJA108(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA108
    -1 * rank(high - min(high, 2)) ^ rank(corr(vwap, mean(volume, 120), 6))

    :param data 默认旋转后，即 data_rotated
    :param window 是一个一维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return ret
    '''

    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]

    ret = pd.DataFrame()
    for w in window:
        _const, N1, N2 = w

        _data = data.loc[futures_universe]
        _data['vwapw'] = (_data.openw + _data.closew + _data.highw + _data.loww) / 4
        _data['const'] = _const
        _data['inner1'] = _data.highw - _data[['highw', 'const']].min(axis=1)
        _data['rank1'] = _data.inner1.unstack().rank(pct=True).stack()

        _data['inner2'] = (_data.volume99.rolling(N1).mean()).rolling(N2).corr(_data.vwapw)
        _data['rank2'] = _data.inner2.unstack().rank(pct=True).stack()

        _data['g108'] = -1 * np.power(_data.rank1, _data.rank2)
        _ret = _data.g108.unstack().T
        _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja108" + str(w)]], names=["futures", "price"])
        ret = pd.concat([ret, _ret], axis=1)
    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret


gtja109_time_lst = [2, 3, 6, 11, 21, 31]
def GTJA109(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA109
    sma(high - low, 10, 2) / sma(sma(high - low, 10, 2), 10, 2)

    :param data 默认旋转后，即 data_rotated
    :param window 是一个一维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return ret
    '''

    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]

    ret = pd.DataFrame()
    for w in window:
        sma_alpha = 1. / w

        _data = data.loc[futures_universe]

        _data['inner1'] = (_data.highw - _data.loww).unstack().ewm(adjust=False, alpha=sma_alpha,axis=1).mean().stack()
        _data['inner2'] = _data.inner1.unstack().ewm(adjust=False, alpha=sma_alpha,axis=1).mean().stack()

        _data['g109'] = - _data.inner1 / _data.inner2
        _ret = _data.g109.unstack().T
        _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja109" + str(w)]], names=["futures", "price"])
        ret = pd.concat([ret, _ret], axis=1)
    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret


gtja110_time_lst = [2, 3, 6, 11, 21, 31]
def GTJA110(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA110
    sum(max(0, high - delay(close, 1)), 20) / sum(max(0, delay(close, 1) - low), 20)

    :param data 默认旋转后，即 data_rotated
    :param window 是一个一维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return ret
    '''

    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]

    ret = pd.DataFrame()
    for w in window:
        N = w

        _data = data.loc[futures_universe]

        _data['const'] = 0

        _data['inner_price_1'] = _data.highw - _data.closew.unstack().shift(1,axis=1).stack()
        _data['inner_price_2'] = _data.closew.unstack().shift(1,axis=1).stack() - _data.loww

        _data['inner1'] = _data[['inner_price_1', 'const']].max(axis=1)
        _data['inner2'] = _data[['inner_price_2', 'const']].max(axis=1)

        _data['g110'] = _data.inner1.unstack().rolling(N,axis=1).sum().stack() / (_data.inner2.unstack().rolling(N,axis=1).sum().stack()+1e-6)
        _ret = _data.g110.unstack().T
        _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja110" + str(w)]], names=["futures", "price"])
        ret = pd.concat([ret, _ret], axis=1)
    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret


gtja111_time_lst = [[1/3, 1/2],
                    [1/4, 1/2],
                    [1/6, 1/3],
                    [1/11, 1/4],
                    [1/21, 1/8],
                    [1/31, 1/12],
                    ]
def GTJA111(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA111
    sma(volume * ((close - low) - (high - close)) / (high - low), 11, 2) - sma(volume * ((close - low) - (high - close)) / (high - low), 4, 2)

    :param data 默认旋转后，即 data_rotated
    :param window 是一个一维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return ret
    '''

    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]

    ret = pd.DataFrame()
    for w in window:
        sma_alpha1, sma_alpha2 = w

        _data = data.loc[futures_universe]

        _data['inner'] = _data.volume99 * ((_data.closew - _data.loww) - (_data.highw - _data.closew)) / (_data.highw - _data.loww)

        _data['g111'] = _data.inner.unstack().ewm(adjust=False, alpha=sma_alpha1,axis=1).mean().stack()
        - _data.inner.unstack().ewm(adjust=False, alpha=sma_alpha2,axis=1).mean().stack()
        _ret = _data.g111.unstack().T
        _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja111" + str(w)]], names=["futures", "price"])
        ret = pd.concat([ret, _ret], axis=1)
    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret


gtja112_time_lst = [2, 3, 6, 11, 21, 31]
def GTJA112(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA112
    sum(delta(close, 1) > 0 ? delta(close, 1) : 0, 12) - sum(delta(close,1) <0 ? -delta(close,1) :0,12) / sum(delta(close,1) >0 ? delta(close,1) :0,12) + sum(delta(close,1) <0 ? -delta(close,1) :0,12)

    :param data 默认旋转后，即 data_rotated
    :param window 是一个一维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return ret
    '''

    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]

    ret = pd.DataFrame()
    for w in window:
        N = w

        _data = data.loc[futures_universe]

        _data['close1'] = _data.closew.unstack().diff(1,axis=1).stack()
        _data['const'] = 0.
        _data['inner_price_1'] = _data[['close1', 'const']].max(axis=1)
        _data['inner_price_2'] = _data[['close1', 'const']].min(axis=1)
        _data['inner1'] = _data.inner_price_1.unstack().rolling(N,axis=1).sum().stack()
        _data['inner2'] = _data.inner_price_2.unstack().rolling(N,axis=1).sum().stack()

        _data['g112'] = (_data.inner1 - _data.inner2) / (_data.inner1 + _data.inner2)
        _ret = _data.g112.unstack().T
        _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja112" + str(w)]], names=["futures", "price"])
        ret = pd.concat([ret, _ret], axis=1)
    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret


gtja113_time_lst = [
    [3,1,6],
    [4,2,6],
    [6,2,6],
    [6,2,10],
    [10,2,10],
    [20,4,20],
    ]

def GTJA113(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA113
    -1 * rank(mean(delay(close, 5), 20)) * corr(close, volume, 2) * rank(corr(sum(close, 5), sum(close, 20), 2))

    :param data 默认旋转后，即 data_rotated
    :param window 是一个一维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return ret
    '''

    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]

    ret = pd.DataFrame()
    for w in window:
        N1, N2, N3 = w

        _data = data.loc[futures_universe]

        _data['inner1'] = (_data.closew.unstack().shift(N2,axis=1)).rolling(N3,axis=1).mean().stack()
        _data['rank1'] = _data.inner1.unstack().rank(pct=True).stack()
        _data['inner2'] = _data.closew.unstack().rolling(N1,axis=1).corr(_data.volume99.unstack()).stack()
        _data['inner3'] = (_data.closew.unstack().rolling(N2,axis=1).sum()).rolling(N1,axis=1).corr(_data.closew.unstack().rolling(N3,axis=1).sum()).stack()
        _data['rank3'] = _data.inner3.unstack().rank(pct=True).stack()

        _data['g113'] = -1 * _data.rank1 * _data.inner2 * _data.rank3
        _ret = _data.g113.unstack().T
        _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja113" + str(w)]], names=["futures", "price"])
        ret = pd.concat([ret, _ret], axis=1)
    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret


gtja114_time_lst = [ [5, 2], [10, 2], [15, 2], [10,5 ], [20, 5], [20, 10]]
def GTJA114(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA114
    rank(delay((high - low) / mean(close, 5), 2)) * rank(volume) / (((high - low) / mean(close, 5)) / (vwap - close))

    :param data 默认旋转后，即 data_rotated
    :param window 是一个一维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return ret
    '''

    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]

    ret = pd.DataFrame()
    for w in window:
        N1, N2 = w

        _data = data.loc[futures_universe]
        _data['vwapw'] = (_data.openw + _data.closew + _data.highw + _data.loww) / 4

        _data['inner'] = (_data.highw - _data.loww) / (_data.closew.unstack().rolling(N1,axis=1).mean().stack()+1e-6)
        _data['inner1'] = _data.inner.unstack().shift(N2,axis=1).stack()
        _data['rank1'] = _data.inner1.unstack().rank(pct=True).stack()

        _data['rank2'] = _data.volume99.unstack().rank(pct=True).stack()

        _data['inner3'] = _data.inner / ((_data.vwapw - _data.closew)+1e-6)

        _data['g114'] = _data.rank1 * _data.rank2 / (_data.inner3+1e-6)
        _ret = _data.g114.unstack().T
        _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja114" + str(w)]], names=["futures", "price"])
        ret = pd.concat([ret, _ret], axis=1)
    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret


gtja116_time_lst = [2, 3, 6, 11, 21, 31]
def GTJA116(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA116
    regbeta(close, sequence, 20)
    其中 regbeta(A, B, n) 是前 n 期 A 对 B 做回归得到的回归系数, 相比 021, 没有使用平均, 但天数更长
    
    :param data 默认旋转后，即 data_rotated
    :param window 是一个一维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return ret
    '''

    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]

    ret = pd.DataFrame()
    for w in window:
        N = w

        _data = data.loc[futures_universe]

        _data['g116'] = _data.closew.unstack().rolling(N,axis=1).apply(lambda y: np.polyfit(y=y, x=np.arange(N), deg=1)[0]).stack()
        _ret = _data.g116.unstack().T
        _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja116" + str(w)]], names=["futures", "price"])
        ret = pd.concat([ret, _ret], axis=1)
    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret


gtja117_time_lst = [
    [6,3],
    [12,6],
    [20,10],
    [32,16],
    [48,24],
    [60,30],
    ]
def GTJA117(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA117
    tsrank(volume, 32) * (1 - tsrank(close + high - low, 16)) * (1 - tsrank(ret, 32))
    
    :param data 默认旋转后，即 data_rotated
    :param window 是一个一维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return ret
    '''

    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]

    def tsrank(array):
        s = pd.Series(array)
        return np.array(s.rank(method="min", ascending=False))[-1]

    ret = pd.DataFrame()
    for w in window:
        N1, N2 = w

        _data = data.loc[futures_universe]
        _data['chl'] = _data.closew + _data.highw -_data.loww
        _data['inner1'] = _data.volume99.unstack().rolling(N1,axis=1).apply(tsrank).stack()
        _data['inner2'] = 1 - _data.chl.unstack().rolling(N2,axis=1).apply(tsrank).stack()
        _data['inner3'] = 1 - _data.closew.unstack().rolling(N1,axis=1).apply(tsrank).stack()

        _data['g117'] = _data.inner1 * _data.inner2 * _data.inner3
        _ret = _data.g117.unstack().T
        _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja117" + str(w)]], names=["futures", "price"])
        ret = pd.concat([ret, _ret], axis=1)
    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret


# gtja118_time_lst = [4, 12, 20, 32, 48, 60]
gtja118_time_lst = [5, 10, 20, 40, 60, 120]
# =============================================================================
# def GTJA118(data, window, futures_universe, time_freq='day'):
#     '''
#     计算 GTJA118
#     sum(high - open, 20) / sum(open - low, 20)
#     
#     :param data 默认旋转后，即 data_rotated
#     :param window 是一个一维数组
#     :param futures_universe 是计算 rank 的范围。默认是 gu24
#     :return ret
#     '''
# 
#     
#     volume = data.volume99[futures_universe]
#     open_price = data.open[futures_universe]
#     openw = data.openw[futures_universe]
#     high = data.high[futures_universe]
#     highw = data.highw[futures_universe]
#     low = data.low[futures_universe]
#     loww = data.loww[futures_universe]
#     close = data.close[futures_universe]
#     closew = data.closew[futures_universe]
# 
#     ret = pd.DataFrame()
#     for w in window:
#         N = w
# 
#         _data = data.loc[futures_universe]
# 
#         _data['g118'] = (_data.highw - _data.openw).unstack().rolling(N,axis=1).sum().stack() / ((_data.openw - _data.loww).unstack().rolling(N,axis=1).sum().stack()+1e-6)
#         _ret = _data.g118.unstack().T
#         _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja118" + str(w)]], names=["futures", "price"])
#         ret = pd.concat([ret, _ret], axis=1)
#     # 把 open 和 openw 贴进去
#     _open = open_price.unstack().T
#     _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
#     _openw = openw.unstack().T
#     _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
#     ret = pd.concat([ret, _open, _openw], axis=1)
# 
#     return ret
# =============================================================================


def GTJA118(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA118
    sum(high - open, 20) / sum(open - low, 20)
    
    :param data 默认旋转后，即 data_rotated
    :param window 是一个一维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return ret
    '''

    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]

    ret = pd.DataFrame()
    for w in window:
        N = w
        for name in futures_universe:
            N= w
    
            _data = data.loc[name]
            _data['g118'] = (_data.highw - _data.openw).rolling(N).sum() / ((_data.openw - _data.loww).rolling(N).sum()+1e-6)
            new_columns = pd.MultiIndex.from_product([[str(name)], ["gtja118" + str(w)]])  
            temp = _data['g118'].to_frame(name='g118').set_index(_data.index)  
            temp.columns = new_columns  
            ret = pd.concat([ret, temp], axis=1)
# =============================================================================
#     for w in window:
#         N = w
# 
#         _data = data.loc[futures_universe]
# 
#         _data['g118'] = (_data.highw - _data.openw).unstack().rolling(N,axis=1).sum().stack() / ((_data.openw - _data.loww).unstack().rolling(N,axis=1).sum().stack()+1e-6)
#         _ret = _data.g118.unstack().T
#         _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja118" + str(w)]], names=["futures", "price"])
#         ret = pd.concat([ret, _ret], axis=1)
# =============================================================================
    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret


gtja120_time_lst = [4, 12, 20, 32, 48, 60]
def GTJA120(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA120
    rank(vwap - close) / rank(vwap + close)
    
    :param data 默认旋转后，即 data_rotated
    :param window 是一个一维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return ret
    '''

    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]

    ret = pd.DataFrame()
    for w in window:
        N = w

        _data = data.loc[futures_universe]
        _data['vwapw'] = (_data.openw + _data.closew + _data.highw + _data.loww) / 4
        _data['rank1'] = (_data.vwapw - _data.closew).unstack().rank(pct=True).stack()
        _data['rank2'] = (_data.vwapw + _data.closew).unstack().rank(pct=True).stack()

        _data['g120'] = _data.rank1 / _data.rank2
        _ret = _data.g120.unstack().T
        _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja120" + str(w)]], names=["futures", "price"])
        ret = pd.concat([ret, _ret], axis=1)
    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret


gtja122_time_lst = [4, 6, 10, 13, 18, 30]
def GTJA122(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA122
    delta(sma(sma(sma(log(close), 13, 2), 13, 2), 13, 2), 1) / delay(sma(sma(sma(log(close), 13, 2), 13, 2), 13, 2), 1)

    :param data 默认旋转后，即 data_rotated
    :param window 是一个一维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return ret
    '''

    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]

    # sma_alpha = 2. / 13

    ret = pd.DataFrame()
    for w in window:
        N1 = w
        sma_alpha = 2./N1
        _data = data.loc[futures_universe]
        _data['log_closew'] = np.log(_data.closew)

        _data['inner'] = (((_data.log_closew.ewm(adjust=False, alpha=sma_alpha).mean()).ewm(adjust=False, alpha=sma_alpha).mean()).ewm(adjust=False, alpha=sma_alpha).mean())

        _data['g122'] = _data.inner.unstack().diff(1,axis=1).stack() / _data.inner.unstack().shift(1,axis=1).stack()
        _ret = _data.g122.unstack().T
        _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja122" + str(w)]], names=["futures", "price"])
        ret = pd.concat([ret, _ret], axis=1)
    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret


gtja123_time_lst = [
    [10,6,3,3],
    [20,8,4,2],
    [30,10,5,3],
    [60,20,9,6],
    [90,30,16,12],
    [120,40,30,20],
    ]
def GTJA123(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA123
    rank(corr(sum((high + low) / 2, 20), sum(mean(volume, 60), 20), 9)) < rank(corr(low, volume, 6))

    :param data 默认旋转后，即 data_rotated
    :param window 是一个一维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return ret
    '''

    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]

    ret = pd.DataFrame()
    for w in window:
        N1, N2, N3, N4 = w

        _data = data.loc[futures_universe]

        _data['inner11'] = ((_data.highw + _data.loww) / 2).unstack().rolling(N2,axis=1).sum().stack()
        _data['inner12'] = (_data.volume99.unstack().rolling(N1,axis=1).mean()).rolling(N2,axis=1).sum().stack()
        _data['inner1'] = _data.inner11.unstack().rolling(N3,axis=1).corr(_data.inner12.unstack()).stack()
        _data['rank1'] = _data.inner1.unstack().rank(pct=True).stack()

        _data['inner2'] = _data.loww.unstack().rolling(N4,axis=1).corr(_data.volume99.unstack()).stack()
        _data['rank2'] = _data.inner2.unstack().rank(pct=True).stack()

        _data['g123'] = _data.rank1 - _data.rank2
        _ret = _data.g123.unstack().T
        _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja123" + str(w)]], names=["futures", "price"])
        ret = pd.concat([ret, _ret], axis=1)
    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret


gtja124_time_lst = [
    [6,2],
    [12,2],
    [18,2],
    [30,2],
    [48,4],
    [60,4],
    ]
def GTJA124(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA124
    (close - vwap) / decaylinear(rank(tsmax(close, 30)), 2)

    :param data 默认旋转后，即 data_rotated
    :param window 是一个一维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return ret
    '''

    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]

    def wma(series, N):
        return series.unstack().rolling(N,axis=1).apply(lambda x: x[:: -1].cumsum().sum() * 2 / N / (N + 1)).stack()

    ret = pd.DataFrame()
    for w in window:
        N1, N2 = w

        _data = data.loc[futures_universe]
        _data['vwapw'] = (_data.openw + _data.closew + _data.highw + _data.loww) / 4
        _data['inner2'] = _data.closew.unstack().rolling(N1,axis=1).max().stack()
        _data['rank2'] = _data.inner2.unstack().rank(pct=True).stack()

        _data['g124'] = (_data.closew - _data.vwapw) / wma(_data.rank2, N2)
        _ret = _data.g124.unstack().T
        _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja124" + str(w)]], names=["futures", "price"])
        ret = pd.concat([ret, _ret], axis=1)
    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret


gtja125_time_lst = [
    [20,3,10,1,10],
    [30,17,20,3,16],
    [40,17,20,3,16],
    [80,17,20,3,16],
    [120,17,20,3,16],
    [180,17,20,3,16],
    ]
def GTJA125(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA125
    rank(decaylinear(corr(vwap, mean(volume, 180), 17), 20)) / rank(decaylinear(delta(close * .5 + vwap * .5, 3), 16))

    :param df 是带有原始数据的 DataFrame
    :param universe 需要计算因子值的股票, string 的 list
    :param start_date 开始计算日期
    :param end_date 结束计算日期
    :return: 因子值, index 为日期, column 为股票代码
    '''
    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]
    ret = pd.DataFrame()
    def wma(series, N):
        return series.unstack().rolling(N,axis=1).apply(lambda x: x[:: -1].cumsum().sum() * 2 / N / (N + 1)).stack()
    for w in window:
        N1,N2,N3,N4,N5 = w
        _data = data.loc[futures_universe]
        _data['vwapw'] = (_data.openw + _data.closew + _data.highw + _data.loww) / 4
        _data['inner1'] = wma(_data.vwapw.unstack().rolling(N2,axis=1).corr(_data.volume99.unstack().rolling(N1,axis=1).mean()).stack(), N3)
        _data['rank1'] = _data.inner1.unstack().rank(pct=True).stack()
    
        _data['inner2'] = wma((_data.closew * .5 + _data.vwapw * .5).unstack().diff(N4,axis=1).stack(), N5)
        _data['rank2'] = _data.inner2.unstack().rank(pct=True).stack()
    
        _data['g125'] = _data.rank1 / _data.rank2
        _ret = _data.g125.unstack().T
        _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja125" + str(w)]], names=["futures", "price"])
        ret = pd.concat([ret, _ret], axis=1)
    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret


gtja126_time_lst = [4, 12, 20, 32, 48, 60]
def GTJA126(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA126
    (close + high + low) / 3

    :param data 默认旋转后，即 data_rotated
    :param window 是一个一维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return ret
    '''

    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]

    ret = pd.DataFrame()
    for w in window:
        _data = data.loc[futures_universe]

        _data['g126'] = (_data.closew + _data.highw + _data.loww) / 3
        _ret = _data.g126.unstack().T
        _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja126" + str(w)]], names=["futures", "price"])
        ret = pd.concat([ret, _ret], axis=1)
    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret


gtja127_time_lst =[
    [2,1],
    [4,1],
    [8,1],
    [12,2],
    [18,3],
    [30,4],
    ]
def GTJA127(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA127
    mean((close - tsmax(close, 12)) / tsmax(close, 12), 1)

    :param data 默认旋转后，即 data_rotated
    :param window 是一个一维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return ret
    '''

    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]

    ret = pd.DataFrame()
    for w in window:
        N1, N2 = w

        _data = data.loc[futures_universe]

        _data['inner'] = (_data.closew - _data.closew.unstack().rolling(N1,axis=1).max().stack()) / _data.closew.unstack().rolling(N1,axis=1).max().stack()

        _data['g127'] = _data.inner.unstack().rolling(N2,axis=1).mean().stack()
        _ret = _data.g127.unstack().T
        _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja127" + str(w)]], names=["futures", "price"])
        ret = pd.concat([ret, _ret], axis=1)
    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret


# gtja128_time_lst = [4, 12, 20, 32, 48, 60]
gtja128_time_lst = [7, 12, 26, 44, 68, 100]
def GTJA128(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA128
    100 - (100 / (1 + sum(high + low + close > delay(high + low + close, 1) ? (high + low + close) / 3 * volume : 0, 14) / sum(high + low + close < delay(high + low + close, 1) ? (high + low + close) / 3 * volume : 0, 14)))

    :param data 默认旋转后，即 data_rotated
    :param window 是一个一维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return ret
    '''

    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]

    ret = pd.DataFrame()
    for w in window:
        N = w

        _data = data.loc[futures_universe]

        _data['inner_price'] = (_data.highw + _data.loww + _data.closew) / 3

        _data['inner1'] = 0
        _data.loc[_data.inner_price > _data.inner_price.shift(1), "inner1"] = _data.inner_price * _data.volume99

        _data['inner2'] = 0
        _data.loc[_data.inner_price < _data.inner_price.shift(1), "inner2"] = _data.inner_price * _data.volume99

        _data['g128'] = 100 - (100 / (1 + _data.inner1.unstack().rolling(N,axis=1).sum().stack() / _data.inner2.unstack().rolling(N,axis=1).sum().stack()))
        _ret = _data.g128.unstack().T
        _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja128" + str(w)]], names=["futures", "price"])
        ret = pd.concat([ret, _ret], axis=1)
    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret


gtja129_time_lst = [3, 6, 12, 20, 32, 48]
def GTJA129(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA129
    sum(delta(close, 1) < 0 ? abs(delta(close, 1)) : 0, 12)

    :param data 默认旋转后，即 data_rotated
    :param window 是一个一维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return ret
    '''

    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]

    ret = pd.DataFrame()
    for w in window:
        N = w

        _data = data.loc[futures_universe]

        _data['inner_price'] = _data.closew.unstack().diff(1,axis=1).stack()

        _data['inner'] = 0
        _data.loc[_data.inner_price < 0, "inner"] = -_data.inner_price

        _data['g129'] = _data.inner.unstack().rolling(N,axis=1).sum().stack()
        _ret = _data.g129.unstack().T
        _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja129" + str(w)]], names=["futures", "price"])
        ret = pd.concat([ret, _ret], axis=1)
    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret


gtja130_time_lst = [
    [20,3,10,7,3],
    [30,17,20,7,3],
    [40,17,20,7,3],
    [80,17,20,7,3],
    [120,17,20,7,3],
    [180,17,20,7,3],
    ]
def GTJA130(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA130
    rank(decaylinear(corr(high + low, mean(volume, 40), 9), 10)) / 
    rank(decaylinear(corr(rank(vwap), rank(volume), 7), 3))

    :param df 是带有原始数据的 DataFrame
    :param universe 需要计算因子值的股票, string 的 list
    :param start_date 开始计算日期
    :param end_date 结束计算日期
    :return: 因子值, index 为日期, column 为股票代码
    '''

    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]

    ret = pd.DataFrame()
    
    def wma(series, N):
        return series.unstack().rolling(N,axis=1).apply(lambda x: x[:: -1].cumsum().sum() * 2 / N / (N + 1)).stack()
    
    for w in window:
        N1,N2,N3,N4,N5 = w
        _data = data.loc[futures_universe]
        _data['vwapw'] = (_data.openw + _data.closew + _data.highw + _data.loww) / 4
        _data['inner1'] = wma((_data.highw + _data.loww).unstack().rolling(N2,axis=1).corr(_data.volume99.unstack().rolling(N1,axis=1).mean()).stack(), N3)
        _data['rank1'] = _data.inner1.unstack().rank(pct=True).stack()
    
        _data['inner2'] = wma((_data.vwapw.unstack().rank(pct=True).stack()).rolling(N4).corr(_data.volume99.unstack().rank(pct=True).stack()), N5)
        _data['rank2'] = _data.inner2.unstack().rank(pct=True).stack()
    
        _data['g130'] = _data.rank1 / _data.rank2
        _ret = _data.g130.unstack().T
        _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja130" + str(w)]], names=["futures", "price"])
        ret = pd.concat([ret, _ret], axis=1)
    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret


gtja131_time_lst =[
    [10,3],
    [20,6],
    [30,9],
    [50,18],
    [70,24],
    [90,30],
    ]

def GTJA131(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA131
    rank(delta(vwap, 1)) ^ tsrank(corr(close, mean(volume, 50), 18), 18)

    :param data 默认旋转后，即 data_rotated
    :param window 是一个一维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return ret
    '''

    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]

    def tsrank(array):
        s = pd.Series(array)
        return np.array(s.rank(method="min", ascending=False))[-1]

    ret = pd.DataFrame()
    for w in window:
        N1, N2 = w

        _data = data.loc[futures_universe]
        _data['vwapw'] = (_data.openw + _data.closew + _data.highw + _data.loww) / 4

        _data['inner1'] = _data.vwapw.unstack().diff(1,axis=1).stack()
        _data['rank1'] = _data.inner1.unstack().rank(pct=True).stack()

        _data['inner2'] = _data.closew.unstack().rolling(N2,axis=1).corr(_data.volume99.unstack().rolling(N1,axis=1).mean()).stack()
        _data['rank2'] = _data.inner2.unstack().rolling(N2,axis=1).apply(tsrank).stack()

        _data['g131'] = np.power(_data.rank1, _data.rank2)
        _ret = _data.g131.unstack().T
        _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja131" + str(w)]], names=["futures", "price"])
        ret = pd.concat([ret, _ret], axis=1)
    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret


gtja132_time_lst = [4, 8, 12, 20, 32, 48]
def GTJA132(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA132
    mean(amount, 20)

    :param data 默认旋转后，即 data_rotated
    :param window 是一个一维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return ret
    '''

    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]

    ret = pd.DataFrame()
    for w in window:
        N = w

        _data = data.loc[futures_universe]

        _data['g132'] = _data.total_turnover.unstack().rolling(N,axis=1).mean().stack()
        _ret = _data.g132.unstack().T
        _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja132" + str(w)]], names=["futures", "price"])
        ret = pd.concat([ret, _ret], axis=1)
    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret


gtja133_time_lst = [4, 8, 12, 20, 32, 48]
def GTJA133(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA133
    (20 - highday(high, 20)) / 20 - (20 - lowday(low, 20)) / 20

    :param data 默认旋转后，即 data_rotated
    :param window 是一个一维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return ret
    '''

    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]

    ret = pd.DataFrame()
    for w in window:
        N = w

        _data = data.loc[futures_universe]

        _data['g133'] = _data.highw.unstack().rolling(N,axis=1).apply(np.argmax).stack() - _data.loww.unstack().rolling(N,axis=1).apply(np.argmin).stack()
        _ret = _data.g133.unstack().T
        _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja133" + str(w)]], names=["futures", "price"])
        ret = pd.concat([ret, _ret], axis=1)
    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret


gtja134_time_lst = [4, 8, 12, 20, 32, 48]
def GTJA134(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA134
    (close - delay(close, 12)) / delay(close, 12) * volume

    :param data 默认旋转后，即 data_rotated
    :param window 是一个一维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return ret
    '''

    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]

    ret = pd.DataFrame()
    for w in window:
        N = w

        _data = data.loc[futures_universe]

        _data['g134'] = (_data.closew - _data.closew.unstack().shift(N,axis=1).stack()) / _data.closew.unstack().shift(N,axis=1).stack() * _data.volume99
        _ret = _data.g134.unstack().T
        _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja134" + str(w)]], names=["futures", "price"])
        ret = pd.concat([ret, _ret], axis=1)
    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret


gtja135_time_lst = [4, 8, 12, 20, 32, 48]
def GTJA135(data, window, futures_universe, time_freq='day'):
    '''
    计算 GTJA135
    sma(delay(close / delay(close, 20), 1), 20, 1)

    :param data 默认旋转后，即 data_rotated
    :param window 是一个一维数组
    :param futures_universe 是计算 rank 的范围。默认是 gu24
    :return ret
    '''

    
    volume = data.volume99[futures_universe]
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    high = data.high[futures_universe]
    highw = data.highw[futures_universe]
    low = data.low[futures_universe]
    loww = data.loww[futures_universe]
    close = data.close[futures_universe]
    closew = data.closew[futures_universe]

    

    ret = pd.DataFrame()
    for w in window:
        N1 = w
        sma_alpha = 1. / N1
        _data = data.loc[futures_universe]

        _data['inner'] = (_data.closew / _data.closew.unstack().shift(N1,axis=1).stack()).unstack().shift(1,axis=1).stack()

        _data['g135'] = _data.inner.unstack().ewm(adjust=False, alpha=sma_alpha,axis=1).mean().stack()
        _ret = _data.g135.unstack().T
        _ret.columns = pd.MultiIndex.from_product([_ret.columns, ["gtja135" + str(w)]], names=["futures", "price"])
        ret = pd.concat([ret, _ret], axis=1)
    # 把 open 和 openw 贴进去
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["futures", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["futures", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret


# Add(Mul(Constant(-1.0),Sub(Greater($closew,Abs($highw)),Constant(10.0))),$highw)
def ALPHAGEN001(data, window, futures_universe, time_freq='day'):
    ret = pd.DataFrame()
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    for w in window:
        _data = data.loc[futures_universe]

        _data['factor'] = _data.groupby('future').apply(lambda x: (
                x['open_interest99'].rolling(window=w).mean() - x['open_interest99'])).reset_index(drop=True, level=0)
        _ret = _data['factor'].unstack().T
        _ret.columns = pd.MultiIndex.from_product([_ret.columns, [f"Factor{w}"]], names=["futures", "price"])
        # _ret.index = data.index  # 确保索引与原始数据相同
        ret = pd.concat([ret, _ret], axis=1)
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["future", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["future", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret


# Add(Mean(Var(Mean($highw,20),50),20),$closew)
# 表现不好
def ALPHAGEN002(data, window, futures_universe, time_freq='day'):
    ret = pd.DataFrame()
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    for w in window:
        _data = data.loc[futures_universe]

        _data['factor'] = _data.groupby('future').apply(lambda x: (
            x['highw'].rolling(w).mean().rolling(w).var())).reset_index(drop=True, level=0)
        _ret = _data['factor'].unstack().T
        _ret.columns = pd.MultiIndex.from_product([_ret.columns, [f"Factor{w}"]], names=["futures", "price"])
        # _ret.index = data.index  # 确保索引与原始数据相同
        ret = pd.concat([ret, _ret], axis=1)
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["future", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["future", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret

# -Div(Div(Mul(Mul(Add(Sum(Mul(Mul(Sub(Constant(1.0),$loww),$volume),
# Constant(1.0)),20),Constant(-10.0)),$highw),Constant(1.0)),$highw),$highw)
# 表现不好
def ALPHAGEN003(data, window, futures_universe, time_freq='day'):
    ret = pd.DataFrame()
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    for w in window:
        _data = data.loc[futures_universe]

        _data['factor'] = (_data.groupby('future').apply(lambda x: (
            ( x['closew'] * x['volume99'] * x['highw']).rolling(w).sum()).reset_index(drop=True, level=0)))
        _ret = _data['factor'].unstack().T
        _ret.columns = pd.MultiIndex.from_product([_ret.columns, [f"Factor{w}"]], names=["futures", "price"])
        # _ret.index = data.index  # 确保索引与原始数据相同
        ret = pd.concat([ret, _ret], axis=1)
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["future", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["future", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret

# -Sum(Med(EMA(Sub(Sub(Add(Add(EMA($open_interest,40),$openw),$loww),$closew),$volume),10),10),40)
def ALPHAGEN004(data, window, futures_universe, time_freq='day'):
    ret = pd.DataFrame()
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    for w in window:
        _data = data.loc[futures_universe]

        _data['factor'] = -((_data.groupby('future').apply(
            lambda x: ((x['open_interest'].ewm(w).mean() - x['volume99']).rolling(w).sum()))).reset_index(
            drop=True, level=0))
        _ret = _data['factor'].unstack().T
        _ret.columns = pd.MultiIndex.from_product([_ret.columns, [f"Factor{w}"]], names=["futures", "price"])
        # _ret.index = data.index  # 确保索引与原始数据相同
        ret = pd.concat([ret, _ret], axis=1)
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["future", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["future", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret


# WMA(Div(Add(Constant(1.0),$volume99),$open_interest),20)
def ALPHAGEN005(data, window, futures_universe, time_freq='day'):
    ret = pd.DataFrame()
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    for w in window:
        _data = data.loc[futures_universe]

        _data['factor'] = _data.groupby('future').apply(lambda x: (
            (x['volume99'] / x['open_interest']).rolling(w).mean())).reset_index(drop=True, level=0)
        _ret = _data['factor'].unstack().T
        _ret.columns = pd.MultiIndex.from_product([_ret.columns, [f"Factor{w}"]], names=["futures", "price"])
        # _ret.index = data.index  # 确保索引与原始数据相同
        ret = pd.concat([ret, _ret], axis=1)
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["future", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["future", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret

# 趋势策略
def Trend(data, window, futures_universe, time_freq='day'):
    ret = pd.DataFrame()
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    for w in window:
        _data = data.loc[futures_universe]

        _data['factor'] = _data.groupby('future').apply(lambda x: (
            (x['closew'] - x['closew'].rolling(15).mean()))).reset_index(drop=True, level=0)
        _ret = _data['factor'].unstack().T
        _ret.columns = pd.MultiIndex.from_product([_ret.columns, [f"Factor{w}"]], names=["futures", "price"])
        # _ret.index = data.index  # 确保索引与原始数据相同
        ret = pd.concat([ret, _ret], axis=1)
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["future", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["future", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret

def Reversion(data, window, futures_universe, time_freq='day'):
    ret = pd.DataFrame()
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    for w in window:
        _data = data.loc[futures_universe]

        _data['factor'] = _data.groupby('future').apply(lambda x: (
           - (x['closew'] - x['closew'].rolling(15).mean()))).reset_index(drop=True, level=0)
        _ret = _data['factor'].unstack().T
        _ret.columns = pd.MultiIndex.from_product([_ret.columns, [f"Factor{w}"]], names=["futures", "price"])
        # _ret.index = data.index  # 确保索引与原始数据相同
        ret = pd.concat([ret, _ret], axis=1)
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["future", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["future", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret

# MA*VHF todo
def MA_VHF(data, window, futures_universe, time_freq='day'):
    ret = pd.DataFrame()
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    for w in window:
        _data = data.loc[futures_universe]

        _data['factor'] = _data.groupby('future').apply(lambda x: (
            - (x['closew'] - x['closew'].rolling(15).mean()))).reset_index(drop=True, level=0)
        _ret = _data['factor'].unstack().T
        _ret.columns = pd.MultiIndex.from_product([_ret.columns, [f"Factor{w}"]], names=["futures", "price"])
        # _ret.index = data.index  # 确保索引与原始数据相同
        ret = pd.concat([ret, _ret], axis=1)
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["future", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["future", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret

# 止盈
# upregion_time_lst = [[100, 5, 0.4, 0.4, 0.32]]
def upregion(data,window,futures_universe):
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]

    df_temp = openw.unstack(level=0).diff(1) / open_price.unstack(level=0)
    df_temp = (df_temp + 1).fillna(1)
    df_temp = np.log(df_temp)
    df_temp = df_temp.cumsum()
    df_temp = np.exp(df_temp)
    # df_temp.columns = pd.MultiIndex.from_product([df_temp.columns, ['cum']])


    ret = pd.DataFrame()
    for w in window:
        window1,variance_root,weight,a,b = w

        min_cum = df_temp.rolling(window=window1,min_periods=1).min()
        normalized_var = (df_temp.var()) ** (1/variance_root) #每列的方差
        days_since_min = (window1 - 1) - df_temp.rolling(window=window1,min_periods=1).apply(lambda x:np.argmin(x))
        relative_change_from_min = (df_temp - min_cum) / min_cum

        slope = (1 + relative_change_from_min) ** (1 / (days_since_min.replace(0, np.nan)+ 12)) -1
        slope = slope.replace([np.inf, -np.inf], np.nan).fillna(0)
        slope.fillna(0, inplace=True)  # 将NaN（原先除数为0的情况）填充为0

        days_since_min.columns = pd.MultiIndex.from_product([df_temp.columns, ['day']])
        slope.columns = pd.MultiIndex.from_product([df_temp.columns, ['slope']])
        relative_change_from_min.columns = pd.MultiIndex.from_product([df_temp.columns, ['change_cum']])
        df = df_temp.copy()
        df.columns = pd.MultiIndex.from_product([df.columns, ['cum']])
        df = pd.concat([df, days_since_min], axis=1)
        df = pd.concat([df, slope], axis=1)
        df = pd.concat([df, relative_change_from_min], axis=1)

        slope_weighted = df.xs('slope', level=1, axis=1) * weight * window1

        scores = df.xs('change_cum', level=1, axis=1) + slope_weighted

        adj_threshold = b + (normalized_var * a).values.reshape(-1,1)
        adj_threshold = adj_threshold.T
        adj_threshold = np.tile(adj_threshold, (scores.shape[0], 1))
        lower_threshold = adj_threshold * 0.5

        conditions_met = pd.DataFrame(False, index=scores.index, columns=scores.columns)
        condition1 = scores >= adj_threshold
        condition2 = scores >= lower_threshold
        conditions_met = condition1.copy()
        previous_conditions_met = conditions_met.iloc[0]
        conditions_met.sort_index(axis=0,inplace=True)

        for i in range(1,conditions_met.shape[0]):
            # 计算需要更新的部分
            needs_update = ~conditions_met.iloc[i]
            update_values = condition2.iloc[i] & previous_conditions_met

            # 更新 conditions_met
            conditions_met.iloc[i] = np.where(needs_update, update_values, conditions_met.iloc[i])

            previous_conditions_met = conditions_met.iloc[i]



        _ret = conditions_met
        _ret.columns = pd.MultiIndex.from_product([_ret.columns.get_level_values(level=0),['upregion'+str(w)]], names=["future", "price"])
        ret = pd.concat([ret, _ret], axis=1)

    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["future", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["future", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret


# 日频因子
def QRS_BETA(data, window, futures_universe, time_freq='day'):
    ret = pd.DataFrame()
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    
    for w in window:
        _data = data.loc[futures_universe]
        _data['factor'] = _data.groupby('future').apply(
            lambda x: RollingOLS(
                endog=x['highw'], 
                exog=x[['loww']], 
                window=w
            ).fit().params['loww']
        ).reset_index(drop=True, level=0)
        
        _ret = _data['factor'].unstack().T
        _ret = _ret.apply(lambda x: (x - x.rolling(600, min_periods=1).mean()) / x.rolling(600, min_periods=1).std(), axis=1)

        _ret.columns = pd.MultiIndex.from_product([_ret.columns, [f"Factor{w}"]], names=["futures", "price"])
        
        ret = pd.concat([ret, _ret], axis=1)

    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["future", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["future", "price"])
    
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret

def vol_std(data, window, futures_universe, time_freq='day'):
    ret = pd.DataFrame()
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    
    for w in window:
        _data = data.loc[futures_universe]
        _data['factor'] = _data.groupby('future').apply(
            lambda x: x['closew'].pct_change().rolling(window=w, min_periods=1).std()
        ).reset_index(drop=True, level=0)

        _ret = _data['factor'].unstack().T
        _ret.columns = pd.MultiIndex.from_product([_ret.columns, [f"Factor{w}"]], names=["futures", "price"])
        ret = pd.concat([ret, _ret], axis=1)
    
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["future", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["future", "price"])
    
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret

def vol_up_std(data, window, futures_universe, time_freq='day'):
    ret = pd.DataFrame()
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    
    for w in window:
        _data = data.loc[futures_universe]
        _data['factor'] = _data.groupby('future').apply(
            lambda x: x['closew'].pct_change() 
            .dropna()
            .loc[lambda r: r > 0]
            .rolling(window=w, min_periods=1)
            .std()
        ).reset_index(drop=True, level=0)
        _ret = _data['factor'].unstack().T
        _ret.columns = pd.MultiIndex.from_product([_ret.columns, [f"Factor{w}"]], names=["futures", "price"])
        ret = pd.concat([ret, _ret], axis=1)
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["future", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["future", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret

def highlow_mean(data, window, futures_universe, time_freq='day'):
    ret = pd.DataFrame()
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    
    for w in window:
        _data = data.loc[futures_universe]
        _data['factor'] = _data.groupby('future').apply(
            lambda x: x['highw'] / x['loww']
            .rolling(window=w, min_periods=1)
            .mean()
        ).reset_index(drop=True, level=0)
        
        _ret = _data['factor'].unstack().T
        _ret.columns = pd.MultiIndex.from_product([_ret.columns, [f"Factor{w}"]], names=["futures", "price"])  
        ret = pd.concat([ret, _ret], axis=1)
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["future", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["future", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret

def highlow_std(data, window, futures_universe, time_freq='day'):
    ret = pd.DataFrame()
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    
    for w in window:
        _data = data.loc[futures_universe]
        _data['factor'] = _data.groupby('future').apply(
            lambda x: x['highw'] / x['loww']
            .rolling(window=w, min_periods=1)
            .std()
        ).reset_index(drop=True, level=0)
        
        _ret = _data['factor'].unstack().T
        _ret.columns = pd.MultiIndex.from_product([_ret.columns, [f"Factor{w}"]], names=["futures", "price"])  
        ret = pd.concat([ret, _ret], axis=1)
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["future", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["future", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret

def oi_vol_corr(data, window, futures_universe, time_freq='day'):
    ret = pd.DataFrame()
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    
    for w in window:
        _data = data.loc[futures_universe]
        _data['factor'] = _data.groupby('future').apply(
            lambda x: x['open_interest99'].rolling(window=w, min_periods=1)
            .corr(x['volume99'])
        ).reset_index(drop=True, level=0)
        
        _ret = _data['factor'].unstack().T
        _ret.columns = pd.MultiIndex.from_product([_ret.columns, [f"Factor{w}"]], names=["futures", "price"])  
        ret = pd.concat([ret, _ret], axis=1)
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["future", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["future", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret

def oi_mmt(data, window, futures_universe, time_freq='day'):
    ret = pd.DataFrame()
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    
    for w in window:
        _data = data.loc[futures_universe]
        _data['momentum'] = _data.groupby('future')['closew'].pct_change()

        # Calculate rolling sum of open_interest99 for weights
        _data['weight'] = _data.groupby('future')['open_interest99'].rolling(window=w, min_periods=1).sum().reset_index(level=0, drop=True)

        # Apply rolling weighted momentum calculation
        _data['factor'] = _data.groupby('future').apply(
            lambda x: (x['momentum'] * x['open_interest99']).rolling(window=w, min_periods=1).sum() / x['weight']
        ).reset_index(drop=True, level=0)
        
        _ret = _data['factor'].unstack().T
        _ret.columns = pd.MultiIndex.from_product([_ret.columns, [f"Factor{w}"]], names=["futures", "price"])  
        ret = pd.concat([ret, _ret], axis=1)
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["future", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["future", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret

def ALPHAGEN907(data, window, futures_universe, time_freq='day'):
    ret = pd.DataFrame()
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    for w in window:
        _data = data.loc[futures_universe]
        _data['factor'] = _data.groupby('future').apply(lambda x: -x['open_interest99'].diff(w)).reset_index(drop=True, level=0)
        _ret = _data['factor'].unstack().T
        _ret.columns = pd.MultiIndex.from_product([_ret.columns, [f"alphagen907{w}"]], names=["futures", "price"])
        ret = pd.concat([ret, _ret], axis=1)
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["future", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["future", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret

def ALPHAGEN904(data, window, futures_universe, time_freq='day'):
    ret = pd.DataFrame()
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    for w in window:
        _data = data.loc[futures_universe]
        _data['factor'] = _data.groupby('future').apply(
    lambda x: (x['open_interest99'].ewm(span=w).mean() - x['open_interest99'])).reset_index(drop=True,
                                                                                              level=0)
        _ret = _data['factor'].unstack().T
        _ret.columns = pd.MultiIndex.from_product([_ret.columns, [f"alphagen904{w}"]], names=["futures", "price"])
        ret = pd.concat([ret, _ret], axis=1)
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["future", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["future", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret

def ALPHAGEN916(data, window, futures_universe, time_freq='day'):
    ret = pd.DataFrame()
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    for w in window:
        _data = data.loc[futures_universe]
        _data['factor'] = _data.groupby('future').apply(
    lambda x: (x['volume99'] * x['total_turnover']).rolling(w).median()).reset_index(drop=True, level=0)
        _ret = _data['factor'].unstack().T
        _ret.columns = pd.MultiIndex.from_product([_ret.columns, [f"alphagen916{w}"]], names=["futures", "price"])
        ret = pd.concat([ret, _ret], axis=1)
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["future", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["future", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret

def ALPHAGEN913(data, window, futures_universe, time_freq='day'):
    # window=[1]
    ret = pd.DataFrame()
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    for w in window:
        m,n = w
        _data = data.loc[futures_universe]

        # 因子表达式部分
        # -Sum(Div(Std($total_turnover,m),$volume99),n)
        _data['factor'] = -((_data.groupby('future').apply(
            lambda x: x['total_turnover'].rolling(m).std().div(x['volume99']).rolling(n).sum())).reset_index(drop=True,
                                                                                                              level=0))
        _ret = _data['factor'].unstack().T
        _ret.columns = pd.MultiIndex.from_product([_ret.columns, [f"alphagen913{w}"]], names=["futures", "price"])
        ret = pd.concat([ret, _ret], axis=1)
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["future", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["future", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret

def ALPHAGEN903(data, window, futures_universe, time_freq='day'):
    ret = pd.DataFrame()
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    for w in window:
        _data = data.loc[futures_universe]
        _data['factor'] = _data.groupby('future').apply(lambda x: (
            (-x['open_interest99']).rolling(window=w).apply(lambda y: pd.Series(y).rank().iloc[-1], raw=True) *
            x['settlementw'])).reset_index(drop=True, level=0)
        _ret = _data['factor'].unstack().T
        _ret.columns = pd.MultiIndex.from_product([_ret.columns, [f"alphagen903{w}"]], names=["futures", "price"])
        ret = pd.concat([ret, _ret], axis=1)
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["future", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["future", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret

def ALPHAGEN902(data, window, futures_universe, time_freq='day'):
    ret = pd.DataFrame()
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    for w in window:
        _data = data.loc[futures_universe]
        _data['factor'] = _data.groupby('future').apply(
    lambda x: (x['volume99'] - x['open_interest']).ewm(span=w).mean()).reset_index(drop=True, level=0)
        _ret = _data['factor'].unstack().T
        _ret.columns = pd.MultiIndex.from_product([_ret.columns, [f"alphagen902{w}"]], names=["futures", "price"])
        ret = pd.concat([ret, _ret], axis=1)
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["future", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["future", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret

def Entropy(data, window, futures_universe, time_freq='day'):
    ret = pd.DataFrame()
    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]
    for w in window:
        _data = data.loc[futures_universe]

        # 因子表达式部分
        _data['entropy'] = _data.groupby('future').apply(
            lambda x: ((x['volume']-x['volume'].shift(1))/x['volume99'].shift(1)).rolling(window=w).apply(
        lambda y: -np.sum((np.histogram(y, bins=5, density=True)[0] *
                          np.log(np.histogram(y, bins=5, density=True)[0] + 1e-10))),
        raw=True
            )
        ).reset_index(drop=True, level=0)
        # _data['factor'] = -_data['entropy'] * (_data['openw']-_data['openw'].shift(3))/_data['open'].shift(3)
        def RSI_compute(_data, w):
            # 计算价格的涨跌幅
            delta = (_data['openw'] - _data['openw'].shift(1)) / _data['open'].shift(1)
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)

            # 计算平均涨幅和平均跌幅
            avg_gain = gain.rolling(w).mean()
            avg_loss = loss.rolling(w).mean()

            # 计算RSI
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        _data['RSI'] = RSI_compute(_data, w)
        _data['factor'] = -_data['entropy'] * _data['RSI']

        _ret = _data['factor'].unstack().T
        _ret.columns = pd.MultiIndex.from_product([_ret.columns, [f"entropy{w}"]], names=["futures", "price"])
        # _ret.index = data.index  # 确保索引与原始数据相同
        ret = pd.concat([ret, _ret], axis=1)
    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["future", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["future", "price"])
    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret
















# 高频（分钟）因子低频化（日）
# 因子名称的最后部分代表此因子在计算出日频数据之后，在日间处理的方式，一般为窗口期内的平均(mean)或标准差(std)
import datetime
# 分钟级数据路径
global DATAPATH
global time_list

DATAPATH = './data/future_all1mdata_20150331-20250304.txt'
time_list_path = './data/time_list.txt'
with open('data/time_list.txt','r') as f:
    lines = f.readlines()
    time_list = lines[0].strip()
time_list = [t.strip() for t in time_list.split(',')]

# 预处理分钟频数据
def get_processed_min_data(futures_universe, time_freq):
    with open(DATAPATH, 'rb') as f:
            high_freq_data = pickle.load(f)

    columns_to_use = high_freq_data.columns[high_freq_data.columns.get_level_values(0).isin(futures_universe)]
    _data = high_freq_data.loc[:, columns_to_use]

    _data = _data.stack(level=0)
    _data.index.set_names(['datetime', 'future'], inplace=True)
    _data = _data.swaplevel('future', 'datetime')
    _data = _data.sort_index(level=['future', 'datetime'])

    _data = _data.reset_index()
    _data['date'] = _data['trading_date'].dt.date
    _data['time'] = _data['datetime'].dt.time

    if time_freq == 'hour':
        # 将date设置为规定的时间节点
        _data['date'] = _data['trading_date'].dt.strftime('%Y-%m-%d ')
        _data['time'] = _data['datetime'].dt.time
        _data['time_mark'] = 0

        time_list_processed = [datetime.datetime.strptime(i, '%H:%M:%S').time() for i in time_list]
        for t in time_list_processed:
            _data.loc[_data['time'] > t,'time_mark'] += 1
        time_map = {i:time_list[i] for i in range(len(time_list))}
        _data['time_mark'] = _data['date'] + _data['time_mark'].map(time_map)
        _data['time_mark'] = pd.to_datetime(_data['time_mark'])
        _data['date'] = _data['time_mark']
    return _data

# 日间滚动平滑mean/std
def high_freq_rolling(factors,factor_name, window, data, futures_universe, method="mean", direction=1):
    ret = pd.DataFrame()
    factors0 = factors.copy()
    for curr_window in window:
        if method == 'mean':
            factors['factor'] = (factors0.groupby('future')['factor'].rolling(window=curr_window, min_periods=1).mean().reset_index(level=0, drop=True))
        elif method == 'std':
            factors['factor'] = (factors0.groupby('future')['factor'].rolling(window=curr_window, min_periods=1).std().reset_index(level=0, drop=True))
        else:
            raise KeyError(f'Unspported method {method}')
        factors['factor'] = factors['factor'] * direction
        _ret = factors.pivot(index='date', columns='future', values='factor')
        _ret.columns = pd.MultiIndex.from_product([_ret.columns, [f"{factor_name}{curr_window}"]], names=["future", "price"])
        ret = pd.concat([ret, _ret], axis=1)

    open_price = data.open[futures_universe]
    openw = data.openw[futures_universe]

    _open = open_price.unstack().T
    _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["future", "price"])
    _openw = openw.unstack().T
    _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["future", "price"])

    ret.index = pd.to_datetime(ret.index)
    _open.index = pd.to_datetime(_open.index)
    _openw.index = pd.to_datetime(_openw.index)

    ret = pd.concat([ret, _open, _openw], axis=1)
    return ret

# 尾盘动量，日间均值平滑
def mmt_last_mean(data, window, futures_universe, time_freq='day'):
    # window控制尾盘w分钟动量 day:30, hour:5
    last_mins = 30 if time_freq == 'day' else 10
    ret = pd.DataFrame()

    FACTOR_PATH = f'./factor_{time_freq}/mmt_last.csv'
    if os.path.exists(FACTOR_PATH):
        factors = pd.read_csv(FACTOR_PATH, parse_dates=['date'])
    else:
        _data = get_processed_min_data(futures_universe, time_freq)

        def get_last_minutes(group, minutes_before = 30):
            # 计算当前bar结束之前的指定分钟数时间点
            end_time = group.iloc[-1]['datetime']
            start_time = end_time - datetime.timedelta(minutes=minutes_before)
            end_time = end_time.time()
            start_time = start_time.time()
            last_minutes = group[(group['time'] > start_time) & (group['time'] <= end_time)]
            return last_minutes

        def compute_momentum(group):
            group = group.sort_values(by='datetime')
            # 计算每分钟的收益率
            group['minute_return'] = group['closew'].pct_change().dropna()
            # 如果收益率序列为空，则直接返回 NaN
            if group['minute_return'].empty:
                return pd.Series({'factor': np.nan})
            # 使用指数衰减权重
            weights = np.exp(np.linspace(-1, 0, len(group['minute_return'])))
            weights /= weights.sum()
            # 计算加权平均收益率
            weighted_momentum = (group['minute_return'] * weights).sum()
            # weighted_momentum = -weighted_momentum
            return pd.Series({'factor': weighted_momentum})
        
        def compute_momentum1(group):
            group = group.sort_values(by='datetime')
            first_price = group.iloc[0]['closew']
            last_price = group.iloc[-1]['closew']
            momentum = (last_price - first_price) / group.iloc[0]['close']
            return pd.Series({'factor': momentum})

        last_minutes_data = _data.groupby(['future', 'date']).apply(lambda x: get_last_minutes(x, last_mins)).reset_index(drop=True)
        factors = last_minutes_data.groupby(['future', 'date']).apply(compute_momentum1).reset_index()
        factors.to_csv(FACTOR_PATH, index=False)
    
    ret = high_freq_rolling(factors, factor_name='mmtlastmean',window=window,data=data,futures_universe=futures_universe, method='mean', direction=1)
    return ret

# 尾盘动量，日间标准差
def mmt_last_std(data, window, futures_universe, time_freq='day'):
    # window控制尾盘w分钟动量 day:30, hour:5
    last_mins = 30 if time_freq == 'day' else 10
    ret = pd.DataFrame()

    FACTOR_PATH = f'./factor_{time_freq}/mmt_last.csv'
    if os.path.exists(FACTOR_PATH):
        factors = pd.read_csv(FACTOR_PATH, parse_dates=['date'])
    else:
        _data = get_processed_min_data(futures_universe, time_freq)

        def get_last_minutes(group, minutes_before = 30):
            # 计算当前bar结束之前的指定分钟数时间点
            end_time = group.iloc[-1]['datetime']
            start_time = end_time - datetime.timedelta(minutes=minutes_before)
            end_time = end_time.time()
            start_time = start_time.time()
            last_minutes = group[(group['time'] > start_time) & (group['time'] <= end_time)]
            return last_minutes

        def compute_momentum(group):
            group = group.sort_values(by='datetime')
            # 计算每分钟的收益率
            group['minute_return'] = group['closew'].pct_change().dropna()
            # 如果收益率序列为空，则直接返回 NaN
            if group['minute_return'].empty:
                return pd.Series({'factor': np.nan})
            # 使用指数衰减权重
            weights = np.exp(np.linspace(-1, 0, len(group['minute_return'])))
            weights /= weights.sum()
            # 计算加权平均收益率
            weighted_momentum = (group['minute_return'] * weights).sum()
            # weighted_momentum = -weighted_momentum
            return pd.Series({'factor': weighted_momentum})
        
        def compute_momentum1(group):
            group = group.sort_values(by='datetime')
            first_price = group.iloc[0]['closew']
            last_price = group.iloc[-1]['closew']
            momentum = (last_price - first_price) / group.iloc[0]['close']
            return pd.Series({'factor': momentum})

        last_minutes_data = _data.groupby(['future', 'date']).apply(lambda x: get_last_minutes(x, last_mins)).reset_index(drop=True)
        factors = last_minutes_data.groupby(['future', 'date']).apply(compute_momentum1).reset_index()
        factors.to_csv(FACTOR_PATH, index=False)
    
    ret = high_freq_rolling(factors, factor_name='mmtlaststd',window=window,data=data,futures_universe=futures_universe, method='std', direction=1)
    return ret

# 开盘动量，日间均值平滑
def mmt_start_mean(data, window, futures_universe, time_freq='day'):
    # window控制开盘w分钟动量
    start_mins = 30 if time_freq == 'day' else 10
    ret = pd.DataFrame()

    FACTOR_PATH = f'./factor_{time_freq}/mmt_start.csv'
    if os.path.exists(FACTOR_PATH):
        factors = pd.read_csv(FACTOR_PATH, parse_dates=['date'])
    else:
        _data = get_processed_min_data(futures_universe, time_freq)

        def get_start_minutes(group, minutes_after = 30):
            # 计算bar开始之后的指定分钟数时间点
            start_time = group.iloc[0]['datetime']
            end_time = start_time + datetime.timedelta(minutes=minutes_after)
            end_time = end_time.time()
            start_time = start_time.time()
            start_minutes = group[(group['time'] >= start_time) & (group['time'] < end_time)]
            return start_minutes.sort_values(by='time')

        def compute_momentum(group):
            # 计算每分钟的收益率
            group['minute_return'] = group['closew'].pct_change().dropna()
            # 如果收益率序列为空，则直接返回 NaN
            if group['minute_return'].empty:
                return pd.Series({'factor': np.nan})
            # 使用指数衰减权重
            weights = np.exp(np.linspace(-1, 0, len(group['minute_return'])))
            weights /= weights.sum()
            # 计算加权平均收益率
            weighted_momentum = (group['minute_return'] * weights).sum()
            weighted_momentum = -weighted_momentum
            return pd.Series({'factor': weighted_momentum})
        
        def compute_momentum1(group):
            first_price = group.iloc[0]['closew']
            last_price = group.iloc[-1]['closew']
            momentum = (last_price - first_price) / first_price
            return pd.Series({'factor': momentum})
        
        start_minutes_data = _data.groupby(['future', 'date']).apply(lambda x: get_start_minutes(x, start_mins)).reset_index(drop=True)
        factors = start_minutes_data.groupby(['future', 'date']).apply(compute_momentum1).reset_index()
        factors.to_csv(FACTOR_PATH, index=False)

    ret = high_freq_rolling(factors, factor_name='mmtstartmean',window=window,data=data,futures_universe=futures_universe, method='mean', direction=1)
    return ret

# 开盘动量，日间标准差平滑
def mmt_start_std(data, window, futures_universe, time_freq='day'):
    # window控制开盘w分钟动量
    start_mins = 30 if time_freq == 'day' else 10
    ret = pd.DataFrame()

    FACTOR_PATH = f'./factor_{time_freq}/mmt_start.csv'
    if os.path.exists(FACTOR_PATH):
        factors = pd.read_csv(FACTOR_PATH, parse_dates=['date'])
    else:
        _data = get_processed_min_data(futures_universe, time_freq)

        def get_start_minutes(group, minutes_after = 30):
            # 计算bar开始之后的指定分钟数时间点
            end_time = group.iloc[0]['datetime']
            end_time = start_time + datetime.timedelta(minutes=minutes_after)
            end_time = end_time.time()
            start_time = start_time.time()
            start_minutes = group[(group['time'] >= start_time) & (group['time'] < end_time)]
            return start_minutes.sort_values(by='time')

        def compute_momentum(group):
            # 计算每分钟的收益率
            group['minute_return'] = group['closew'].pct_change().dropna()
            # 如果收益率序列为空，则直接返回 NaN
            if group['minute_return'].empty:
                return pd.Series({'factor': np.nan})
            # 使用指数衰减权重
            weights = np.exp(np.linspace(-1, 0, len(group['minute_return'])))
            weights /= weights.sum()
            # 计算加权平均收益率
            weighted_momentum = (group['minute_return'] * weights).sum()
            weighted_momentum = -weighted_momentum
            return pd.Series({'factor': weighted_momentum})
        
        def compute_momentum1(group):
            first_price = group.iloc[0]['closew']
            last_price = group.iloc[-1]['closew']
            momentum = (last_price - first_price) / first_price
            return pd.Series({'factor': momentum})
        
        start_minutes_data = _data.groupby(['future', 'date']).apply(lambda x: get_start_minutes(x, start_mins)).reset_index(drop=True)
        factors = start_minutes_data.groupby(['future', 'date']).apply(compute_momentum1).reset_index()
        factors.to_csv(FACTOR_PATH, index=False)

    ret = high_freq_rolling(factors, factor_name='mmtstartstd',window=window,data=data,futures_universe=futures_universe, method='std', direction=1)
    return ret

# # 尾盘动量，表现暂时不好，目前窗口是尾盘w分钟动量，可以加入日间平滑，成为2D窗口再尝试
# def mmt_last(data, window, futures_universe, time_freq='day'):
#     # window控制尾盘w分钟动量
#     window = [30]
#     ret = pd.DataFrame()
#     with open(DATAPATH, 'rb') as f:
#         high_freq_data = pickle.load(f)
#     f.close()

#     columns_to_use = high_freq_data.columns[high_freq_data.columns.get_level_values(0).isin(futures_universe)]
#     _data = high_freq_data.loc[:, columns_to_use]

#     _data = _data.stack(level=0)
#     _data.index.set_names(['datetime', 'future'], inplace=True)
#     _data = _data.swaplevel('future', 'datetime')
#     _data = _data.sort_index(level=['future', 'datetime'])

#     _data = _data.reset_index()
#     _data['date'] = _data['trading_date'].dt.date
#     _data['time'] = _data['datetime'].dt.time

#     def get_last_minutes(group, minutes_before = 30):
#         # 计算15:00之前的指定分钟数时间点
#         end_time = pd.to_datetime('15:15')
#         start_time = end_time - pd.Timedelta(minutes=minutes_before)
#         end_time = end_time.time()
#         start_time = start_time.time()
#         last_minutes = group[(group['time'] > start_time) & (group['time'] <= end_time)]
#         return last_minutes

#     def compute_momentum(group):
#         group = group.sort_values(by='datetime')
#         # 计算每分钟的收益率
#         group['minute_return'] = group['closew'].pct_change().dropna()
#         # 如果收益率序列为空，则直接返回 NaN
#         if group['minute_return'].empty:
#             return pd.Series({'factor': np.nan})
#         # 使用指数衰减权重
#         weights = np.exp(np.linspace(-1, 0, len(group['minute_return'])))
#         weights /= weights.sum()
#         # 计算加权平均收益率
#         weighted_momentum = (group['minute_return'] * weights).sum()
#         # weighted_momentum = -weighted_momentum
#         return pd.Series({'factor': weighted_momentum})
    
#     def compute_momentum1(group):
#         group = group.sort_values(by='datetime')
#         first_price = group.iloc[0]['closew']
#         last_price = group.iloc[-1]['closew']
#         momentum = (last_price - first_price) / group.iloc[0]['close']
#         return pd.Series({'factor': momentum})
    
#     for w in window:
#         last_minutes_data = _data.groupby(['future', 'date']).apply(lambda x: get_last_minutes(x, w)).reset_index(drop=True)
#         momentum_factors = last_minutes_data.groupby(['future', 'date']).apply(compute_momentum1).reset_index()

#         _ret = momentum_factors.pivot(index='date', columns='future', values='factor')
#         _ret.columns = pd.MultiIndex.from_product([_ret.columns, [f"Factor{w}"]], names=["future", "price"])
#         ret = pd.concat([ret, _ret], axis=1)

#     open_price = data.open[futures_universe]
#     openw = data.openw[futures_universe]

#     _open = open_price.unstack().T
#     _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["future", "price"])
#     _openw = openw.unstack().T
#     _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["future", "price"])

#     ret.index = pd.to_datetime(ret.index)
#     _open.index = pd.to_datetime(_open.index)
#     _openw.index = pd.to_datetime(_openw.index)

#     ret = pd.concat([ret, _open, _openw], axis=1)
#     return ret

# # 开盘动量，表现暂时不好，目前窗口是开盘w分钟动量，可以加入日间平滑，成为2D窗口再尝试
# def mmt_start(data, window, futures_universe, time_freq='day'):
#     # window控制开盘w分钟动量
#     # window = [30, 60, 120]
#     ret = pd.DataFrame()
#     with open(DATAPATH, 'rb') as f:
#         high_freq_data = pickle.load(f)
#     f.close()

#     columns_to_use = high_freq_data.columns[high_freq_data.columns.get_level_values(0).isin(futures_universe)]
#     _data = high_freq_data.loc[:, columns_to_use]

#     _data = _data.stack(level=0)
#     _data.index.set_names(['datetime', 'future'], inplace=True)
#     _data = _data.swaplevel('future', 'datetime')
#     _data = _data.sort_index(level=['future', 'datetime'])

#     _data = _data.reset_index()
#     _data['date'] = _data['trading_date'].dt.date
#     _data['time'] = _data['datetime'].dt.time

#     def get_start_minutes(group, minutes_after = 30):
#         # 计算21:01之后的指定分钟数时间点
#         start_time = pd.to_datetime('21:01')
#         end_time = start_time + pd.Timedelta(minutes=minutes_after)
#         end_time = end_time.time()
#         start_time = start_time.time()
#         start_minutes = group[(group['time'] >= start_time) & (group['time'] < end_time)]
#         return start_minutes.sort_values(by='time')

#     def compute_momentum(group):
#         # 计算每分钟的收益率
#         group['minute_return'] = group['closew'].pct_change().dropna()
#         # 如果收益率序列为空，则直接返回 NaN
#         if group['minute_return'].empty:
#             return pd.Series({'factor': np.nan})
#         # 使用指数衰减权重
#         weights = np.exp(np.linspace(-1, 0, len(group['minute_return'])))
#         weights /= weights.sum()
#         # 计算加权平均收益率
#         weighted_momentum = (group['minute_return'] * weights).sum()
#         weighted_momentum = -weighted_momentum
#         return pd.Series({'factor': weighted_momentum})
    
#     def compute_momentum1(group):
#         first_price = group.iloc[0]['closew']
#         last_price = group.iloc[-1]['closew']
#         momentum = (last_price - first_price) / first_price
#         return pd.Series({'factor': momentum})
    
#     for w in window:
#         start_minutes_data = _data.groupby(['future', 'date']).apply(lambda x: get_start_minutes(x, w)).reset_index(drop=True)
#         momentum_factors = start_minutes_data.groupby(['future', 'date']).apply(compute_momentum).reset_index()

#         _ret = momentum_factors.pivot(index='date', columns='future', values='factor')
#         _ret.columns = pd.MultiIndex.from_product([_ret.columns, [f"Factor{w}"]], names=["future", "price"])
#         ret = pd.concat([ret, _ret], axis=1)

#     open_price = data.open[futures_universe]
#     openw = data.openw[futures_universe]

#     _open = open_price.unstack().T
#     _open.columns = pd.MultiIndex.from_product([_open.columns, ["open"]], names=["future", "price"])
#     _openw = openw.unstack().T
#     _openw.columns = pd.MultiIndex.from_product([_openw.columns, ["openw"]], names=["future", "price"])

#     ret.index = pd.to_datetime(ret.index)
#     _open.index = pd.to_datetime(_open.index)
#     _openw.index = pd.to_datetime(_openw.index)

#     ret = pd.concat([ret, _open, _openw], axis=1)
#     return ret

# 成交量激增因子，表现好，但是和全一的相关性较高
def vol_volumesurge_mean(data, window, futures_universe, time_freq='day'):
    ret = pd.DataFrame()
    FACTOR_PATH = f'./factor_{time_freq}/vol_volumesurge.csv'

    if os.path.exists(FACTOR_PATH):
        factors = pd.read_csv(FACTOR_PATH, parse_dates=['date'])
    else:
        _data = get_processed_min_data(futures_universe, time_freq)

        def compute_factor(group):
            rolling_sd = group['volume99'].rolling(window=5, min_periods=1).mean()
            sd = (group['volume99'] / rolling_sd).max()
            return pd.Series({'factor': sd})
        
        factors = _data.groupby(['future', 'date']).apply(compute_factor).reset_index()
        factors.to_csv(FACTOR_PATH, index=False)
    
    ret = high_freq_rolling(factors, factor_name='volumesurgemean',window=window,data=data,futures_universe=futures_universe, method='mean', direction=1)
    return ret

# 日内收益率的标准差因子
def vol_return1minstd_std(data, window, futures_universe, time_freq='day'):
    ret = pd.DataFrame()
    FACTOR_PATH = f'./factor_{time_freq}/vol_return1minstd.csv'

    if os.path.exists(FACTOR_PATH):
        factors = pd.read_csv(FACTOR_PATH, parse_dates=['date'])
    else:
        _data = get_processed_min_data(futures_universe, time_freq)

        def compute_factor(group):
            group = group.sort_values(by='datetime')
            group['minute_return'] = (group['closew'] - group['closew'].shift(1)) / group['close'].shift(1)
            group['minute_return'] = group['minute_return'].dropna()
            sd = group['minute_return'].std()
            return pd.Series({'factor': sd})
        factors = _data.groupby(['future', 'date']).apply(compute_factor).reset_index()
        factors.to_csv(FACTOR_PATH, index=False)
    
    ret = high_freq_rolling(factors, factor_name='return1minstdstd',window=window,data=data,futures_universe=futures_universe, method='std', direction=1)
    return ret

# 日内收益率的标准差因子
def vol_return1minstd_mean(data, window, futures_universe, time_freq='day'):
    ret = pd.DataFrame()
    FACTOR_PATH = f'./factor_{time_freq}/vol_return1minstd.csv'

    if os.path.exists(FACTOR_PATH):
        factors = pd.read_csv(FACTOR_PATH, parse_dates=['date'])
    else:
        _data = get_processed_min_data(futures_universe, time_freq)

        def compute_factor(group):
            group = group.sort_values(by='datetime')
            group['minute_return'] = (group['closew'] - group['closew'].shift(1)) / group['close'].shift(1)
            group['minute_return'] = group['minute_return'].dropna()
            sd = group['minute_return'].std()
            return pd.Series({'factor': sd})
        factors = _data.groupby(['future', 'date']).apply(compute_factor).reset_index()
        factors.to_csv(FACTOR_PATH, index=False)
    
    ret = high_freq_rolling(factors, factor_name='return1minstdmean',window=window,data=data,futures_universe=futures_universe, method='mean', direction=1)
    return ret

# 日内上行波动率因子，取收益率大于0的点计算收益率的标准差
def vol_upVol_mean(data, window, futures_universe, time_freq='day'):
    ret = pd.DataFrame()
    FACTOR_PATH = f'./factor_{time_freq}/vol_upVol.csv'

    if os.path.exists(FACTOR_PATH):
        factors = pd.read_csv(FACTOR_PATH, parse_dates=['date'])
    else:
        _data = get_processed_min_data(futures_universe, time_freq)

        def compute_factor(group):
            group = group.sort_values(by='datetime')
            returns = (group['closew'] - group['closew'].shift(1)) / group['close'].shift(1)
            returns = returns.dropna()
            up_returns = returns[returns > 0]
            if up_returns.empty:
                sd = 0
            else:
                sd = up_returns.std()
            return pd.Series({'factor': sd})
        factors = _data.groupby(['future', 'date']).apply(compute_factor).reset_index()
        factors.to_csv(FACTOR_PATH, index=False)
    ret = high_freq_rolling(factors, factor_name='upvolmean',window=window,data=data,futures_universe=futures_universe, method='mean', direction=1)
    return ret

# 日内上行波动率因子，取收益率大于0的点计算收益率的标准差
def vol_upVol_std(data, window, futures_universe, time_freq='day'):
    ret = pd.DataFrame()
    FACTOR_PATH = f'./factor_{time_freq}/vol_upVol.csv'

    if os.path.exists(FACTOR_PATH):
        factors = pd.read_csv(FACTOR_PATH, parse_dates=['date'])
    else:
        _data = get_processed_min_data(futures_universe, time_freq)

        def compute_factor(group):
            group = group.sort_values(by='datetime')
            returns = (group['closew'] - group['closew'].shift(1)) / group['close'].shift(1)
            returns = returns.dropna()
            up_returns = returns[returns > 0]
            if up_returns.empty:
                sd = 0
            else:
                sd = up_returns.std()
            return pd.Series({'factor': sd})
        factors = _data.groupby(['future', 'date']).apply(compute_factor).reset_index()
        factors.to_csv(FACTOR_PATH, index=False)

    ret = high_freq_rolling(factors, factor_name='upvolstd',window=window,data=data,futures_universe=futures_universe, method='std', direction=1)
    return ret

# 日内下行波动率因子，取收益率小于0的点计算收益率的标准差
def vol_downVol_mean(data, window, futures_universe, time_freq='day'):
    ret = pd.DataFrame()
    FACTOR_PATH = f'./factor_{time_freq}/vol_downVol.csv'

    if os.path.exists(FACTOR_PATH):
        factors = pd.read_csv(FACTOR_PATH, parse_dates=['date'])
    else:
        _data = get_processed_min_data(futures_universe, time_freq)

        def compute_factor(group):
            group = group.sort_values(by='datetime')
            returns = (group['closew'] - group['closew'].shift(1)) / group['close'].shift(1)
            returns = returns.dropna()
            down_returns = returns[returns < 0]
            if down_returns.empty:
                sd = 0
            else:
                sd = down_returns.std()
            return pd.Series({'factor': sd})
        factors = _data.groupby(['future', 'date']).apply(compute_factor).reset_index()
        factors.to_csv(FACTOR_PATH, index=False)

    ret = high_freq_rolling(factors, factor_name='downvolmean',window=window,data=data,futures_universe=futures_universe, method='mean', direction=1)
    return ret

# 日内下行波动率因子，取收益率小于0的点计算收益率的标准差
def vol_downVol_std(data, window, futures_universe, time_freq='day'):
    ret = pd.DataFrame()
    FACTOR_PATH = f'./factor_{time_freq}/vol_downVol.csv'

    if os.path.exists(FACTOR_PATH):
        factors = pd.read_csv(FACTOR_PATH, parse_dates=['date'])
    else:
        _data = get_processed_min_data(futures_universe, time_freq)

        def compute_factor(group):
            group = group.sort_values(by='datetime')
            returns = (group['closew'] - group['closew'].shift(1)) / group['close'].shift(1)
            returns = returns.dropna()
            down_returns = returns[returns < 0]
            if down_returns.empty:
                sd = 0
            else:
                sd = down_returns.std()
            return pd.Series({'factor': sd})
        factors = _data.groupby(['future', 'date']).apply(compute_factor).reset_index()
        factors.to_csv(FACTOR_PATH, index=False)

    ret = high_freq_rolling(factors, factor_name='downvolstd',window=window,data=data,futures_universe=futures_universe, method='std', direction=1)
    return ret

# 日内closew和volume99的相关系数
def corr_pv_std(data, window, futures_universe, time_freq='day'):
    ret = pd.DataFrame()
    FACTOR_PATH = f'./factor_{time_freq}/corr_pv.csv'
    if os.path.exists(FACTOR_PATH):
        factors = pd.read_csv(FACTOR_PATH, parse_dates=['date'])
    else:
        _data = get_processed_min_data(futures_universe, time_freq)

        def compute_factor(group):
            group = group.sort_values(by='datetime')
            cor = group['closew'].corr(group['volume99'])
            if pd.isna(cor) or np.isinf(cor):
                cor = 0
            return pd.Series({'factor': cor})
        factors = _data.groupby(['future', 'date']).apply(compute_factor).reset_index()
        factors.to_csv(FACTOR_PATH, index=False)

    ret = high_freq_rolling(factors, factor_name='corrpvstd',window=window,data=data,futures_universe=futures_universe, method='std', direction=-1)
    return ret

# 日内closew和volume99的相关系数
def corr_pv_mean(data, window, futures_universe, time_freq='day'):
    ret = pd.DataFrame()
    FACTOR_PATH = f'./factor_{time_freq}/corr_pv.csv'
    if os.path.exists(FACTOR_PATH):
        factors = pd.read_csv(FACTOR_PATH, parse_dates=['date'])
    else:
        _data = get_processed_min_data(futures_universe, time_freq)

        def compute_factor(group):
            group = group.sort_values(by='datetime')
            cor = group['closew'].corr(group['volume99'])
            if pd.isna(cor) or np.isinf(cor):
                cor = 0
            return pd.Series({'factor': cor})
        factors = _data.groupby(['future', 'date']).apply(compute_factor).reset_index()
        factors.to_csv(FACTOR_PATH, index=False)

    ret = high_freq_rolling(factors, factor_name='corrpvmean',window=window,data=data,futures_universe=futures_universe, method='mean', direction=-1)
    return ret

def corr_prv_mean(data, window, futures_universe, time_freq='day'):
    ret = pd.DataFrame()
    FACTOR_PATH = f'./factor_{time_freq}/corr_prv.csv'
    if os.path.exists(FACTOR_PATH):
        factors = pd.read_csv(FACTOR_PATH, parse_dates=['date'])
    else:
        _data = get_processed_min_data(futures_universe, time_freq)

        def compute_factor(group):
            group = group.sort_values(by='datetime')
            returns = (group['closew'] - group['closew'].shift(1)) / group['close'].shift(1)
            cor = (returns.iloc[1:]).corr(group['volume99'].iloc[1:])
            if pd.isna(cor) or np.isinf(cor):
                cor = 0
            return pd.Series({'factor': cor})
        factors = _data.groupby(['future', 'date']).apply(compute_factor).reset_index()
        factors.to_csv(FACTOR_PATH, index=False)

    ret = high_freq_rolling(factors, factor_name='corrprvmean',window=window,data=data,futures_universe=futures_universe, method='mean', direction=-1)
    return ret

def corr_prv_std(data, window, futures_universe, time_freq='day'):
    ret = pd.DataFrame()
    FACTOR_PATH = f'./factor_{time_freq}/corr_prv.csv'
    if os.path.exists(FACTOR_PATH):
        factors = pd.read_csv(FACTOR_PATH, parse_dates=['date'])
    else:
        _data = get_processed_min_data(futures_universe, time_freq)

        def compute_factor(group):
            group = group.sort_values(by='datetime')
            returns = (group['closew'] - group['closew'].shift(1)) / group['close'].shift(1)
            cor = (returns.iloc[1:]).corr(group['volume99'].iloc[1:])
            if pd.isna(cor) or np.isinf(cor):
                cor = 0
            return pd.Series({'factor': cor})
        factors = _data.groupby(['future', 'date']).apply(compute_factor).reset_index()
        factors.to_csv(FACTOR_PATH, index=False)

    ret = high_freq_rolling(factors, factor_name='corrprvstd',window=window,data=data,futures_universe=futures_universe, method='std', direction=-1)
    return ret

def GTJA060_min_mean(data, window, futures_universe, time_freq='day'):
    ret = pd.DataFrame()
    FACTOR_PATH = f'./factor_{time_freq}/GTJA060_min.csv'
    if os.path.exists(FACTOR_PATH):
        factors = pd.read_csv(FACTOR_PATH, parse_dates=['date'])
    else:
        _data = get_processed_min_data(futures_universe, time_freq)

        def compute_factor(group):
            group = group.sort_values(by='datetime')
            group['result'] = (((group['closew'] - group['loww']) / (group['highw'] - group['closew'] + 1e-6)) /
                            (group['highw'] - group['loww'] + 1e-6)) * group['volume99']
            daily_avg = group['result'].mean()
            return pd.Series({'factor': daily_avg})
        factors = _data.groupby(['future', 'date']).apply(compute_factor).reset_index()
        factors.to_csv(FACTOR_PATH, index=False)

    ret = high_freq_rolling(factors, factor_name='gtja060minmean',window=window,data=data,futures_universe=futures_universe, method='mean', direction=1)
    return ret

def GTJA060_min_std(data, window, futures_universe, time_freq='day'):
    ret = pd.DataFrame()
    FACTOR_PATH = f'./factor_{time_freq}/GTJA060_min.csv'
    if os.path.exists(FACTOR_PATH):
        factors = pd.read_csv(FACTOR_PATH, parse_dates=['date'])
    else:
        _data = get_processed_min_data(futures_universe, time_freq)

        def compute_factor(group):
            group = group.sort_values(by='datetime')
            group['result'] = (((group['closew'] - group['loww']) / (group['highw'] - group['closew'] + 1e-6)) /
                            (group['highw'] - group['loww'] + 1e-6)) * group['volume99']
            daily_avg = group['result'].mean()
            return pd.Series({'factor': daily_avg})
        factors = _data.groupby(['future', 'date']).apply(compute_factor).reset_index()
        factors.to_csv(FACTOR_PATH, index=False)

    ret = high_freq_rolling(factors, factor_name='gtja060minstd',window=window,data=data,futures_universe=futures_universe, method='std', direction=1)
    return ret

def GTJA052_min_mean(data, window, futures_universe, time_freq='day'):
    ret = pd.DataFrame()
    FACTOR_PATH = f'./factor_{time_freq}/GTJA052_min.csv'
    if os.path.exists(FACTOR_PATH):
        factors = pd.read_csv(FACTOR_PATH, parse_dates=['date'])
    else:
        _data = get_processed_min_data(futures_universe, time_freq)

        def compute_daily_factor(group):
            group['const'] = 0
            group['inner11'] = group['highw'] - ((group['highw'] + group['loww'] + group['closew']) / 3).shift(1)
            group['inner21'] = ((group['highw'] + group['loww'] + group['closew']) / 3).shift(1) - group['loww']
            group['inner1'] = group[['const', 'inner11']].max(axis=1)
            group['inner2'] = group[['const', 'inner21']].max(axis=1)
            daily_factor = group['inner1'].sum() / group['inner2'].sum()
            return pd.Series({'factor': daily_factor})
        factors = _data.groupby(['future', 'date']).apply(compute_daily_factor).reset_index()
        factors.to_csv(FACTOR_PATH, index=False)

    ret = high_freq_rolling(factors, factor_name='gtja052minmean',window=window,data=data,futures_universe=futures_universe, method='mean', direction=1)
    return ret

def GTJA118_min_mean(data, window, futures_universe, time_freq='day'):
    ret = pd.DataFrame()
    FACTOR_PATH = f'./factor_{time_freq}/GTJA118_min.csv'
    if os.path.exists(FACTOR_PATH):
        factors = pd.read_csv(FACTOR_PATH, parse_dates=['date'])
    else:
        _data = get_processed_min_data(futures_universe, time_freq)

        def compute_factor(group):
            group = group.sort_values(by='datetime')
            daily_avg = (group['highw'] - group['openw']).sum() / ((group['openw']-group['loww']).sum()+1e-6)
            return pd.Series({'factor': daily_avg})
        factors = _data.groupby(['future', 'date']).apply(compute_factor).reset_index()
        factors.to_csv(FACTOR_PATH, index=False)

    ret = high_freq_rolling(factors, factor_name='gtja118minmean',window=window,data=data,futures_universe=futures_universe, method='mean', direction=1)
    return ret

def mmt_ols_beta_mean(data, window, futures_universe, time_freq='day'):
    #日内rolling的长度，后续可以作为2Dwindow
    N = 50 if time_freq == 'day' else 30
    ret = pd.DataFrame()
    FACTOR_PATH = f'./factor_{time_freq}/mmt_ols_beta.csv'
    if os.path.exists(FACTOR_PATH):
        factors = pd.read_csv(FACTOR_PATH, parse_dates=['date'])
    else:
        _data = get_processed_min_data(futures_universe, time_freq)

        def compute_factor(group):
            group = group.sort_values(by='datetime')
            y = group['highw'] * 100
            X = group['loww'] * 100
            X = sm.add_constant(X)
            model = RollingOLS(y, X, window=N)
            rolling_results = model.fit()
            avg = rolling_results.params['loww'].mean()
            return pd.Series({'factor': avg})
        factors = _data.groupby(['future', 'date']).apply(compute_factor).reset_index()
        factors.to_csv(FACTOR_PATH, index=False)

    ret = high_freq_rolling(factors, factor_name='mmtolsbetamean',window=window,data=data,futures_universe=futures_universe, method='mean', direction=1)
    return ret

def GTJA129_min_mean(data, window, futures_universe, time_freq='day'):
    ret = pd.DataFrame()
    FACTOR_PATH = f'./factor_{time_freq}/GTJA129_min.csv'
    if os.path.exists(FACTOR_PATH):
        factors = pd.read_csv(FACTOR_PATH, parse_dates=['date'])
    else:
        _data = get_processed_min_data(futures_universe, time_freq)

        def compute_factor(group):
            group = group.sort_values(by='datetime')
            group['inner_price'] = group['closew'].diff().fillna(0)
            group['inner'] = group['inner_price'].where(group['inner_price'] < 0, 0).abs()
            daily_factor = group['inner'].sum()
            return pd.Series({'factor': daily_factor})
        factors = _data.groupby(['future', 'date']).apply(compute_factor).reset_index()
        factors.to_csv(FACTOR_PATH, index=False)

    ret = high_freq_rolling(factors, factor_name='gtja129minmean',window=window,data=data,futures_universe=futures_universe, method='mean', direction=1)
    return ret

def GTJA129_min_std(data, window, futures_universe, time_freq='day'):
    ret = pd.DataFrame()
    FACTOR_PATH = f'./factor_{time_freq}/GTJA129_min.csv'
    if os.path.exists(FACTOR_PATH):
        factors = pd.read_csv(FACTOR_PATH, parse_dates=['date'])
    else:
        _data = get_processed_min_data(futures_universe, time_freq)

        def compute_factor(group):
            group = group.sort_values(by='datetime')
            group['inner_price'] = group['closew'].diff().fillna(0)
            group['inner'] = group['inner_price'].where(group['inner_price'] < 0, 0).abs()
            daily_factor = group['inner'].sum()
            return pd.Series({'factor': daily_factor})
        factors = _data.groupby(['future', 'date']).apply(compute_factor).reset_index()
        factors.to_csv(FACTOR_PATH, index=False)

    ret = high_freq_rolling(factors, factor_name='gtja129minstd',window=window,data=data,futures_universe=futures_universe, method='std', direction=1)
    return ret

def GTJA053_min_mean(data, window, futures_universe, time_freq='day'):
    ret = pd.DataFrame()
    FACTOR_PATH = f'./factor_{time_freq}/GTJA053_min.csv'
    if os.path.exists(FACTOR_PATH):
        factors = pd.read_csv(FACTOR_PATH, parse_dates=['date'])
    else:
        _data = get_processed_min_data(futures_universe, time_freq)

        def compute_minute_factor(group):
            group = group.sort_values(by='datetime')
            group['inner'] = group['closew'] > group['closew'].shift(1)
            daily_factor = group['inner'].sum()
            return pd.Series({'factor': daily_factor})
        
        factors = _data.groupby(['future', 'date']).apply(compute_minute_factor).reset_index()
        factors.to_csv(FACTOR_PATH, index=False)

    ret = high_freq_rolling(factors, factor_name='gtja053minmean',window=window,data=data,futures_universe=futures_universe, method='mean', direction=1)
    return ret

def ALPHAGEN904_min_mean(data, window, futures_universe, time_freq='day'):
    ret = pd.DataFrame()
    FACTOR_PATH = f'./factor_{time_freq}/ALPHAGEN904_min.csv'
    if os.path.exists(FACTOR_PATH):
        factors = pd.read_csv(FACTOR_PATH, parse_dates=['date'])
    else:
        _data = get_processed_min_data(futures_universe, time_freq)

        def compute_daily_factor(group):
            daily_factor = group['open_interest99'].ewm(span=len(group)).mean().iloc[-1] - group['open_interest99'].mean()
            return pd.Series({'factor': daily_factor})

        factors = _data.groupby(['future', 'date']).apply(compute_daily_factor).reset_index()
        factors.to_csv(FACTOR_PATH, index=False)

    ret = high_freq_rolling(factors, factor_name='alphagen904minmean',window=window,data=data,futures_universe=futures_universe, method='mean', direction=1)
    return ret

def ALPHAGEN907_min_mean(data, window, futures_universe, time_freq='day'):
    ret = pd.DataFrame()
    FACTOR_PATH = f'./factor_{time_freq}/ALPHAGEN907_min.csv'
    if os.path.exists(FACTOR_PATH):
        factors = pd.read_csv(FACTOR_PATH, parse_dates=['date'])
    else:
        _data = get_processed_min_data(futures_universe, time_freq)

        def compute_daily_factor(group):
            group = group.sort_values(by='datetime')
            daily_factor = -(group['open_interest99'].iloc[-1] - group['open_interest99'].iloc[0])
            return pd.Series({'factor': daily_factor})
        
        factors = _data.groupby(['future', 'date']).apply(compute_daily_factor).reset_index()
        factors.to_csv(FACTOR_PATH, index=False)

    ret = high_freq_rolling(factors, factor_name='alphagen907minmean',window=window,data=data,futures_universe=futures_universe, method='mean', direction=1)
    return ret

def vol_upshadow_mean_std(data, window, futures_universe, time_freq='day'):
    ret = pd.DataFrame()
    FACTOR_PATH = f'./factor_{time_freq}/vol_upshadow_mean.csv'
    if os.path.exists(FACTOR_PATH):
        factors = pd.read_csv(FACTOR_PATH, parse_dates=['date'])
    else:
        _data = get_processed_min_data(futures_universe, time_freq)

        def compute_factor(group):
            group = group.sort_values(by='datetime')
            group['upshadow'] = (group['highw'] - np.maximum(group['closew'], group['openw'])) / group['low']
            avg = group['upshadow'].mean()
            return pd.Series({'factor': avg})
        factors = _data.groupby(['future', 'date']).apply(compute_factor).reset_index()
        factors.to_csv(FACTOR_PATH, index=False)

    ret = high_freq_rolling(factors, factor_name='upshadowmeanstd',window=window,data=data,futures_universe=futures_universe, method='std', direction=1)
    return ret

def vol_upshadow_std_mean(data, window, futures_universe, time_freq='day'):
    ret = pd.DataFrame()
    FACTOR_PATH = f'./factor_{time_freq}/vol_upshadow_std.csv'
    if os.path.exists(FACTOR_PATH):
        factors = pd.read_csv(FACTOR_PATH, parse_dates=['date'])
    else:
        _data = get_processed_min_data(futures_universe, time_freq)

        def compute_factor(group):
            group = group.sort_values(by='datetime')
            group['upshadow'] = (group['highw'] - np.maximum(group['closew'], group['openw'])) / group['low']
            sd = group['upshadow'].std()
            return pd.Series({'factor': sd})
        factors = _data.groupby(['future', 'date']).apply(compute_factor).reset_index()
        factors.to_csv(FACTOR_PATH, index=False)

    ret = high_freq_rolling(factors, factor_name='upshadowstdmean',window=window,data=data,futures_universe=futures_universe, method='mean', direction=1)
    return ret

def vol_upshadow_mean_mean(data, window, futures_universe, time_freq='day'):
    ret = pd.DataFrame()
    FACTOR_PATH = f'./factor_{time_freq}/vol_upshadow_mean.csv'
    if os.path.exists(FACTOR_PATH):
        factors = pd.read_csv(FACTOR_PATH, parse_dates=['date'])
    else:
        _data = get_processed_min_data(futures_universe, time_freq)

        def compute_factor(group):
            group = group.sort_values(by='datetime')
            group['upshadow'] = (group['highw'] - np.maximum(group['closew'], group['openw'])) / group['low']
            avg = group['upshadow'].mean()
            return pd.Series({'factor': avg})
        factors = _data.groupby(['future', 'date']).apply(compute_factor).reset_index()
        factors.to_csv(FACTOR_PATH, index=False)

    ret = high_freq_rolling(factors, factor_name='upshadowmeanmean',window=window,data=data,futures_universe=futures_universe, method='mean', direction=1)
    return ret

def vol_highlow_std(data, window, futures_universe, time_freq='day'):
    ret = pd.DataFrame()
    FACTOR_PATH = f'./factor_{time_freq}/vol_highlow.csv'
    if os.path.exists(FACTOR_PATH):
        factors = pd.read_csv(FACTOR_PATH, parse_dates=['date'])
    else:
        _data = get_processed_min_data(futures_universe, time_freq)

        def compute_factor(group):
            group = group.sort_values(by='datetime')
            group['highlow'] = group['high'] / group['low']
            avg = group['highlow'].mean()
            return pd.Series({'factor': avg})
        factors = _data.groupby(['future', 'date']).apply(compute_factor).reset_index()
        factors.to_csv(FACTOR_PATH, index=False)

    ret = high_freq_rolling(factors, factor_name='highlowstd',window=window,data=data,futures_universe=futures_universe, method='std', direction=1)
    return ret

def vol_downshadow_mean_std(data, window, futures_universe, time_freq='day'):
    ret = pd.DataFrame()
    FACTOR_PATH = f'./factor_{time_freq}/vol_downshadow_mean.csv'
    if os.path.exists(FACTOR_PATH):
        factors = pd.read_csv(FACTOR_PATH, parse_dates=['date'])
    else:
        _data = get_processed_min_data(futures_universe, time_freq)

        def compute_factor(group):
            group = group.sort_values(by='datetime')
            group['upshadow'] = (np.minimum(group['closew'], group['openw']) - group['loww']) / group['low']
            avg = group['upshadow'].mean()
            return pd.Series({'factor': avg})
        factors = _data.groupby(['future', 'date']).apply(compute_factor).reset_index()
        factors.to_csv(FACTOR_PATH, index=False)

    ret = high_freq_rolling(factors, factor_name='downshadowmeanstd',window=window,data=data,futures_universe=futures_universe, method='std', direction=1)
    return ret

def vol_downshadow_std_mean(data, window, futures_universe, time_freq='day'):
    ret = pd.DataFrame()
    FACTOR_PATH = f'./factor_{time_freq}/vol_downshadow_std.csv'
    if os.path.exists(FACTOR_PATH):
        factors = pd.read_csv(FACTOR_PATH, parse_dates=['date'])
    else:
        _data = get_processed_min_data(futures_universe, time_freq)

        def compute_factor(group):
            group = group.sort_values(by='datetime')
            group['upshadow'] = (np.minimum(group['closew'], group['openw']) - group['loww']) / group['low']
            sd = group['upshadow'].std()
            return pd.Series({'factor': sd})
        factors = _data.groupby(['future', 'date']).apply(compute_factor).reset_index()
        factors.to_csv(FACTOR_PATH, index=False)
    factors0 = factors.copy()

    ret = high_freq_rolling(factors, factor_name='downshadowstdmean',window=window,data=data,futures_universe=futures_universe, method='mean', direction=1)
    return ret

def vol_downshadow_mean_mean(data, window, futures_universe, time_freq='day'):
    ret = pd.DataFrame()
    FACTOR_PATH = f'./factor_{time_freq}/vol_downshadow_mean.csv'
    if os.path.exists(FACTOR_PATH):
        factors = pd.read_csv(FACTOR_PATH, parse_dates=['date'])
    else:
        _data = get_processed_min_data(futures_universe, time_freq)

        def compute_factor(group):
            group = group.sort_values(by='datetime')
            group['upshadow'] = (np.minimum(group['closew'], group['openw']) - group['loww']) / group['low']
            avg = group['upshadow'].mean()
            return pd.Series({'factor': avg})
        factors = _data.groupby(['future', 'date']).apply(compute_factor).reset_index()
        factors.to_csv(FACTOR_PATH, index=False)

    ret = high_freq_rolling(factors, factor_name='downshadowmeanmean',window=window,data=data,futures_universe=futures_universe, method='mean', direction=1)
    return ret

def liq_vstd_mean(data, window, futures_universe, time_freq='day'):
    ret = pd.DataFrame()
    FACTOR_PATH = f'./factor_{time_freq}/liq_vstd.csv'
    if os.path.exists(FACTOR_PATH):
        factors = pd.read_csv(FACTOR_PATH, parse_dates=['date'])
    else:
        _data = get_processed_min_data(futures_universe, time_freq)

        def compute_factor(group):
            group = group.sort_values(by='datetime')
            group['return'] = (group['closew'] - group['closew'].shift(1)) / group['close'].shift(1)
            group['return'] = group['return'].dropna()
            return_std = group['return'].std()
            vstd = group['total_turnover'].sum() / return_std
            return pd.Series({'factor': vstd})
        factors = _data.groupby(['future', 'date']).apply(compute_factor).reset_index()
        factors.to_csv(FACTOR_PATH, index=False)

    ret = high_freq_rolling(factors, factor_name='liqvstdmean',window=window,data=data,futures_universe=futures_universe, method='mean', direction=1)
    return ret

def liq_amihud_mean_mean(data, window, futures_universe, time_freq='day'):
    ret = pd.DataFrame()
    FACTOR_PATH = f'./factor_{time_freq}/liq_amihud_mean.csv'
    if os.path.exists(FACTOR_PATH):
        factors = pd.read_csv(FACTOR_PATH, parse_dates=['date'])
    else:
        _data = get_processed_min_data(futures_universe, time_freq)

        def compute_factor(group):
            group = group.sort_values(by='datetime')
            returns = (group['closew'] - group['closew'].shift(1)) / group['close'].shift(1)
            valid_data = group.iloc[1:]
            amihud_values = returns.abs().iloc[1:] / valid_data['volume99']
            amihud_liquidity = amihud_values.mean()
            return pd.Series({'factor': amihud_liquidity})
        factors = _data.groupby(['future', 'date']).apply(compute_factor).reset_index()
        factors.to_csv(FACTOR_PATH, index=False)

    ret = high_freq_rolling(factors, factor_name='liqamihudmeanmean',window=window,data=data,futures_universe=futures_universe, method='mean', direction=1)
    return ret

def liq_amihud_mean_std(data, window, futures_universe, time_freq='day'):
    ret = pd.DataFrame()
    FACTOR_PATH = f'./factor_{time_freq}/liq_amihud_mean.csv'
    if os.path.exists(FACTOR_PATH):
        factors = pd.read_csv(FACTOR_PATH, parse_dates=['date'])
    else:
        _data = get_processed_min_data(futures_universe, time_freq)

        def compute_factor(group):
            group = group.sort_values(by='datetime')
            returns = (group['closew'] - group['closew'].shift(1)) / group['close'].shift(1)
            valid_data = group.iloc[1:]
            amihud_values = returns.abs().iloc[1:] / valid_data['volume99']
            amihud_liquidity = amihud_values.mean()
            return pd.Series({'factor': amihud_liquidity})
        factors = _data.groupby(['future', 'date']).apply(compute_factor).reset_index()
        factors.to_csv(FACTOR_PATH, index=False)

    ret = high_freq_rolling(factors, factor_name='liqamihudmeanstd',window=window,data=data,futures_universe=futures_universe, method='std', direction=1)
    return ret

def liq_amihud_std_mean(data, window, futures_universe, time_freq='day'):
    ret = pd.DataFrame()
    FACTOR_PATH = f'./factor_{time_freq}/liq_amihud_std.csv'
    if os.path.exists(FACTOR_PATH):
        factors = pd.read_csv(FACTOR_PATH, parse_dates=['date'])
    else:
        _data = get_processed_min_data(futures_universe, time_freq)

        def compute_factor(group):
            group = group.sort_values(by='datetime')
            returns = (group['closew'] - group['closew'].shift(1)) / group['close'].shift(1)
            valid_data = group.iloc[1:]
            amihud_values = returns.abs().iloc[1:] / valid_data['volume99']
            amihud_liquidity = amihud_values.std()
            return pd.Series({'factor': amihud_liquidity})
        factors = _data.groupby(['future', 'date']).apply(compute_factor).reset_index()
        factors.to_csv(FACTOR_PATH, index=False)

    ret = high_freq_rolling(factors, factor_name='liqamihudstdmean',window=window,data=data,futures_universe=futures_universe, method='mean', direction=1)
    return ret

def liq_shortcut_mean_mean(data, window, futures_universe, time_freq='day'):
    ret = pd.DataFrame()
    FACTOR_PATH = f'./factor_{time_freq}/liq_shortcut_mean.csv'
    if os.path.exists(FACTOR_PATH):
        factors = pd.read_csv(FACTOR_PATH, parse_dates=['date'])
    else:
        _data = get_processed_min_data(futures_universe, time_freq)

        def compute_factor(group):
            group = group.sort_values(by='datetime')
            group['shortcut'] = 2*(group['highw']-group['loww'])-abs(group['openw']-group['closew'])
            avg = (group['shortcut'] / group['total_turnover']).mean()
            return pd.Series({'factor': avg})
        factors = _data.groupby(['future', 'date']).apply(compute_factor).reset_index()
        factors.to_csv(FACTOR_PATH, index=False)

    ret = high_freq_rolling(factors, factor_name='liqshortcutmeanmean',window=window,data=data,futures_universe=futures_universe, method='mean', direction=1)
    return ret

def liq_shortcut_mean_std(data, window, futures_universe, time_freq='day'):
    ret = pd.DataFrame()
    FACTOR_PATH = f'./factor_{time_freq}/liq_shortcut_mean.csv'
    if os.path.exists(FACTOR_PATH):
        factors = pd.read_csv(FACTOR_PATH, parse_dates=['date'])
    else:
        _data = get_processed_min_data(futures_universe, time_freq)

        def compute_factor(group):
            group = group.sort_values(by='datetime')
            group['shortcut'] = 2*(group['highw']-group['loww'])-abs(group['openw']-group['closew'])
            avg = (group['shortcut'] / group['total_turnover']).mean()
            return pd.Series({'factor': avg})
        factors = _data.groupby(['future', 'date']).apply(compute_factor).reset_index()
        factors.to_csv(FACTOR_PATH, index=False)

    ret = high_freq_rolling(factors, factor_name='liqshortcutmeanstd',window=window,data=data,futures_universe=futures_universe, method='std', direction=1)
    return ret

def liq_shortcut_std_mean(data, window, futures_universe, time_freq='day'):
    ret = pd.DataFrame()
    FACTOR_PATH = f'./factor_{time_freq}/liq_shortcut_std.csv'
    if os.path.exists(FACTOR_PATH):
        factors = pd.read_csv(FACTOR_PATH, parse_dates=['date'])
    else:
        _data = get_processed_min_data(futures_universe, time_freq)

        def compute_factor(group):
            group = group.sort_values(by='datetime')
            group['shortcut'] = 2*(group['highw']-group['loww'])-abs(group['openw']-group['closew'])
            sd = (group['shortcut'] / group['total_turnover']).std()
            return pd.Series({'factor': sd})
        factors = _data.groupby(['future', 'date']).apply(compute_factor).reset_index()
        factors.to_csv(FACTOR_PATH, index=False)

    ret = high_freq_rolling(factors, factor_name='liqshortcutstdmean',window=window,data=data,futures_universe=futures_universe, method='mean', direction=1)
    return ret

def GTJA084_min_mean(data, window, futures_universe, time_freq='day'):
    ret = pd.DataFrame()
    FACTOR_PATH = f'./factor_{time_freq}/GTJA084_min.csv'
    if os.path.exists(FACTOR_PATH):
        factors = pd.read_csv(FACTOR_PATH, parse_dates=['date'])
    else:
        _data = get_processed_min_data(futures_universe, time_freq)

        def compute_factor(group):
            group = group.sort_values(by='datetime')
            group['inner'] = ((group['closew'] > group['closew'].shift(1)) * group['volume99'] - 
                                (group['closew'] < group['closew'].shift(1)) * group['volume99'])
            avg = group['inner'].mean()
            return pd.Series({'factor': avg})
        factors = _data.groupby(['future', 'date']).apply(compute_factor).reset_index()
        factors.to_csv(FACTOR_PATH, index=False)

    ret = high_freq_rolling(factors, factor_name='gtja084minmean',window=window,data=data,futures_universe=futures_universe, method='mean', direction=1)
    return ret

def GTJA049_min_mean(data, window, futures_universe, time_freq='day'):
    ret = pd.DataFrame()
    FACTOR_PATH = f'./factor_{time_freq}/GTJA049_min.csv'
    if os.path.exists(FACTOR_PATH):
        factors = pd.read_csv(FACTOR_PATH, parse_dates=['date'])
    else:
        _data = get_processed_min_data(futures_universe, time_freq)

        def compute_factor(group):
            group = group.sort_values(by='datetime')
            group['high_diff'] = np.abs(group['highw'].diff(1))
            group['low_diff'] = np.abs(group['loww'].diff(1))
            group['inner_price'] = group[['high_diff', 'low_diff']].max(axis=1)
            group['cond1'] = group['highw'] + group['loww'] >= group['highw'].shift(1) + group['loww'].shift(1)
            group['cond2'] = group['highw'] + group['loww'] <= group['highw'].shift(1) + group['loww'].shift(1)

            group['inner1'] = group['inner_price']
            group.loc[group['cond1'], 'inner1'] = 0
            
            group['inner2'] = group['inner_price']
            group.loc[group['cond2'], 'inner2'] = 0
            
            g049 = group['inner1'].sum() / (group['inner1'].sum() + group['inner2'].sum())
            
            return pd.Series({'factor': g049})

        factors = _data.groupby(['future', 'date']).apply(compute_factor).reset_index()
        factors.to_csv(FACTOR_PATH, index=False)

    ret = high_freq_rolling(factors, factor_name='gtja049minmean',window=window,data=data,futures_universe=futures_universe, method='mean', direction=1)
    return ret

def shape_skew_mean(data, window, futures_universe, time_freq='day'):
    ret = pd.DataFrame()
    FACTOR_PATH = f'./factor_{time_freq}/shape_skew.csv'
    if os.path.exists(FACTOR_PATH):
        factors = pd.read_csv(FACTOR_PATH, parse_dates=['date'])
    else:
        _data = get_processed_min_data(futures_universe, time_freq)

        def compute_factor(group):
            group = group.sort_values(by='datetime')
            returns = (group['closew'] - group['closew'].shift(1)) / group['close'].shift(1)
            returns = returns.dropna()
            skew = returns.skew()
            return pd.Series({'factor': skew})
        factors = _data.groupby(['future', 'date']).apply(compute_factor).reset_index()
        factors.to_csv(FACTOR_PATH, index=False)

    ret = high_freq_rolling(factors, factor_name='shapeskewmean',window=window,data=data,futures_universe=futures_universe, method='mean', direction=1)
    return ret

def shape_skew_std(data, window, futures_universe, time_freq='day'):
    ret = pd.DataFrame()
    FACTOR_PATH = f'./factor_{time_freq}/shape_skew.csv'
    if os.path.exists(FACTOR_PATH):
        factors = pd.read_csv(FACTOR_PATH, parse_dates=['date'])
    else:
        _data = get_processed_min_data(futures_universe, time_freq)

        def compute_factor(group):
            group = group.sort_values(by='datetime')
            returns = (group['closew'] - group['closew'].shift(1)) / group['close'].shift(1)
            returns = returns.dropna()
            skew = returns.skew()
            return pd.Series({'factor': skew})
        factors = _data.groupby(['future', 'date']).apply(compute_factor).reset_index()
        factors.to_csv(FACTOR_PATH, index=False)

    ret = high_freq_rolling(factors, factor_name='shapeskewstd',window=window,data=data,futures_universe=futures_universe, method='std', direction=1)
    return ret

def shape_kurt_mean(data, window, futures_universe, time_freq='day'):
    ret = pd.DataFrame()
    FACTOR_PATH = f'./factor_{time_freq}/shape_kurt.csv'
    if os.path.exists(FACTOR_PATH):
        factors = pd.read_csv(FACTOR_PATH, parse_dates=['date'])
    else:
        _data = get_processed_min_data(futures_universe, time_freq)

        def compute_factor(group):
            group = group.sort_values(by='datetime')
            returns = (group['closew'] - group['closew'].shift(1)) / group['close'].shift(1)
            returns = returns.dropna()
            kurt = returns.kurt()
            return pd.Series({'factor': kurt})
        factors = _data.groupby(['future', 'date']).apply(compute_factor).reset_index()
        factors.to_csv(FACTOR_PATH, index=False)

    ret = high_freq_rolling(factors, factor_name='shapekurtmean',window=window,data=data,futures_universe=futures_universe, method='mean', direction=1)
    return ret

def shape_kurt_std(data, window, futures_universe, time_freq='day'):
    ret = pd.DataFrame()
    FACTOR_PATH = f'./factor_{time_freq}/shape_kurt.csv'
    if os.path.exists(FACTOR_PATH):
        factors = pd.read_csv(FACTOR_PATH, parse_dates=['date'])
    else:
        _data = get_processed_min_data(futures_universe, time_freq)

        def compute_factor(group):
            group = group.sort_values(by='datetime')
            returns = (group['closew'] - group['closew'].shift(1)) / group['close'].shift(1)
            returns = returns.dropna()
            kurt = returns.kurt()
            return pd.Series({'factor': kurt})
        factors = _data.groupby(['future', 'date']).apply(compute_factor).reset_index()
        factors.to_csv(FACTOR_PATH, index=False)

    ret = high_freq_rolling(factors, factor_name='shapekurtstd',window=window,data=data,futures_universe=futures_universe, method='std', direction=1)
    return ret

def mmt_top20VolumeRet_std(data, window, futures_universe, time_freq='day'):
    N = 20
    ret = pd.DataFrame()
    FACTOR_PATH = f'./factor_{time_freq}/mmt_top20VolumeRet.csv'
    if os.path.exists(FACTOR_PATH):
        factors = pd.read_csv(FACTOR_PATH, parse_dates=['date'])
    else:
        _data = get_processed_min_data(futures_universe, time_freq)

        def compute_factor(group):
            top20_volume_group = group.nlargest(N, 'volume99')
            top20_volume_group = top20_volume_group.sort_values(by='datetime')
            return_series = (top20_volume_group['closew'] - top20_volume_group['closew'].shift(1)) / top20_volume_group['close'].shift(1)
            return_series = return_series.iloc[1:]
            # weights = np.ones(len(return_series))
            # weights = weights / sum(weights)
            weights = np.exp(np.linspace(-1, 0, len(return_series)))
            weights /= weights.sum()
            daily_momentum = np.dot(return_series, weights)
            return pd.Series({'factor': daily_momentum})
        factors = _data.groupby(['future', 'date']).apply(compute_factor).reset_index()
        factors.to_csv(FACTOR_PATH, index=False)

    ret = high_freq_rolling(factors, factor_name='mmttop20volumeretstd',window=window,data=data,futures_universe=futures_universe, method='std', direction=1)
    return ret

def mmt_top20VolumeRet_mean(data, window, futures_universe, time_freq='day'):
    N = 20
    ret = pd.DataFrame()
    FACTOR_PATH = f'./factor_{time_freq}/mmt_top20VolumeRet.csv'
    if os.path.exists(FACTOR_PATH):
        factors = pd.read_csv(FACTOR_PATH, parse_dates=['date'])
    else:
        _data = get_processed_min_data(futures_universe, time_freq)

        def compute_factor(group):
            top20_volume_group = group.nlargest(N, 'volume99')
            top20_volume_group = top20_volume_group.sort_values(by='datetime')
            return_series = (top20_volume_group['closew'] - top20_volume_group['closew'].shift(1)) / top20_volume_group['close'].shift(1)
            return_series = return_series.iloc[1:]
            # weights = np.ones(len(return_series))
            # weights = weights / sum(weights)
            weights = np.exp(np.linspace(-1, 0, len(return_series)))
            weights /= weights.sum()
            daily_momentum = np.dot(return_series, weights)
            return pd.Series({'factor': daily_momentum})
        factors = _data.groupby(['future', 'date']).apply(compute_factor).reset_index()
        factors.to_csv(FACTOR_PATH, index=False)

    ret = high_freq_rolling(factors, factor_name='mmttop20volumeretmean',window=window,data=data,futures_universe=futures_universe, method='mean', direction=1)
    return ret

def mmt_last30_mean(data, window, futures_universe, time_freq='day'):
    ret = pd.DataFrame()
    FACTOR_PATH = f'./factor_{time_freq}/mmt_last30.csv'
    if os.path.exists(FACTOR_PATH):
        factors = pd.read_csv(FACTOR_PATH, parse_dates=['date'])
    else:
        _data = get_processed_min_data(futures_universe, time_freq)

        def get_last_minutes(group):
            # 计算当前bar结束之前的指定分钟数时间点
            minutes_before = 30
            end_time = group.iloc[-1]['datetime']
            start_time = end_time - datetime.timedelta(minutes=minutes_before)
            end_time = end_time.time()
            start_time = start_time.time()
            last_minutes = group[(group['time'] > start_time) & (group['time'] <= end_time)]
            return last_minutes

        def compute_momentum(group):
            group = group.sort_values(by='datetime')
            # 计算每分钟的收益率
            group['minute_return'] = (group['closew'] - group['closew'].shift(1)) / group['close'].shift(1)
            group['minute_return'] = group['minute_return'].dropna()
            # 如果收益率序列为空，则直接返回 NaN
            if group['minute_return'].empty:
                return pd.Series({'factor': np.nan})
            # 使用指数衰减权重
            weights = np.exp(np.linspace(-1, 0, len(group['minute_return'])))
            weights /= weights.sum()
            # 计算加权平均收益率
            weighted_momentum = (group['minute_return'] * weights).sum()
            # weighted_momentum = -weighted_momentum
            return pd.Series({'factor': weighted_momentum})
        
        def compute_momentum1(group):
            group = group.sort_values(by='datetime')
            first_price = group.iloc[0]['closew']
            last_price = group.iloc[-1]['closew']
            momentum = -(last_price - first_price) / first_price
            return pd.Series({'factor': momentum})
        
        last_minutes_data = _data.groupby(['future', 'date']).apply(get_last_minutes).reset_index(drop=True)
        factors = last_minutes_data.groupby(['future', 'date']).apply(compute_momentum1).reset_index()
        factors.to_csv(FACTOR_PATH, index=False)

    ret = high_freq_rolling(factors, factor_name='mmtlast30mean',window=window,data=data,futures_universe=futures_universe, method='mean', direction=1)
    return ret

def mmt_last30_std(data, window, futures_universe, time_freq='day'):
    ret = pd.DataFrame()
    FACTOR_PATH = f'./factor_{time_freq}/mmt_last30.csv'
    if os.path.exists(FACTOR_PATH):
        factors = pd.read_csv(FACTOR_PATH, parse_dates=['date'])
    else:
        _data = get_processed_min_data(futures_universe, time_freq)

        def get_last_minutes(group):
            # 计算当前bar结束之前的指定分钟数时间点
            minutes_before = 30
            end_time = group.iloc[-1]['datetime']
            start_time = end_time - datetime.timedelta(minutes=minutes_before)
            end_time = end_time.time()
            start_time = start_time.time()
            last_minutes = group[(group['time'] > start_time) & (group['time'] <= end_time)]
            return last_minutes

        def compute_momentum(group):
            group = group.sort_values(by='datetime')
            # 计算每分钟的收益率
            group['minute_return'] = (group['closew'] - group['closew'].shift(1)) / group['close'].shift(1)
            group['minute_return'] = group['minute_return'].dropna()
            # 如果收益率序列为空，则直接返回 NaN
            if group['minute_return'].empty:
                return pd.Series({'factor': np.nan})
            # 使用指数衰减权重
            weights = np.exp(np.linspace(-1, 0, len(group['minute_return'])))
            weights /= weights.sum()
            # 计算加权平均收益率
            weighted_momentum = (group['minute_return'] * weights).sum()
            # weighted_momentum = -weighted_momentum
            return pd.Series({'factor': weighted_momentum})
        
        def compute_momentum1(group):
            group = group.sort_values(by='datetime')
            first_price = group.iloc[0]['closew']
            last_price = group.iloc[-1]['closew']
            momentum = (last_price - first_price) / first_price
            return pd.Series({'factor': momentum})
        
        last_minutes_data = _data.groupby(['future', 'date']).apply(get_last_minutes).reset_index(drop=True)
        factors = last_minutes_data.groupby(['future', 'date']).apply(compute_momentum).reset_index()
        factors.to_csv(FACTOR_PATH, index=False)

    ret = high_freq_rolling(factors, factor_name='mmtlast30std',window=window,data=data,futures_universe=futures_universe, method='std', direction=1)
    return ret

def mmt_start30_mean(data, window, futures_universe, time_freq='day'):
    ret = pd.DataFrame()
    FACTOR_PATH = f'./factor_{time_freq}/mmt_start30.csv'
    if os.path.exists(FACTOR_PATH):
        factors = pd.read_csv(FACTOR_PATH, parse_dates=['date'])
    else:
        _data = get_processed_min_data(futures_universe, time_freq)

        def get_start_minutes(group):
            # 计算当前bar之后的指定分钟数时间点
            minutes_after = 30
            start_time = group.iloc[0]['datetime']
            end_time = start_time + datetime.timedelta(minutes=minutes_after)
            end_time = end_time.time()
            start_time = start_time.time()
            start_minutes = group[(group['time'] >= start_time) & (group['time'] < end_time)]
            return start_minutes.sort_values(by='time')

        def compute_momentum(group):
            group = group.sort_values(by='datetime')
            # 计算每分钟的收益率
            group['minute_return'] = (group['closew'] - group['closew'].shift(1)) / group['close'].shift(1)
            group['minute_return'] = group['minute_return'].dropna()
            # 如果收益率序列为空，则直接返回 NaN
            if group['minute_return'].empty:
                return pd.Series({'factor': np.nan})
            # 使用指数衰减权重
            weights = np.exp(np.linspace(-1, 0, len(group['minute_return'])))
            weights /= weights.sum()
            # 计算加权平均收益率
            weighted_momentum = (group['minute_return'] * weights).sum()
            # weighted_momentum = -weighted_momentum
            return pd.Series({'factor': weighted_momentum})
        
        def compute_momentum1(group):
            group = group.sort_values(by='datetime')
            first_price = group.iloc[0]['closew']
            last_price = group.iloc[-1]['closew']
            momentum = (last_price - first_price) / first_price
            return pd.Series({'factor': momentum})
        
        last_minutes_data = _data.groupby(['future', 'date']).apply(get_start_minutes).reset_index(drop=True)
        factors = last_minutes_data.groupby(['future', 'date']).apply(compute_momentum).reset_index()
        factors.to_csv(FACTOR_PATH, index=False)

    ret = high_freq_rolling(factors, factor_name='mmtstart30mean',window=window,data=data,futures_universe=futures_universe, method='mean', direction=1)
    return ret

def mmt_start30_std(data, window, futures_universe, time_freq='day'):
    ret = pd.DataFrame()
    FACTOR_PATH = f'./factor_{time_freq}/mmt_start30.csv'
    if os.path.exists(FACTOR_PATH):
        factors = pd.read_csv(FACTOR_PATH, parse_dates=['date'])
    else:
        _data = get_processed_min_data(futures_universe, time_freq)

        def get_start_minutes(group):
            # 计算当前bar之后的指定分钟数时间点
            minutes_after = 30
            start_time = group.iloc[0]['datetime']
            end_time = start_time + datetime.timedelta(minutes=minutes_after)
            end_time = end_time.time()
            start_time = start_time.time()
            start_minutes = group[(group['time'] >= start_time) & (group['time'] < end_time)]
            return start_minutes.sort_values(by='time')

        def compute_momentum(group):
            group = group.sort_values(by='datetime')
            # 计算每分钟的收益率
            group['minute_return'] = (group['closew'] - group['closew'].shift(1)) / group['close'].shift(1)
            group['minute_return'] = group['minute_return'].dropna()
            # 如果收益率序列为空，则直接返回 NaN
            if group['minute_return'].empty:
                return pd.Series({'factor': np.nan})
            # 使用指数衰减权重
            weights = np.exp(np.linspace(-1, 0, len(group['minute_return'])))
            weights /= weights.sum()
            # 计算加权平均收益率
            weighted_momentum = (group['minute_return'] * weights).sum()
            # weighted_momentum = -weighted_momentum
            return pd.Series({'factor': weighted_momentum})
        
        def compute_momentum1(group):
            group = group.sort_values(by='datetime')
            first_price = group.iloc[0]['closew']
            last_price = group.iloc[-1]['closew']
            momentum = (last_price - first_price) / first_price
            return pd.Series({'factor': momentum})
        
        last_minutes_data = _data.groupby(['future', 'date']).apply(get_start_minutes).reset_index(drop=True)
        factors = last_minutes_data.groupby(['future', 'date']).apply(compute_momentum).reset_index()
        factors.to_csv(FACTOR_PATH, index=False)

    ret = high_freq_rolling(factors, factor_name='mmtstart30std',window=window,data=data,futures_universe=futures_universe, method='std', direction=1)
    return ret

def mmt_ols_beta_std(data, window, futures_universe, time_freq='day'):
    N = 50 if time_freq =='day'else 30 #日内rolling的长度，后续可以作为2Dwindow
    ret = pd.DataFrame()
    FACTOR_PATH = f'./factor_{time_freq}/mmt_ols_beta.csv'
    if os.path.exists(FACTOR_PATH):
        factors = pd.read_csv(FACTOR_PATH, parse_dates=['date'])
    else:
        _data = get_processed_min_data(futures_universe, time_freq)

        def compute_factor(group):
            group = group.sort_values(by='datetime')
            y = group['highw']
            X = group['loww']
            X = sm.add_constant(X)
            model = RollingOLS(y, X, window=N)
            rolling_results = model.fit()
            avg = rolling_results.params['loww'].mean()
            return pd.Series({'factor': avg})
        factors = _data.groupby(['future', 'date']).apply(compute_factor).reset_index()
        factors.to_csv(FACTOR_PATH, index=False)

    ret = high_freq_rolling(factors, factor_name='mmtolsbetastd',window=window,data=data,futures_universe=futures_universe, method='std', direction=1)
    return ret

def mmt_ols_corr_mean(data, window, futures_universe, time_freq='day'):
    N = 50 if time_freq =='day'else 30 #日内rolling的长度，后续可以作为2Dwindow
    ret = pd.DataFrame()
    FACTOR_PATH = f'./factor_{time_freq}/mmt_ols_corr.csv'
    if os.path.exists(FACTOR_PATH):
        factors = pd.read_csv(FACTOR_PATH, parse_dates=['date'])
    else:
        _data = get_processed_min_data(futures_universe, time_freq)

        def compute_factor(group):
            group = group.sort_values(by='datetime')
            y = group['highw']
            X = group['loww']
            rolling_corr = y.rolling(window=N).corr(X)
            avg_corr = rolling_corr.mean()
            return pd.Series({'factor': avg_corr})
        factors = _data.groupby(['future', 'date']).apply(compute_factor).reset_index()
        factors.to_csv(FACTOR_PATH, index=False)

    ret = high_freq_rolling(factors, factor_name='mmtolscorrmean',window=window,data=data,futures_universe=futures_universe, method='mean', direction=1)
    return ret

def mmt_ols_corr_std(data, window, futures_universe, time_freq='day'):
    N = 50 if time_freq =='day'else 30 #日内rolling的长度，后续可以作为2Dwindow
    ret = pd.DataFrame()
    FACTOR_PATH = f'./factor_{time_freq}/mmt_ols_corr.csv'
    if os.path.exists(FACTOR_PATH):
        factors = pd.read_csv(FACTOR_PATH, parse_dates=['date'])
    else:
        _data = get_processed_min_data(futures_universe, time_freq)

        def compute_factor(group):
            group = group.sort_values(by='datetime')
            y = group['highw']
            X = group['loww']
            rolling_corr = y.rolling(window=N).corr(X)
            avg_corr = rolling_corr.mean()
            return pd.Series({'factor': avg_corr})
        factors = _data.groupby(['future', 'date']).apply(compute_factor).reset_index()
        factors.to_csv(FACTOR_PATH, index=False)

    ret = high_freq_rolling(factors, factor_name='mmtolscorrstd',window=window,data=data,futures_universe=futures_universe, method='std', direction=1)
    return ret

def GTJA093_min_mean(data, window, futures_universe, time_freq='day'):
    ret = pd.DataFrame()
    FACTOR_PATH = f'./factor_{time_freq}/GTJA093_min.csv'
    if os.path.exists(FACTOR_PATH):
        factors = pd.read_csv(FACTOR_PATH, parse_dates=['date'])
    else:
        _data = get_processed_min_data(futures_universe, time_freq)

        def compute_factor(group):
            group = group.sort_values(by='datetime')
            group['inner1'] = group['openw'] - group['loww']
            group['inner2'] = group['openw'].diff(1)
            group['inner'] = 0
            group.loc[group['inner2'] < 0, 'inner'] = group[['inner1', 'inner2']].max(axis=1)
            g093 = group['inner'].sum()
            return pd.Series({'factor': g093})
        factors = _data.groupby(['future', 'date']).apply(compute_factor).reset_index()
        factors.to_csv(FACTOR_PATH, index=False)

    ret = high_freq_rolling(factors, factor_name='gtja093minmean',window=window,data=data,futures_universe=futures_universe, method='mean', direction=1)
    return ret

def GTJA128_min_mean(data, window, futures_universe, time_freq='day'):
    ret = pd.DataFrame()
    FACTOR_PATH = f'./factor_{time_freq}/GTJA128_min.csv'
    if os.path.exists(FACTOR_PATH):
        factors = pd.read_csv(FACTOR_PATH, parse_dates=['date'])
    else:
        _data = get_processed_min_data(futures_universe, time_freq)

        def compute_factor(group):
            group = group.sort_values(by='datetime')
            group['inner_price'] = (group['highw'] + group['loww'] + group['closew']) / 3
            group['inner1'] = 0
            group['inner2'] = 0
            group.loc[group['inner_price'] > group['inner_price'].shift(1), 'inner1'] = group['inner_price'] * group['volume99']
            group.loc[group['inner_price'] < group['inner_price'].shift(1), 'inner2'] = group['inner_price'] * group['volume99']
            g128 = 100 - (100 / (1 + group['inner1'].sum() / group['inner2'].sum()))
            return pd.Series({'factor': g128})
        factors = _data.groupby(['future', 'date']).apply(compute_factor).reset_index()
        factors.to_csv(FACTOR_PATH, index=False)

    ret = high_freq_rolling(factors, factor_name='gtja128minmean',window=window,data=data,futures_universe=futures_universe, method='mean', direction=1)
    return ret