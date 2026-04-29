import sys
import os
import numpy as np
import bottleneck as bn
from numpy.lib.stride_tricks import sliding_window_view as sliding
import pandas as pd
# from .gp_config import *
username = os.getlogin()
sys.path.append(rf'/home/{username}/torch_operators_user')
# from .torch_operators import *
# user_name = os.getlogin()
# sys.path.append(rf'/home/{user_name}/torch_operators_user/')
from torch_operators_my import *

func_map_dict = {
    ##简单算子
    'add': (add, ['basic', 'basic']),
    'sub': (sub, ['basic', 'basic']),
    'mul': (mul, ['basic', 'basic']),
    'div': (div, ['basic', 'basic']),
    'Max': (Max, ['basic', 'basic']),
    'Min': (Min, ['basic', 'basic']),
    'Abs': (Abs, ['basic']),
    'neg': (neg, ['basic']),
    'sqrt': (sqrt, ['basic']),
    'log': (log, ['basic']),
    'Pow': (Pow, ['basic', 'const_exp']),
    # 'sign_square': (sign_square, ['basic','const_exp']),
    'inv': (inv, ['basic']),
    'sigmoid': (sigmoid, ['basic']),
    'sign': (sign, ['basic']),
    'ceil_scale':(ceil_scale, ['basic', 'const_exp']),
    'floor_scale':(floor_scale, ['basic', 'const_exp']),
    'round_zero_scale':(round_zero_scale, ['basic', 'const_exp']),
    # 'ceil_scale':(ceil_scale, ['basic']),
    

    # 时序算子
    # 'ts_diff_abs_sum': (ts_diff_abs_sum, ['basic', 'const_delay']),
    'ts_delay': (ts_delay, ['basic', 'const_delay']),
    'ts_delta': (ts_delta, ['basic', 'const_delay']),
    'ts_log_diff': (ts_log_diff, ['basic', 'const_delay']),
    'ts_corr': (ts_corr, ['basic', 'basic', 'const_delay']),
    'ts_rankcorr': (ts_rankcorr, ['basic', 'basic', 'const_delay']),
    'ts_cov': (ts_cov, ['basic', 'basic', 'const_delay']),
    'ts_decay_linear': (ts_decay_linear, ['basic', 'const_delay']),
    'ts_autocorr': (ts_autocorr, ['basic', 'const_delay', 'const_delay']),

    'ts_rank': (ts_rank, ['basic', 'const_delay']),
    'ts_min': (ts_min, ['basic', 'const_delay']),
    'ts_max': (ts_max, ['basic', 'const_delay']),
    'ts_argmin': (ts_argmin, ['basic', 'const_delay']),
    'ts_argmax': (ts_argmax, ['basic', 'const_delay']),
    'ts_sum': (ts_sum, ['basic', 'const_delay']),
    'ts_product': (ts_product, ['basic', 'const_delay']),
    'ts_mean': (ts_mean, ['basic', 'const_delay']),
    'ts_med': (ts_med, ['basic', 'const_delay']),
    'ts_mad_mean': (ts_mad_mean, ['basic', 'const_delay']),
    'ts_mad_med': (ts_mad_med, ['basic', 'const_delay']),
    'ts_var': (ts_var, ['basic', 'const_delay']),
    'ts_stddev': (ts_stddev, ['basic', 'const_delay']),
    'ts_kurt': (ts_kurt, ['basic', 'const_delay']),
    'ts_skew': (ts_skew, ['basic', 'const_delay']),
    'ts_weighted_skew': (ts_weighted_skew, ['basic', 'basic', 'const_delay']),
    'ts_triple_corr': (ts_triple_corr, ['basic', 'basic', 'basic', 'const_delay']),
    'ts_stddev_selq': (ts_stddev_selq, ['basic', 'basic', 'const_delay', 'const_quan', 'const_quan']),
    'ts_skew_selq': (ts_skew_selq, ['basic', 'basic', 'const_delay', 'const_quan', 'const_quan']),
    'ts_kurt_selq': (ts_kurt_selq, ['basic', 'basic', 'const_delay', 'const_quan', 'const_quan']),
    'ts_mean_selq': (ts_mean_selq, ['basic', 'basic', 'const_delay', 'const_quan', 'const_quan']),
    'ts_median_selq': (ts_median_selq, ['basic', 'basic', 'const_delay', 'const_quan', 'const_quan']),
    'ts_sumprod_selq': (ts_sumprod_selq, ['basic', 'basic', 'basic', 'const_delay', 'const_quan', 'const_quan']),
    'ts_cond_quantile_mean': (ts_cond_quantile_mean, ['basic', 'basic', 'const_delay', 'const_quan', 'const_quan']),
    'ts_cond_quantile_stddev': (ts_cond_quantile_stddev, ['basic', 'basic', 'const_delay', 'const_quan', 'const_quan']),
    'ts_cond_quantile_skew': (ts_cond_quantile_skew, ['basic', 'basic', 'const_delay', 'const_quan', 'const_quan']),
    'ts_cond_quantile_kurt': (ts_cond_quantile_kurt, ['basic', 'basic', 'const_delay', 'const_quan', 'const_quan']),
    'ts_obp': (ts_obp, ['basic', 'basic', 'const_delay']),
    'ts_obn': (ts_obn, ['basic', 'basic', 'const_delay']),
    'ts_obpm': (ts_obpm, ['basic', 'basic', 'const_delay']),
    'ts_obnm': (ts_obnm, ['basic', 'basic', 'const_delay']),

    'ts_quantile': (ts_quantile, ['basic', 'const_quan', 'const_delay']),
    'ts_qua': (ts_qua, ['basic', 'const_quan', 'const_delay']),
    'ts_qua_selq': (ts_qua_selq, ['basic', 'basic',  'const_delay', 'const_quan', 'const_quan', 'const_quan']),
    'ts_return': (ts_return, ['basic', 'const_delay']),
    'ts_mean_return': (ts_mean_return, ['basic', 'const_delay']),
    'ts_spread': (ts_spread, ['basic', 'const_delay', 'const_quan', 'const_quan']),
    'ts_spread_selq': (ts_spread_selq, ['basic', 'basic', 'const_delay', 'const_quan', 'const_quan']),
    'ts_spreadw_selq': (ts_spreadw_selq, ['basic', 'basic', 'const_delay', 'const_quan', 'const_quan']),

    'ts_demean': (ts_demean, ['basic', 'const_delay']),
    'ts_zscore': (ts_zscore, ['basic', 'const_delay']),
    'ts_regbeta': (ts_regbeta, ['basic', 'basic', 'const_delay']),
    'ts_regres': (ts_regres, ['basic', 'basic', 'const_delay']),
    'ts_reg_r2': (ts_reg_r2, ['basic', 'basic', 'const_delay']),
    'ts_reg_tval': (ts_reg_tval, ['basic', 'basic', 'const_delay']),
    'ts_reg_betase': (ts_reg_betase, ['basic', 'basic', 'const_delay']),
    'ts_reg_resse': (ts_reg_resse, ['basic', 'basic', 'const_delay']),
    'ts_reg_rsquare': (ts_reg_rsquare, ['basic', 'basic', 'const_delay']),
    'ts_reg_tvalue': (ts_reg_tvalue, ['basic', 'basic', 'const_delay']),
    'ts_reg_Fvalue': (ts_reg_Fvalue, ['basic', 'basic', 'const_delay']),
    'ts_autocorr_reg': (ts_autocorr_reg, ['basic', 'const_delay', 'const_delay']),
    'ts_std_ratio':(ts_std_ratio, ['basic', 'const_delay']),
    'ts_mean_gt_pct': (ts_mean_gt_pct, ['basic', 'basic', 'const_quan', 'const_delay']),
    'ts_sum_gt_pct': (ts_sum_gt_pct, ['basic', 'basic', 'const_quan', 'const_delay']),
    'ts_stddev_gt_pct': (ts_stddev_gt_pct, ['basic', 'basic', 'const_quan', 'const_delay']),    
    # 'ts_rankregbeta': (ts_rankregbeta, ['basic', 'basic', 'const_delay']),
    # 'ts_rankregres': (ts_rankregres, ['basic', 'basic', 'const_delay']),
    'ts_ema': (ts_ema, ['basic', 'const_delay']),
    'ts_ewma': (ts_ewma, ['basic', 'const_delay', 'const_quan']),
    'ts_scale':(ts_scale,  ['basic','const_delay']),
    'ts_min_max_norm':(ts_min_max_norm,  ['basic','const_delay']),
    'ts_entropy':(ts_entropy,  ['basic','const_delay']),
    'ts_cross_entropy':(ts_cross_entropy,  ['basic', 'basic', 'const_delay']),
    'ts_kl_diverg':(ts_kl_diverg,  ['basic', 'basic', 'const_delay']),
    'ts_first_deriv':(ts_first_deriv,  ['basic', 'const_delay']),
    'ts_second_deriv':(ts_second_deriv,  ['basic', 'const_delay', 'const_delay']),
    'ts_partial_corr': (ts_partial_corr, ['basic', 'basic', 'basic', 'const_delay']),
    'ts_gmean': (ts_gmean, ['basic', 'const_delay']),
    'ts_hmean': (ts_hmean, ['basic', 'const_delay']),
    'ts_gmean_amean_diff': (ts_gmean_amean_diff, ['basic', 'const_delay']),
    'ts_hmean_amean_diff': (ts_hmean_amean_diff, ['basic', 'const_delay']),
    'ts_gmean_hmean_diff': (ts_gmean_hmean_diff, ['basic', 'const_delay']),
    'ts_max_change': (ts_max_change, ['basic', 'basic', 'const_delay']),
    'ts_sort_diffcumsum': (ts_sort_diffcumsum, ['basic', 'basic', 'const_delay']),
    'ts_sort_diff_tailsubhead': (ts_sort_diff_tailsubhead, ['basic', 'basic', 'const_delay']),
    'ts_poly2_trend': (ts_poly2_trend, ['basic', 'const_delay']),
    'ts_max_drawdown': (ts_max_drawdown, ['basic', 'const_delay']),
    'ts_all_volatility': (ts_all_volatility, ['basic', 'const_delay']),
    'ts_downside_volatility': (ts_downside_volatility, ['basic', 'const_delay']),
    'ts_upside_volatility': (ts_upside_volatility, ['basic', 'const_delay']),
    'ts_ewmv': (ts_ewmv, ['basic', 'const_delay']),
    'ts_ewmstd': (ts_ewmstd, ['basic', 'const_delay']),
    
    
    # 概率类
    'normal_log_prob': (normal_log_prob, ['basic']),
    'normal_cdf': (normal_cdf, ['basic']),

    # 截面算子
    'cs_rank': (cs_rank, ['basic']),
    'cs_orth': (cs_orth, ['basic', 'basic']),
    'cs_regpred': (cs_regpred, ['basic', 'basic']),
    'cs_leftright_range1': (cs_leftright_range1, ['basic']),
    'cs_regres': (cs_regres, ['basic', 'basic']),
    'cs_normalize': (cs_normalize, ['basic']),
    'cs_demean': (cs_demean, ['basic']),
    'cs_min': (cs_min, ['basic']),
    'cs_sum': (cs_sum, ['basic']),
    'cs_median': (cs_median, ['basic']),
    'cs_winsorize':(cs_winsorize, ['basic']),
    'cs_mean':(cs_mean, ['basic']),
    'cs_market':(cs_market, ['basic', 'basic']),
    'cs_abs_deviation': (cs_abs_deviation, ['basic']),
    'cs_mad_winsorize': (cs_mad_winsorize, ['basic']),
    'cs_group_rank_neutralize': (cs_group_rank_neutralize, ['basic', 'basic', 'const_delay']),
    'cs_salience': (cs_salience, ['basic']),
    'cs_marketcorr': (cs_marketcorr, ['basic', 'const_delay']),
    'cs_stddev':(cs_stddev, ['basic']),
    'cs_min_max_norm':(cs_min_max_norm, ['basic']),
    'cs_rank_mad': (cs_rank_mad, ['basic',]),
    'cs_rank_quantile': (cs_rank_quantile, ['basic']),
    'cs_group_demean': (cs_group_demean, ['basic', 'basic']),
    'cs_group_demean_2': (cs_group_demean_2, ['basic', 'basic']),
    'cs_group_norm': (cs_group_norm, ['basic', 'basic']),

    'VOI_B': (VOI_B, ['basic', 'basic']),
    'VOI_S': (VOI_S, ['basic', 'basic']),
    'taylor_log_diff': (taylor_log_diff, ['basic']),
    'taylor_exp_diff': (taylor_exp_diff, ['basic']),
    
    # 元素算子
    'boxcox': (boxcox, ['basic', 'const_quan']),
    'rs_Hurst': (rs_Hurst, ['basic']),
    'ts_rs_Hurst': (ts_rs_Hurst, ['basic', 'const_delay']),
    'compare_and_return': (compare_and_return, ['basic', 'basic', 'basic', 'basic', 'basic']),
    'cs_mean_distance': (cs_mean_distance, ['basic', 'basic']),
    
    # 'tanh': (tanh, ['basic']),
    # 'cs_highlowtag': (cs_highlowtag, ['basic']),
    'indicator': (indicator, ['basic']),
    # 'cs_min_max_normal': (cs_min_max_normal, ['basic']),
    # 'ts_arg_outline': (ts_arg_outline, ['basic', 'basic']),
    
    
    'arctan': (arctan, ['basic']),
    'erf': (erf, ['basic']),

    
    # 'ts_ridegeregression_pred': (ts_ridegeregression_pred, ['regression', 'ridegeregression']),
    # 'ts_ridegeregression_res': (ts_ridegeregression_res, ['regression', 'ridegeregression']),
    
    
    'is_zero': (is_zero, ['basic']),
    'is_nan': (is_nan, ['basic']),
}

function_set = set(func_map_dict.keys())

