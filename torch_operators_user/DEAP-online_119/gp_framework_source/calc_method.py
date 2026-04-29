# %%
import operator
import traceback
import sys
import os
user = os.getlogin()
sys.path.append(f'/home/{user}/data_share/FactorFramework/GpFramework/gp_support/')
sys.path.append(f'/home/{user}/torch_operators_user')
import gc
from _deap import creator, base, tools, gp 
from .basic_functions import *
from .basic_functions import __TIME_OUT__
from gp_functions import *
import pickle as pkl
import re

import warnings
import os
import torch
import re
from io import StringIO
import contextlib
import pandas as pd
from abc import ABC, abstractmethod
import hashlib, sys
import time

warnings.filterwarnings("ignore")
username = os.getlogin()
sys.path.append(rf'/home/{username}/data_share')
import multiprocessing

import FactorFramework.FactorFramework as ff

ff.set_gpu_device('cuda:0')


class Context:
    """
    上下文管理对象
    """
    pass


def md5_32(s: str) -> str:
    """
    返回字符串 s 的 32 位 MD5 哈希值（小写十六进制）

    Args:
        s (str): 输入字符串

    Returns:
        str: 32位MD5值（十六进制字符串）
    """
    return hashlib.md5(s.encode('utf-8')).hexdigest()


def capture_backtest_results(ff_obj, factor_name):
    """捕获回测结果的标准输出并解析为结构化DataFrame"""
    # 使用缓冲区捕获控制台输出
    with StringIO() as buf, contextlib.redirect_stdout(buf):
        ff_obj.get_all_backtest_result(
            fac_name=factor_name,
            label_name_list=['label4', 'alpha4', 'label5'],
            print_tabel=True,  # 注意参数名可能是拼写错误
            print_figure=False
        )
        raw_output = buf.getvalue()

    # 使用正则表达式清理输出
    cleaned_output = re.sub(r'\x1b\[.*?m', '', raw_output)  # 去除ANSI颜色代码
    cleaned_output = re.sub(r'[\r]+', '', cleaned_output)  # 去除回车符

    # 解析结构化数据
    def parse_block(block):
        metadata = {
            'fac_name': re.search(r'fac_name: (\w+)', block).group(1),
            'label_name': re.search(r'label_name: (\w+)', block).group(1)
        }

        # 提取并处理表格数据
        table_match = re.search(r'(period_name.*?)(?=fac_name|\Z)', block, re.DOTALL)
        if not table_match:
            return None

        df = pd.read_csv(
            StringIO(table_match.group(1).strip()),
            sep='\s+',
            engine='python'
        )
        # 处理列名
        df.columns = df.columns.str.strip()
        # 合并元数据
        return df.assign(**metadata)

    # 分割不同数据块
    blocks = re.split(r'(?=fac_name:)', cleaned_output)
    results = [parse_block(b) for b in blocks if b.strip()]

    return [df for df in results if df is not None]


class BaseFitEvaluation(ABC):
    """评估个体适应度基类

    Returns:
        _type_: _description_
    """

    def __init__(self, context=Context(), fit_eval_config=None):
        self.fit_eval_config = fit_eval_config  # 将GpConfig写在基类，方便后续调用
        self.context = context

    def get_basic_metrics(self, ind):
        """调用因子框架，

        Args:
            ind (_type_): _description_

        Returns:
            _type_: _description_
        """
        def md5_32(s: str) -> str:
            """
            返回字符串 s 的 32 位 MD5 哈希值（小写十六进制）

            Args:
                s (str): 输入字符串

            Returns:
                str: 32位MD5值（十六进制字符串）
            """
            return hashlib.md5(s.encode('utf-8')).hexdigest()
        fac_exp = str(ind)  # 获得字符串表达式
        try:
            # 如果已经测试过因子就直接读取
            backtest_results = capture_backtest_results(ff, f'test_{md5_32(fac_exp)}')
        except Exception as e:
            original_stderr = sys.stderr
            original_stdout = sys.stdout
            
            sys.stderr = StreamToLogger(gp_logger, logging.ERROR)
            sys.stdout = StreamToLogger(gp_logger, logging.INFO)
            # 将异常信息输出到日志 
            ff.fac_backtest(fac_exp=fac_exp, fac_name=f'test_{md5_32(fac_exp)}',
                            calc_mode='daily', intraday_data_only=True, check_fac_name_duplicate=False,
                            save_fac_data=False, just_return_backtest_result=False)
            # 恢复标准输出
            sys.stderr = original_stderr
            sys.stdout = original_stdout
            
            backtest_results = capture_backtest_results(ff, f'test_{md5_32(fac_exp)}')

        basic_metrics = {
            'fac_exp': fac_exp,
            'label4': backtest_results[0],
            'alpha4': backtest_results[1],
            'label5': backtest_results[2],
            'formula_len': calc_formula_length(fac_exp),
            'formula_exp': fac_exp
        }

        return basic_metrics
    
    def safe_get_basic_metrics(self, ind, timeout=3):
        """防止运算超时

        Args:
            evaluate_func (_type_): _description_
            arg (_type_): _description_
            timeout (int, optional): _description_. Defaults to 3.
        """
        def target(q, ind):
            try:
                basic_metrics = self.get_basic_metrics(ind)
                q.put(("ok", basic_metrics))
            except Exception as e:
                q.put(("err", e))

        q = multiprocessing.Queue()
        p = multiprocessing.Process(target=target, args=(q, ind))
        p.daemon = True
        p.start()
        p.join(timeout=timeout)
        
        if p.is_alive():
            p.terminate()
            p.join()
            return "Timeout", None

        if not q.empty():
            status, data = q.get()
            return status, data
        else:
            return "Empty", None  # 不太可能，但兜底

    def fit_evaluation(self, ind):
        """_summary_

        Args:
            ind (_type_): _description_

        Returns:
            dict: {'fitness': fitness, 'indicators': indicators, 'basic_metrics' : basic_metrics}
        """
        fac_exp = str(ind)  # 获得字符串表达式
        formula_length = calc_formula_length(fac_exp)
        if (formula_length < self.fit_eval_config.min_depth
                or formula_length > self.fit_eval_config.max_depth):
            """
            超过公式长度有效范围，直接返回
            """
            return None

        basic_metrics_start = time.time()
        # ------ 计算基本评估指标开始 ------ #
        status, basic_metrics = self.safe_get_basic_metrics(ind, timeout=__TIME_OUT__)
        # ------ 计算基本评估指标结束 ------ #
        basic_metrics_stop = time.time()
        gp_logger.debug(f"计算基本评估指标，用时: {basic_metrics_stop - basic_metrics_start:.3f} s")
        gp_logger.debug(f"基本评估指标计算状态: {status}")
        # 根据状态进行处理
        if status == 'err':
        # 运算异常
            raise basic_metrics
        elif status == "Timeout":
            # 运算超时
            raise TimeoutException("基本评估指标计算超时")

        # indicators 因子各项关键指标
        indicators = {
            'formula_len': basic_metrics['formula_len'],

            'ic_label5': basic_metrics['label5'].loc[
                basic_metrics['label5']['period_name'] == 'all_period_mean', 'ic'].mean(),

            'rank_ic_label5': basic_metrics['label5'].loc[
                basic_metrics['label5']['period_name'] == 'all_period_mean', 'rank_ic'].mean(),

            'head50_label5': basic_metrics['label5'].loc[
                basic_metrics['label5']['period_name'] == 'all_period_mean', 'head50'].mean(),

            'head100_label5': basic_metrics['label5'].loc[
                basic_metrics['label5']['period_name'] == 'all_period_mean', 'head100'].mean(),

            'tail50_label5': basic_metrics['label5'].loc[
                basic_metrics['label5']['period_name'] == 'all_period_mean', 'tail50'].mean(),

            'tail100_label5': basic_metrics['label5'].loc[
                basic_metrics['label5']['period_name'] == 'all_period_mean', 'tail100'].mean(),
        }
        
        comprehensive_metrics_start = time.time()
        # ------ 计算综合评估指标开始 ------ #
        # 整理返回值
        fitness_status, fitness_result = self.safe_cal_fitness(basic_metrics, timeout=__TIME_OUT__)
        # ------ 计算综合评估指标结束 ------ #
        comprehensive_metrics_stop = time.time()
        # 根据状态进行处理
        if fitness_status == 'err':
        # 运算异常
            raise fitness_result
        elif fitness_status == "Timeout":
            pprint("综合评估指标计算超时")
            # 运算超时
            raise TimeoutException("综合评估指标计算超时")
        res = {
            'fitness': (fitness_result,),  # 注意返回值为元组
            'indicators': indicators,
            'basic_metrics': basic_metrics
        }
        gp_logger.debug(f"计算综合评估指标，用时: {comprehensive_metrics_stop - comprehensive_metrics_start:.3f} s")
        gp_logger.debug(f"综合评估指标计算状态: {status}")

        return res

    @abstractmethod
    def cal_fitness(self, basic_metrics) -> float:
        """
        适应度函数未定义
        Args:
            basic_metrics:

        Returns:

        """
        raise AttributeError("适应度函数cal_fitness没有被定义")
    
    def safe_cal_fitness(self, basic_metrics, timeout=3):
        """防止适应度函数超时

        Args:
            basic_metrics (dict): 基本指标
            timeout (int, optional): 超时时间（秒）

        Returns:
            (status, result): status为'ok'/'err'/'Timeout'/'Empty'，result为结果或异常
        """
        def target(q, basic_metrics):
            try:
                result = self.cal_fitness(basic_metrics)
                q.put(("ok", result))
            except Exception as e:
                q.put(("err", e))

        q = multiprocessing.Queue()
        p = multiprocessing.Process(target=target, args=(q, basic_metrics))
        p.daemon = True
        p.start()
        p.join(timeout=timeout)

        if p.is_alive():
            p.terminate()
            p.join()
            return "Timeout", None

        if not q.empty():
            status, data = q.get()
            return status, data
        else:
            return "Empty", None  # 兜底

