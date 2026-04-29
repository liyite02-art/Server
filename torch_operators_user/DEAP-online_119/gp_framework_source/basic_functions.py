# %%
import multiprocessing
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import datetime
import operator
from multiprocessing import Pool
import re
import sys, time
from tqdm import tqdm
import joblib
import random
import math
import torch
import warnings
from .gp_config import *  # 基本配置
torch.multiprocessing.set_sharing_strategy('file_system')
warnings.filterwarnings("ignore")
import traceback
from queue import Empty
import traceback


#%% 将输出重定向到日志文件
class StreamToLogger:
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self._buffer = ''

    def write(self, message):
        # 有时候print带换行，这里缓冲，保证逐行写日志
        message = message.strip()
        if message:
            self.logger.log(self.log_level, message)

    def flush(self):
        pass  # 兼容文件对象接口

#%% 超时处理
__TIME_OUT__ = 180  # 超时设置

class TimeoutException(Exception):
    """自定义超时异常"""
    pass


def timeout_handler(signum, frame):
    # 超时处理
    raise TimeoutException("计算超时")


# 定义一些常用的函数
def pprint(*args):
    # print with UTC+8 time
    time = '\n[' + str(datetime.datetime.utcnow() +
                       datetime.timedelta(hours=8))[:19] + '] -'
    print(time, *args, flush=True)


#%% 定义种群数据类 为后续做dataloader打好基础
class PopDataset(torch.utils.data.Dataset):
    '''
    多进程计算适应度
    '''

    def __init__(self, pop, evaluate_func) -> None:
        self.pop = pop
        self.evaluate_func = evaluate_func
        self.timeout = __TIME_OUT__ * 2 + 5 # 最大回测时间2min

    #TODO: 逻辑设计错误，只应该包括调用FF框架的时间，相关性检测时间不能包括
    def safe_evaluate(self, evaluate_func, arg, timeout=3):
        """防止运算超时"""
        def target(q, arg):
            try:
                result = evaluate_func(arg)
                q.put(("ok", result))
            except Exception as e:
                q.put(("err", e))
                
        q = multiprocessing.Queue()
        p = multiprocessing.Process(target=target, args=(q, arg))
        p.daemon = False
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

    def __getitem__(self, index):
        ind = self.pop[index]
        try:
            status, fitness = self.safe_evaluate(
                self.evaluate_func, ind, self.timeout)

            if status == "ok":
                # 正常运算
                return index, fitness
            elif status == "err":
                # 运算异常
                raise fitness
            elif status == "Timeout":
                # 运算超时
                raise TimeoutException("总评估计算超时")
            else:
                raise ValueError("Unknown status: " + str(status))
                
        except Exception as e:
            gp_logger.warning(f"公式: \"{str(ind)}\" ")
            gp_logger.warning("异常: " + str(e))
            return index, None

    def __len__(self):
        return len(self.pop)


class PopDataLoader():
    """自定义DataLoader
    不使手动使用守护进程方式加载数据，进行手动精确控制，从而能够在DataSet中实现超时控制
    """

    def __init__(self, dataset, num_workers=2, shuffle=False):
        """
        初始化一个支持多进程加载的自定义数据加载器。

        Args:
            dataset (Sequence): 支持索引访问的数据集，如 list、numpy 数组、或自定义 Dataset 类。
            num_workers (int, optional): 同时启动的数据加载进程数。默认为 2。
            shuffle (bool, optional): 是否对数据索引进行乱序处理。默认为 False。
            timeout (int, optional): 主进程等待数据时的超时时间（秒）。默认为 5。
        """
        self.dataset = dataset
        self.num_workers = num_workers
        self.shuffle = shuffle

        self.index_queue = multiprocessing.Queue()
        self.data_queue = multiprocessing.Queue()
        self.workers = []
        self._init_indices()
        self._start_workers()

    def _init_indices(self):
        """初始化索引队列
        """
        self.indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(self.indices)
        for idx in self.indices:
            self.index_queue.put(idx)

    def _worker_loop(self, dataset, index_queue, data_queue):
        """工作进程的主循环
        """
        while True:

            if index_queue.empty():
                # 如果索引队列为空，退出循环
                break
            try:
                idx = index_queue.get(timeout=1)
            except Exception as e:
                # 其他异常是严重问题，记录并退出
                err_msg = traceback.format_exc()
                data_queue.put(RuntimeError(
                    f"[Worker Error] Unexpected error in index_queue.get():\n{err_msg}"
                ))
                break

            try:
                sample = dataset[idx]
                data_queue.put(sample, timeout=3)
            except Exception:
                err_msg = traceback.format_exc()
                data_queue.put(RuntimeError(
                    f"[Worker Error] Failed at idx {idx}:\n{err_msg}"
                ))
                break

    def _start_workers(self):
        """启动工作进程
        """
        for _ in range(self.num_workers):
            p = multiprocessing.Process(target=self._worker_loop, args=(
                self.dataset, self.index_queue, self.data_queue
            ))
            p.daemon = False  # 设置为非守护进程，手动处理防止出现僵尸进程
            p.start()
            self.workers.append(p)

    def __iter__(self):
        return self

    def __next__(self):
        """获取下一个数据
        """
        time.sleep(5)
        # print([w.is_alive() for w in self.workers])
        if all(not w.is_alive() for w in self.workers) and self.data_queue.empty():
                # 正常结束
                raise StopIteration
            
        try:
            data = self.data_queue.get(timeout=__TIME_OUT__ * 2 + 5)
            if isinstance(data, Exception):
                gp_logger.warning(str(data))
            else:
                return data
        except Empty:
            self._shutdown()
            gp_logger.error("进程非正常结束...")
            raise StopIteration

    def _shutdown(self):
        """关闭所有工作进程
        """
        for w in self.workers:
            if w.is_alive():
                w.terminate()
                w.join()
        self.workers.clear()
        self.index_queue.close()
        self.data_queue.close()

    def __del__(self):
        """析构时全部关闭进程
        """
        self._shutdown()


def calc_formula_length(formula):
    '''识别公式长度: num('(')+num(',')  '''
    code_list = re.split('([\(,\)\s])', formula)
    return len([x for x in code_list if x in ['(', ',']])
