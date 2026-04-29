import os
import sys
import logging

user_name = os.getlogin()
from argparse import ArgumentParser
import numpy as np
import shutil


class GpConfig:
    """
    配置类
    """

    def __init__(
            self,
            # 抽样及训练相关
            factor_input_ratio=1.0,
            num_workers=14,

            # 过检阈值相关
            ic_thres=2.0,  # Label 5 IC阈值
            rank_ic_thres=2.0,  # Label 5 rank IC阈值

            # Gplearn参数
            gens=128,
            populations=128,

            # 终端参数
            delay_list=[1, 3, 5, 15],
            quan_list=[0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95],
            exp_list=[-3, -2, -1, -0.5, 0.5, 2, 3],

            # 长度限制
            min_depth=2,
            max_depth=16,

            # 交叉概率
            p_crossover=0.8,
            p_mutation=0.2,

            # 初始化
            init_depth=[2, 4],
            init_method='half and half',

            # 路径配置
            work_path=rf'/home/{user_name}/MyWorkSpace/08_gp_v2/',
            suffix=None,
    ):
        """配置基本的参数
        """
        self.factor_input_ratio = factor_input_ratio
        self.num_workers = num_workers
        self.ic_thres = ic_thres
        self.rank_ic_thres = rank_ic_thres
        self.gens = gens
        self.populations = populations
        self.delay_list = delay_list
        self.quan_list = quan_list
        self.exp_list = exp_list
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.p_crossover = p_crossover
        self.p_mutation = p_mutation
        self.init_depth = init_depth
        self.work_path = work_path
        self.init_method = init_method
        self.suffix = suffix

        # 衍生参数
        self.gp_result_path = rf'{self.work_path}/gp_result'
        self.summary_path = self.gp_result_path + r'/LeaderBoard_%s/' % self.__get_file_name()
        self.gp_global_factor_black_list_path = self.summary_path

    def __get_file_name(self):
        return '-'.join(filter(None, [  # Remove empty string by filtering
            f'{self.suffix}',
        ])).replace(' ', '')  # 通过suffix作为结果文件的唯一索引


# %% 配置logging日志输出模块
gp_logger = logging.getLogger('GeneticProgramming')
gp_logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
