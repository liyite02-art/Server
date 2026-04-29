"""
遗传规划主程序
"""
import operator
import traceback
import sys
import gc

import numpy as np

# sys.path.append("..")
import os
user = os.getlogin()
sys.path.append(f'/home/{user}/data_share/FactorFramework/GpFramework/gp_support/')
sys.path.append(f'/home/{user}/torch_operators_user')
from _deap import creator, base, tools, gp  # 遗传规划函数
from .basic_functions import *
from gp_functions import *
from .gp_config import *  # 基本配置
from .calc_method import *
import pickle as pkl
import re
import os
import random
import csv
from .calc_method import *


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# import multiprocessing
# lock = multiprocessing.Lock()
from tqdm import tqdm
import numpy as np

class GPLearn:
    """遗传算法核心类
    """

    def __init__(self, args,
                 function_set, func_map_dict,
                 random_state, factor_lst, suffix,
                 fit_valuation=BaseFitEvaluation,
                 **kwargs):

        self.args = args  # 全局参数
        self.function_set = function_set
        self.func_map_dict = func_map_dict

        np.random.seed(random_state)
        random.seed(random_state)
        self.random_state = random_state

        self.pset = None
        self.tool = None
        self.logbook = None
        self.best_inds = []
        self.best_ind = None
        self.factor_lst = factor_lst
        self.fit_valuation = fit_valuation
        self.suffix = suffix
        fit_valuation.context.random_state = random_state

    def _generate_pset(self, function_set, func_map_dict):
        """生成premitiveSet

        Args:
            function_set (_type_): 算子集合
            func_map_dict (_type_): 建立从算子到因子回测框架的映射

        Returns:
            _type_: premitiveSet实例对象
        """
        # 1. 输入因子集合
        factor_names = self.factor_lst
        # factor_names = [f"cs_normalize({i})" for i in factor_names]  # 注释时，使用原始数据进行遗传规划
        factor_input_nums = int(len(factor_names) * self.args.factor_input_ratio)  # 对因子选取一个子集进行规划
        pset = gp.My_PrimitiveSet(
            'main', factor_input_nums, input_type='basic')
        factor_nums_list = random.sample(
            range(len(factor_names)), factor_input_nums)
        for i, nums in enumerate(factor_nums_list):
            pset.renameArguments(**{f'ARG{i}': factor_names[nums]})  # 重命名参数类型
            # exec("pset.renameArguments(ARG%i='%s')" % (i, factor_names[nums]))  # 重命名参数类型，表达式与Cython不兼容

        # 2. 添加算子集合
        for symb in function_set:
            pset.addPrimitive(func_map_dict[symb][0], len(
                func_map_dict[symb][1]), args=func_map_dict[symb][1], name=symb)

        # 3. 常数集合
        pset.addConstTerminal(
            'rand_d', self.args.delay_list, 'const_delay', True)
        pset.addConstTerminal('rand_e', self.args.exp_list, 'const_exp', True)
        pset.addConstTerminal(
            'rand_q', self.args.quan_list, 'const_quan', True)

        return pset

    def _generate_fitness_individual(self):
        """定义适应度和个体，由于每个人关注的权重不一样，将适应度函数封装至FitEvaluation
        """
        # 创建fitness类、individual类
        creator.create('FitnessMax', base.Fitness, weights=(1.0,))
        creator.create('Individual', gp.PrimitiveTree, fitness=creator.FitnessMax)

    def _register_init_methods(self, tool, pset):
        """ 定义个体生成方法，种群生成方法 """
        init_method_dict = {
            'grow': gp.genGrow,
            'full': gp.genFull,
            'half and half': gp.genHalfAndHalf
        }
        if self.args.init_method not in init_method_dict:
            raise ValueError(f"Unsupported init method: {self.args.init_method}")

        tool.register('expr', init_method_dict[self.args.init_method], pset=pset,
                      min_=self.args.init_depth[0], max_=self.args.init_depth[1])
        tool.register('individual', tools.initIterate, creator.Individual, tool.expr)
        tool.register('population', tools.initUniqueRepeat, list, tool.individual)  # 确保生成的初始种群是没有重复的

    def _register_evaluate_select_methods(self, tool, metric):
        """ 定义evaluate、select """
        tool.register('evaluate', metric)  # 对个体进行fitness计算
        # tool.register('selectTournament', tools.selTournament, tournsize=self.args.tournament_size)  # 选择最优N个
        tool.register('selectBest', tools.selBest)  # 选择最优N个，默认按照适应度

    def _register_mate_mutate_methods(self, tool, pset):
        """ 定义 mate、mutate """
        tool.register('mate', gp.cxOnePoint)  # 单点交叉 会产生两棵树
        tool.register('expr_mut', gp.genFull, pset=pset, min_=0, max_=3)  # 生成一个subtree，且长度至少为1
        tool.register('mutUniform', gp.mutUniform, expr=tool.expr_mut, pset=pset)  # subtree mutation
        tool.register('mutInsert', gp.mutInsert, pset=pset)  # subtree mutation
        tool.register('mutNodeReplacement', gp.mutNodeReplacement, pset=pset)  # subtree mutation

        # 限制一下交叉变异后的树深度，最大12
        tool.decorate('mate', gp.staticLimit(key=operator.attrgetter('height'), max_value=self.args.max_depth))
        tool.decorate('mutUniform', gp.staticLimit(key=operator.attrgetter('height'), max_value=self.args.max_depth))
        tool.decorate('mutInsert', gp.staticLimit(key=operator.attrgetter('height'), max_value=self.args.max_depth))
        tool.decorate('mutNodeReplacement',
                      gp.staticLimit(key=operator.attrgetter('height'), max_value=self.args.max_depth))

    def _fit(self, tool):
        """
        进行种群迭代，计算适应度
        Args:
            tool:

        Returns:

        """
        popSize = self.args.populations
        ngen = self.args.gens
        cxpb = self.args.p_crossover
        mutpb = self.args.p_mutation

        # 生成初始种群
        pop = tool.population(n=popSize)

        # 先去除黑名单内样本
        black_list_path = os.path.join(
            self.args.gp_global_factor_black_list_path + "global_factor_back_list.txt"
        )
        if os.path.exists(black_list_path):
            with open(black_list_path, mode='r', encoding="utf-8") as f:
                black_list = f.read().split("\n")
        else:
            black_list = list()
        pop = [i for i in filter(lambda x: md5_32(str(x)) not in black_list, pop)]
        # 确保生成的样本中是互相不重复的
        attempts = 0
        seen_ = set(pop)
        while len(seen_) < popSize and attempts <= 50:

            len_before = len(seen_)

            new_pop = tool.population(n=1)  # 保证种群的数量不会发生大的变化

            # 检查样本是否位于黑名单内
            if str(new_pop[0]) in black_list:
                gp_logger.info("检测到黑名单内样本: 正在重新生成...")
                continue  # 重新生成

            seen_.update(new_pop)
            len_after = len(seen_)

            if len_after > len_before:
                attempts = 0
            else:
                attempts += 1

        pop = list(seen_)
        if len(pop) < popSize:
            gp_logger.warning(f"Only {len(seen_)} unique individuals generated after {attempts} attempts (target={popSize}).")
            
        # 进行拷贝
        pop = list(map(tool.clone, pop))

        # 开始迭代
        gen_count = -1  # 记录当前代数
        log_book = tools.Logbook()  # 迭代日志记录
        log_book.header = log_book.header = [
            "gen",  # 当前代数
            "eval",  # 当前评估的个体数
            "Avg_Length",  # 平均表达式长度
            "Avg_Fitness",  # 平均适应度
            "Best_Ind_Length",  # 最佳个体的表达式长度
            "Best_Ind_Fitness",  # 最佳个体的适应度
            "Best_Ind_Formula"  # 最佳个体的公式
        ]

        # 用于父子竞争机制的HOF，同时用于确保每一代都存在有效个体
        hof_save = tools.HallOfFame(int(np.ceil(popSize * 1 / 4)))
        # 开始迭代
        pprint('......开始迭代......')
        while gen_count < ngen:
            gen_count += 1
            pprint('第', gen_count, '次迭代...')

            # 更新
            pds = PopDataset(pop, tool.evaluate)
            dataloader = PopDataLoader(dataset=pds, num_workers=self.args.num_workers, shuffle=False)
            # dataloader = DataLoader(pds, collate_fn=lambda x: x[0], batch_size=1, num_workers=self.args.num_workers,
            #                         shuffle=False, drop_last=False)

            # 计算当前每个个体的适应度
            valid_pop = list()
            for batch in tqdm(dataloader, total=len(pds)):
                i = batch[0]  # 个体索引
                if batch[1] is not None and not np.isnan(batch[1]['fitness'][0]):
                    pop[i].basic_metrics = batch[1]['basic_metrics']
                    pop[i].fitness.values = batch[1]['fitness']  # 个体适应度
                    pop[i].indicators = batch[1]['indicators']  # 个体各项指标(IC, Rank IC...)
            
                    gp_logger.info(
                        "Factor Hash: {} Factor IC Label 5: {} \tFactor IC Label 5: {}".format(
                            md5_32(str(pop[i])),
                            batch[1]['indicators']['ic_label5'],
                            batch[1]['indicators']['rank_ic_label5']
                        )
                    )

                    valid_pop.append(tool.clone(pop[i]))
                else:
                    # 维护一个表达式黑名单
                    black_list_path = os.path.join(
                        self.args.gp_global_factor_black_list_path + "global_factor_back_list.txt"
                    )
                    with open(black_list_path, mode='a', encoding="utf-8") as f:
                        f.write(md5_32(str(pop[i])) + "\n")

            # 去除无效的个体
            pop = list(map(tool.clone, valid_pop))

            # 最佳个体集合 TODO: 没有必要只保留最佳结果，全部进行检查
            hof = tools.HallOfFame(int(popSize * 1 / 2))  # 如果有效数量不足1/2全部放入
            hof.clear()
            hof.update(pop)
            best_ind = hof.items[0]

            # 跨越代际更新最优
            hof_save.update(pop)
            
            gp_logger.debug(f"""
                            当前总体Pop数量: {len(pds)}, 
                            当前有效个体数量: {len(pop)}, 
                            当前Hof数量: {len(hof)}, 
                            当前Hof_save数量: {len(hof_save)} 
                            """)

            if len(pop) == 0:  # 防止出现有效个体不为零
                # 从加入Hof中的元素，即引入父子竞争机制
                pop = pop + [i for i in hof_save]
                pop = list(map(tool.clone, set(pop)))  # 去重
                gp_logger.error("当前种群没有有效个体，恢复hof_save中个体...")
                continue
                
            # 从hof中筛选过检因子
            for ind in hof:
                this_indicators = ind.indicators
                # pprint("Hof筛选检查，当前IC: ", this_indicators)
                if (
                        abs(this_indicators['ic_label5']) >= self.args.ic_thres
                ) and (
                        this_indicators['ic_label5'] * this_indicators['rank_ic_label5'] > 0
                ) and (
                        abs(this_indicators['rank_ic_label5']) > self.args.rank_ic_thres
                ):
                    # 使用IC作为过检因子标准
                    # with lock:
                    # print(this_indicators)
                    self.save_formula(ind, gen_count)

            # 本轮迭代性能评估
            record = {
                'Avg_Length': bn.nanmean([p.basic_metrics['formula_len'] for p in pop]),
                'Avg_Fitness': bn.nanmean([p.fitness.values for p in pop]),
                'Best_Ind_Length': best_ind.basic_metrics['formula_len'],
                'Best_Ind_Fitness': best_ind.fitness.values[0],
                'Best_Ind_Formula': str(best_ind)
            }
            log_book.record(gen=gen_count, eval=popSize, **record)

            # 生成下一代
            off_spring = tool.selectBest(pop, round(popSize * 1 / 4))  # 添加随机性

            # 全局因子黑名单
            black_list_path = os.path.join(
                self.args.gp_global_factor_black_list_path + "global_factor_back_list.txt"
            )
            if os.path.exists(black_list_path):
                with open(black_list_path, mode='r', encoding="utf-8") as f:
                    black_list = f.read().split("\n")
            else:
                black_list = list()

            # 确保生成的样本中是互相不重复的
            attempts = 0
            seen_ = set(off_spring)
            while len(seen_) < popSize and attempts <= 50:

                len_before = len(seen_)

                new_pop = tool.population(n=1)  # 保证种群的数量不会发生大的变化

                # 检查样本是否位于黑名单内
                if md5_32(str(new_pop[0])) in black_list:
                    gp_logger.warning("检测到黑名单内样本: 正在重新生成...")
                    continue  # 重新生成

                seen_.update(new_pop)
                len_after = len(seen_)

                if len_after > len_before:
                    attempts = 0
                else:
                    attempts += 1

            off_spring = list(seen_)
            if len(off_spring) < popSize:
                gp_logger.warning(f"Only {len(seen_)} unique individuals generated after {attempts} attempts (target={popSize}).")
            # 进行拷贝，进入下一轮迭代
            pop = list(map(tool.clone, off_spring))

            # 交叉操作
            random.shuffle(pop)  # 进行打乱
            for child1, child2 in zip(pop[1::2], pop[::2]):
                if np.random.random() < cxpb:
                    tool.mate(child1, child2)

            # 变异概率
            for child in pop:  # 每个个体都有概率进行变异
                if np.random.random() < mutpb:
                    # TODO: 变异方法可以自定义
                    mutate = tool.mutUniform
                    mutate(child)

            # 从加入Hof中的元素，即引入父子竞争机制
            pop = pop + [i for i in hof_save]
            pop = list(map(tool.clone, set(pop)))  # 去重
            
            # 迭代结束
            pprint("......迭代结束......")

            # 输出log book
            if len(log_book) > 0:
                print(log_book)
                # 将LookBook写入到文件
                log_book_path = os.path.join(self.args.summary_path,
                                             "GP_log_Book_Round_{}_{}.csv".format(self.random_state, self.suffix))
                with open(log_book_path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(log_book.header))
                    writer.writeheader()
                    for record in log_book:
                        writer.writerow(record)
            else:
                pprint("没有找到过检因子")

    def fit(self):
        """
        迭代训练
        Returns:

        """
        # 1. 生成 primitive Set
        self.pset = self._generate_pset(self.function_set, self.func_map_dict)
        self.tool = base.Toolbox()

        # 2. Fitness
        self._generate_fitness_individual()

        # 3. 定义个体（Individual）、总体（Population）生成方式
        self._register_init_methods(self.tool, self.pset)

        # 4. 定义进化（Evaluate）、选择（Select）方式
        self._register_evaluate_select_methods(self.tool, self.fit_valuation.fit_evaluation)

        # 5. 定义交叉（Mate）、变异（Mutate）的方式
        self._register_mate_mutate_methods(self.tool, self.pset)

        # 6.  迭代训练
        self._fit(self.tool)

    def save_formula(self, ind, gen_num):
        """
        对公式进行保存
        Args:
            ind: 个体
            gen_num: 当前代数

        Returns: None

        """

        summary_file = os.path.join(self.args.summary_path, "GP_Formula_Round_{}_{}.csv".format(self.random_state, self.suffix))
        ind_indicators = ind.indicators  # 获取当前个体的indicator字典

        summary = pd.DataFrame({
            'round': [self.random_state],
            'gen': [gen_num],
            'formula': [str(ind)],
            'formula_hash': [md5_32(str(ind))],
            'ic_label5': ind_indicators['ic_label5'],
            'rank_ic_label5': ind_indicators['rank_ic_label5'],
            'head50_label5': ind_indicators['head50_label5'],
            'head100_label5': ind_indicators['head100_label5'],
            'tail50_label5': ind_indicators['tail50_label5'],
            'tail100_label5': ind_indicators['tail100_label5'],
        })

        if os.path.isfile(summary_file):
            summary.to_csv(summary_file, index=False, header=False, mode='a+')
        else:
            summary.to_csv(summary_file, index=False, header=True, mode='a+')
