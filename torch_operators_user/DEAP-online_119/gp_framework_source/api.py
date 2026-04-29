from .gp_config import *
import os, sys
import random

__RANDOM_STATE__ = 100
username = os.getlogin()
sys.path.append(rf'/home/{username}/data_share')
from .program_scond import *
import numpy as np, re
import FactorFramework.FactorFramework as ff
from sklearn.preprocessing import MinMaxScaler
from .mc_corr_estimator import *
import pickle
import matplotlib.pyplot as plt

# %% 基本配置文件
work_path = rf'/home/{username}/GP_Output/'
if os.path.exists(work_path) is False:
    os.makedirs(work_path)

# 相关性矩阵，需要提前计算好，放在共享目录中
# with open("/home/user61/MyWorkSpace/08_gp_v2/correlation_matrix_122_demo_127.pkl", 'rb') as f:
with open(f"/home/{username}/data_share/FactorFramework/GpFramework/gp_support/corr_matrix/correlation_matrix_119_demo_937.pkl", 'rb') as f:
    corr_matrix = pickle.load(f)

# 本轮使用算子
sample_keys = random.sample(func_map_dict.keys(), int(np.floor(len(func_map_dict) * 1.0)))
batch_function_set = {k: func_map_dict[k] for k in sample_keys}


# %% 重写关键的类
class FitEvaluation(BaseFitEvaluation):
    """
    继承FitEvaluation类重写适应度评估方法
    """

    def __init__(self, context=Context(), fit_eval_config=None):
        """
        初始化方法
        Args:
            context: 上下文管理
        """
        super().__init__(context, fit_eval_config)

    def cal_fitness(self, basic_metrics) -> float:
        """
        重写适应度函数
        Args:
            basic_metrics:

        Returns:

        """

        def calculate_score(df):
            """
               计算综合评分 `score`：
               1. 使用 ic、rank_ic 和 group1、group20、tail50、tail100、head50 等特征进行评分。
               2. 取 ic 和 rank_ic 的绝对值。
               3. 返回一个浮动的 score（去除标准化）。

               Args:
                 df (pd.DataFrame): 包含回测数据的 DataFrame。

               Returns:
                 float: 计算得到的 `score`。
               """
            # 仅筛选出 all_period_mean 行

            df_all_period_mean = df[df['period_name'] == 'all_period_mean']

            # 计算 ic 和 rank_ic 的绝对值
            abs_ic = abs(df_all_period_mean['ic'].iloc[0])
            abs_rank_ic = abs(df_all_period_mean['rank_ic'].iloc[0])

            # 选择 group1、group20、tail50、tail100、head50 等字段
            group_columns = ['group1', 'group20', 'tail50', 'tail100', 'head50']

            # 进行 Min-Max 标准化（将数据缩放到 [0, 100] 的范围）
            scaler = MinMaxScaler(feature_range=(0, 100))
            df_group_values = df_all_period_mean[group_columns].values

            # 标准化
            df_group_scaled = scaler.fit_transform(df_group_values)

            # 将标准化后的数据和 ic、rank_ic 加入到 `score` 计算中
            score = (abs_ic + abs_rank_ic + np.sum(df_group_scaled) / 100.0) / (
                    len(group_columns) + 2)  # 加上2代表ic和rank_ic的两项

            return score

        # 计算当前因子重复次数，和本轮次所有的公式进行比较
        sum_file_path = os.path.join(self.context.args.summary_path,
                                     "GP_Formula_Round_{}_{}.csv".format(self.context.random_state,
                                                                         self.context.suffix))
        repeat_indicator = 0
        # 检验是否已经有过检的因子
        if os.path.exists(sum_file_path):
            sum_file = pd.read_csv(sum_file_path).tail(100)
            existed_formulas = sum_file["formula"].to_list()

            for formula in existed_formulas:
                # 计算库内累计因子相关性
                repeat_indicator += abs(
                    cal_corr_estimate_field_related(basic_metrics["formula_exp"], formula, corr_matrix))  # 需要注意相关性绝对值
        else:
            pass

        gp_logger.info(f"当前因子累计相关性: {repeat_indicator:.3f}, 因子: {basic_metrics['formula_exp']}")

        return calculate_score(basic_metrics['label5']) - repeat_indicator * 0.2


# %% 训练gp
def fit_gp_formula(
        test_name: str = "dev",
        field_lst: list = None,
        overwrite: bool = False,
        # GP参数
        ic_thres: float = 1.5,  # Label 5 IC阈值
        rank_ic_thres: float = 1.5,  # Label 5 rank IC阈值
        gens: int = 32,
        populations: int = 64,
        min_depth: int = 2,
        max_depth: int = 16,
        p_crossover: float = 0.8,
        p_mutation: float = 0.2,
        init_depth: list = [2, 4],
):
    """
    GP训练
    Args:
        test_name: 本轮测试名称
        field_lst: 字段选用
        overwrite: 是否覆盖
        ic_thres: IC过检阈值（绝对值）
        rank_ic_thres: Rank IC过检阈值（绝对值）
        gens: 代数
        populations: 种群数量
        min_depth: 最小深度
        max_depth: 最大深度
        p_crossover: 交叉概率
        p_mutation: 变异概率
        init_depth: 长度范围

    Returns:

    """
    # 配置本轮参数
    config_args = GpConfig(
        work_path=work_path,
        suffix=test_name,
        # GP训练参数
        ic_thres=ic_thres,
        rank_ic_thres=rank_ic_thres,
        gens=gens,
        populations=populations,
        min_depth=min_depth,
        max_depth=max_depth,
        p_crossover=p_crossover,
        p_mutation=p_mutation,
        init_depth=init_depth
    )
    random_state = __RANDOM_STATE__
    if os.path.exists(config_args.summary_path):
        # 如果目录存在，查看用户是否覆写
        if overwrite:
            shutil.rmtree(config_args.summary_path)
        else:
            raise PermissionError("结果已经存在，如果需要覆盖，请手动指`overwrite`参数为True")
    os.makedirs(config_args.summary_path, exist_ok=True)

    # 为日志添加处理器
    file_handler = logging.FileHandler(os.path.join(config_args.summary_path, "GeneticProgramming.log"), mode='a',
                                       encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    gp_logger.addHandler(file_handler)

    # 上下文对象
    current_context = Context()
    current_context.random_state = random_state
    current_context.suffix = test_name
    current_context.args = config_args

    gp_logger.info("本轮选用字段: {}".format(",".join(field_lst)))

    # 定义训练器
    deap_gp = GPLearn(
        config_args, function_set=set(batch_function_set.keys()), func_map_dict=batch_function_set,
        random_state=random_state,
        factor_lst=field_lst, suffix=current_context.suffix,
        fit_valuation=FitEvaluation(current_context, fit_eval_config=config_args)
    )
    deap_gp.fit()


def standard_fields_evaluator(test_name: str = "dev", field_lst: list = None, overwrite: bool = False):
    """
    标准化测试函数
    Args:
        test_name: 本轮测试名称
        field_lst: 字段选用
        overwrite: 是否覆盖

    Returns:

    """
    # 固定化部分参数
    fit_gp_formula(
        test_name=test_name,
        field_lst=field_lst,
        overwrite=overwrite,
        ic_thres=1.5,
        rank_ic_thres=1.5,
        gens=32,
        populations=64,
        min_depth=2,
        max_depth=16,
        p_crossover=0.8,
        p_mutation=0.2,
        init_depth=[2, 4],
    )


# 获取GP统计量
def eva_gp_performance(test_name: str = "dev", plot_data: bool = False):
    """统计 指定轮次的GP模型性能

    Args:
        test_name (str, optional): 系列命名. Defaults to "dev".
    """
    # 配置本轮参数
    config_args = GpConfig(
        work_path=work_path,
        suffix=test_name,
    )

    # 统计所有的Formula
    formula_path = os.path.join(config_args.summary_path,
                                "GP_Formula_Round_{}_{}.csv".format(__RANDOM_STATE__, test_name))
    log_book_path = os.path.join(config_args.summary_path,
                                 "GP_log_Book_Round_{}_{}.csv".format(__RANDOM_STATE__, test_name))

    res = dict()
    # 1. 处理公式文件
    if not os.path.exists(formula_path):
        # 如果没有找到公式文件，抛出异常
        pprint(f"本轮没有因子检出")
    else:
        formula_df = pd.read_csv(formula_path)
        # 处理公式数据
        formula_df = formula_df.drop_duplicates(subset=["formula_hash"])
        formula_df['abs_ic_label5'] = formula_df['ic_label5'].abs()
        formula_df = formula_df.sort_values(by="abs_ic_label5", ascending=False).reset_index(drop=True)

        # 统计量
        stat_df = formula_df['abs_ic_label5'].describe().to_frame().T
        res['formula_df'] = formula_df
        print(stat_df)

    # 2. 处理日志数据
    log_book_df = pd.read_csv(log_book_path)
    res['log_book_df'] = log_book_df
    if plot_data:
        fig, axs = plt.subplots(1, 2, figsize=(20, 5), dpi=150)
        axs[0].plot(log_book_df["gen"], log_book_df["Avg_Fitness"], label="Avg_Fitness", color='blue', marker='o')
        axs[0].plot(log_book_df["gen"], log_book_df["Best_Ind_Fitness"], label="Best_Ind_Fitness", color='red',
                    marker='^')
        axs[0].set_xlabel("Generation")
        axs[0].set_ylabel("Fitness")
        axs[0].set_title("GP Fitness Evolution")
        axs[0].legend()
        axs[0].grid(True)

        axs[1].plot(log_book_df["gen"], log_book_df["Avg_Length"], label="Avg_Length", color='blue', marker='o')
        axs[1].plot(log_book_df["gen"], log_book_df["Best_Ind_Length"], label="Best_Ind_Length", color='red',
                    marker='^')
        axs[1].set_xlabel("Generation")
        axs[1].set_ylabel("Length")
        axs[1].set_title("GP Length Evolution")
        axs[1].legend()
        axs[1].grid(True)

        # plt.tight_layout()
        plt.show()

    # 输出结果
    return res
