import warnings
import os
import sys
warnings.filterwarnings("ignore")
username = os.getlogin()
sys.path.append(rf'/home/{username}/data_share/FactorFramework/GpFramework')

from gp_framework.api import fit_gp_formula, eva_gp_performance, standard_fields_evaluator

if __name__ == "__main__":
    # Evaluate GP performance
    with open("/home/user61/MyWorkSpace/08_gp_v2/DEAP/valid_field_list_119.txt", "r") as f:
        field_lst = f.read().split("\n")

    fit_gp_formula(
        test_name = "dev_0606",
        field_lst = field_lst,
        overwrite = True,
        # GP参数
        ic_thres = 2.0, 
        rank_ic_thres = 2.0, 
        gens = 128,
        populations = 64,
        min_depth = 2,
        max_depth = 16,
        p_crossover = 0.8,
        p_mutation = 0.2,
        init_depth = [2, 4],
    )

    
