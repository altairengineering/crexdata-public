from plot_utils import *
import pickle
import matplotlib.pyplot as plt
from math import inf


"""
Plot Results

This example plot the results for the versions of the Optimizer with different number of initial micro-benchmarks,
the improvement of the Optimizer over the random evaluations and the comparison between the Optimizer and the exhaustive search.

"""

"""
Base directory for the results of the experiments of the Optimizer
"""

base_directory = ""

"""
experiment_folders contains the experiment folders for the results of the different Optimizer
"""
experiment_folders = ["2","4","8","16"]

"""
Number of initial micro_benchmarks corresponding to the experiments of the different versions of the Optimizer. Must follow the same order as the versions in experiment_folders
"""
n_inits = [2,4,8,16]
random_seeds = [1234,1280,1300,1400,1500]
target_func = "comb"
n_calls = 32
res_list = []
str_seeds = ["Seed_{}".format(i) for i in range(1,6)]

"""
Best version of the Optimizer
"""
best_version = experiment_folders[2]

log_log_list = []
log_log_list_bo = []

cnt = 0
for exp_fold in experiment_folders:
    res_temp = []
    logs_list = []
    for seed in random_seeds:
        print("{}/{}/{}_macro/{}_opt_{}.pkl".format(base_directory,exp_fold,seed,target_func,seed))
        with open("{}/{}/{}_macro/{}_opt_{}.pkl".format(base_directory,exp_fold,seed,target_func,seed), "rb") as f:
            opt_restored = pickle.load(f)
            res=opt_restored.get_result()
            res_temp.append(res)
            logs_list.append("{}/{}/{}_macro/{}_{}.txt".format(base_directory,exp_fold,seed,target_func,seed))
            print("restored: "+"{}/{}/{}_macro/{}_opt_{}.pkl".format(base_directory,exp_fold,seed,target_func,seed))
            plot_results_grid(res,"{}/{}/{}_macro/bo_grid.png".format(base_directory,exp_fold,seed),levels=100)
    if exp_fold==best_version:
        log_list_best_bo = logs_list

    log_log_list_bo.append(logs_list)
    res_list.append(res_temp)

    plot_convergence(logs_list,"{}/{}/converge_focused.png".format(base_directory,exp_fold),n_init=n_inits[cnt])
    cnt+=1
plot_comparison_bars(log_log_list_bo,"{}/comparison_bar.png".format(base_directory))
plot_best_score_initial(log_log_list_bo,"{}/comparison_init_bar.png".format(base_directory))



"""
Base directory for the results of random evaluations
"""
base_random = ""

log_list_rand = []

for seed in random_seeds:
    logs_list.append("{}/{}_macro/{}_{}.txt".format(base_random,seed,target_func,seed))
    print("restored: "+"{}/{}_macro/{}_{}.txt".format(base_random,seed,target_func,seed))
    log_list_rand.append(logs_list)

plot_results_bar_with_random(log_list_best_bo,log_list_rand,"random_bo_best_new.png",str_seeds,["Random","Optimizer_16"],fixed=True,ranges=[0.4,0.6])

"""
Exhaustive Search Plot
"""
base_directory_exh = ""

log_exh_list = []
for order in range(1,6):
    log_exh_list.append("{}/Order_{}/Order_{}.txt".format(base_directory_exh,order,order))
    print("restore: {}/Order_{}/Order_{}.txt".format(base_directory_exh,order,order))

names_exh = ["Exhaustive \n Search"] + ["Optimizer_{}\nSeed{}".format(best_init,i) for i in range(1,6)]
plot_proximity_bar(log_exh_list,log_list_best_bo,"exhaustive_bo.png",names_exh)
plot_results_grid_exhaustive(base_directory_exh,"exhaustive_search_black_new.png")
