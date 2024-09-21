import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.pyplot import cm
import matplotlib.pylab as pl
from scipy.optimize import OptimizeResult
import re
from math import inf
from skopt import expected_minimum
import os
import matplotlib.ticker as ticker


"""
Return whether the target is to maximize or not the target function

Parameters
------------
target_func_name: The name of the target function

Return
------------
A boolean value of whether our target is to maximize the function
"""

def get_target(target_func_name):
    return (target_func_name != "Training Time")

"""
Function to plot the a grid estimation of the Optimizer

Parameters
------------
res: The result object of the Optimizer
target_func_name: The name of the target function
plot_dims: The names of the dimensions. Not necesseary if given when creating the space for the Optimizer
size: The size of the sub-figures
levels: The number of levels to plot on the contour plot
fixed: If true ranges of colorbar is fixed
ranges: Range of colorbar if fixed is True

Return
------------
Save the figure at source directory under the name results_{$target_func_name}.png
"""
def plot_results_grid(res,filename,target_func_name=None,plot_dims=None,size=6,levels=20,fixed=False,ranges=None):

    flag_indicator=False
    #Get space and dimensions
    space = res.space

    if plot_dims is None:
        plot_dims = []
        for row in range(space.n_dims):
            if space.dimensions[row].is_constant:
                continue
            plot_dims.append((row, space.dimensions[row]))
    else:
        plot_dims = space[plot_dims]


    target_maximize = get_target(target_func_name)

    n_dims = len(plot_dims)
    x_min = res.x
    samples = space.transform([x_min])
    model = res.models[-1]

    fig = plt.figure(figsize=(size * n_dims, size * n_dims))
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"

    ax = []
    g_lvl=5
    gamma_levels = list(np.linspace(0.0001,0.01,g_lvl))
    max_order = 5
    n_gamma_lvls = len(gamma_levels)

    for row in range(max_order):
        temp_row = []
        for col in range(n_gamma_lvls):
            temp_row.append(plt.subplot2grid((max_order, n_gamma_lvls), (row, col)))
        ax.append(temp_row)

    if fixed:
        if isinstance(ranges,list):
            vmin = ranges[0]
            vmax = ranges[1]
        else:
            raise Exception ("Ranges must be a list [min,max]")
    else:
        z = []
        for order in range(max_order):
            for gamma_level in range(n_gamma_lvls):
                xi, yi, zi = calculate_values(space,model,order+1,gamma_levels[gamma_level],samples,n_points=100)
                z.append(zi.flatten())

        z_final = np.asarray(z).flatten()
        vmin = np.min(z_final)
        vmax = np.max(z_final)
        print("Min is: {}".format(vmin))
        print("Max is: {}".format(vmax))

    if vmin==vmax:
        vmin=vmin-0.00000001
        flag_indicator = True

    cmap='viridis_r'
    if target_maximize:
        norm = plt.cm.colors.Normalize(vmin = -vmax, vmax = -vmin)
        sc = plt.cm.ScalarMappable(norm = norm, cmap =  cmap)
    else:
        norm = plt.cm.colors.Normalize(vmin = vmin, vmax = vmax)
        sc = plt.cm.ScalarMappable(norm = norm, cmap =  cmap)

    for order in range(max_order):
        for gamma_level in range(n_gamma_lvls):

            xi, yi, zi = calculate_values(space,model,order+1,gamma_levels[gamma_level],samples,n_points=100)

            if target_maximize:
                contourf_ = ax[max_order-1-order][gamma_level].contourf(xi, yi, -zi, levels = np.linspace(-vmax, -vmin, levels), cmap=cmap, vmin=-vmax, vmax=-vmin)
            else:
                contourf_ = ax[max_order-1-order][gamma_level].contourf(xi, yi, zi, levels = np.linspace(vmin, vmax, levels), cmap=cmap, vmin=vmin, vmax=vmax)

            ax[max_order-1-order][gamma_level].xaxis.set_major_locator(ticker.FixedLocator([0.0001,0.002575, 0.005050, 0.007525,0.01]))
            ax[max_order-1-order][gamma_level].yaxis.set_major_locator(ticker.FixedLocator([0.0,0.2,0.4,0.6,0.8,1.0]))
            ax[max_order-1-order][gamma_level].tick_params(axis='x',labelsize=30,labelrotation=90)
            ax[max_order-1-order][gamma_level].tick_params(axis='y',labelsize=30)

            if(order==max_order-1):
                ax[max_order-1-order][gamma_level].set_title('{}'.format("%.6f" % round(gamma_levels[gamma_level],6)),fontsize=30,weight='bold')
            if(order!=0 and gamma_level!=0):
                ax[max_order-1-order][gamma_level].set_xticks([])
                ax[max_order-1-order][gamma_level].set_yticks([])
            elif(order!=0 and gamma_level==0):
                ax[max_order-1-order][gamma_level].set_xticks([])
            elif(order==0 and gamma_level!=0):
                ax[max_order-1-order][gamma_level].set_yticks([])

        ax_temp = ax[max_order-1-order][gamma_level].twinx()
        ax_temp.set_ylabel('{}'.format(order+1),fontsize=30,weight='bold')
        ax_temp.set_yticks([])


    cax = plt.axes([0.95, 0.2, 0.015, 0.6])
    cbar = plt.colorbar(sc,cax=cax)
    cbar.ax.tick_params(labelsize=20)
    cbar.ax.set_ylabel('Score', rotation=270, fontsize=30)
    if target_func_name is not None:
        plt.suptitle("Metric: {}".format(target_func_name),fontsize=25)
    if flag_indicator:
        plt.suptitle("Metric: {}. Note Vmin==Vmax".format(target_func_name),fontsize=25)
    fig.text(0.5, -0.02, 'PMin', ha='center',fontsize=40)
    fig.text(0.06, 0.5, 'Confidence Threshold', va='center', rotation='vertical',fontsize=40)
    fig.text(0.5, 0.91, 'Gamma', ha='center',fontsize=40)
    fig.text(0.92, 0.5, 'Order', va='center', rotation='vertical',fontsize=40)
    plt.savefig(filename,bbox_inches='tight')
    plt.close()
    return

"""
Calculate the predicted value of the target function at sampled points

Parameters
------------
space: The parameter space
model: The surrogate model
order: Maximum order of the VMM
gamma: Gamma
samples: The sample points to use as base. Currently used for reference in creating a new point.
n_points: The number of points at which to evaluate the function along each dimension

Return
------------
pm: Array of sampled PMin values
con: Array of sampled Confidence Threshold values
zi: Array of the estimations
"""

def calculate_values(space, model, order, gamma, samples, n_points=40):

    def _calc(con, pm, ord_transformed,gamma_tranformed):
        rvs_ = np.array(samples)
        rvs_[:, 0] = ord_transformed
        rvs_[:, 1] = con
        rvs_[:, 2] = pm
        rvs_[:, 3] = gamma_tranformed
        return np.mean(model.predict(rvs_))

    ord_transformed = tranform_dim_point(space.dimensions[0],order)
    gamma_tranformed = tranform_dim_point(space.dimensions[3],gamma)
    pm, pm_transformed = sample_points(space.dimensions[2], n_points)
    con, con_transformed = sample_points(space.dimensions[1], n_points)

    zi = [[_calc(con, pm, ord_transformed,gamma_tranformed) for pm in pm_transformed] for con in con_transformed]

    zi = np.array(zi)

    return pm, con, zi

"""
Tranform a point of a given dimension

Parameters
-------------
dim: The dimension
point: The point to transform

Return
-------------
The transformed point
"""

def tranform_dim_point(dim,point):
    point_transformed = dim.transform(point)
    return point_transformed

"""
Sample points along the given dimension

Parameters
-----------
dim: The dimension
n_points: Number of samples

Return
----------
The sampled points
"""

def sample_points(dim,n_points):
    bounds = dim.bounds
    xi = np.linspace(bounds[0], bounds[1], n_points)
    xi_transformed = dim.transform(xi)
    return xi, xi_transformed


"""
Plot Convergence Plot

Parameters
-----------
dim: The optimization results (Single result or list of results)
names: Names of labels
mean: If true plot the mean of the best evaluated score in each iteration
fixed: If true ranges of colorbar is fixed
ranges: Range of colorbar if fixed is True
n_init: Number of initial points. If equal to zero, the initial points are included in the plot
n_calls: Number of total evaluations

Return
----------
Save the figure at source directory under the name conv_plot.png
"""

def plot_convergence(log_list,filename_to_save,n_init=8):

    n_calls=32
    n_of_informed = n_calls-n_init

    conv = []

    for log in log_list:
        conv_temp = []

        with open(log) as f:
            lines = f.readlines()

            for n in range(n_of_informed):
                estimated_score = eval(lines[8*n_calls+11+n])
                conv_temp.append(estimated_score)
                print(estimated_score)

        conv.append(conv_temp)

    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    ax = plt.gca()
    ax.set_xlabel("Number of evaluations $n$",fontsize=16)
    ax.set_ylabel(r"$\max Score $ after $n$ evaluations",fontsize=16)
    ax.grid()

    colors = cm.viridis(np.linspace(0.25, 1.0, 5))


    iterations = range(n_init+1, n_calls + 1)
    iterations_temp = range(1, n_calls+1)
    names = ["Seed_{}".format(i) for i in range(1,6)]

    for m,color,name in zip(conv,colors,names):
        ax.plot(iterations, m, c=color, alpha=1, label=name)

    ax.set_xlim([n_init+1,n_calls])
    ax.set_ylim([0.25,0.55])
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(filename_to_save)
    plt.close()
    return



"""
Function to plot the results of the exhaustive search

Parameters
------------
base_directory: Directory with the experiments of the exhaustive search
filename_to_save: Filename to save the figure
target_func_name: The name of the target function
size: The size of the sub-figures
conf_lv: number of sampled confidence level values
p_lvl: number of sampled Pmin values
g_lvl: number of sampled gamma values
fixed_colorbar: [vmin,vmax]. List containing the mininum and maximum value for the range of colors. Use to set the colormap the same with the grid of bayesian optimization

Return
------------
Save the figure at source directory under the name filename.png
"""

def plot_results_grid_exhaustive(base_directory,filename_to_save,target_func_name=None,size=6,conf_lv=11,p_lvl=5,g_lvl=5,n_dims=4,fixed_colorbar=None):

    max_order = 5
    max_score = -inf
    min_score = inf
    max_score_point = [-1,-1,-1,-1]
    min_score_point = [-1,-1,-1,-1]
    gamma_levels = list(np.linspace(0.0001,0.01,g_lvl))
    conf_levels = list(np.linspace(0.0,1.0,conf_lv))
    pmin_levels = list(np.linspace(0.0001,0.01,p_lvl))

    fig = plt.figure(figsize=(size * n_dims, size * n_dims))
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    ax = []

    for row in range(max_order):
        temp_row = []
        for col in range(g_lvl):
            temp_row.append(plt.subplot2grid((max_order, g_lvl), (row, col)))
        ax.append(temp_row)

    z = []
    for order in range(max_order):
        with open("{}/Order_{}/New_Order_{}.txt".format(base_directory,order+1,order+1), "r") as f:
            lines = f.readlines()

        total_points = [[order+1,conf,pmin,gamma] for conf in conf_levels for pmin in pmin_levels for gamma in gamma_levels]
        n_calls = len(total_points)

        indexes_points = [1+8*i for i in range(0,n_calls)]
        indexes_scores = [8*i for i in range(1,n_calls+1)]

        for index_p, index_s in zip(indexes_points,indexes_scores):
            z.append(eval(lines[index_s].split()[2]))
            temp_point = eval(lines[index_p])
            temp_score = eval(lines[index_s].split()[2])

            if temp_score>max_score:
                max_score = temp_score
                max_score_point = temp_point
            if temp_score<min_score:
                min_score = temp_score
                min_score_point = temp_point

    z_final = np.asarray(z)
    vmin = np.min(z_final)
    vmax = np.max(z_final)

    print("Max Score is: {}".format(vmax))
    print(max_score_point)
    print("Min Score is: {}".format(vmin))
    print(min_score_point)

    if fixed_colorbar!=None:
        vmin = fixed_colorbar[0]
        vmax = fixed_colorbar[1]

    cmap='viridis_r'
    norm = plt.cm.colors.Normalize(vmin = vmin, vmax = vmax)
    sc = plt.cm.ScalarMappable(norm = norm, cmap =  cmap)

    for order in range(max_order):

        points = []
        scores = []
        with open("{}/Order_{}/New_Order_{}.txt".format(base_directory,order+1,order+1), "r") as f:
            lines = f.readlines()

        total_points = [[order+1,conf,pmin,gamma] for conf in conf_levels for pmin in pmin_levels for gamma in gamma_levels]
        n_calls = len(total_points)

        indexes_points = [1+8*i for i in range(0,n_calls)]
        indexes_scores = [8*i for i in range(1,n_calls+1)]

        for index_p in indexes_points:
            points.append(eval(lines[index_p]))

        for index_s in indexes_scores:
            scores.append(eval(lines[index_s].split()[2]))

        for gamma_level in range(g_lvl):
            gamma = gamma_levels[gamma_level]

            scores_gamma = [scores[i] for i in range(n_calls) if points[i][3]==gamma]
            pmin = [points[i][2] for i in range(n_calls) if points[i][3]==gamma]
            confs = [points[i][1] for i in range(n_calls) if points[i][3]==gamma]

            ax[max_order-1-order][gamma_level].scatter(pmin, confs, c=scores_gamma, cmap=cmap, vmin=vmin, vmax=vmax, s=100)

            ax[max_order-1-order][gamma_level].xaxis.set_major_locator(ticker.FixedLocator([0.0001,0.002575, 0.005050, 0.007525,0.01]))
            ax[max_order-1-order][gamma_level].yaxis.set_major_locator(ticker.FixedLocator([0.0,0.2,0.4,0.6,0.8,1.0]))
            ax[max_order-1-order][gamma_level].tick_params(axis='x',labelsize=30,labelrotation=90)
            ax[max_order-1-order][gamma_level].tick_params(axis='y',labelsize=30)

            if(order==max_order-1):
                ax[max_order-1-order][gamma_level].set_title('{}'.format("%.6f" % round(gamma_levels[gamma_level],6)),fontsize=30,weight='bold')
            if(order!=0 and gamma_level!=0):
                ax[max_order-1-order][gamma_level].set_xticks([])
                ax[max_order-1-order][gamma_level].set_yticks([])
            elif(order!=0 and gamma_level==0):
                ax[max_order-1-order][gamma_level].set_xticks([])
            elif(order==0 and gamma_level!=0):
                ax[max_order-1-order][gamma_level].set_yticks([])

            ax[max_order-1-order][gamma_level].set_facecolor('black')

        ax_temp = ax[max_order-1-order][gamma_level].twinx()
        ax_temp.set_ylabel('{}'.format(order+1),fontsize=30,weight='bold')
        ax_temp.set_yticks([])

    cax = plt.axes([0.95, 0.2, 0.015, 0.6])
    cbar = plt.colorbar(sc,cax=cax)
    cbar.remove()
    fig.text(0.5, -0.02, 'PMin', ha='center',fontsize=40)
    fig.text(0.06, 0.5, 'Confidence Threshold', va='center', rotation='vertical',fontsize=40)
    fig.text(0.5, 0.91, 'Gamma', ha='center',fontsize=40)
    fig.text(0.92, 0.5, 'Order', va='center', rotation='vertical',fontsize=40)
    plt.savefig(os.path.join(base_directory,filename_to_save),bbox_inches='tight')
    plt.close()
    return


"""
Function to plot the improvement achieved by the optimizer over the random evaluations

Parameters
------------
log_list_bo: List containing the log files of the Optimizer for different seeds
log_list_rand: List containing the log files of the Random Evaluations for different seeds
filename_to_save: Filename to save the figure
n_calls: Number of total evaluations. Default 32
fixed: If true range of y axis is fixed
ranges: Range of y axis if fixed is True
title: Title for the figure. Default None.

Return
------------
Save the figure at source directory under the name filename.png
"""

def plot_results_bar_with_random(log_list_bo,log_list_rand,filename_to_save,n_calls=32,fixed=False,ranges=None,title=None):
    scores_bo = []
    scores_rand = []


    for log in log_list_bo:
        with open(log) as f:
            lines = f.readlines()
            score = eval(lines[8*n_calls+9].split()[3])
        scores_bo.append(score)

    for log in log_list_rand:
        with open(log) as f:
            lines = f.readlines()
            score = eval(lines[8*n_calls+8].split()[3])
        scores_rand.append(score)

    n_comparables = len(scores_bo)
    iterations = np.arange(1,n_comparables+1)

    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.xticks([r for r in iterations], names)
    ax = plt.gca()

    if title!=None:
        ax.set_title(title)

    ax.set_axisbelow(True)
    ax.yaxis.grid(color='gray', linestyle='dashed')

    width = 0.3

    colors = cm.viridis(np.linspace(0.25, 1.0, n_comparables))
    cls = [colors[i] for i in range(n_comparables)]

    improv = [((bo/rand)-1)*100 for rand,bo in zip(scores_rand,scores_bo)]
    ax.bar(iterations,improv,width,color=cls,edgecolor = "black")

    for i in range(n_comparables):
        plt.annotate("%.0f" % round(improv[i],0), xy=(iterations[i],improv[i]), ha='center', va='bottom')

    if fixed==True:
        ax.set_ylim(ranges)

    plt.ylabel("Improvement (%)",fontsize=16)
    plt.tight_layout()
    plt.savefig(filename_to_save)
    plt.close()

    return

"""
Function to plot the proxity of the results of the Optimizer to the best score achieved by the exhaustive search

Parameters
------------
log_list_bo: List containing the log files of the Optimizer for different seeds
log_list_rand: List containing the log files of the Random Evaluations for different seeds
filename_to_save: Filename to save the figure
names: Names of seeds
n_calls: Number of total evaluations. Default 32
conf_lv: Number of sampled confidence threshold levels
p_lvl: Number of sampled pMin levels
g_lvl: Number of sampled gamma levels
fixed: If true range of y axis is fixed
ranges: Range of y axis if fixed is True
title: Title for the figure. Default None.

Return
------------
Save the figure at source directory under the name filename.png
"""

def plot_proximity_bar(list_log_exhaustive,log_list_bo,filename_to_save,names,n_calls=32,conf_lv=11,p_lvl=5,g_lvl=5,fixed=False,ranges=None,title=None):

    n_comparables = 1 + len(log_list_bo)
    n_seeds = len(log_list_bo)
    max_order = 5
    best_score_exh = -inf
    best_score_bo = -inf
    best_score_bo_mcc = -inf
    best_score_bo_time = inf

    gamma_levels = list(np.linspace(0.0001,0.01,g_lvl))
    conf_levels = list(np.linspace(0.0,1.0,conf_lv))
    pmin_levels = list(np.linspace(0.0001,0.01,p_lvl))

    best_scores = []


    for log_ex in list_log_exhaustive:
        with open(log_ex) as f:
            lines = f.readlines()

            total_points = [[1,conf,pmin,gamma] for conf in conf_levels for pmin in pmin_levels for gamma in gamma_levels]
            n_points_grid = len(total_points)

            indexes_scores = [8*i for i in range(1,n_points_grid+1)]

            for index_s in indexes_scores:
                temp_score = eval(lines[index_s].split()[2])

                if temp_score>best_score_exh:
                    best_score_exh = temp_score

    best_scores.append(best_score_exh)

    print(best_score_exh)

    for log in log_list_bo:
        with open(log) as f:
            lines = f.readlines()
            score = eval(lines[8*n_calls+9].split()[3])
        best_scores.append(score)

    print(best_scores)

    iterations = np.arange(1,n_comparables+1)

    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"

    plt.xticks(rotation = 70)
    plt.xticks([r for r in iterations], names)

    ax = plt.gca()
    if title!=None:
        ax.set_title(title)

    ax.set_axisbelow(True)
    ax.yaxis.grid(color='gray', linestyle='dashed')

    width = 0.3

    c_exhaustive = ['m']
    colors = cm.viridis(np.linspace(0.25, 1.0, n_seeds))
    cls = [colors[i] for i in range(n_seeds)]
    cls= c_exhaustive+cls
    ax.bar(iterations, best_scores, width, color=cls, edgecolor='black')

    if fixed==True:
        ax.set_ylim(ranges)

    plt.ylabel('Score',fontsize=16)

    plt.tight_layout()
    plt.savefig(filename_to_save)
    plt.close()

    return

"""
Function to plot the highest estimated scores for each random seed for different versions of the optimizer

Parameters
------------
list_log_to_read: [[Log of 1st version of the Optimizer for 1st seed, Log of 1st version of the Optimizer for 2nd seed, ... ], [Log of 2st version of the Optimizer for 1st seed, ...], ...]
                    List containing lists with the log files of the different versions of the Optimizer.

filename_to_save: Filename to save the figure
title: Title for the figure. Default None.

Return
------------
Save the figure at source directory under the name filename.png
"""

def plot_comparison_bars(list_log_to_read,filename_to_save,title=None):

    n_seeds = len(list_log_to_read[0])
    n_comparables = len(list_log_to_read)

    index_score = 2
    results = []

    scores = []
    cnt=0

    n_calls = 32
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    init_points = [2,4,8,16]
    str_seeds = ["Seed_{}".format(i) for i in range(1,6)]
    str_init = [str(init) for init in init_points]
    estimated = []

    for log_list in list_log_to_read:
        estimated_temp = []

        for log in log_list:

            with open(log) as f:
                lines = f.readlines()

                estimated_score = eval(lines[8*n_calls+9].split()[3])

                print(estimated_score)

            estimated_temp.append(estimated_score)

        estimated.append(estimated_temp)

    iterations = np.arange(1,n_seeds+1)
    width = 1/(n_comparables+1)
    ax = plt.gca()

    ax.set_axisbelow(True)
    ax.yaxis.grid(color='gray', linestyle='dashed')

    hatches = [ "++" , "oo" , "//" , ".." , "+" , "x", "o", "O", ".", "*" ]
    colors = cm.viridis(np.linspace(0.25, 1.0, n_seeds))


    for i in range(n_comparables):
        ax.bar(iterations + (i-n_seeds//n_comparables)*width, estimated[i], width, edgecolor='black', color=colors, label=str_init[i], hatch=hatches[i])

    ax.set_ylabel("Score",fontsize=16)
    legend = ax.legend(title="Initial micro-benchmarks",loc="upper left",mode="expand",ncol=n_comparables,fontsize="small",edgecolor='black')

    handles = legend.legendHandles

    for i,handle in enumerate(handles):
        handle.set_facecolor('white')
        handle.set_hatch(hatches[i])

    ax.set_ylim([0.0,0.58])
    plt.xticks([r + width*np.median(np.arange(0,n_comparables)-1) for r in iterations], str_seeds)
    if title!=None:
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig(filename_to_save)
    plt.close()


    return
