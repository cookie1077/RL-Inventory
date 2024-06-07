import pandas as pd
import ray
import os.path

import numpy as np
import glob

from learning_extend import run_rl_plus
from agent_extend import Agent
from itertools import product
from utils import experiment_name


ray.init(num_cpus=8)
assert ray.is_initialized() == True



version = 5.1
experiment_num = 11

fine_tune = True

load_experiment_num = 10
load_mean = 5.0
load_std = 1.0


if not os.path.isdir(f'./experiment_{experiment_num}'):
    os.makedirs(f'./experiment_{experiment_num}')
    os.makedirs(f'./experiment_{experiment_num}/checkpoints_cost')
    os.makedirs(f'./experiment_{experiment_num}/checkpoints_policy')
    os.makedirs(f'./experiment_{experiment_num}/data')
    os.makedirs(f'./experiment_{experiment_num}/summary data')

final_experiment_num = 3
final_experiment_ids = []
df_final_experiment = pd.read_csv('./Final Experiment_trial.csv')
print(ray.is_initialized())
print(os.getcwd())

for index, row in df_final_experiment.iterrows():
    lead_time, mean, std, p, alpha, algorithm, x_actor_lr, x_critic_lr, x_tau =\
        row[['lead_time','mean', 'std', 'p', 'alpha', 'algorithm', 'x_actor_lr', 'x_critic_lr', 'x_tau']]
    
    lead_time, mean, std, p, alpha, x_actor_lr, x_critic_lr, x_tau =\
        int(lead_time), float(mean),float(std), float(p), float(alpha), float(x_actor_lr), float(x_critic_lr), float(x_tau)
    

    for step in range(final_experiment_num):
        csv_file_name = './experiment_' + str(experiment_num) + '/data/' + \
                        experiment_name(version=version, lead_time=lead_time, mean = mean, std=std, p=p, alpha=alpha,
                                        algorithm=algorithm, x_actor_lr=x_actor_lr, x_critic_lr=x_critic_lr,
                                        x_tau=x_tau,
                                        step=step) + '.csv'
        
        weight_path = None
        if fine_tune:
             glob_path = f'./experiment_{load_experiment_num}/checkpoints_cost/exp_id=0_Experiment_ver_.5.1(l=0, mean={load_mean}, std={load_std},*_(A={algorithm}, *)*.pth.tar'
             weight_path = glob.glob(glob_path)[-1]
             print(weight_path)
             print(experiment_name(version=version, lead_time=lead_time, mean = mean, std=std, p=p, alpha=alpha,
                                            algorithm=algorithm, x_actor_lr=x_actor_lr, x_critic_lr=x_critic_lr,
                                            x_tau=x_tau,
                                            step=step))

        if os.path.isfile(csv_file_name):
            print('exist')
            continue
           

        final_experiment_ids.append(run_rl_plus.remote(version=version, experiment_num=experiment_num,
                                                  lead_time=lead_time, mean = mean, std=std, p=p, alpha=alpha,
                                                  algorithm=algorithm, x_actor_lr=x_actor_lr,
                                                  x_critic_lr=x_critic_lr, x_tau=x_tau,
                                                  step=step,fine_tune = fine_tune, load_path = weight_path
                                                  ))

ray.get(final_experiment_ids)

final_experiment_results = [0] * (len(df_final_experiment) * final_experiment_num)
final_experiment_summary = [0] * len(df_final_experiment)
for index, row in df_final_experiment.iterrows():
    lead_time, mean, std, p, alpha, algorithm, x_actor_lr, x_critic_lr, x_tau = \
        row[['lead_time', 'mean', 'std', 'p', 'alpha', 'algorithm', 'x_actor_lr', 'x_critic_lr', 'x_tau']]
    
    lead_time, mean, std, p, alpha, x_actor_lr, x_critic_lr, x_tau =\
        int(lead_time), float(mean), float(std), float(p), float(alpha), float(x_actor_lr), float(x_critic_lr), float(x_tau)
    
    list_expected_cost = np.array([0.0] * final_experiment_num)
    list_expected_diff = np.array([0.0] * final_experiment_num)
    for step in range(final_experiment_num):
        csv_file_name = './experiment_' + str(experiment_num) + '/data/' + \
                        experiment_name(version=version, lead_time=lead_time, mean = mean, std=std, p=p, alpha=alpha,
                                        algorithm=algorithm, x_actor_lr=x_actor_lr, x_critic_lr=x_critic_lr,
                                        x_tau=x_tau,
                                        step=step) + '.csv'
        df = pd.read_csv(csv_file_name)
        list_expected_cost[step] = df.nsmallest(10, 'expected_cost')['expected_cost'].mean()
        list_expected_diff[step] = df.nsmallest(10, 'expected_cost')['expected_diff'].mean()
        final_experiment_results[final_experiment_num * index + step] =\
            [lead_time, mean, std, p, alpha, algorithm, x_actor_lr, x_critic_lr, x_tau, step, list_expected_cost[step], list_expected_diff[step]]

    expected_cost = np.mean(list_expected_cost)
    expected_diff = np.mean(list_expected_diff)
    final_experiment_summary[index] = [lead_time, mean, std, p, alpha, algorithm, x_actor_lr, x_critic_lr, x_tau,
                               expected_cost, expected_diff]

df_final_results = pd.DataFrame(final_experiment_results,
                                columns=['lead_time', 'mean','std', 'p', 'alpha',
                                         'algorithm', 'x_actor_lr', 'x_critic_lr', 'x_tau', 'step',
                                         'expected_cost', 'expected_diff'])

df_final_summary = pd.DataFrame(final_experiment_summary,
                                columns=['lead_time','mean', 'std', 'p', 'alpha',
                                         'algorithm', 'x_actor_lr', 'x_critic_lr', 'x_tau',
                                         'expected_cost', 'expected_diff'])

df_final_results.to_csv('./experiment_' + str(experiment_num) + '/summary data/' + 'Final Experiment Results.csv')
df_final_summary.to_csv('./experiment_' + str(experiment_num) + '/summary data/' + 'Final Experiment Summary.csv')




ray.shutdown()
assert ray.is_initialized() == False