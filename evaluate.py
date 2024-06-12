import pandas as pd
import ray
import os.path

import numpy as np
import glob

from learning import run_rl
from agent import Agent
from itertools import product
from utils import experiment_name


ray.init(num_cpus=8)
assert ray.is_initialized() == True



version = 5.0
load_experiment_num=15

if not os.path.isdir(f'./experiment_{load_experiment_num}'):
    os.makedirs(f'./experiment_{load_experiment_num}')
    os.makedirs(f'./experiment_{load_experiment_num}/checkpoints_cost')
    os.makedirs(f'./experiment_{load_experiment_num}/checkpoints_policy')
    os.makedirs(f'./experiment_{load_experiment_num}/data')
    os.makedirs(f'./experiment_{load_experiment_num}/summary data')

#lr_list = [[5.5, 3.5], [6.0, 4.0], [6.5, 4.5]]
lr_list = [[5.5, 3.5]]

df_final_experiment = pd.read_csv('./Evaluate_env.csv')
df_final_models = pd.read_csv('./Evaluate_model.csv')

print(ray.is_initialized())
print(os.getcwd())
step=0
        

for i in range(0, len(df_final_experiment)):
    row_env = df_final_experiment.iloc[i]
    row_model = df_final_models.iloc[i]

    for i in range(len(lr_list)):

        exp_id, lead_time, mean, std, p, alpha, algorithm, x_actor_lr, x_critic_lr, x_tau =\
            row_model[['exp_id', 'lead_time','mean', 'std', 'p', 'alpha', 'algorithm', 'x_actor_lr', 'x_critic_lr', 'x_tau']]
        
        exp_id, lead_time, mean, std, p, alpha, x_actor_lr, x_critic_lr, x_tau =\
            int(exp_id), int(lead_time), float(mean),float(std), float(p), float(alpha), float(x_actor_lr), float(x_critic_lr), float(x_tau)
        
        x_actor_lr = lr_list[i][0]
        x_critic_lr = lr_list[i][1]
    
        #glob_path = f'./experiment_{load_experiment_num}/checkpoints_cost/' + experiment_name(exp_id = exp_id, version=version, lead_time=lead_time, mean = mean, std=std, p=p, alpha=alpha,
        #                                   algorithm=algorithm, x_actor_lr=x_actor_lr, x_critic_lr=x_critic_lr, x_tau=x_tau,step=step) + "*.pth.tar"
        
        glob_path = './inference/' + experiment_name(exp_id = exp_id, version=version, lead_time=lead_time, mean = mean, std=std, p=p, alpha=alpha,
                                           algorithm=algorithm, x_actor_lr=x_actor_lr, x_critic_lr=x_critic_lr, x_tau=x_tau,step=2) + "*.pth.tar"


        weight_path = glob_path
        # Use glob to find files that match the pattern
        matching_files = glob.glob(weight_path)

        # Check if any matching files exist
        if matching_files:
            for file in matching_files:
                if os.path.isfile(file):
                    print(f"File {file} exists.")
        else:
            print("No matching files found.")
            print(weight_path)
            print()
            continue

        lead_time, mean, std, p, alpha, algorithm, x_actor_lr, x_critic_lr, x_tau =\
            row_env[['lead_time','mean', 'std', 'p', 'alpha', 'algorithm', 'x_actor_lr', 'x_critic_lr', 'x_tau']]
        
        lead_time, mean, std, p, alpha, x_actor_lr, x_critic_lr, x_tau =\
            int(lead_time), float(mean),float(std), float(p), float(alpha), float(x_actor_lr), float(x_critic_lr), float(x_tau)
        
        
        result = run_rl.remote(exp_id= exp_id, version=version, experiment_num=load_experiment_num,
                                                    lead_time=lead_time, mean = mean, std=std, p=p, alpha=alpha,
                                                    algorithm=algorithm, x_actor_lr=x_actor_lr,
                                                    x_critic_lr=x_critic_lr, x_tau=x_tau,
                                                    step=step,fine_tune = False, load_path = file, test=True, freeze=True)
        
        cost_value, policy_value = ray.get(result)
        
        print('Testing for env', exp_id, lead_time, mean, std, p, alpha, algorithm, x_actor_lr, x_critic_lr, x_tau,
                        '|  cost_value: {:.4f}  |  policy_value: {:.4f}'.format(cost_value, policy_value))
        print('---------------------------------------------------------------------------------')

ray.shutdown()
assert ray.is_initialized() == False