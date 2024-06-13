import time
import csv
import gym
import ray
import numpy as np
import random
import pandas as pd
import datetime
import argparse


from ray import tune
from agent import Agent
from utils import experiment_name
from environment import InventoryEnv
import matplotlib.pyplot as plt


@ray.remote
def run_rl(version=4.0,
           exp_id = 0,
           experiment_num=0,
           algorithm='ddpg',
           fixed_cost=0,
           lead_time=0,
           mean=5,
           std=1,
           p=4.0,
           alpha=1,
           gamma=0.99,
           x_actor_lr=4.0,
           x_critic_lr=3.0,
           x_tau=3.0,
           sigma=0.1,
           hidden_layers=[150, 120, 80, 20],
           max_iter=50000,
           max_ep_len=500,
           eval_interval=1000,
           start_train=10000,
           train_interval=50,
           buffer_size=1e6,
           fill_buffer=1000,
           batch_size=64,
           num_checkpoints=3,
           render=False,
           step=0,
           fine_tune = False,
           freeze = False,
           load_path = None,
           test = False):
    
    print('started')
    
    max_iter = int(max_iter)
    max_ep_len = int(max_ep_len)
    buffer_size = int(buffer_size)

    actor_lr = 10 ** (- x_actor_lr)
    critic_lr = 10 ** (- x_critic_lr)
    tau = 10 ** (- x_tau)

    if algorithm == 'ddpg':
        penalty = False
    else:
        penalty = True

    env = InventoryEnv(fixed_cost, lead_time, mean, std, p, alpha)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = Agent(algorithm=algorithm,
                  dimS=state_dim,
                  dimA=action_dim,
                  gamma=gamma,
                  actor_lr=actor_lr,
                  critic_lr=critic_lr,
                  tau=tau,
                  sigma=sigma,
                  hidden_layers=hidden_layers,
                  buffer_size=buffer_size,
                  batch_size=batch_size,
                  render=render,
                  freeze = freeze)
    
    if fine_tune:
        agent.load_model(load_path)
        agent.buffer.set_finetune()

    if test:
        agent.load_model(load_path)

        costs = []
        policies = []

        for i in range(5):
            cost_value = eval_cost(agent, lead_time, mean, std, p, alpha)
            policy_value = eval_policy(agent, lead_time, mean, std, p, alpha)

            costs.append(cost_value)
            policies.append(policy_value)
        
        cost_value = np.mean(np.array(costs))
        policy_value = np.mean(np.array(policies))

        print('Testing for exp', exp_id, lead_time, mean, std, p, alpha, algorithm, x_actor_lr, x_critic_lr, x_tau,
                      '|  cost_value: {:.4f}  |  policy_value: {:.4f}'.format(cost_value, policy_value))
        
        return (cost_value, policy_value)

    # name
    name = experiment_name(version=version,
                           lead_time=lead_time, mean = mean, std=std, p=p, alpha=alpha,
                           algorithm=algorithm, x_actor_lr=x_actor_lr, x_critic_lr=x_critic_lr, x_tau=x_tau,
                           step=step)
    print(name)
    

    # initialize
    state = env.reset()
    step_count = 0
    ep_reward = 0

    list_t = list(range(0, max_iter + 1, eval_interval))
    list_cost = [100.0] * len(list_t)
    list_policy = [100.0] * len(list_t)

    list_cost_t = [0] * num_checkpoints
    list_cost_value = [np.inf] * num_checkpoints
    list_cost_model = [None] * num_checkpoints

    list_policy_t = [0] * num_checkpoints
    list_policy_value = [np.inf] * num_checkpoints
    list_policy_model = [None] * num_checkpoints

    old_ratio = 0.1
    eval_count = 0

    # main loop
    for t in range(max_iter + 1):
        if t < fill_buffer:
            action = env.action_space.sample()
            action = env.reverse_action(action)
        else:
            action = agent.get_action(state)

        # environment step
        next_state, reward, done, _ = env.step(action, translate=True, evaluate=False, penalty=penalty)
        step_count += 1

        if step_count == max_ep_len:
            done = False

        agent.buffer.append(state, action, reward, next_state, done)

        state = next_state
        ep_reward += reward

        if done or (step_count == max_ep_len):
            state = env.reset()
            step_count = 0
            ep_reward = 0

        if (t >= start_train) and (t % train_interval == 0):
            for _ in range(train_interval):
                agent.train(old_ratio=old_ratio)

        if t % 10000 == 0:
            old_ratio += 0.05

        if t % eval_interval == 0:
            eval_t = int(t/eval_interval)
            cost_value = eval_cost(agent, lead_time, mean, std, p, alpha)
            policy_value = eval_policy(agent, lead_time, mean, std, p, alpha)

            list_t[eval_t] = t
            list_cost[eval_t] = cost_value
            list_policy[eval_t] = policy_value

            eval_count += 1

            if t % (1 * eval_interval) == 0:
                print(lead_time, mean, std, p, alpha, algorithm, x_actor_lr, x_critic_lr, x_tau, 
                      '|  step {} cost_value: {:.4f}  |  policy_value: {:.4f}'.format(t, cost_value, policy_value)) 
                
            if eval_count >= 6 :  # Ensure we have enough data points to compare
                # Calculate the average of the last 3 policy values
                recent_average = sum(list_policy[eval_t-1:eval_t+1]) / 2.0
                # Compare to the policy value 3 periods ago
                past_value = list_policy[eval_t-2]
                
                if recent_average < 0.2 and recent_average > 0.005 + past_value:
                    print(f"Stopping training at t={t} due to stopping condition.")
                    break
            
            if max(list_cost_value) > cost_value:
                k = np.argmax(list_cost_value)
                list_cost_t[k] = t
                list_cost_value[k] = cost_value
                list_cost_model[k] = agent.save_dict()

            if max(list_policy_value) > policy_value:
                k = np.argmax(list_policy_value)
                list_policy_t[k] = t
                list_policy_value[k] = policy_value
                list_policy_model[k] = agent.save_dict()

    csv_file_name = './experiment_' + str(experiment_num) + '/data/' + name + '.csv'
    df = pd.DataFrame({'step': list_t, 'expected_cost': list_cost, 'expected_diff': list_policy})
    df.to_csv(csv_file_name, index=False)

    for i in range(num_checkpoints):
        cost_path = './experiment_' + str(experiment_num) + '/checkpoints_cost/' + name + '_(iter={})'.format(list_cost_t[i])
        policy_path = './experiment_' + str(experiment_num) + '/checkpoints_policy/' + name + '_(iter={})'.format(list_policy_t[i])
        agent.save_model(model_dict=list_cost_model[i], path=cost_path)
        agent.save_model(model_dict=list_policy_model[i], path=policy_path)

    expected_cost = df.nlargest(10, 'expected_cost')['expected_cost'].mean()
    expected_diff = df.nlargest(10, 'expected_diff')['expected_diff'].mean()

    return [lead_time, std, p, alpha, algorithm, x_actor_lr, x_critic_lr, x_tau, expected_cost, expected_diff]


def eval_cost(agent, lead_time, mean, std, p, alpha, eval_num=100, render=False):
    list_cost = []

    for ep in range(eval_num):
        env = InventoryEnv(lead_time=lead_time, mean = mean, std=std, p=p, alpha=alpha)

        state = env.reset_eval()
        step_count = 0
        ep_reward = 0
        done = False

        while done is False and step_count < 500:
            if render and ep == 0:
                env.render()

            action = agent.get_action(state, eval=True)

            next_state, reward, done, _ = env.step(action, evaluate=True)
            step_count += 1
            state = next_state

            ep_reward += reward

        if render and ep == 0:
            env.close()
        list_cost.append(- ep_reward / step_count)
    avg_cost = float(np.mean(list_cost))

    return avg_cost


def eval_policy(agent, lead_time, mean, std, p, alpha, eval_num=5000):
    list_policy = [100] * eval_num
    list_action_optimal = [100] * eval_num
    env = InventoryEnv(lead_time=lead_time, mean = mean, std=std, p=p, alpha=alpha)

    for i in range(eval_num):
        state = env.reset_eval()
        action_agent = env.trans_action(agent.get_action(state, eval=True))
        action_order = env.feasible_action(action_agent)
        action_optimal = env.optimal_action(state)
        diff_action = abs(action_order - action_optimal)
        list_policy[i] = diff_action
        list_action_optimal[i] = action_optimal

    avg_policy = float(np.mean(list_policy) / (np.mean(list_action_optimal) + 0.001))

    return avg_policy
