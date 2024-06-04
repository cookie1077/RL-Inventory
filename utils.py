import os
import math
import numpy as np

from random import uniform
from scipy.stats import norm

def set_dir(env_id):
    os.makedirs('./train_log/' + env_id + '/', exist_ok=True)
    os.makedirs('./eval_log/' + env_id + '/', exist_ok=True)
    os.makedirs('./checkpoints/' + env_id + '/', exist_ok=True)
    return


def get_th(x, y):
    # transform cartesian coordinate of a given point into polar coordinate
    # here we only compute \th
    if x > 0.:
        return np.arctan(y / x) - (np.pi / 2.)
    elif x == 0.:
        return 0. if y > 0. else -np.pi
    else:
        return np.arctan(y / x) + (np.pi / 2.)


def obs2state(obs):
    # transform an observation (\cos(\th), \sin(th), \dot\th) to the corresponding state (\th, \thdot)
    # observation : a point on the unit circle
    th = get_th(obs[0], obs[1])
    th_dot = obs[2]
    state = (th, th_dot)
    return state


def environment_name(prefix='Environment_', version=4.0, lead_time=4, mean = 5,std=1, p=4.0, alpha=1):
    name = prefix + \
           'ver_.' + str(version) + \
           '[l=' + str(lead_time) + \
           ', mean=' + str(mean) + \
           ', std=' + str(std) + \
           ', p=' + str(p) + \
           ', alpha=' + str(alpha) + ']'

    return name


def experiment_name(prefix='_Experiment_', exp_id=0, version=4.0, lead_time=4, mean = 5, std=1, p=4.0, alpha=1,
                    algorithm='ddpg', x_actor_lr=4.0, x_critic_lr=3.0, x_tau=3.0,
                    step=0):
    name = 'exp_id=' + str(exp_id) + \
            prefix + \
           'ver_.' + str(version) + \
           '(l=' + str(lead_time) + \
           ', mean=' + str(mean) + \
           ', std=' + str(std) + \
           ', p=' + str(p) + \
           ', alpha=' + str(alpha) + ')_' + \
           '(A=' + str(algorithm) + \
           ', al=' + str(x_actor_lr) + \
           ', cl=' + str(x_critic_lr) + \
           ', tau=' + str(x_tau) + \
           ', step=' + str(step) + ')'

    return name


def get_optimal_S(lead_time, mean, std, p):

    key = (lead_time, std, p)
    S_optimal = 0 
    if lead_time ==0 : 
        S_optimal  = mean + std*norm.cdf(p/(p+1))
    elif key == (0, 1.0, 4.0):
        S_optimal = 5.84186715599139 # mean + sigma*z_a
    elif key == (2, 1.0, 4.0):
        S_optimal = 16.4577841578795
    elif key == (4, 1.0, 4.0):
        S_optimal = 26.8825070963673
    elif key == (6, 1.0, 4.0):
        S_optimal = 37.2265153828729
    elif key == (8, 1.0, 4.0):
        S_optimal = 47.5254051815305
    elif key == (4, 0.5, 4.0):
        S_optimal = 25.9412279404479
    elif key == (4, 1.5, 4.0):
        S_optimal = 27.828053329363
    elif key == (4, 2.0, 4.0):
        S_optimal = 28.8569317715825
    elif key == (4, 2.5, 4.0):
        S_optimal = 30.1194534250778
    elif key == (4, 1.0, 1.0):
        S_optimal = 25.0004296573742
    elif key == (4, 1.0, 9.0):
        S_optimal = 27.8661320439789
    elif key == (4, 1.0, 16.0):
        S_optimal = 28.4983337238988
    elif key == (4, 1.0, 25.0):
        S_optimal = 28.95550369735
    else:
        print("Key Error")

    return S_optimal, S_optimal


def get_optimal_cost(lead_time, std, p):
    key = (lead_time, std, p)
    cost = 100
    if key == (0, 1.0, 4.0):
        cost = 1.4030929675879065
    elif key == (2, 1.0, 4.0):
        cost = 2.512479873519656
    elif key == (4, 1.0, 4.0):
        cost = 3.3191018739412343
    elif key == (6, 1.0, 4.0):
        cost = 4.0083888816209585
    elif key == (8, 1.0, 4.0):
        cost = 4.623523115064558
    elif key == (4, 0.5, 4.0):
        cost = 1.7510661684010176
    elif key == (4, 1.5, 4.0):
        cost = 4.9112846369756245
    elif key == (4, 2.0, 4.0):
        cost = 6.373503541957382
    elif key == (4, 2.5, 4.0):
        cost = 7.76143157810546
    elif key == (4, 1.0, 1.0):
        cost = 1.825911660086072
    elif key == (4, 1.0, 9.0):
        cost = 4.328554193544493
    elif key == (4, 1.0, 16.0):
        cost = 5.221101138964448
    elif key == (4, 1.0, 25.0):
        cost = 6.151887599655934
    else:
        print("Key Error")

    return cost





