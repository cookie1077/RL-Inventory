import gym
import math
import random
import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from os import path
from gym import spaces

from random import uniform
from scipy.stats import norm
from gym.utils import seeding
from itertools import product
from utils import get_optimal_S


class InventoryEnv(gym.Env):
    def __init__(self, fixed_cost=0, lead_time=0, mean=5, std=1, p=4.0, alpha=1):
        # system parameters
        self.fixed = fixed_cost
        self.leadtime = lead_time

        self.price = 0
        self.overage = 1.
        self.underage = p
        self.ordering = 0

        self.demand_mean = mean
        self.demand_std = std
        self.demand_lower_bound = 0
        self.demand_upper_bound = None
        self.u_lower = norm.cdf(self.demand_lower_bound, self.demand_mean, self.demand_std)
        self.u_upper = 1

        self.min_order = - alpha * self.demand_mean
        self.max_order = float(math.ceil((self.leadtime+1)*self.demand_mean + 3*math.sqrt(self.leadtime+1)*self.demand_std))

        self.min_inv = - self.max_order
        self.max_inv = (1 + self.leadtime/2) * self.max_order

        if self.leadtime == 0:
            self.state = np.zeros(1)
        else:
            self.state = np.zeros(self.leadtime)

        # setting viewer
        self.viewer = None

        # setting action space
        self.action_space = spaces.Box(
            low=np.float32(self.min_order),
            high=np.float32(self.max_order),
            shape=(1,)
        )

        # setting observation space
        # inventory level, on-order items
        self.obs_low = np.array([self.min_inv])
        self.obs_high = np.array([self.max_inv])

        # self.obs_low = np.append(self.obs_low, [self.min_order] * (self.leadtime - 1))
        if self.leadtime > 1:
            self.obs_low = np.append(self.obs_low, [0] * (self.leadtime - 1))
            self.obs_high = np.append(self.obs_high, [self.max_order] * (self.leadtime - 1))

        self.observation_space = spaces.Box(
            low=np.float32(self.obs_low),
            high=np.float32(self.obs_high)
        )

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action, translate=True, evaluate=False, penalty=False):
        """
        ==============================================================================================================
        Case 1: fixed cost = 0 & lead time = 0

        state is 1-dimensional
        s[0] = inventory level at the end of period (t.py-1)

        The sequence of events in each period t.py is as follows
        
        1. The inventory level is observed.
        2. A replenishment order of size q_t is placed and is received instantly.
        3. Demand d_t occurs; as much as possible is satisﬁed from inventory, and the rest is backordered.
        4. Holding and stockout costs are assessed based on the ending inventory level.

        ==============================================================================================================
        Case 2: fixed cost = 0 & lead time = l
        Case 3: fixed cost = K & lead time = l

        state is l-dimensional (l = leadtime)

        s[0] = inventory level at the end of period (t.py-1) + order quantity at period (t.py-l)
        s[1] = order quantity at period (t.py-l+1)
        ...
        s[l-1] = order quantity at period (t.py-1)

        The sequence of events in each period t.py is as follows

        1. The inventory level is observed.
        2. A replenishment order of size q_t is placed.
        3. Demand d_t occurs; as much as possible is satisﬁed from inventory, and the rest is backordered.
        4. Holding and stockout costs are assessed based on the ending inventory level.
        5. A replenishment order of size q_(t.py-l+1) is received.
        ==============================================================================================================
        At the beginning of any period t.py
        the order quantity, q_t ≥ 0, must be decided
        knowing the last observed inventory on hand, I_{t.py−1},
        and outstanding receipts or “pipeline” vector Q_{t.py−1} =(q_{t.py−l}, q_{t.py−l+1},···,q_{t.py−1}).
        ==============================================================================================================
        """

        if translate:
            order = self.trans_action(action[0])
        else:
            order = action[0]
        order = self.feasible_action(order)

        # truncated normal distribution
        rn_unif = uniform(self.u_lower, self.u_upper)
        demand = float(norm.ppf(rn_unif, self.demand_mean, self.demand_std))

        if self.leadtime == 0:
            inv_s = self.state[0] + order
        elif self.leadtime > 0:
            inv_s = self.state[0]
        else:
            inv_s = 0
            print("leadtime error")

        inv_f = inv_s - demand
        satisfied = min(max(0, inv_s), demand)
        on_hand = max(0, inv_f)
        backorder = max(0, -inv_f)

        reward = self.price * satisfied - self.ordering * order - self.overage * on_hand - self.underage * backorder
        if order > 0:
            reward -= self.fixed

        if self.leadtime == 0:
            observation = np.array([inv_f])
        elif self.leadtime == 1:
            inv_f += order
            observation = np.array([inv_f])
        elif self.leadtime == 2:
            inv_f += self.state[1]
            pipeline = order
            observation = np.append(inv_f, pipeline)
        elif self.leadtime > 2:
            inv_f += self.state[1]
            pipeline = np.append(self.state[2:], order)
            observation = np.append(inv_f, pipeline)
        else:
            observation = 0
            print("error")

        observation[0], done = self.feasible_inv(observation[0])
        if done and (not evaluate) and penalty:
            if self.leadtime == 0:
                reward -= 100.0
            elif self.leadtime == 2:
                reward -= 500.0
            elif self.leadtime == 4:
                reward -= 1000.0
            elif self.leadtime == 6:
                reward -= 5000.0
            elif self.leadtime == 8:
                reward -= 10000.0

        self.state = observation
        info = None

        return observation, reward, done, info

    def reset(self):
        # small size setting
        # low = np.array([-self.demand_mean])
        # high = np.array([self.demand_mean])

        # big size setting
        # low = np.array([self.min_inv])
        # high = np.array([self.max_inv])

        # proper size setting
        if self.leadtime == 0:
            low = np.array([self.min_inv + self.demand_mean])
            high = np.array([self.max_order - self.demand_mean])
        else:
            low = np.array([self.min_inv + self.leadtime * self.demand_mean])
            high = np.array([self.max_order - self.leadtime * self.demand_mean])

        if self.leadtime > 1:
            low = np.append(low, [self.min_order] * (self.leadtime - 1))
            high = np.append(high, [2 * self.demand_mean] * (self.leadtime - 1))

        self.state = self.np_random.uniform(low=low, high=high)
        self.last_action = None

        return self._get_obs()

    def reset_eval(self):
        if self.leadtime == 0:
            inventory = np.random.choice(np.arange(self.min_inv + self.demand_mean,
                                                   self.max_order - self.demand_mean + 0.01, 1), 1)
        else:
            inventory = np.random.choice(np.arange(self.min_inv + self.leadtime * self.demand_mean,
                                                   self.max_order - self.leadtime * self.demand_mean + 0.01, 1), 1)

        if self.leadtime > 1:
            on_order = np.random.choice(np.arange(0, 2 * self.demand_mean + 0.01, 1), self.leadtime-1)
            self.state = np.append(inventory, on_order)
        else:
            self.state = inventory

        self.last_action = None
        return self._get_obs()

    def reset_grid(self, state):
        self.state = state
        self.last_action = None

        return self._get_obs()

    def optimal_solution(self):
        s_optimal, S_optimal = get_optimal_S(lead_time=self.leadtime, mean=self.demand_mean, std=self.demand_std, p=self.underage)

        return s_optimal, S_optimal

    def optimal_action(self, state):
        _, S_optimal = self.optimal_solution()
        action = max(0.0, min(self.max_order, S_optimal - state.sum()))

        return action

    def _get_obs(self):
        return self.state

    def trans_action(self, action):
        act_k = (self.action_space.high - self.action_space.low) / 2.
        act_b = (self.action_space.high + self.action_space.low) / 2.
        return float(act_k * action + act_b)

    def reverse_action(self, action):
        act_k_inv = 2. / (self.action_space.high - self.action_space.low)
        act_b = (self.action_space.high + self.action_space.low) / 2.
        return act_k_inv * (action - act_b)

    def feasible_action(self, action):
        feasible_action = action
        if action < 0:
            feasible_action = 0
        elif action > self.max_order:
            feasible_action = self.max_order
        return feasible_action

    def feasible_inv(self, inv):
        feasible_inv = inv
        done = False
        if inv < self.min_inv:
            feasible_inv = self.min_inv
            done = True
        elif inv > self.max_inv:
            feasible_inv = self.max_inv
            done = True

        return feasible_inv, done
