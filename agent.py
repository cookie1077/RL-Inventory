import torch.optim
from torch.nn import MSELoss
import numpy as np
import copy
from buffer import ReplayBuffer
from model import Actor, Critic


class Agent:
    def __init__(self,
                 algorithm='ddpg',
                 dimS=0,
                 dimA=0,
                 gamma=0.99,
                 actor_lr=1e-4,
                 critic_lr=1e-3,
                 tau=1e-3,
                 sigma=0.5,
                 hidden_layers=[150, 120, 80, 20],
                 buffer_size=int(1e6),
                 batch_size=128,
                 render=False):

        # original case: hidden_layers= [400, 300]
        self.algorithm = algorithm
        self.dimS = dimS
        self.dimA = dimA
        
        self.gamma = gamma
        self.pi_lr = actor_lr
        self.q_lr = critic_lr
        self.tau = tau
        self.sigma = sigma

        self.batch_size = batch_size

        # networks definition
        # pi : actor network, Q : critic network
        if algorithm == 'ddpg':
            self.pi = Actor(dimS, dimA, hidden_layers)
        elif algorithm == 'iddpg':
            self.pi = Actor(1, dimA, hidden_layers)
        elif algorithm == 'inddpg':
            self.pi = Actor(dimS, dimA, hidden_layers)
        else:
            print("error")
        
        self.Q = Critic(dimS, dimA, hidden_layers)

        # target networks
        self.targ_pi = copy.deepcopy(self.pi)
        self.targ_Q = copy.deepcopy(self.Q)

        # buffer setting
        self.buffer = ReplayBuffer(algorithm, dimS, dimA, limit=buffer_size)

        # optimizer setting
        self.Q_optimizer = torch.optim.Adam(self.Q.parameters(), lr=self.q_lr)
        self.pi_optimizer = torch.optim.Adam(self.pi.parameters(), lr=self.pi_lr)

        self.render = render

    def target_update(self):
        # soft-update for both actors and critics
        # \theta^\prime = \tau * \theta + (1 - \tau) * \theta^\prime
        for th, targ_th in zip(self.pi.parameters(), self.targ_pi.parameters()):        # th : theta
            targ_th.data.copy_(self.tau * th.data + (1.0 - self.tau) * targ_th.data)

        for th, targ_th in zip(self.Q.parameters(), self.targ_Q.parameters()):
            targ_th.data.copy_(self.tau * th.data + (1.0 - self.tau) * targ_th.data)

    def get_action(self, state, eval=False):
        if self.algorithm == 'ddpg':
            state = torch.tensor(state, dtype=torch.float)
        elif self.algorithm == 'iddpg':
            state = [state.sum()]
            state = torch.tensor(state, dtype=torch.float)
        elif self.algorithm == 'inddpg':
            state = [state.sum()] * len(state)
            sigma = np.arange(0, 0.01 * len(state), 0.01)
            noise = sigma * np.random.randn(len(state))
            state = state + state * noise
            state = torch.tensor(state, dtype=torch.float)
        else:
            print("error")

        with torch.no_grad():
            action = self.pi(state)
            action = action.numpy()
        if not eval:
            # for exploration, we use a behavioral policy of the form
            # \beta(s) = \pi(s) + N(0, \sigma^2)
            noise = self.sigma * np.random.randn(self.dimA)
            self.sigma = max(0.0, self.sigma - 0.5/100000)
            return action + noise
        else:
            return action

    def get_value(self, state, action):
        with torch.no_grad():
            value = self.Q(state, action)
            value = value.numpy()
        return value

    def train(self):
        """
        train actor-critic network using DDPG
        """

        batch = self.buffer.sample_batch(batch_size=self.batch_size)

        # unroll batch
        observations = torch.tensor(batch['state'], dtype=torch.float)
        actions = torch.tensor(batch['action'], dtype=torch.float)
        rewards = torch.tensor(batch['reward'], dtype=torch.float)
        next_observations = torch.tensor(batch['next_state'], dtype=torch.float)
        terminal_flags = torch.tensor(batch['done'], dtype=torch.float)

        if self.algorithm == 'iddpg':
            observations_reduced = torch.tensor(batch['state_reduced'], dtype=torch.float)
            next_observations_reduced = torch.tensor(batch['next_state_reduced'], dtype=torch.float)
        elif self.algorithm == 'inddpg':
            observations_changed = torch.tensor(batch['state_changed'], dtype=torch.float)
            next_observations_changed = torch.tensor(batch['next_state_changed'], dtype=torch.float)

        mask = torch.tensor([1.]) - terminal_flags

        # compute TD targets based on target networks
        # if done, set target value to reward
        if self.algorithm == 'ddpg':
            target = rewards + self.gamma * mask * self.targ_Q(next_observations, self.targ_pi(next_observations))
        elif self.algorithm == 'iddpg':
            target = rewards + self.gamma * mask *\
                     self.targ_Q(next_observations, self.targ_pi(next_observations_reduced))
        elif self.algorithm == 'inddpg':
            target = rewards + self.gamma * mask *\
                     self.targ_Q(next_observations, self.targ_pi(next_observations_changed))
        else:
            print("error")

        out = self.Q(observations, actions)
        loss_ftn = MSELoss()
        loss = loss_ftn(out, target)
        self.Q_optimizer.zero_grad()
        loss.backward()
        self.Q_optimizer.step()

        if self.algorithm == 'ddpg':
            pi_loss = - torch.mean(self.Q(observations, self.pi(observations)))
        elif self.algorithm == 'iddpg':
            pi_loss = - torch.mean(self.Q(observations, self.pi(observations_reduced)))
        elif self.algorithm == 'inddpg':
            pi_loss = - torch.mean(self.Q(observations, self.pi(observations_changed)))
        else:
            print("error")

        self.pi_optimizer.zero_grad()
        pi_loss.backward()
        self.pi_optimizer.step()

        self.target_update()

    def save_dict(self):
        model_dict = {'actor': self.pi.state_dict(),
                      'critic': self.Q.state_dict(),
                      'target_actor': self.targ_pi.state_dict(),
                      'target_critic': self.targ_Q.state_dict(),
                      'actor_optimizer': self.pi_optimizer.state_dict(),
                      'critic_optimizer': self.Q_optimizer.state_dict()
                      }
                      
        return model_dict

    def save_model(self, model_dict, path):
        checkpoint_path = path + 'model.pth.tar'
        torch.save(model_dict, checkpoint_path)

        return


    def load_model(self, path):
        checkpoint = torch.load(path)

        self.pi.load_state_dict(checkpoint['actor'])
        self.Q.load_state_dict(checkpoint['critic'])
        self.targ_pi.load_state_dict(checkpoint['target_actor'])
        self.targ_Q.load_state_dict(checkpoint['target_critic'])
        self.pi_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.Q_optimizer.load_state_dict(checkpoint['critic_optimizer'])

        return


if __name__ == '__main__':
    agent = Agent(3, 2, 1)
    print(agent.pi.state_dict())

    # if self.algorithm == 'ddpg':
    #     pass
    # elif self.algorithm == 'iddpg':
    #     pass
    # else:
    #     print("error")
