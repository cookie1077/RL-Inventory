import numpy as np
import pickle
from torch.utils.data import Dataset
import torch


class ReplayBuffer:
    def __init__(self, algorithm, state_dim, action_dim, limit):
        self.states = Memory(shape=(state_dim,), limit=limit)
        self.actions = Memory(shape=(action_dim,), limit=limit)
        self.rewards = Memory(shape=(1,), limit=limit)
        self.next_states = Memory(shape=(state_dim,), limit=limit)
        self.terminals = Memory(shape=(1,), limit=limit)

        # setting for iddpg
        self.algorithm = algorithm
        if self.algorithm == 'iddpg':
            self.states_reduced = Memory(shape=(1,), limit=limit)
            self.next_states_reduced = Memory(shape=(1,), limit=limit)
        elif self.algorithm == 'inddpg':
            self.states_changed = Memory(shape=(state_dim,), limit=limit)
            self.next_states_changed = Memory(shape=(state_dim,), limit=limit)

        self.limit = limit
        self.size = 0
        self.old_size = 0
       

    def set_finetune(self):
        self.old_size = self.size

    def append(self, state, action, reward, next_state, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.terminals.append(done)

        if self.algorithm == 'iddpg':
            self.states_reduced.append(np.array([state.sum()]))
            self.next_states_reduced.append(np.array([next_state.sum()]))
        elif self.algorithm == 'inddpg':
            self.states_changed.append(np.array([state.sum()] * len(state)))
            self.next_states_changed.append(np.array([next_state.sum()] * len(next_state)))

        self.size = self.states.size

    def sample_batch(self, batch_size, old_ratio=0):
        old_data_size = int(batch_size * old_ratio)
        new_data_size = batch_size - old_data_size
        
        rng = np.random.default_rng()
        
        # Sample old data
        old_idxs = rng.choice(self.old_size, old_data_size, replace=False)
        
        # Sample new data
        new_idxs = rng.choice(range(self.old_size, self.size), new_data_size, replace=False)
        
        # Combine indices
        idxs = np.concatenate((old_idxs, new_idxs))

        # get batch from each buffer
        states = self.states.get_batch(idxs)
        actions = self.actions.get_batch(idxs)
        rewards = self.rewards.get_batch(idxs)
        next_states = self.next_states.get_batch(idxs)
        terminal_flags = self.terminals.get_batch(idxs)

        if self.algorithm == 'ddpg':
            batch = {'state': states,
                     'action': actions,
                     'reward': rewards,
                     'next_state': next_states,
                     'done': terminal_flags}
        elif self.algorithm == 'iddpg':
            states_reduced = self.states_reduced.get_batch(idxs)
            next_states_reduced = self.next_states_reduced.get_batch(idxs)

            batch = {'state': states,
                     'state_reduced': states_reduced,
                     'action': actions,
                     'reward': rewards,
                     'next_state': next_states,
                     'next_state_reduced': next_states_reduced,
                     'done': terminal_flags}

        elif self.algorithm == 'inddpg':
            states_changed = self.states_changed.get_batch(idxs)
            next_states_changed = self.next_states_changed.get_batch(idxs)

            batch = {'state': states,
                     'state_changed': states_changed,
                     'action': actions,
                     'reward': rewards,
                     'next_state': next_states,
                     'next_state_changed': next_states_changed,
                     'done': terminal_flags}
        else:
            print("error")

        return batch
    
    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)


class Memory:
    """
    implementation of a circular buffer
    """

    def __init__(self, shape, limit=1000000):
        self.start = 0
        self.data_shape = shape
        self.size = 0
        self.limit = limit
        self.data = np.zeros((self.limit,) + shape)

    def append(self, data):
        if self.size < self.limit:
            self.size += 1
        else:
            self.start = (self.start + 1) % self.limit

        self.data[(self.start + self.size - 1) % self.limit] = data

    def get_batch(self, idxs):

        return self.data[(self.start + idxs) % self.limit]

    def __len__(self):
        return self.size
    

class ReplayDataset(Dataset):
    def __init__(self, replay_buffer):
        self.buffer = replay_buffer

    def __len__(self):
        return self.buffer.size

    def __getitem__(self, idx):
        state = self.buffer.states.data[idx]
        action = self.buffer.states.data[idx]
        reward = self.buffer.rewards.data[idx]
        next_state = self.buffer.next_states.data[idx]
        done = self.buffer.terminals.data[idx]
        
        return (torch.tensor(state, dtype=torch.float32),
                torch.tensor(action, dtype=torch.float32),
                torch.tensor(reward, dtype=torch.float32),
                torch.tensor(next_state, dtype=torch.float32),
                torch.tensor(done, dtype=torch.float32))

