import numpy as np


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
            self.states_reduced = Memory(shape=(3,), limit=limit)
            self.next_states_reduced = Memory(shape=(3,), limit=limit)
        elif self.algorithm == 'inddpg':
            self.states_changed = Memory(shape=(state_dim,), limit=limit)
            self.next_states_changed = Memory(shape=(state_dim,), limit=limit)

        self.limit = limit
        self.size = 0

    def append(self, state, action, reward, next_state, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.terminals.append(done)

        if self.algorithm == 'iddpg':
            self.states_reduced.append(np.array([state[:-2].sum(),state[-2],state[-1]]))
            self.next_states_reduced.append(np.array([next_state[:-2].sum(),next_state[-2],next_state[-1]]))
        elif self.algorithm == 'inddpg':
            self.states_changed.append(np.array([state.sum()] * len(state)))
            self.next_states_changed.append(np.array([next_state.sum()] * len(next_state)))

        self.size = self.states.size

    def sample_batch(self, batch_size):
        rng = np.random.default_rng()
        idxs = rng.choice(self.size, batch_size)

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
