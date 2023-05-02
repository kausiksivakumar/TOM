import random
import numpy as np
from operator import itemgetter

class ReplayMemory:
    def __init__(self, capacity, exp_size = 1000):
        self.capacity               = capacity
        self.buffer                 =   []
        self.position               =   0
        self.init_state             =   []
        self.exp_size               =   exp_size
        self.disc_rewards           =   np.zeros((len(self.buffer),1))
        self.prob                   =   None
        self.num_curpol             =   0

    def update_disc_reward(self):
        self.disc_rewards           =   np.zeros((len(self.buffer),1))
        self.disc_rewards[-self.exp_size:]   =   1

    def push(self, state, action, reward, next_state, done):
        self.num_curpol +=  1
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position]  =   (state, action, reward, next_state, done)
        self.position               =   (self.position + 1) % self.capacity
        # TODO: Update binary disc reward
        self.update_disc_reward()

    def push_batch(self, batch):
        if len(self.buffer) < self.capacity:
            append_len              =   min(self.capacity - len(self.buffer), len(batch))
            self.buffer.extend([None] * append_len)

        if self.position + len(batch) < self.capacity:
            self.buffer[self.position : self.position + len(batch)] = batch
            self.position           +=  len(batch)
        else:
            self.buffer[self.position : len(self.buffer)] = batch[:len(self.buffer) - self.position]
            self.buffer[:len(batch) - len(self.buffer) + self.position] = batch[len(self.buffer) - self.position:]
            self.position = len(batch) - len(self.buffer) + self.position
        # TODO: Update binary disc reward
        self.update_disc_reward()

    def sample(self, batch_size):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        # batch = random.sample(self.buffer, int(batch_size))
        batch_idxs = np.random.randint(len(self.buffer),size = int(batch_size))
        batch      = [self.buffer[idxs] for idxs in batch_idxs]
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        disc_reward     =   self.disc_rewards[batch_idxs]
        # This will only happen if litm is True 
        if(self.prob is not None):
            prob        =   self.prob[batch_idxs]
            return state, action, reward, next_state, done, disc_reward,prob
        return state, action, reward, next_state, done, disc_reward, None

    def sample_all_batch(self, batch_size, litm = False):
        idxes = np.random.randint(0, len(self.buffer), batch_size)
        if litm:
            idxes   =   np.random.choice(len(self.buffer),batch_size,p=self.prob)
        batch = list(itemgetter(*idxes)(self.buffer))
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def return_all(self):
        state, action, reward, next_state, done = map(np.stack, zip(*self.buffer))
        return state, action, reward, next_state, done

    def save_buffer(self, path='dataset/'):
        state, action, reward, next_state, done = map(np.stack, zip(*self.buffer))
        dataset = {}
        dataset['observations'] = state
        dataset['actions'] = action
        dataset['rewards'] = reward
        dataset['next_observations'] = next_state
        dataset['terminals'] = done
        np.save(path, dataset)

    def __len__(self):
        return len(self.buffer)
