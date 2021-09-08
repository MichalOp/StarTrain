import torch.multiprocessing as mp
from random import shuffle
from util import chunk_it
from replay_roller import ReplayRoller
import torch
from observations import concat_along_axis_tensor, concat_lstm_hidden


class BatchSeqLoader():
    '''
    This loader attempts to diversify loaded samples by keeping a pool of open
    replays and randomly selecting several to load a sequence from at each
    training step.
    '''
    def __init__(self, envs, names, steps, batch, model):

        self.main_sem = mp.Semaphore(0)
        self.rollers = []
        self.batch_size = batch

        names = chunk_it(names, envs)

        for i in range(envs):
            self.rollers.append(
                ReplayRoller(names[i], model, self.main_sem, steps, 1))

    def batch_lstm(self, states):
        states = zip(*states)
        return tuple([torch.cat(s, 1) for s in states])

    def unbatch_lstm(self, state):
        states = []
        for i, roller in enumerate(self.current_rollers):
            extracted = tuple()
            for val in state:
                size = val.shape[1] // self.batch_size
                extracted += (val[:, i * size:(i + 1) * size, :].detach(), )

            states.append(extracted)
        return states

    def get_batch(self):

        shuffle(self.rollers)
        inputs, targets, masks, hiddens, current_rollers = [], [], [], [], []

        while len(inputs) < self.batch_size:
            self.main_sem.acquire()
            for roller in self.rollers:
                maybe_data = roller.get()
                if len(maybe_data) > 0:
                    input, target, mask, hidden, _ = maybe_data
                    inputs.append(input)
                    targets.append(target)
                    masks.append(mask)
                    hiddens.append(hidden)
                    current_rollers.append(roller)
                    if len(inputs) == self.batch_size:
                        break

        inputs = concat_along_axis_tensor(inputs, 1)
        targets = concat_along_axis_tensor(targets, 1)

        masks = concat_along_axis_tensor(masks, 1)
        hiddens = concat_lstm_hidden(hiddens)

        self.current_rollers = current_rollers
        return inputs, targets, masks, hiddens

    def put_back(self, lstm_state):
        lstm_state = self.unbatch_lstm(lstm_state)
        for i, roller in enumerate(self.current_rollers):
            roller.set_hidden(lstm_state[i])

    def kill(self):
        for roller in self.rollers:
            roller.kill()
