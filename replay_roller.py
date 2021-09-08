import torch.multiprocessing as mp
import torch
import os
from pickle import load, loads
from zstd import decompress
from random import random, seed
import pysc2.lib.actions as ac
from observations import (load_global_stats, to_cuda, shift_last_action,
                          extend_batch, concat_along_axis_tensor)
from util import get_names
from copy import deepcopy
ROLLS = 1


def loader(files, pipe, sem, sem2, sem_consumed, seq_length):
    torch.set_num_threads(1)
    seed(2137)
    length = len(files)
    while True:

        for d_id, directory in enumerate(files):
            names = get_names(directory)
            if len(names) < 2:
                continue

            stats = load_global_stats(directory)
            # print(stats)
            build_visible = random() < 0.9

            last_action = {
                'function': torch.tensor([[0]]).long(),
                str(ac.TYPES.screen): torch.tensor([[0]]).long(),
                str(ac.TYPES.screen2): torch.tensor([[0]]).long(),
                str(ac.TYPES.minimap): torch.tensor([[0]]).long()
            }
            rep_count = (len(names) - 1)
            multi_load = seq_length // 32
            for i in range(0, rep_count, multi_load):
                batch = []
                for j in range(multi_load):
                    if i + j < rep_count:
                        with open(os.path.join(directory, str(i + j)),
                                  "rb") as f:
                            loaded = loads(decompress(load(f)))
                        batch.append(loaded)
                    else:
                        ins, ts, masks = deepcopy(batch[-1])
                        for k in masks:
                            masks[k] *= 0
                        batch.append((ins, ts, masks))

                inputs, targets, masks = zip(*batch)

                inputs = concat_along_axis_tensor(inputs, 0)
                targets = concat_along_axis_tensor(targets, 0)
                masks = concat_along_axis_tensor(masks, 0)

                batch = (inputs, targets, masks)

                prev_actions, last_action = shift_last_action(
                    batch[1], last_action, seq_length)
                batch = extend_batch(batch, stats, seq_length, prev_actions,
                                     build_visible)
                flags = None
                sem2.release()
                sem.release()
                if i >= rep_count - multi_load and d_id == length - 1:
                    flags = "END"
                pipe.send(batch + (flags, ))
                sem_consumed.acquire()

            pipe.send(("RESET", None, None, None))


class NotAvailable:
    pass


class ReplayRoller():
    def __init__(self, files_queue, model, sem, seq_length, prefetch):
        self.seq_length = seq_length
        self.sem = sem
        self.model = model
        self.in_sem = mp.Semaphore(0)
        self.sem_consumed = mp.Semaphore(prefetch)
        self.data = []
        self.hidden = self.model.get_hidden(1)
        self.hidden = tuple([hid.cuda() for hid in self.hidden])
        self.pipe_my, pipe_other = mp.Pipe()
        self.files = files_queue
        self.loader = mp.Process(target=loader,
                                 args=(self.files, pipe_other, self.sem,
                                       self.in_sem, self.sem_consumed,
                                       self.seq_length))
        self.loader.start()

    def get(self):

        if not self.in_sem.acquire(block=False):
            return []

        inputs, target, masks, flags = self.pipe_my.recv()

        while inputs == "RESET":
            self.hidden = self.model.get_hidden(1)
            self.hidden = tuple([hid.cuda() for hid in self.hidden])
            inputs, target, masks, flags = self.pipe_my.recv()

        return to_cuda(inputs), to_cuda(target), to_cuda(
            masks), self.hidden, flags

    def set_hidden(self, new_hidden):
        self.sem_consumed.release()
        self.hidden = new_hidden
