from pickle import dump, dumps
import pysc2.lib.units as un
import torch
from zstd import compress
from os import mkdir, path
from observations import (
    get_observation,
    to_tensor,
    concat_along_axis,
    extract_action,
    generate_zeroed_dicts,
    get_skip,
    set_skip,
)

MAX_NOPS = 128

def get_unit_type(unit_name):
    for race in (un.Neutral, un.Protoss, un.Terran, un.Zerg):
        try:
            return race[unit_name]
        except KeyError:
            pass  # Wrong race.


class ReplayState():
    def __init__(self, batch_size, compress=False, store_dir=None, name=None):
        self.batch_size = batch_size
        self.last_action = 0
        self.steps = 0
        self.mmr = 0
        self.result = 0
        self.last_obs = None
        self.replay_data = []
        self.nops = MAX_NOPS
        self.total_nops = 0
        self.max_nops = 128
        self.last_action_time = 0
        self.last_not_camera = 0
        self.last_action = 0
        self.ops = 0
        self.send_data = []
        self.stored_count = 0
        self.compress = compress
        self.build = []
        if self.compress:
            store_path = path.join(store_dir, name)
            mkdir(store_path)

            self.store_dir = store_path
            print(self.store_dir)

    def save_global_info(self):
        data = {"mmr": self.mmr, "build": self.build, "result": self.result}
        with open(self.store_dir + "/stats.pkl", "wb") as f:
            dump(data, f)

    def get_batch(self):
        batch = self.replay_data[:self.batch_size]
        self.replay_data = self.replay_data[self.batch_size:]

        while len(batch) < self.batch_size:
            rd, md = generate_zeroed_dicts()
            batch.append((batch[-1][0], rd, md))

        inputs, target, masks = concat_along_axis(batch, 0)
        inputs = to_tensor(inputs)
        target = to_tensor(target, torch.int64)
        masks = to_tensor(masks)
        if not self.compress:
            self.send_data.append((inputs, target, masks))
        else:
            compressed = compress(dumps((inputs, target, masks)))

            with open(self.store_dir + "/" + str(self.stored_count),
                      "wb") as f:
                dump(compressed, f)
            self.stored_count += 1

    def stats(self):
        print(
            f"steps: {self.steps}, batches: {self.stored_count}, valid: {self.ops}"
        )

    def step(self, observation, actions, features):
        if self.last_obs is not None:
            true_action = None
            if len(actions) > 0:
                action = actions[0]
                try:
                    true_action = features.reverse_action(action)
                except Exception:
                    true_action = None

            if true_action is not None:
                rd, md = extract_action(true_action)

                new_obs = observation

                func_name = true_action.function.name
                if ("Train" in func_name
                        or "Build" in func_name) and len(self.build) < 20:
                    self.build.append(get_unit_type(func_name.split("_")[1]))

                if true_action.function == 1:
                    if self.last_action == 1:
                        # remove last camera so that it doesn't clutter the training data
                        if self.steps - self.last_not_camera < MAX_NOPS:
                            self.last_action_time = self.last_not_camera
                            del self.replay_data[-1]
                            self.ops -= 1
                            new_obs = self.last_obs
                        else:
                            self.last_not_camera = self.steps  # not actually a camera, but we need to update
                else:
                    self.last_not_camera = self.steps  # we are not moving camera

                assert true_action.function < 573, "wrong action number"

                self.replay_data.append(
                    (get_observation(features.transform_obs(self.last_obs)),
                     rd, md))
                self.ops += 1
                assert self.steps - self.last_action_time < 128, "too many nops"
                set_skip(self.replay_data[-1],
                         self.steps - self.last_action_time)

                self.last_obs = new_obs
                self.last_action = true_action.function
                self.last_action_time = self.steps

                if self.last_action != 1:
                    self.last_not_camera = self.steps
                    while len(self.replay_data) > self.batch_size:
                        self.get_batch()
            else:
                self.total_nops += 1

                if self.steps - self.last_action_time == MAX_NOPS - 1:
                    self.last_action = 0
                    rd, md = extract_action(None)
                    self.replay_data.append((get_observation(
                        features.transform_obs(self.last_obs)), rd, md))
                    self.last_obs = observation
                    set_skip(self.replay_data[-1], MAX_NOPS - 1)
                    self.last_action_time = self.steps
        else:
            self.last_obs = observation
        self.steps += 1
