import pysc2.lib.actions as ac
from pysc2.env.environment import StepType
from observations import (get_observation, to_tensor, extend_batch,
                          extract_action, set_skip)
import torch
import time
from pysc2.lib.features import ScoreCumulative as sc
import numpy as np

VALUE_DIM = 5


def weighted_score(score, reward):
    return np.array([
        score[sc.killed_value_units] * 0.03,
        score[sc.killed_value_structures]*0.1,
        score[sc.collected_minerals] * 0.01,
        score[sc.collected_vespene] * 0.05,
        reward * 1000.0
    ])


value_names = ["killed_units", "killed_structures",
               "collected_minerals", "collected_vespene", "game_result"]


class SC2EnvWrapper:
    def __init__(self, env, stats, timeout, env_name="env"):
        self.env_name = env_name
        self.timeout = timeout
        self.stats = stats
        self.env = env
        self.minute = 0

    def gen_obs(self, ts):
        inputs = get_observation(ts.observation)
        inputs = to_tensor(inputs)
        inputs = extend_batch((inputs, ), self.stats, 1, self.last_action,
                              True)[0]
        return inputs

    def extract_action(self, outputs):
        action = outputs["function_sampled"].squeeze().item()
        true_delay = outputs["time_skip_sampled"].squeeze().item()
        delay = max(true_delay - 1, 0)

        func = ac.FUNCTIONS[action]
        action_data = []
        last_action = {}
        for x in outputs:
            last_action[x.replace("_sampled", "")] = outputs[x].detach().cpu()

        for x in ac.FUNCTION_TYPES[func.function_type]:
            sub_action = outputs[str(x) + "_sampled"].squeeze().item()

            if "screen" in str(x) or "minimap" in str(x):
                sub_action = (sub_action // 64, sub_action % 64)
            else:
                sub_action = (sub_action, )
            action_data.append(sub_action)

        skip_target = self.time_step.observation["game_loop"][0] + delay

        return ([ac.FunctionCall(action, action_data)], skip_target,
                last_action, true_delay)

    def reset(self):
        self.minute = 0
        self.scheduled = [ac.FunctionCall(0, [])]
        obs = self.env.reset()
        self.time_step = obs[0]
        self.last_score = weighted_score(
            self.time_step.observation['score_cumulative'], 0)
        self.last_action = {
            'function': torch.tensor([[0]]).long(),
            str(ac.TYPES.screen): torch.tensor([[0]]).long(),
            str(ac.TYPES.screen2): torch.tensor([[0]]).long(),
            str(ac.TYPES.minimap): torch.tensor([[0]]).long()
        }

        self.t0 = time.time()

        return self.gen_obs(self.time_step)

    def game_end_return(self, time_step, delta, end_action):
        print(f"{self.env_name}: GAME END", time_step.reward)
        return self.gen_obs(time_step), \
            delta, True, time_step.reward, end_action

    def step_wrap(self, action, delay):
        obs = self.env.step(action, delay)
        time_step = obs[0]
        game_loop = time_step.observation["game_loop"][0]
        mins = game_loop / 22.4 / 60

        min_val = int(mins)

        if min_val > self.minute:
            self.minute = min_val
            print(f"{self.env_name}: minute ", min_val,
                  f"x{game_loop /(time.time()-self.t0)/ 22.4} realtime")

        end = time_step.step_type == StepType.LAST or mins > self.timeout
        return time_step, end

    def step(self, action):
        action, skip_target, self.last_action, true_skip = self.extract_action(
            action)

        act, md = extract_action(action[0])
        set_skip([None, act], true_skip)
        act = to_tensor(act, type=torch.int64)
        md = to_tensor(md)

        self.time_step, end = self.step_wrap(action, max(1, true_skip))
        score = weighted_score(
            self.time_step.observation['score_cumulative'], self.time_step.reward)
        delta = score - self.last_score
        self.last_score = score

        if end:
            return self.game_end_return(self.time_step, delta, (act, md))
        return self.gen_obs(self.time_step), delta, False, 0, (act, md)
