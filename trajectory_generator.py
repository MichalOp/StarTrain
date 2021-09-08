import os
from model import compute_action_prob
import pysc2.lib.actions as ac
import pysc2.lib.features as ft
from pysc2.env.sc2_env import SC2Env, Agent, Race, Bot, Difficulty
from observations import (get_observation, to_tensor, to_cuda,
                          load_global_stats, extend_batch,
                          concat_along_axis_tensor, concat_lstm_hidden,
                          interface_format)
import torch
import torch.nn.functional
import torch.multiprocessing as mp
from queue import Empty
from absl import flags
import sys
from copy import deepcopy
from torch.cuda.amp import autocast
from game_env import SC2EnvWrapper, VALUE_DIM
from looptime import log_by_tag, add_t, reset_t, clear_t
from run_game import get_stats
from util import Request

FLAGS = flags.FLAGS
PATH = "model/ModelData.tm"
VISUALIZE = False

FLAGS(sys.argv[:1])


REWARD_SCALING = 0.001


class WorkerImpl:
    def __init__(self, pipe, queue, env_type, id_, data_queue, score_queue,
                 hidden, ref_replay):
        self.pipe = pipe
        self.queue = queue
        self.id_ = id_
        self.data_queue = data_queue
        self.score_queue = score_queue
        self.zero_hidden = hidden
        self.stats = get_stats(ref_replay)
        torch.set_num_threads(1)

        env = SC2Env(
            map_name="Acropolis",
            players=[Agent(Race.protoss, "StarTrain"), env_type],
            agent_interface_format=interface_format,
            visualize=VISUALIZE,
            step_mul=1,
            ensure_available_actions=False,
            realtime=False,
            game_steps_per_episode=0,
            save_replay_episodes=1,
            score_index=-1,
            replay_dir="/home/michal/AI_research/StarTrain/showcase_replays")
        self.env = SC2EnvWrapper(env, self.stats, 60)
        self.pipe.send((self.env.reset(), hidden))
        self.input_buffer, self.hidden_buffer = self.pipe.recv()

    def run_games(self):
        # try:
        while True:
            self.run_game()
        # except Exception as e:
        #     print(e)
        #     self.queue.put((Request.exception, self.id_))

    def generate_trajectory(self, buffer, last_hidden, bootstrap_value):
        with torch.no_grad():
            trajectory = []
            true_value = bootstrap_value
            for obs, action, probs, value, reward, ref_output in reversed(buffer):
                true_value = true_value * 0.999 + reward
                advantage = true_value - value
                ac, ms = action
                trajectory.append((obs, ac, ms, probs, true_value, advantage,
                                   torch.ones((1, 1, 1)), ref_output))

            trajectory = list(reversed(trajectory))

            for i in range(32 - len(buffer)):
                trajectory.append((obs, ac, ms, probs, true_value, advantage,
                                   torch.zeros((1, 1, 1)), ref_output))

            obs, ac, ms, probs, vals, advs, active, ref_output = zip(*trajectory)

            obs = concat_along_axis_tensor(obs, 0)
            ac = concat_along_axis_tensor(ac, 0)
            ms = concat_along_axis_tensor(ms, 0)
            probs = torch.cat(probs, dim=0)
            vals = torch.cat(vals, dim=0)
            advs = torch.cat(advs, dim=0)
            active = torch.cat(active, dim=0)
            ref_output = concat_along_axis_tensor(ref_output, 0)

        return [obs, ac, ms, probs, vals, advs, active, ref_output, last_hidden]

    def place_inputs(self, obs, hidden):
        for key in obs:
            self.input_buffer[key][:, self.id_] = obs[key][:, 0]
        for i in range(len(hidden)):
            self.hidden_buffer[i][:, self.id_] = hidden[i] 

    def run_game(self):
        hidden = self.zero_hidden
        last_hidden = self.zero_hidden
        buffer = []
        obs = self.env.reset()
        total_rewards = 0
        for i in range(1000000):
            self.place_inputs(obs, hidden)
            self.queue.put((Request.process, self.id_))
            outputs, ref_output, hidden = self.pipe.recv()
            new_obs, reward, done, game_result, action = self.env.step(outputs)
            reward = reward * REWARD_SCALING
            probs = compute_action_prob(outputs, action[0], action[1])
            if len(buffer) == 32:
                t = self.generate_trajectory(buffer, last_hidden,
                                             outputs['value'])
                self.data_queue.put(t)
                buffer = []
                last_hidden = hidden

            buffer.append((obs, action, probs, outputs['value'], reward, ref_output))
            obs = new_obs
            total_rewards += reward
            if done:
                t = self.generate_trajectory(buffer, last_hidden,
                                             torch.zeros((1, 1, VALUE_DIM)))
                self.data_queue.put(t)
                self.score_queue.put((total_rewards, game_result))
                return


class Worker:
    def __init__(self, request_queue, zero_hidden, ref_replay, id_, env_type, data_queue,
                 score_queue):
        self.pipe, pipe2 = mp.Pipe()
        self.proc = mp.Process(target=Worker.run_games,
                               args=(pipe2, request_queue, env_type, id_,
                                     data_queue, score_queue, zero_hidden, ref_replay))
        self.proc.start()
        self.id = id_

    # obs, hidden
    def observe(self):
        return self.pipe.recv()

    def send(self, action, ref_output, hidden):
        self.pipe.send((action, ref_output, hidden))

    def join(self):
        self.proc.join()

    def send_buffers(self, obs, hiddens):
        return self.pipe.send((obs, hiddens))

    @staticmethod
    def run_games(pipe, queue, env_type, id_, data_queue, score_queue, hidden, ref_replay):
        w = WorkerImpl(pipe, queue, env_type, id_, data_queue, score_queue,
                        hidden, ref_replay)
        w.run_games()


default_test_groups = [[20, 8, Bot(Race.zerg, Difficulty.easy), "easy"],
                       [20, 8,
                        Bot(Race.zerg, Difficulty.medium), "medium"]]


class TrajectoryGenerator:
    def start_worker(self, i):
        return Worker(
            self.request_queue,
            self.zero_hidden,
            self.ref_replay,
            i,
            self.env_types[i%len(self.env_types)],
            self.data_queue,
            self.score_queue
        )

    def __init__(self,
                 model,
                 ref_model,
                 ref_replay,
                 env_types=[Bot(Race.random, Difficulty.easy),
                            Bot(Race.random, Difficulty.medium)],
                 num_envs=1,
                 max_batch=1):
        self.model = model
        self.ref_model = ref_model
        self.ref_replay = ref_replay
        self.data_queue = mp.Queue()
        self.score_queue = mp.Queue()
        self.env_types = env_types
        self.max_batch = max_batch
        with torch.no_grad():
            self.model.cuda()
            self.model.eval()
            self.ref_model.train()

            self.zero_hidden = model.get_hidden(1)

            self.envs_dict = {}
            self.request_queue = mp.Queue()

            self.rewards = 0
            for i in range(num_envs):
                self.envs_dict[i] = self.start_worker(i)

            obs, hiddens = [], []
            for i in range(num_envs):
                o, h = self.envs_dict[i].observe()
                obs.append(o)
                hiddens.append(h)

            self.obs_buffer = concat_along_axis_tensor(obs, 1)
            self.hidden_buffer = list(concat_lstm_hidden(hiddens))
            self.hidden_buffer2 = list(concat_lstm_hidden(hiddens))

            for k in self.obs_buffer:
                self.obs_buffer[k] = self.obs_buffer[k].clone(
                    memory_format=torch.contiguous_format)

            for i in range(len(self.hidden_buffer)):
                v = self.hidden_buffer[i].clone(
                    memory_format=torch.contiguous_format)
                self.hidden_buffer[i] = v.reshape(v.shape[0], num_envs, -1,
                                                  v.shape[2])

            for i in range(len(self.hidden_buffer2)):
                v = self.hidden_buffer2[i].clone(
                    memory_format=torch.contiguous_format)
                self.hidden_buffer2[i] = v.reshape(v.shape[0], num_envs, -1,
                                                  v.shape[2]).cuda()

            self.hidden_buffer = tuple(self.hidden_buffer)
            self.hidden_buffer2 = tuple(self.hidden_buffer2)

            for i in range(num_envs):
                self.envs_dict[i].send_buffers(self.obs_buffer,
                                               self.hidden_buffer)

    def gather_inputs(self, envs_list):
        envs = torch.tensor(envs_list, dtype=torch.int64)
        obs_dict = {}
        hidden, hidden2 = [], []

        for k in self.obs_buffer:
            obs_dict[k] = torch.index_select(self.obs_buffer[k], 1,
                                             envs).cuda()

        for i in range(len(self.hidden_buffer)):
            selected = torch.index_select(self.hidden_buffer[i], 1,
                                          envs).cuda()
            shape = selected.shape
            hidden.append(selected.reshape(shape[0], -1, shape[3]))

        for i in range(len(self.hidden_buffer2)):
            selected = torch.index_select(self.hidden_buffer2[i], 1,
                                          envs.cuda())
            shape = selected.shape
            hidden2.append(selected.reshape(shape[0], -1, shape[3]))

        return obs_dict, hidden, hidden2

    def step(self):
        with torch.no_grad():
            envs = []
            envs_ids = []
            reset_t("waiting")
            while len(envs) < self.max_batch:
                req = self.request_queue.get()
                if req[0] == Request.exception:
                    self.envs_dict[req[1]].join()
                    self.envs_dict[req[1]] = self.start_worker(req[1])
                    continue
                g = self.envs_dict[req[1]]
                envs.append(g)
                envs_ids.append(req[1])
            add_t("waiting")
            reset_t("gathering")
            inputs, hiddens, hiddens2 = self.gather_inputs(envs_ids)
            add_t("gathering")
            reset_t("running")
            with autocast():
                output, next_state = self.model(inputs, hiddens)
                target = {k.replace("_sampled",""):output[k] for k in output if "_sampled" in k}
                ref_output, next_state2 = self.ref_model(inputs, hiddens2, target)

            add_t("running")
            reset_t("sending")
            length = len(envs)

            for k in output:
                output[k] = output[k].cpu()
            
            for k in ref_output:
                ref_output[k] = ref_output[k].cpu()

            next_state = list(next_state)
            for i in range(len(next_state)):
                next_state[i] = next_state[i].cpu()
            
            for i, idx in enumerate(envs_ids):
                for buf, val in zip(self.hidden_buffer2, next_state2):
                    size = val.shape[1] // length
                    buf[:, idx] = val[:, i * size:i * size + size, :]
                
            for i, env in enumerate(envs):
                new_hidden = tuple()
                for val in next_state:
                    size = val.shape[1] // length
                    new_hidden += (val[:, i * size:i * size + size, :], )
                to_send = {}
                ref_to_send = {}
                for key in output:
                    to_send[key] = output[key][:, i:i + 1]
                
                for key in ref_output:
                    ref_to_send[key] = ref_output[key][:, i:i + 1]

                env.send(to_send, ref_to_send, new_hidden)
            add_t("sending")

    def generate_trajectories(self, min_traj=100):
        self.model.eval()
        trajectories = []

        while len(trajectories) < min_traj:
            reset_t("trajectories")
            while len(trajectories) < min_traj:
                try:
                    trajectories.append(self.data_queue.get(False))
                    print(len(trajectories))
                except Empty:
                    break
            add_t("trajectories")
            self.step()

        log_by_tag()
        clear_t()

        return trajectories

    def step_score_queue(self):
        try:
            return self.score_queue.get(False)
        except Empty:
            return None
