import os
from model import Model, compute_action_prob
import pysc2.lib.actions as ac
import pysc2.lib.features as ft
from pysc2.env.sc2_env import SC2Env, Agent, Race, Bot, Difficulty
from pysc2.env.lan_sc2_env import LanSC2Env
from pysc2.env.environment import StepType
from observations import (get_observation, to_tensor, to_cuda,
                          load_global_stats, extend_batch,
                          concat_along_axis_tensor, concat_lstm_hidden,
                          interface_format)
import torch
import torch.nn.functional
from enum import Enum
import torch.multiprocessing as mp
# from torch.distributions import Categorical
from absl import flags
from time import time
import sys
from copy import deepcopy
from torch.cuda.amp import autocast
from game_env import SC2EnvWrapper
import argparse
from util import Request

FLAGS = flags.FLAGS
FLAGS(sys.argv[:1])
PATH = "model/best_saved.tm"
VISUALIZE = False

def model_load_from_checkpoint(model_path):
    model = Model()
    checkpoint = torch.load(model_path)
    if any(["value" in k for k in checkpoint['model_state_dict']]):
        model.add_value()
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def get_stats(stats_replay):
    stats = load_global_stats(stats_replay)
    stats['mmr'] = 6000
    # stats['build'] = [0]*20
    return stats


class Game:
    def __init__(self,
                 games_count,
                 request_queue,
                 zero_hidden,
                 id_,
                 ref_replay_path,
                 group_name="",
                 bot=Bot(Race.zerg, Difficulty.easy),
                 data_collection_queue=None):
        self.bot = bot
        self.pipe, pipe2 = mp.Pipe()
        self.proc = mp.Process(target=Game.run_games,
                               args=(pipe2, request_queue, games_count, bot,
                                     id_, ref_replay_path))
        self.proc.start()
        self.id = id_
        self.group_name = group_name
        self.zero_hidden = zero_hidden
        self.hidden = zero_hidden

    def observe(self):
        reward, done = self.pipe.recv()
        if done:
            shutdown = self.pipe.recv()
            self.hidden = self.zero_hidden
        else:
            shutdown = False
        if shutdown:
            return None, None, reward, done, shutdown
        else:
            return self.pipe.recv(), self.hidden, reward, done, shutdown

    def send(self, action, hidden):
        self.hidden = hidden
        self.pipe.send(action)

    def join(self):
        self.proc.join()

    @staticmethod
    def run_games(pipe, queue, games_count, bot_type, id_, ref_replay_path):

        stats = get_stats(ref_replay_path)

        torch.set_num_threads(1)
        try:
            env = SC2Env(
                map_name="Acropolis",
                players=[Agent(Race.protoss, "StarTrain"), bot_type],
                agent_interface_format=interface_format,
                visualize=VISUALIZE,
                step_mul=1,
                ensure_available_actions=False,
                realtime=False,
                game_steps_per_episode=0,
                save_replay_episodes=1,
                score_index=-1,
                replay_dir=os.path.abspath("showcase_replays")
            )
            wrapper = SC2EnvWrapper(env, stats, 60)
            pipe.send((0, True))
            while True:
                if games_count.acquire(block=False):
                    pipe.send(False)
                    Game.run_game(wrapper, pipe, queue, id_, game_id=id_)
                else:
                    pipe.send(True)
                    queue.put((Request.process, id_))
                    break
        except Exception as e:
            print(e)
            queue.put((Request.exception, id_))

    @staticmethod
    def run_game(env, pipe, queue, id_, timeout=60, game_id=0):
        obs = env.reset()
        for i in range(1000000):
            queue.put((Request.process, id_))
            pipe.send(obs)
            outputs = pipe.recv()
            new_obs, reward, done, game_result, action = env.step(outputs)
            obs = new_obs

            if done:
                pipe.send((1 if game_result > 0 else 0, True))
                return

            pipe.send((0, False))


def run_game(env, model, stats):
    with torch.no_grad():

        hidden = model.get_hidden(1)
        hidden = tuple([hid.cuda() for hid in hidden])
        scheduled = [ac.FunctionCall(0, [])]
        obs = env.reset()

        skip_target = 0
        time_step = obs[0]
        t0 = time()

        last_action = {
            'function': torch.tensor([[0]]).long(),
            str(ac.TYPES.screen): torch.tensor([[0]]).long(),
            str(ac.TYPES.screen2): torch.tensor([[0]]).long(),
            str(ac.TYPES.minimap): torch.tensor([[0]]).long()
        }

        for i in range(1000000):
            if i % (22 * 60) == 0:
                print("minute ", i // 22 // 60,
                      f"x{i /(time()-t0)/ 22.4} realtime")

            if skip_target > time_step.observation["game_loop"][0]:
                obs = env.step([ac.FunctionCall(0, [])])

                time_step = obs[0]

                if time_step.step_type == StepType.LAST:
                    print("GAME END", time_step.reward)
                    if time_step.reward > 0:
                        return 1
                    break

                continue

            obs = env.step(scheduled)

            time_step = obs[0]

            if time_step.step_type == StepType.LAST:
                print("GAME END", time_step.reward)
                if time_step.reward > 0:
                    return 1
                break

            inputs = get_observation(time_step.observation)
            inputs = to_tensor(inputs)

            inputs = extend_batch((inputs, ), stats, 1, last_action, True)[0]
            inputs = to_cuda(inputs)
            outputs, next_hidden = model(inputs, hidden)

            hidden = tuple([next_h.detach() for next_h in next_hidden])

            action = outputs["function_sampled"].squeeze().item()
            skip_target = max(
                outputs["time_skip_sampled"].squeeze().item() - 1,
                0) + time_step.observation["game_loop"][0]

            func = ac.FUNCTIONS[action]
            action_data = []
            last_action = {}
            for x in outputs:
                last_action[x.replace("_sampled",
                                      "")] = outputs[x].detach().cpu()

            for x in ac.FUNCTION_TYPES[func.function_type]:
                sub_action = outputs[str(x) + "_sampled"].squeeze().item()

                if "screen" in str(x) or "minimap" in str(x):
                    sub_action = (sub_action // 64, sub_action % 64)
                else:
                    sub_action = (sub_action, )
                action_data.append(sub_action)

            scheduled = [ac.FunctionCall(action, action_data)]

    return 0


default_test_groups = [[20, 8, Bot(Race.zerg, Difficulty.easy), "easy"],
                       [20, 8,
                        Bot(Race.zerg, Difficulty.medium), "medium"]]


def launch_games2(model,
                  ref_replay_path,
                  response_pipe=None,
                  test_groups=default_test_groups,
                  max_batch=8,
                  data_queue=None,
                  ):

    with torch.no_grad():
        model.cuda()
        model.eval()

        zero_hidden = model.get_hidden(1)
        zero_hidden = tuple([hid.cuda() for hid in zero_hidden])

        envs_dict = {}
        worker_id = 0
        rewards = {}
        sems = []
        request_queue = mp.Queue()

        for games, workers, opponent, group_name in test_groups:
            rewards[group_name] = 0
            games_counter = mp.Semaphore(games)
            sems.append(games_counter)
            for i in range(workers):
                envs_dict[worker_id] = Game(games_counter, request_queue,
                                            zero_hidden, worker_id,
                                            ref_replay_path,
                                            group_name, opponent)
                worker_id += 1

        envs_left = worker_id
        while True:

            envs = []
            observations = []
            states = []

            while len(envs) < min(max_batch, envs_left):
                req = request_queue.get()
                if req[0] == Request.exception:
                    envs_left -= 1
                    continue
                g = envs_dict[req[1]]
                obs, s, reward, reset, stop = g.observe()
                rewards[g.group_name] += reward
                if reward > 0:
                    print(rewards)
                if not stop:
                    envs.append(g)
                    states.append(s)
                    observations.append(to_cuda(obs))
                else:
                    envs_left -= 1

            if envs_left == 0:
                break

            inputs = concat_along_axis_tensor(observations, 1)
            hiddens = concat_lstm_hidden(states)
            with autocast():
                output, next_state = model(inputs, hiddens)

            states = []
            length = len(envs)
            for i, env in enumerate(envs):
                new_hidden = tuple()
                for val in next_state:
                    size = val.shape[1] // length
                    new_hidden += (val[:, i * size:i * size + size, :], )
                to_send = {}
                for key in output:
                    to_send[key] = output[key][:, i:i + 1].cpu()

                env.send(to_send, new_hidden)

        print("okay")
        if response_pipe is not None:
            response_pipe.send(rewards)
            print(response_pipe.recv())

        for id_ in envs_dict.keys():
            envs_dict[id_].join()


class Evaluator:
    def __init__(self, reference_replay, best_score=0):
        self.reference_replay = reference_replay
        self.running = False
        self.ctx = mp.get_context("spawn")
        self.best_score = best_score

    def start_eval(self, model, turn, test_groups=default_test_groups):
        model = deepcopy(model)
        model.cpu()
        self.stored_model = model
        self.pipe, pipe2 = self.ctx.Pipe()
        self.turn = turn
        self.proc = self.ctx.Process(target=launch_games2,
                                     args=(
                                         model,
                                         self.reference_replay,
                                         pipe2,
                                         test_groups))
        self.proc.start()
        self.running = True

    def try_get_results(self, block=0, save=True):
        if self.running and self.pipe.poll(block):
            self.running = False
            val = self.pipe.recv()
            self.pipe.send("quit pls")
            self.proc.join()

            score = (val["easy"] + val["medium"] * 2)

            if score > self.best_score and save:
                self.best_score = score
                torch.save(
                    {'model_state_dict': self.stored_model.state_dict()},
                    "model/best.tm")

            return val, self.turn
        else:
            return None


def run_lan_game(model_path, reference_replay, host, config_port):
    stats = get_stats(reference_replay)
    torch.set_num_threads(1)
    model = model_load_from_checkpoint(model_path)
    model.cuda()
    model.eval()
    env = LanSC2Env(
        host=host,
        config_port=int(config_port),
        race=Race.protoss,
        name="StarTrain",
        agent_interface_format=interface,
        visualize=False,
        step_mul=1,
        realtime=True,
        replay_dir=os.path.abspath("showcase_replays"))

    wins = 0

    while True:
        wins += run_game(env, model, stats)
        print("wins:", wins)



def evaluate(model_path, reference_replay, race):
    model = model_load_from_checkpoint(model_path)
    mp.set_start_method("spawn")
    test_groups = [[100, 8, Bot(race, Difficulty.very_easy), "very_easy"],
                   [100, 8, Bot(race, Difficulty.easy), "easy"],
                   [100, 8, Bot(race, Difficulty.medium), "medium"],
                   [100, 8, Bot(race, Difficulty.hard), "hard"]]
    ev = Evaluator(reference_replay)
    ev.start_eval(model, 0, test_groups)
    print(ev.try_get_results(block=None, save=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="A path to the neural network model to play the game")
    parser.add_argument("mode", help="The mode to run the games in, 'BATCH' - run games against bots in large number or 'LAN' - connect to lan game against a human")
    parser.add_argument("reference_replay", help="A single processed replay that will be used in evaluation")
    parser.add_argument("-r", "--race", help="Race of the bots in evaluation to play against (if BATCH selected). Random if not specified.")
    parser.add_argument("-a", "--address", help="IP address of the LAN game to join, default: 127.0.0.1. (if LAN game selected)")
    parser.add_argument("-p", "--port", help="Listening port on which to try to connect to the LAN game, default: 14380.(if LAN game selected)")
    args = parser.parse_args()

    race = Race.random
    race_d = {"terran":Race.terran, "protoss":Race.protoss, "zerg":Race.zerg, "random":Race.random}
    if args.race:
        race = race_d[args.race]
    #run_lan_game()
    if args.mode == 'BATCH':
        evaluate(args.model_path, args.reference_replay, race)
    elif args.mode == 'LAN':
        host = "127.0.0.1"
        config_port = 14380 # pysc2 default
        if args.address:
            host = args.address
        if args.port:
            config_port = args.config_port
        run_lan_game(args.model_path, args.reference_replay, host, config_port)

