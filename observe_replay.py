from pysc2.lib import features
from absl import flags
from pysc2 import run_configs
from pysc2.env import sc2_env
from s2clientprotocol import sc2api_pb2 as sc_pb, common_pb2
import os
import sys
from replay_state import ReplayState
from observations import interface_format

screen_size = (64, 64)
minimap_size = (64, 64)

version = "4.9.3"

class WrongReplayException(Exception):
    pass

def valid_replay(info, ping):
    if (info.HasField("error") or info.base_build != ping.base_build or 
        info.game_duration_loops < 1000 or len(info.player_info) != 2):
        return False
    return True

def run_replay(replay_path, replay_state):

    run_config = run_configs.get(version=version)
    sc2_proc = run_config.start()
    controller = sc2_proc.controller
    replay_data = run_config.replay_data(replay_path)
    ping = controller.ping()

    try:
        info = controller.replay_info(replay_data)
    except Exception:
        sc2_proc.close()
        return False

    good = False
    for p in info.player_info:
        if p.player_info.race_actual == common_pb2.Race.Protoss and\
            p.player_mmr > 2400 and p.player_apm > 20:
            good = True
            player_id = p.player_info.player_id
            replay_state.mmr = p.player_mmr
            replay_state.result = p.player_result.result

    if not valid_replay(info, ping) or not good:
        sc2_proc.close()
        return False

    replay_state.save_global_info()
    interface = sc2_env.SC2Env._get_interface(interface_format, False)

    map_data = None
    if info.local_map_path:
        print(info)
        map_data = run_config.map_data(info.local_map_path)

    controller.start_replay(
        sc_pb.RequestStartReplay(
            replay_data=replay_data,
            map_data=map_data,
            options=interface,
            observed_player_id=player_id,
        ))

    fts = features.features_from_game_info(controller.game_info())

    while True:
        controller.step(1)
        obs = controller.observe()
        replay_state.step(obs, obs.actions, fts)
        if obs.player_result:
            break
    
    sc2_proc.close()
    return True


def get_data(seq_length, replay_path, compress=False, target_dir=None):

    FLAGS = flags.FLAGS
    print(sys.argv)
    FLAGS(sys.argv[:1])

    state = ReplayState(seq_length, compress, target_dir,
                          os.path.basename(replay_path))

    if run_replay(replay_path, state):
        while len(state.replay_data) > 0:
            state.get_batch()
        state.stats()
        if not compress:
            return state.send_data
        else:
            state.save_global_info()
    else:
        raise WrongReplayException()
