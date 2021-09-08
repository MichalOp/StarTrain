import torch
from torch.nn.functional import one_hot
import numpy as np
from pickle import loads, load
from zstd import decompress
import pysc2.lib.actions as ac
import pysc2.lib.features as ft
from collections import namedtuple
import os
from util import get_names


interface_format = ft.AgentInterfaceFormat(
    feature_dimensions=ft.Dimensions((64, 64),
                                     (64, 64)),
    rgb_dimensions=None,
    raw_resolution=None,
    action_space=ac.ActionSpace.FEATURES,
    camera_width_world_units=None,
    use_feature_units=False,
    use_raw_units=False,
    use_raw_actions=False,
    max_raw_actions=512,
    max_selected_units=30,
    use_unit_counts=False,
    use_camera_position=True,
    show_cloaked=False,
    show_burrowed_shadows=False,
    show_placeholders=False,
    hide_specific_actions=False,
    action_delay_fn=None,
)


_TRACKED_FEATURES = \
    {'build_queue': (0, 7),
     'cargo': (0, 7),
     'control_groups': (10, 2),
     'multi_select': (0, 7),
     'player': (11,),
     'production_queue': (0, 2),
     'single_select': (0, 7),
     'feature_screen': (27, 64, 64),
     'feature_minimap': (11, 64, 64),
     'game_loop': (1,),
     'available_actions': (573,)}

TARGET_COUNT = 32


def insert_tcount(t):
    out = []
    for x in t:
        if x == 0:
            out.append(TARGET_COUNT)
        else:
            out.append(x)
    return tuple(out)


TRACKED_FEATURES = {
    x: insert_tcount(_TRACKED_FEATURES[x])
    for x in _TRACKED_FEATURES
}

Feat = namedtuple("Feat", ["id", "scale"])


def get_categorical_scalar(features):
    categorical = []
    scalar = []

    for feat in features:
        if feat.type == ft.FeatureType.CATEGORICAL:
            categorical.append(Feat(feat.index, feat.scale))
        else:
            scalar.append(Feat(feat.index, feat.scale))

    return categorical, scalar


screen_categorical, screen_scalar = get_categorical_scalar(ft.SCREEN_FEATURES)
minimap_categorical, minimap_scalar = get_categorical_scalar(
    ft.MINIMAP_FEATURES)


def get_max_val(feats):
    m = 0
    for f in feats:
        m = max(m, f.scale)
    return m


def get_type(feats):
    x = get_max_val(feats)
    if x < 256:
        return np.uint8
    else:
        return np.int16


t_screen_cat = get_type(screen_categorical)
t_screen_sca = get_type(screen_scalar)
t_minimap_cat = get_type(minimap_categorical)
t_minimap_sca = get_type(minimap_scalar)


def get_features_by_type(input, feat_list, normalize=False):
    output = []

    for feat in feat_list:
        if normalize:
            norm = feat.scale
        else:
            norm = 1
        output.append(input[feat.id] / norm)

    return np.asarray(output)


def get_observation(obs):
    out_dict = {}
    for x in TRACKED_FEATURES:
        shape = TRACKED_FEATURES[x]
        if "screen" in x:
            screen = obs[x]
            sc = get_features_by_type(screen, screen_scalar)
            cat = get_features_by_type(screen, screen_categorical)

            sc = sc.reshape((1, 1) + sc.shape)
            cat = cat.reshape((1, 1) + cat.shape)

            out_dict[x + "_scalar"] = sc.astype(t_screen_sca)
            out_dict[x + "_categorical"] = cat.astype(t_screen_cat)
        elif "minimap" in x:
            minimap = obs[x]
            sc = get_features_by_type(minimap, minimap_scalar)
            cat = get_features_by_type(minimap, minimap_categorical)

            sc = sc.reshape((1, 1) + sc.shape)
            cat = cat.reshape((1, 1) + cat.shape)

            out_dict[x + "_scalar"] = sc.astype(t_minimap_sca)
            out_dict[x + "_categorical"] = cat.astype(t_minimap_cat)
        elif "available_actions" in x:
            thing = np.zeros((573, ), dtype=np.float32)
            for data in obs[x]:
                thing[data] = 1

            out_dict[x] = thing.reshape((1, 1, 573))
        elif "game_loop" in x:
            thing = np.zeros((1, ), dtype=np.float32)
            thing[0] = np.minimum(obs[x] / (22.4 * 60 * 20), 10)
            out_dict[x] = thing.reshape((1, 1, 1))
        else:
            out_dict[x] = obs[x].copy()
            if "player" in x:
                out_dict[x] = np.log(out_dict[x] + 1)

            out_dict[x].resize((1, 1) + shape)

    return out_dict


f_count = len(ac.FUNCTIONS)


def to_tensor(x, type=torch.float32):
    result = {}
    for field in x:
        if "feature" in field:
            result[field] = torch.tensor(x[field], dtype=None)
        else:
            result[field] = torch.tensor(x[field], dtype=type)
    return result


def to_cuda(x):
    for field in x:
        x[field] = x[field].cuda()
    return x


def concat_along_axis(x, axis):
    result = []
    swapped = zip(*x)
    for field in swapped:
        output = {}
        for entry in field[0]:
            output[entry] = np.concatenate([p[entry] for p in field],
                                           axis=axis)
        result.append(output)

    return tuple(result)


def concat_along_axis_tensor(x, axis):
    output = {}
    for entry in x[0]:
        output[entry] = torch.cat([p[entry] for p in x], axis=axis)

    return output


def load_global_stats(directory):
    with open(os.path.join(directory, "stats.pkl"), "rb") as f:
        stats = load(f)

    names = get_names(directory)
    half = (len(names) - 1) // 2
    with open(os.path.join(directory, str(half)), "rb") as f:
        batch = loads(decompress(load(f)))
        stats["control_groups"] = batch[0]["control_groups"][0:1, :, :, :]

    return stats


def screen_from_id(active, screen_pos):
    screen = one_hot(torch.zeros(1, 1).long(), num_classes=64 * 64)
    screen = screen.short()
    screen = screen * active
    screen = screen.reshape(screen_pos.shape + (1, 64, 64))
    return screen


def shift_last_action(actions, last_action, length):

    new_last_action = {}
    new_actions = {}

    for key in last_action:
        data = actions[key]
        new_last_action[key] = data[length - 1:length]
        new_actions[key] = torch.cat([last_action[key], data[:length - 1]],
                                     dim=0)

    return new_actions, new_last_action


def extend_batch(batch, stats, batch_size, prev_action, visible):
    mmr = np.zeros((6, ))
    mmr_val = min(max(int(stats['mmr'] - 2500) // 1000, 0), 5)
    # print(mmr_val, stats['mmr'])
    mmr[mmr_val] = 1
    build_order = np.zeros(20, dtype=np.long)

    if visible:
        for i, x in enumerate(stats['build']):
            build_order[i] = int(x)

    mmr = np.resize(mmr, (batch_size, 1, 6))
    build_order = np.resize(build_order, (batch_size, 1, 20))

    uses_screen = np.zeros((batch_size, 1, 1), dtype=np.int16)
    uses_screen2 = np.zeros((batch_size, 1, 1), dtype=np.int16)
    uses_minimap = np.zeros((batch_size, 1, 1), dtype=np.int16)
    functions = prev_action['function'].reshape((-1, )).numpy()
    for i in range(batch_size):
        inputs = ac.FUNCTION_TYPES[ac.FUNCTIONS[functions[i]].function_type]
        if ac.TYPES.screen in inputs:
            uses_screen[i] = 1
        if ac.TYPES.minimap in inputs:
            uses_minimap[i] = 1
        if ac.TYPES.screen2 in inputs:
            uses_screen2[i] = 1
    uses_screen = torch.tensor(uses_screen)
    uses_minimap = torch.tensor(uses_minimap)
    uses_screen2 = torch.tensor(uses_screen2)
    screen = screen_from_id(uses_screen, prev_action[str(ac.TYPES.screen)])
    screen2 = screen_from_id(uses_screen2, prev_action[str(ac.TYPES.screen2)])
    minimap = screen_from_id(uses_minimap, prev_action[str(ac.TYPES.minimap)])

    if visible:
        batch[0]['control_groups_hint'] = stats["control_groups"]
    else:
        batch[0]['control_groups_hint'] = torch.zeros(
            stats["control_groups"].shape)
    batch[0]['feature_screen_categorical'] = torch.cat(
        [batch[0]['feature_screen_categorical'], screen, screen2], 2)
    batch[0]['feature_minimap_categorical'] = torch.cat(
        [batch[0]['feature_minimap_categorical'], minimap], 2)

    batch[0]['prev_action'] = prev_action['function']
    batch[0]['mmr'] = torch.tensor(mmr, dtype=torch.float32)
    batch[0]['build_order'] = torch.tensor(build_order, dtype=torch.int64)
    return batch


def concat_lstm_hidden(x):
    result = tuple()
    swapped = zip(*x)
    for field in swapped:
        output = torch.cat(field, axis=1)
        result = result + (output, )

    return result


def generate_zeroed_dicts():
    mask_dict = {}
    result_dict = {}

    result_dict["function"] = np.zeros((1, 1), dtype=np.int32)
    mask_dict["function"] = np.zeros((1, 1))
    result_dict["time_skip"] = np.zeros((1, 1), dtype=np.int32)
    mask_dict["time_skip"] = np.zeros((1, 1))

    for t in ac.TYPES:
        x = str(t)
        result_dict[x] = np.zeros((1, 1), dtype=np.int32)
        mask_dict[x] = np.zeros((1, 1))

    return result_dict, mask_dict


def extract_action(fun_call):
    if fun_call is None:
        fun_id = 0
    else:
        fun_id = fun_call.function

    rd, md = generate_zeroed_dicts()

    rd["function"][0, 0] = fun_id
    md["function"][0, 0] = 1.0
    md["time_skip"][0, 0] = 1.0
    rd["time_skip"][0, 0] = 0
    if fun_id == 0:
        return rd, md

    for data, argt in zip(
            fun_call.arguments,
            ac.FUNCTION_TYPES[ac.FUNCTIONS[int(fun_id)].function_type]):
        a_id = data[0]
        shape = argt.sizes
        if len(shape) > 1:
            a_id = data[0] * 64 + data[1]

        name = str(argt)
        rd[name][0, 0] = a_id
        md[name][0, 0] = 1.0

    return rd, md


def get_skip(data):
    return data[1]["time_skip"][0, 0]


def set_skip(data, value):
    data[1]["time_skip"][0, 0] = value
