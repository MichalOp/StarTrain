import pysc2.lib.actions as ac
import pysc2.lib.static_data as sd
import torch
import torch.nn as nn
from torch.distributions import Categorical
from observations import (screen_categorical, screen_scalar,
                          minimap_categorical, minimap_scalar, Feat)
from game_env import VALUE_DIM, value_names

INPUT_LAYERS_SPATIAL = 2
OUTPUT_LAYERS_SPATIAL = 3

SPATIAL_EMBEDDING = 16

SPATIAL_FILTERS = 32
REDUCE_LAYERS_SPATIAL = 6
AGGREGATOR_LAYER = 64
SIMPLE_LAYER = 64
LSTM_INTERNAL = 512
LSTM_LAYERS = 2
LSTM_SIZE = 1024
SPATIAL_LSTM_SUPPLEMENT = 32
DISCRETE_OUTPUT_COUNT = 128
ACTION_INTERNAL_SIZE = 256
UNIT_EMBEDDING = 10

BATCH_SIZE = 16

MAX_NOPS = 128

TRACKED_FEATURES = \
  {'build_queue': (0, 7),          # ok
   'cargo': (0, 7),                # ok
   'control_groups': (10, 2),      # ok
   'multi_select': (0, 7),         # ok
   'player': (11,),                # ok?
   'production_queue': (0, 2),     # ok
   'single_select': (0, 7),        # ok
   'feature_screen': (27, 64, 64),  # ok
   'feature_minimap': (11, 64, 64),  # ok
   'game_loop': (1,),              # not really
   'available_actions': (573,),    # ok
   'prev_action': (1,),            # ok?
   'build_order': (20,),           # ok
   'mmr': (6,)}                    # ok

AUTOREGRESSIVE_CHAIN = ["function", "time_skip", str(ac.TYPES.queued)]

DEBUG = False


def debug(*x):
    if DEBUG:
        print(*x)


class GLU(nn.Module):
    def __init__(self, input_size, gating_size, output_size):
        super().__init__()
        self.gate = nn.Linear(gating_size, input_size)
        self.lin = nn.Linear(input_size, output_size)

    def forward(self, x, gating):
        g = torch.sigmoid(self.gate(gating))
        return self.lin(g * x)


class ResidualBlock(nn.Module):
    def __init__(self, in_layers, size):
        super().__init__()
        self.c1 = nn.Sequential(
            nn.Conv2d(in_layers, in_layers, (3, 3), padding=1), nn.ELU())
        self.n1 = nn.LayerNorm((in_layers, size, size))
        self.c2 = nn.Sequential(
            nn.Conv2d(in_layers, in_layers, (3, 3), padding=1), nn.ELU())
        self.n2 = nn.LayerNorm((in_layers, size, size))

    def forward(self, x):
        old_x = x
        x = self.n1(x)
        x = self.c1(x)
        x = self.n2(x)
        x = self.c2(x)

        return x + old_x


class FiLM(nn.Module):
    def __init__(self, output_size, gating_size):
        super().__init__()
        self.scale = nn.Linear(gating_size, output_size[0])
        self.shift = nn.Linear(gating_size, output_size[0])

    def forward(self, x, gating):
        scale = self.scale(gating).unsqueeze(-1).unsqueeze(-1)
        shift = self.shift(gating).unsqueeze(-1).unsqueeze(-1)
        return scale * x + shift


class ResidualFiLMBlock(nn.Module):
    def __init__(self, in_layers, size, gating_size):
        super().__init__()
        self.c1 = nn.Conv2d(in_layers, in_layers, (3, 3), padding=1)
        self.n1 = nn.LayerNorm((in_layers, size, size))
        self.c2 = nn.Conv2d(in_layers, in_layers, (3, 3), padding=1)
        self.n2 = nn.LayerNorm((in_layers, size, size))
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.film = FiLM((in_layers, size, size), gating_size)

    def forward(self, x, gating):
        old_x = x
        x = self.n1(x)
        x = self.c1(x)
        x = self.relu1(x)
        x = self.c2(x)
        x = self.n2(x)
        x = self.film(x, gating)
        x = self.relu2(x)

        return x + old_x


class ResidualDenseBlock(nn.Module):
    def __init__(self, in_layers):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(in_layers, in_layers), nn.ELU())
        self.n1 = nn.LayerNorm((in_layers, ))
        self.l2 = nn.Sequential(nn.Linear(in_layers, in_layers), nn.ELU())
        self.n2 = nn.LayerNorm((in_layers, ))

    def forward(self, x):
        old_x = x
        x = self.n1(x)
        x = self.l1(x)
        x = self.n2(x)
        x = self.l2(x)

        return x + old_x


class CategoryEmbedder(nn.Module):
    def __init__(self, categories):
        super().__init__()

        self.category_embeddings = []
        one_dim = 0
        high_dim = 0
        for i, cat in enumerate(categories):
            if cat.scale == 2:
                em = None
                one_dim += 1
            else:
                high_dim += 1
                em = nn.Embedding(cat.scale, SPATIAL_EMBEDDING)
                self.add_module(f"cat {i}", em)

            self.category_embeddings.append(em)

        self.output_shape = one_dim + SPATIAL_EMBEDDING
        self.relu = nn.ELU()

    def forward(self, inputs):
        float_inputs = inputs.float()
        unbound = inputs.unbind(
            dim=1) 
        f_unbound = float_inputs.unbind(dim=1)
        result = []
        two_value = []
        for u, f, em in zip(unbound, f_unbound, self.category_embeddings):
            if em is None:
                two_value.append(f.unsqueeze(1))
            else:
                result.append(
                    em(u).permute(0, 3, 1, 2).unsqueeze(0)
                )

        reduced = torch.cat(result, dim=0).sum(dim=0)
        extended = torch.cat([reduced] + two_value, dim=1)

        return extended


SCALAR_LAYER = 512


class SpatialInput(nn.Module):
    def __init__(self, scalar_input, categorical_input, name="in"):
        super().__init__()
        self.name = name
        self.input_column = []

        self.cat_embedder = CategoryEmbedder(categorical_input)
        scales = []
        for x in scalar_input:
            scales.append(x.scale - 1)

        self.scales = torch.tensor(scales).unsqueeze(0).unsqueeze(0).unsqueeze(
            -1).unsqueeze(-1).cuda()

        self.reshaper = nn.Sequential(
            nn.Conv2d(len(scalar_input) + self.cat_embedder.output_shape,
                      SPATIAL_FILTERS, (1, 1),
                      padding=0), nn.ELU())

        self.reduce1 = nn.Sequential(
            nn.Conv2d(SPATIAL_FILTERS, 64, (4, 4), stride=2, padding=1),
            nn.ELU())  # 32x32
        self.reduce2 = nn.Sequential(
            nn.Conv2d(64, 128, (4, 4), stride=2, padding=1), nn.ELU())  # 16x16
        self.reduce3 = nn.Sequential(
            nn.Conv2d(128, 128, (4, 4), stride=2, padding=1), nn.ELU())  # 8x8

        filters = 128
        for i in range(INPUT_LAYERS_SPATIAL):
            self.input_column.append(
                ResidualFiLMBlock(filters, 8, SCALAR_LAYER))

        self.layer_norm_out = nn.LayerNorm((128, 8, 8))

        for i, mod in enumerate(self.input_column):
            self.add_module(f"layer {i}", mod)

    def forward(self, scalar, categorical, gating):

        shape = scalar.shape
        shape2 = categorical.shape
        scalar = scalar / self.scales
        scalar = scalar.view((shape[0] * shape[1], ) + shape[2:])
        categorical = categorical.view((shape2[0] * shape2[1], ) + shape2[2:])
        categorical = self.cat_embedder(categorical)
        bypass = []

        x = torch.cat([scalar, categorical], dim=1)
        x = self.reshaper(x)
        bypass.append(x)
        x = self.reduce1(x)
        bypass.append(x)
        x = self.reduce2(x)
        bypass.append(x)
        x = self.reduce3(x)
        gating = gating.view(-1, SCALAR_LAYER)
        i = 0
        for layer in self.input_column:
            x = layer(x, gating)
            i += 1

        return x.view(shape[:2] + (128, 8, 8)), bypass


class ListInput(nn.Module):
    def __init__(self, in_shape, embedder_size, name="in"):
        super().__init__()
        self.name = name
        self.encoder = nn.Linear(in_shape[1] + UNIT_EMBEDDING - 1,
                                 AGGREGATOR_LAYER)
        self.unit_embedder = nn.Embedding(embedder_size, UNIT_EMBEDDING)
        self.layer_norm = nn.LayerNorm((AGGREGATOR_LAYER, ))
        self.modifier = nn.Sequential(
            nn.Linear(AGGREGATOR_LAYER, AGGREGATOR_LAYER), nn.ELU())

    def forward(self, x, norm=True):
        embedded = self.unit_embedder(x[:, :, :, 0].long())

        x = torch.cat([embedded, x[:, :, :, 1:].float()], dim=-1)
        x = self.encoder(x)
        v = x.max(dim=-2, keepdim=False).values

        x = self.modifier(v)
        if norm:
            x = self.layer_norm(x)
        return x


CONTROL_GROUP_SIZE = 2 * UNIT_EMBEDDING + 1


class ControlGroupsInput(nn.Module):
    def __init__(self, name="in"):
        super().__init__()
        self.name = name
        self.layer_size = CONTROL_GROUP_SIZE
        self.unit_embedder = nn.Embedding(
            max(sd.UNIT_TYPES) + 1, UNIT_EMBEDDING)
        self.layer_norm = nn.LayerNorm((SIMPLE_LAYER, ))
        self.layer = nn.Sequential(
            nn.Linear(self.layer_size * 10, SIMPLE_LAYER), nn.ReLU())

    def forward(self, x, hint, norm=True):
        embedded = self.unit_embedder(x[:, :, :, 0].long())
        embedded_hint = self.unit_embedder(hint[:, :, :, 0].long())
        embedded_hint = embedded_hint.repeat(
            embedded.shape[0] // embedded_hint.shape[0], 1, 1, 1)
        x = torch.cat([embedded, x[:, :, :, 1:].float(), embedded_hint],
                      dim=-1)
        bypass = x
        s = x.shape
        x = self.layer(x.view(s[:2] + (10 * self.layer_size, )))

        if norm:
            x = self.layer_norm(x)
        debug(self.name, x.std().item())
        return x, bypass


class SimpleInput(nn.Module):
    def __init__(self, in_shape, name="in"):
        super().__init__()
        self.name = name
        self.cut = len(in_shape)
        self.size = 1
        self.layer_norm = nn.LayerNorm((SIMPLE_LAYER, ))
        for x in in_shape:
            self.size *= x
        self.layer = nn.Sequential(nn.Linear(self.size, SIMPLE_LAYER),
                                   nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(SIMPLE_LAYER, SIMPLE_LAYER),
                                    nn.ReLU())

    def forward(self, x, norm=True):
        s = x.shape[:-self.cut] + (self.size, )
        x = self.layer(x.view(s).float())
        x = self.layer2(x)
        if norm:
            x = self.layer_norm(x)
        debug(self.name, x.std().item())
        return x


class RelationalCore(nn.Module):
    def __init__(self):
        super().__init__()

        self.reduction1 = nn.Sequential(
            nn.Conv2d(128, 32, (4, 4), stride=2, padding=1), nn.ELU(),
            nn.LayerNorm((32, 4, 4)))
        self.reduction2 = nn.Sequential(
            nn.Conv2d(128, 32, (4, 4), stride=2, padding=1), nn.ELU(),
            nn.LayerNorm((32, 4, 4)))
        self.layer_norm = nn.LayerNorm(128)
        self.transform1 = nn.TransformerEncoderLayer(128, 4, 512, dropout=0.0)
        self.lstm = nn.LSTM(128, 128, 1)
        self.transform2 = nn.TransformerEncoderLayer(128, 4, 512, dropout=0.0)
        self.transform3 = nn.TransformerEncoderLayer(128, 4, 512, dropout=0.0)

    def forward(self, screen, minimap, state):
        shape = screen.shape

        screen = screen.transpose(2, 4).reshape(shape[0] * shape[1], -1,
                                                shape[2])
        minimap = minimap.transpose(2, 4).reshape(shape[0] * shape[1], -1,
                                                  shape[2])

        x = torch.cat([screen, minimap], dim=1)
        x = self.layer_norm(x)
        x = x.transpose(0, 1)
        x = self.transform1(x)

        x = x.transpose(0, 1)

        lstm_in = x.reshape(shape[0], shape[1] * shape[3] * shape[4] * 2,
                            shape[2])

        x, nextstate = self.lstm(lstm_in, state)

        x = x + lstm_in

        x = x.reshape(shape[0] * shape[1], -1, shape[2])
        x = self.layer_norm(x)
        x = x.transpose(0, 1)

        x = self.transform2(x)
        x = self.transform3(x)
        x = x.transpose(0, 1)
        after = x.reshape(shape[0] * shape[1], 2, shape[3], shape[4],
                          shape[2]).transpose(2, 4)
        screenout = after[:, 0]
        mapout = after[:, 1]
        reduced1 = self.reduction1(screenout).view(shape[0], shape[1], -1)
        reduced2 = self.reduction2(mapout).view(shape[0], shape[1], -1)
        out_reduced = torch.cat([reduced1, reduced2], dim=2)

        return screenout.view(shape), mapout.view(
            shape), out_reduced, nextstate


class SpatialOutput(nn.Module):
    def __init__(self):
        super().__init__()
        self.output_column = []
        self.deconv_layers = []

        self.reencode = nn.Conv2d(128 + 16, 128, kernel_size=(1, 1))

        for i in range(OUTPUT_LAYERS_SPATIAL):
            self.output_column.append(ResidualFiLMBlock(128, 8, LSTM_SIZE))
            self.add_module(f"layer {i}", self.output_column[i])

        self.deconv_layer1 = nn.Sequential(
            nn.ConvTranspose2d(128,
                               128,
                               kernel_size=(4, 4),
                               stride=2,
                               padding=1), nn.ELU())
        self.deconv_layer2 = nn.Sequential(
            nn.LayerNorm((128, 16, 16)),
            nn.ConvTranspose2d(128,
                               64,
                               kernel_size=(4, 4),
                               stride=2,
                               padding=1), nn.ELU())

        self.deconv_layer3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=2,
                               padding=1))
        self.output_layer = nn.Sequential(
            nn.ELU(), nn.Conv2d(32, 1, kernel_size=(1, 1), stride=1,
                                padding=0))

    def forward(self, spatial_in, lstm_in):
        spatial_in, bypass = spatial_in
        shape = spatial_in.shape
        lstm_in_shredded = lstm_in.view((-1, 16, 8, 8))

        spatial_in = spatial_in.view((shape[0] * shape[1], ) + shape[2:])

        shape2 = lstm_in.shape
        lstm_in = lstm_in.view((shape2[0] * shape2[1], ) + shape2[2:])
        x = torch.cat((spatial_in, lstm_in_shredded), dim=1)
        x = self.reencode(x)

        debug(x.shape)

        i = 0
        for layer in self.output_column:
            x = layer(x, lstm_in)
            i += 1

        x = self.deconv_layer1(x)
        x += bypass[2]
        x = self.deconv_layer2(x)
        x += bypass[1]
        x = self.deconv_layer3(x)
        x += bypass[0]
        x = self.output_layer(x)

        return x.view(shape[:-3] + (64 * 64, ))


class DiscreteOutput(nn.Module):
    def __init__(self, count, addition=0, name="out"):
        super().__init__()
        self.name = name
        self.l1 = nn.Sequential(
            nn.Linear(LSTM_SIZE + addition, DISCRETE_OUTPUT_COUNT), nn.ELU())
        self.l2 = nn.Linear(DISCRETE_OUTPUT_COUNT, count)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        return x


CONTROL_GROUPS_SELECT = 10
CONTROL_GROUPS_ACT = 5


class ControlGroupsOutput(nn.Module):
    def __init__(self, addition=0, name="out"):
        super().__init__()
        self.name = name
        self.l1 = nn.Sequential(
            nn.Linear(LSTM_SIZE + addition,
                      DISCRETE_OUTPUT_COUNT * 2), nn.ELU()
        )  # doubled because this is effectively two layers merged into one
        self.l2 = nn.Linear(DISCRETE_OUTPUT_COUNT * 2,
                            CONTROL_GROUPS_SELECT + 4)
        self.encoder = nn.Linear(CONTROL_GROUP_SIZE, CONTROL_GROUPS_SELECT)
        self.position_encoding = nn.Parameter(torch.randn(1, 1, 10, 4))
        self.action_output = nn.Linear(
            CONTROL_GROUP_SIZE + DISCRETE_OUTPUT_COUNT * 2, CONTROL_GROUPS_ACT)

    def forward(self, x, control_groups_data):
        x = self.l1(x)
        select = self.l2(x)

        values = self.encoder(control_groups_data)
        values = torch.cat(
            [values,
             self.position_encoding.repeat(values.shape[:2] + (1, 1))],
            dim=-1)
        attention = values @ select.unsqueeze(-1)
        selected = nn.functional.softmax(attention,
                                         dim=-2).detach() * control_groups_data
        selected = selected.sum(-2)
        attention = attention.view(attention.shape[:-1])

        act_in = torch.cat([x, selected], dim=-1)
        act_out = self.action_output(act_in)
        return attention, act_out


class ActionOutput(nn.Module):
    def __init__(self, input_size, count, gating_size):
        super().__init__()
        self.in_layer = nn.Linear(input_size, ACTION_INTERNAL_SIZE)

        self.layers = []

        for i in range(4):
            self.layers.append(ResidualDenseBlock(ACTION_INTERNAL_SIZE))
            self.add_module(f"layer {i}", self.layers[i])

        self.output = GLU(ACTION_INTERNAL_SIZE, gating_size, count)

    def forward(self, x, gating):
        x = self.in_layer(x)

        for layer in self.layers:
            x = layer(x)

        return self.output(x, gating)


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self._cuda = False
        self.training = True
        self.input_columns = {}

        scalar_input_size = 0

        # spatial input extended with previous actions
        in_type = {
            "feature_screen":
            (screen_scalar,
             screen_categorical + [Feat(123, 2), Feat(124, 2)]),
            "feature_minimap":
            (minimap_scalar, minimap_categorical + [Feat(125, 2)])
        }

        for name in TRACKED_FEATURES:
            shape = TRACKED_FEATURES[name]
            if "screen" in name or "minimap" in name:
                continue
            elif "build_order" in name:
                self.input_columns[name] = nn.Sequential(
                    nn.Embedding(max(sd.UNIT_TYPES) + 1, 10),
                    SimpleInput((20, 10), name))
                scalar_input_size += SIMPLE_LAYER
            elif "control_group" in name:
                self.input_columns[name] = ControlGroupsInput(name)
                scalar_input_size += SIMPLE_LAYER

            elif "prev_action" in name:
                self.input_columns[name] = nn.Sequential(
                    nn.Embedding(573, 16), SimpleInput((16, ), name))
                scalar_input_size += SIMPLE_LAYER

            elif len(shape) == 2 and shape[-2] == 0:  # one type of list inputs

                if shape[-1] == 2:  # production queue
                    embedder_len = max(ac.ABILITY_IDS) + 1
                else:  # unit list
                    embedder_len = max(sd.UNIT_TYPES) + 1

                self.input_columns[name] = ListInput(shape, embedder_len, name)
                scalar_input_size += AGGREGATOR_LAYER
            else:
                debug(shape)
                self.input_columns[name] = SimpleInput(shape, name)
                scalar_input_size += SIMPLE_LAYER

            self.add_module(name + " column", self.input_columns[name])

        sc, cat = in_type["feature_screen"]
        self.screen_input = SpatialInput(sc, cat, "feature_screen")
        sc, cat = in_type["feature_minimap"]
        self.minimap_input = SpatialInput(sc, cat, "feature_minimap")

        self.scalar_input_size = scalar_input_size
        self.scalar_reshape = nn.Linear(scalar_input_size, SCALAR_LAYER)

        self.core = RelationalCore()
        self.lstm = nn.LSTM(SCALAR_LAYER + 512 + 512, LSTM_INTERNAL,
                            LSTM_LAYERS)
        after_lstm = LSTM_INTERNAL + SCALAR_LAYER + 512 + 512
        self.lstm_reshape_gating = GLU(after_lstm, 64 * 4, LSTM_SIZE)
        self.output_columns = {}
        self.embeddings = {}

        self.output_columns["function"] = ActionOutput(after_lstm,
                                                       len(ac.FUNCTIONS),
                                                       64 * 4)
        self.embeddings["function"] = nn.Embedding(len(ac.FUNCTIONS),
                                                   LSTM_SIZE)
        with torch.no_grad():
            self.embeddings["function"].weight *= 1 / LSTM_SIZE
        self.output_columns["time_skip"] = DiscreteOutput(MAX_NOPS)
        self.embeddings["time_skip"] = nn.Embedding(MAX_NOPS, LSTM_SIZE)
        with torch.no_grad():
            self.embeddings["time_skip"].weight *= 1 / LSTM_SIZE

        for out in self.output_columns:
            self.add_module(out + " column", self.output_columns[out])
            self.add_module(out + " embedding", self.embeddings[out])

        for x in ac.TYPES:
            u = str(x)
            if u in AUTOREGRESSIVE_CHAIN:
                self.embeddings[u] = nn.Embedding(x.sizes[0], LSTM_SIZE)
                with torch.no_grad():
                    self.embeddings[u].weight *= 1 / LSTM_SIZE
                self.add_module(u + " embedding", self.embeddings[u])

            if "screen" in u or "minimap" in u:
                self.output_columns[u] = SpatialOutput()
            elif "control_group_id" in u:
                self.control_group_id_name = u
                u = "control_group"
                self.output_columns["control_group"] = ControlGroupsOutput()
            elif "control_group_act" in u:
                self.control_group_act_name = u
                continue
            else:
                self.output_columns[u] = DiscreteOutput(x.sizes[0])

            self.add_module(u + " column", self.output_columns[u])

    # because I am not particularly smart
    def add_value(self):
        self.output_columns["value"] = DiscreteOutput(VALUE_DIM, name="value")
        self.add_module("value column", self.output_columns["value"])

    def get_hidden(self, batch_size):
        return (torch.zeros(1, batch_size * 8 * 8 * 2,
                            128), torch.zeros(1, batch_size * 8 * 8 * 2, 128),
                torch.zeros(LSTM_LAYERS, batch_size, LSTM_INTERNAL),
                torch.zeros(LSTM_LAYERS, batch_size, LSTM_INTERNAL))

    def train(self, value=True):
        self.training = value

    def cuda(self):
        super().cuda()
        self._cuda = True

    def sample_action(self, ps, name):
        ps = torch.softmax(ps, dim=-1)
        indices = Categorical(ps.squeeze(0)).sample()
        action = indices.unsqueeze(0)
        return action

    def autoregressive_chain(self, lstm_output, gating, targets=None):
        outputs = {}

        # processing the action first
        key = "function"
        result = self.output_columns[key](lstm_output, gating)
        lstm_output = self.lstm_reshape_gating(lstm_output, gating)

        if "value" in self.output_columns:
            outputs["value"] = self.output_columns["value"](lstm_output)

        if self.training:
            action = targets[key]
        else:
            action = self.sample_action(result, key)

        lstm_output = lstm_output + self.embeddings[key](action)
        outputs[key] = result
        outputs[key + "_sampled"] = action
        # remaining outputs

        for key in AUTOREGRESSIVE_CHAIN[1:]:
            result = self.output_columns[key](lstm_output)

            if self.training:
                action = targets[key]
            else:
                action = self.sample_action(result, key)

            lstm_output = lstm_output + self.embeddings[key](action)

            outputs[key] = result
            outputs[key + "_sampled"] = action

        return outputs, lstm_output

    def forward(self, inputs, hidden, targets=None):
        sample = next(iter(inputs.values()))
        scalar_input = torch.zeros(sample.shape[:2] + (0, ))
        gating_input = torch.zeros(sample.shape[:2] + (0, ))
        if self._cuda:
            scalar_input = scalar_input.cuda()
            gating_input = gating_input.cuda()
        screen = None
        minimap = None
        control_groups = None
        for name in sorted(TRACKED_FEATURES):
            if "screen" in name or "minimap" in name:
                continue
            elif "control_group" in name:
                x, control_groups = self.input_columns[name](
                    inputs[name], inputs[name + "_hint"])
                scalar_input = torch.cat((scalar_input, x), 2)

            elif ("available_actions" in name or "build_order" in name
                  or "select" in name):
                x = self.input_columns[name](inputs[name])
                scalar_input = torch.cat((scalar_input, x), 2)
                gating_input = torch.cat((gating_input, x), 2)
            else:
                x = self.input_columns[name](inputs[name])
                scalar_input = torch.cat((scalar_input, x), 2)

        scalar_input = self.scalar_reshape(scalar_input)

        screen_input, screen_bypass = self.screen_input(
            inputs["feature_screen_scalar"].float(),
            inputs["feature_screen_categorical"].long(), scalar_input)
        minimap_input, minimap_bypass = self.minimap_input(
            inputs["feature_minimap_scalar"].float(),
            inputs["feature_minimap_categorical"].long(), scalar_input)
        hidden_1 = hidden[:2]
        hidden_2 = hidden[2:]

        screen_output, minimap_output, lstm_input, next_hidden1 = self.core(
            screen_input, minimap_input, hidden_1)
        lstm_input = torch.cat([lstm_input, scalar_input], dim=2)
        lstm_output, next_hidden2 = self.lstm(lstm_input, hidden_2)
        lstm_output = torch.cat([lstm_output, lstm_input], dim=2)

        screen = (screen_output, screen_bypass)
        minimap = (minimap_output, minimap_bypass)

        output_dict, lstm_output = self.autoregressive_chain(lstm_output,
                                                             gating_input,
                                                             targets=targets)

        for x in self.output_columns:
            if x in AUTOREGRESSIVE_CHAIN or x == "value":
                continue
            if "screen" in x:
                output_dict[x] = self.output_columns[x](screen, lstm_output)
            elif "minimap" in x:
                output_dict[x] = self.output_columns[x](minimap, lstm_output)
            elif "control_group" in x:
                output_dict[self.control_group_id_name], output_dict[
                    self.control_group_act_name] = self.output_columns[x](
                        lstm_output, control_groups)
            else:
                output_dict[x] = self.output_columns[x](lstm_output)

        keys = list(output_dict.keys())
        if not self.training:
            for x in keys:
                if x in AUTOREGRESSIVE_CHAIN or "_sampled" in x:
                    continue
                output_dict[x + "_sampled"] = self.sample_action(
                    output_dict[x], x)
        return output_dict, next_hidden1 + next_hidden2


def compute_loss(prediction, target, masks):

    losses_dict = {}
    scores_dict = {}
    loss = 0
    loss_op = nn.CrossEntropyLoss(reduction="none")
    for t in target:

        s1 = prediction[t].shape
        loss_value = loss_op(prediction[t].view((-1, ) + (s1[-1], )),
                             target[t].view((-1, )))
        loss_value = masks[t].view((-1, )) * loss_value

        losses_dict[str(t)] = loss_value.sum()
        scores_dict[str(t)] = masks[t].sum()
        loss += loss_value.sum()

    return loss.sum(), losses_dict, scores_dict


def compute_action_prob(prediction, target, masks, debug=False):

    log_prob = 0
    for t in target:
        log_p = torch.log_softmax(prediction[t].float(), dim=-1)
        value = torch.gather(log_p, -1, target[t].unsqueeze(-1))
        value = value * masks[t].unsqueeze(-1)
        log_prob = log_prob + value

    return torch.exp(log_prob)

def compute_kl_div(prediction, target, masks, keys, debug=False):

    sum = 0
    for t in keys:
        log_p = torch.log_softmax(prediction[t].float(), dim=-1)
        log_p_target = torch.log_softmax(target[t].float(), dim=-1)
        value = nn.functional.kl_div(log_p, log_p_target, reduce=False, log_target=True)
        value = value * masks[t].unsqueeze(-1)
        sum = sum + value.sum()

    return sum  

def PPO_loss(predictions,
             actions,
             masks,
             values,
             advantages,
             actives,
             old_probs,
             ref_outputs,
             train_policy=True):

    probs = compute_action_prob(predictions, actions, masks, debug=True)

    ratios = probs / old_probs
    clipped_ratios = torch.clamp(ratios, 0.9, 1.1)

    policy = (((clipped_ratios * advantages.sum(dim=-1,keepdim=True))) * actives).sum()
    value = (((predictions["value"] - values)**2) * actives).reshape(
        -1, VALUE_DIM).sum(dim=0)
    count = actives.sum()

    policy_train = 0
    if train_policy:
        policy_train = policy

    values_dict = {
        "value_" + value_names[i]: value[i]
        for i in range(VALUE_DIM)
    }
    values_counts = {
        "value_" + value_names[i]: count
        for i in range(VALUE_DIM)
    }

    kl_div = compute_kl_div(predictions, ref_outputs, masks, actions.keys())

    return -policy_train + value.sum() * 1.0 + kl_div*0.004, {
        "kl_div":kl_div,
        "policy": policy,
        "value": value.sum()
    } | values_dict, {
        "kl_div":count,
        "policy": count,
        "value": count
    } | values_counts
