import torch
from torch.optim import Adam
from model import Model, compute_loss, PPO_loss
from random import seed
from util import get_names
from clearml import Task
from torch.cuda.amp import autocast, GradScaler
from loader import BatchSeqLoader
from trajectory_generator import TrajectoryGenerator
from observations import (concat_along_axis_tensor, concat_lstm_hidden,
                          to_cuda)
from game_env import value_names, VALUE_DIM
import numpy as np
import argparse
import os

ENVS = 32
STEPS = 32
BATCH = 16
WARMUP = 10
OPT_STEPS = 32
TRAJECTORIES = 512
LEARNING_RATE = 0.001
SAVE_PATH = "model/ModelRL.tm"

PROFILE = False
WRITE = True


def selective_load(source_dict, target):
    '''
    load_dict = {}
    target_dict = target.state_dict()
    for x in target_dict:
        print(x)
        if x in source_dict and source_dict[x].shape == target_dict[x].shape:
            load_dict[x] = source_dict[x]
    '''
    target.load_state_dict(source_dict)


def join_trajectories(trajectories):
    obs, ac, ms, probs, vals, advs, active, ref_out, last_hidden = zip(*trajectories)

    obs = to_cuda(concat_along_axis_tensor(obs, 1))
    ac = to_cuda(concat_along_axis_tensor(ac, 1))
    ms = to_cuda(concat_along_axis_tensor(ms, 1))
    ref_out = to_cuda(concat_along_axis_tensor(ref_out, 1))
    probs = torch.cat(probs, dim=1).cuda()
    vals = torch.cat(vals, dim=1).cuda()
    advs = torch.cat(advs, dim=1).cuda()
    active = torch.cat(active, dim=1).cuda()
    hs = concat_lstm_hidden(last_hidden)
    hs = tuple([x.cuda() for x in hs])

    return obs, ac, ms, probs, vals, advs, active, ref_out, hs


def rl_training(base_model, reference_replay):
    torch.manual_seed(2138)
    seed(2137)

    model = Model()
    start_epoch = 0

    checkpoint = torch.load(base_model)
    selective_load(checkpoint['model_state_dict'], model)
    frozen_model = Model()
    selective_load(checkpoint['model_state_dict'], frozen_model)

    model.add_value()

    frozen_model.cuda()
    frozen_model.train()
    model.cuda()

    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-7)
    generator = TrajectoryGenerator(model, frozen_model, reference_replay, num_envs=24, max_batch=12)

    if WRITE:
        task = Task.init(project_name='StarTrain',
                         task_name='rl batch 512')
        logger = task.get_logger()

    total_epoch_loss = 0
    total_losses_dict = {}
    total_scores_dict = {}
    steps = 0

    optimizer.zero_grad()
    scaler = GradScaler()

    game_scores = [(np.zeros((VALUE_DIM,)), -1)]

    for epoch in range(start_epoch, 1000000, 1):
        running_loss = 0

        trajectories = generator.generate_trajectories(TRAJECTORIES)

        model.train()

        for repeat in range(2):
            for i in range(0, TRAJECTORIES, BATCH):
                steps += STEPS * BATCH

                losses_dict = {}
                scores_dict = {}

                obs, ac, ms, probs, vals, advs, actives, ref_output, hiddens = join_trajectories(
                    trajectories[i:i + BATCH])

                with autocast():
                    output, new_hiddens = model(obs, hiddens, ac)
                    loss_rl, losses_dict_rl, scores_dict_rl = PPO_loss(
                        output, ac, ms, vals, advs, actives, probs, ref_output, epoch > WARMUP)

                reduced_loss = (loss_rl * 1.0) / \
                    BATCH / STEPS / OPT_STEPS
                scaler.scale(reduced_loss).backward()

                losses_dict |= losses_dict_rl
                scores_dict |= scores_dict_rl

                loss_val = loss_rl.item()
                running_loss += loss_val
                total_epoch_loss += loss_val


                if len(total_losses_dict) > 0:
                    for x in total_losses_dict:
                        total_losses_dict[x] += losses_dict[x].item()
                        total_scores_dict[x] += scores_dict[x].item()
                else:
                    total_losses_dict = {
                        x: losses_dict[x].item()
                        for x in losses_dict
                    }
                    total_scores_dict = {
                        x: scores_dict[x].item()
                        for x in scores_dict
                    }
                if (i // BATCH) % OPT_STEPS == OPT_STEPS - 1:
                    print('[%d, %d, %5d] loss: %.3f' %
                          (epoch, repeat, i, running_loss / STEPS / BATCH / OPT_STEPS))
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    running_loss = 0.0

        weights = 0
        for param in model.parameters():
            weights += (param * param).sum().item()

        logger.report_scalar(title="Stats",
                             series="Weight Magnitude",
                             value=weights,
                             iteration=epoch)
        # writer.add_scalar("Stats/Gradient Magnitude",
        #                   grad_magnitudes/steps,epoch)
        print(f"avg loss in epoch {total_epoch_loss/steps}")
        logger.report_scalar(title="Losses",
                             series="Total",
                             value=total_epoch_loss / steps,
                             iteration=epoch)
        for x in total_losses_dict:
            logger.report_scalar(title="Losses",
                                 series=x,
                                 value=total_losses_dict[x] /
                                 (total_scores_dict[x] + 0.001),
                                 iteration=epoch)

        game_score = generator.step_score_queue()
        while game_score is not None:
            game_scores.append(game_score)
            game_score = generator.step_score_queue()

        while len(game_scores) > 100:
            del game_scores[0]

        rewards, wins = zip(*game_scores)
        wins = [max(0, x) for x in wins]

        rewards = np.array(rewards)
        for i, name in enumerate(value_names):
            logger.report_scalar(title="Rewards",
                                 series="reward_"+name,
                                 value=sum(rewards[:, i]) / len(rewards),
                                 iteration=epoch)

        logger.report_scalar(title="Rewards",
                             series="wins",
                             value=sum(wins) / len(wins),
                             iteration=epoch)

        steps = 0

        total_losses_dict = {}
        total_scores_dict = {}
        total_epoch_loss = 0

        print("SAVING")

        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, SAVE_PATH)

        print("SAVED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source_model", help="A pre-trained model to start training from.")
    parser.add_argument("reference_replay", help="A single processed replay from which features will be provided to the model. It will quickly start ignoring them.")
    args = parser.parse_args()
    
    os.makedirs("model", exist_ok=True)
    rl_training(args.source_model, args.reference_replay)
