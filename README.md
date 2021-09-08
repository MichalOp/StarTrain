# StarTrain - a neural network bot for StarCraft II

## What is this?

A neural network bot for StarCraft II, playing the game through the PySC2 library. Heavily inspired by AlphaStar, but uses the feature interface instead of the raw interface. It uses a network combined from parts of AlphaStar with parts from Relational Deep Reinforcement Learning, and is trainable on a single reasonably powerful GPU.

This project is a part of my bachelor's thesis, *Applying supervised and reinforcement learning methods to create neural-network-based agents for playing Starcraft II* at the Faculty of Mathematics and Computer Science, University of Wrocław.

## Installation

It should be possible, but it might not be easy.

The project was developed on Ubuntu 18.04, and might or might not work on other systems. Theoretically there is nothing preventing it from running on Windows or Mac, but it was not tested.

Installation steps on Ubuntu:
1. Ensure you have an NVIDIA GPU with at least 3GB of VRAM (for running games) or preferably above 11GB (for training). 
2. Ensure you have NVIDIA drivers with CUDA support.
3. Install Python 3.9.
4. Install [PyTorch](https://pytorch.org/), and ensure CUDA acceleration works.
5. Install [PySC2](https://github.com/deepmind/pysc2).
6. From [sc2client-proto](https://github.com/Blizzard/s2client-proto) download and unzip Linux SC2 package for version 4.9.3 (theoretically you can use different version, but the networks were trained for this version and they might work or not on others.
7. Also from [sc2client-proto](https://github.com/Blizzard/s2client-proto), download map pack for 2019 Season 3. Place the unzipped pack in the Maps folder of your SC2 installation.
8. Ensure you can run example commands from [PySC2](https://github.com/deepmind/pysc2). 
9. Install the remaining packages from `requirements.txt` (installing `sc2reader zstd` might be enough).

## Playing against the bot

You will need to download two things: A model file, and a folder with data from a single processed replay (used as hint for the model on how to use control groups and what build order to use, that it ignores anyway, but I forgot to train without it). 

Both of those can be downloaded from this [Google drive link](https://drive.google.com/drive/folders/1NjqgrlDvIUidXfkfGDGVMXdL-qyWX72k?usp=sharing). Download the chosen model and `stats_replay.zip` and unzip the `stats_replay` in any location. 

Now, you should have everything needed to run the game. Open two terminals, and in one start your game instance: 
`python3 -m pysc2.bin.play_vs_agent --human --map Acropolis --realtime`

In the other one, start the bot by running:
`python3 run_game.py /path/to/model.tm LAN /path/to/stats_replay`

The game should start up. 

## Training the model yourself

Apart from all steps from "Installation", you will need to install and configure [ClearML](https://github.com/allegroai/clearml). The logs from training will show on your selected ClearML server.

There are two files for running the training: `run_training.py` which runs the supervised part of the training, and `rl_training.py` which does the reinforcement learning part of the training. 

Training configuration can be changed by editing those files. Depending on the amount of VRAM your GPU has, you might want to change the BATCH in them to a lower value. You will need roughly 3GB + 0.5 GB per 1 element in batch, so ~5GB for BATCH=4 and ~19GB for BATCH=32.

To train in the same effective batch size as in the current configuration, each time you divide BATCH by 2, multiply the OPT_STEPS by 2.

You might also need to change the number of game instances for reinforcement learning, as the current setting will start 32 games and use 64 GB of CPU RAM.

### RL training
For reinforcement learning using the already trained models as a starting point (right now this will only work for supervised learning model, continuing the RL training won't work), you need the same downloads as in "Playing against the bot".

Then, you can simply run
`python3 rl_training.py /path/to/start/model.tm /path/to/stats_replay`

It will run the training against a set of Easy and Medium AIs. After 2 or so days the model should have around 90%-95% win rate against the entire pool, which translates to around 99% winrate against Easy and 65% against Medium. The trained network will be stored as `model/ModelRL.tm`.

### Supervised training

You will also need to get a large number of replays. There is [a somewhat hidden script in sc2client-proto](https://github.com/Blizzard/s2client-proto/tree/master/samples/replay-api) that allows you to download hundreds of thousands of anonymized replays for the selected versions of SC2. To replicate my training setup, download all replays for version 4.9.3. (Should take < 100GB of disk space.)

After you have the replays, you need to select the ones for training. To do it, run

`python3 replay_preprocessing.py \path\to\replay\dir \path\to\selected\replay\dir`

It will select replays that contain a Protoss player, one player above 2500 MMR, are played on the 'Acropolis' map, and are longer than 1 minute, and copy them to the designated directory.

After that, you need to process the replays. To do it, run:

`python3 process_replays.py \path\to\selected\replay\dir \path\to\processed\replay\dir`

If needed, you can change the number of worker by adding `-n NUM_WORKERS`. The default number of workers is 14, and it should take around 2 days if you have 16 threads.

After that, you are ready for training. Run

`python3 run_training.py \path\to\processed\replay\dir \path\to\stats_replay`

The speed of this step is mostly dependent on your GPU. On 1080 Ti you should get a network with a reasonable performance (40% win rate against Easy, 10-20% against Medium) after around 3-5 days. 

The process will periodically run a test vs 20 Easy AIs and 20 Medium AIs and store the best network from those tests as `model/best.tm`. It will also store the current model as `model/ModelData.tm` after each iteration.

