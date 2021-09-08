# A (fast) script that selects the replays for training. 
# Also automatically discovers all map hashes and names those maps
# have in the replays. (Useful if you want to train on a different subset of maps)

import sc2reader
from replay_roller import get_names
import shutil
from tqdm import tqdm
from os import path
import os
import argparse

hashes = [
    '2cdc76cc03983839743dc49360f95460fc17241c2d3da2722746cadd1ba89ad9',
    '2d3ebe581a5ad3a6dfcf0b11292e2ca42dd1ae350db96a0f2148808db54b11fd',
    '0f9b14e5e71133ca4db0e059eed96c4f17d2224d159eb7257e096288a6416eaf'
]

DIR = "/home/michal/SC2Replays/"

TARGET_DIR = "/home/michal/StarCraftII/GoodReplays2/"

def process_replays(DIR, TARGET_DIR):
    os.makedirs(TARGET_DIR, exist_ok=True) 
    maps = {}
    for name in tqdm(get_names(DIR)):
        try:
            replay = sc2reader.load_replay(name, load_level=2)
        except Exception:
            continue

        if replay.map_hash not in maps:
            maps[replay.map_hash] = set()
        maps[replay.map_hash].add(replay.map_name)
        good = True
        race_good = False
        for x in replay.players:
            if (x.detail_data["race"] == "Protoss"):
                race_good = True
            if x.init_data["scaled_rating"] < 2500:
                good = False

        good = good and race_good and (replay.map_hash in hashes)

        if good:
            filename = path.split(name)[-1]
            shutil.copy(DIR + filename, path.join(TARGET_DIR, filename))

    for m in maps:
        print(m, maps[m])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source_dir", help="A directory from which the replays will be analysed")
    parser.add_argument("target_dir", help="A directory in which the selected replays will be stored")
    args = parser.parse_args()
    
    process_replays(args.source_dir, args.target_dir)