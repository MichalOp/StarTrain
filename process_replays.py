# plays the replays to generate the features for the network, and stores them in compressed form.

import os
from torch.multiprocessing import Process
from torch import set_num_threads
from observe_replay import get_data
from util import chunk_it, get_names
import argparse
import shutil

NUM_WORKERS = 14
SEQ_LENGTH = 32


def process_replays(names, target_dir):
    target_dir = os.path.abspath(target_dir)
    set_num_threads(1)
    for i, name in enumerate(names):
        print(i, name)
        store_path = os.path.join(target_dir, os.path.basename(name))
        try:
            get_data(SEQ_LENGTH, os.path.abspath(name), True, os.path.abspath(target_dir))
        except Exception as e:
            print(store_path)
            if os.path.exists(store_path):
                shutil.rmtree(store_path)
            print(e)


def run(replay_dir, target_dir, num_workers):

    files = get_names(replay_dir)

    files = chunk_it(files, num_workers)

    for group in files:
        worker = Process(target=process_replays, args=(group, target_dir))
        worker.start()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source_dir", help="A directory from which the replays will be analysed")
    parser.add_argument("target_dir", help="A directory in which the selected replays will be stored")
    parser.add_argument("-n", "--num_workers", help="Number of workers to use - default: 14")
    args = parser.parse_args()
    
    workers = NUM_WORKERS
    if args.num_workers:
        workers = int(args.num_workers)

    os.makedirs(args.target_dir, exist_ok=True)

    run(args.source_dir, args.target_dir, workers)