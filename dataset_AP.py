import os
import json
from os.path import join as pjoin
from tqdm import tqdm
import numpy as np
import gym
from graph_dataset import GraphDataset

class APData(gym.Env):
    FILENAMES_MAP = {
        "full": {
            "train": "train.full.json",
            "valid": "valid.full.json",
            "test": "test.full.json"
            },
        "seen": {
            "train": "train.seen.json",
            "valid": "valid.seen.json",
            "test": "test.seen.json"
            }
        }

    def __init__(self, config):
        self.rng = None
        self.config = config
        self.read_config()
        self.seed(self.random_seed)

        # Load dataset splits.
        self.dataset = {}
        for split in ["train", "valid", "test"]:
            self.dataset[split] = {
                "current_graph": [],
                "previous_graph": [],
                "target_action": [],
                "action_choices": []
                }
            self.load_dataset_for_ap(split)
        self.train_size = len(self.dataset["train"]["current_graph"])
        self.valid_size = len(self.dataset["valid"]["current_graph"])
        self.test_size = len(self.dataset["test"]["current_graph"])
        print("Train: {}\nValid: {}\nTest: {}".format(self.train_size, self.valid_size, self.test_size))
        self.batch_pointer = None
        self.data_size, self.batch_size, self.data = None, None, None
        self.split = "train"

    def read_config(self):
        self.data_path = self.config["ap"]["data_path"]
        self.graph_type = self.config["ap"]["graph_type"]
        self.random_seed = self.config["general"]["random_seed"]
        self.training_batch_size = self.config["general"]["training"]["batch_size"]
        self.evaluate_batch_size = self.config["general"]["evaluate"]["batch_size"]
        assert (self.graph_type == "full")
        print("--- Train batch size: {}".format(self.training_batch_size))
        print("--- Valid/test batch size: {}".format(self.evaluate_batch_size))

    def load_dataset_for_ap(self, split):
        file_path = pjoin(self.data_path, self.FILENAMES_MAP[self.graph_type][split])
        with open(file_path) as f:
            data = json.load(f)
        graph_dataset = GraphDataset.loads(data["graph_index"])
        self.dataset[split]["graph_dataset"] = graph_dataset
        desc = "Loading {}".format(os.path.basename(file_path))
        for example in tqdm(data["examples"], desc=desc):
            target_action = example["target_action"]
            curr_graph = example["current_graph"]
            prev_graph = example["previous_graph"]
            candidates = example["action_choices"]
            self.dataset[split]["current_graph"].append(curr_graph)
            self.dataset[split]["previous_graph"].append(prev_graph)
            self.dataset[split]["target_action"].append(target_action)
            self.dataset[split]["action_choices"].append(candidates)

    def split_reset(self, split):
        print("--- Switch to {} set".format(split))
        if split == "train":
            self.data_size = self.train_size
            self.batch_size = self.training_batch_size
        elif split == "valid":
            self.data_size = self.valid_size
            self.batch_size = self.evaluate_batch_size
        else:
            self.data_size = self.test_size
            self.batch_size = self.evaluate_batch_size
        self.data = self.dataset[split]
        self.split = split
        self.batch_pointer = 0


    def get_batch(self):
        if self.split == "train":
            indices = self.rng.choice(self.data_size, self.training_batch_size)
        else:
            start = self.batch_pointer
            end = min(start + self.evaluate_batch_size, self.data_size)
            indices = np.arange(start, end)
            self.batch_pointer += self.evaluate_batch_size
            if self.batch_pointer >= self.data_size:
                self.batch_pointer = 0
        current_graph, previous_graph, target_action, action_choices = [], [], [], []
        decompress = self.dataset[self.split]["graph_dataset"].decompress
        for idx in indices:
            target_action.append(self.data["target_action"][idx])
            action_choices.append(self.data["action_choices"][idx])
            current_graph.append(decompress(self.data["current_graph"][idx]))
            previous_graph.append(decompress(self.data["previous_graph"][idx]))
        return current_graph, previous_graph, target_action, action_choices

    def render(self, mode='human'):
        return

    def close(self):
        return

    def seed(self, seed):
        self.rng = np.random.RandomState(seed)
