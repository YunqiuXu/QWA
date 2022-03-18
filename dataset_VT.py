import os
import json
from os.path import join as pjoin
from tqdm import tqdm
import numpy as np
import gym
from glob import glob

class VTData(gym.Env):
    def __init__(self, config, data_path_dict):
        self.rng = None
        self.config = config
        self.read_config()
        self.seed(self.random_seed)
        self.id2kg = {
                'train':{},
                'valid':{},
                'test':{}
        }
        self.dataset = {}
        for split in ["train", "valid", "test"]:
            print("----- Building {} dataset -----".format(split))
            self.dataset[split] = {
                "graph": [],
                "task": [],
                "label": []
                }
            self.load_dataset(split, data_path_dict[split])
            print("----- {}: {} samples".format(split, len(self.dataset[split]["graph"])))
            print("----- ----- ----- ----- -----")
        self.train_size = len(self.dataset["train"]["graph"])
        self.valid_size = len(self.dataset["valid"]["graph"])
        self.test_size = len(self.dataset["test"]["graph"])
        self.batch_pointer = None
        self.data_size, self.batch_size, self.data = None, None, None
        self.split = "train"

    def read_config(self):
        self.random_seed = self.config["general"]["random_seed"]
        self.training_batch_size = self.config["general"]["training"]["batch_size"]
        self.evaluate_batch_size = self.config["general"]["evaluate"]["batch_size"]

    def seed(self, seed):
        self.rng = np.random.RandomState(seed)

    def load_dataset(self, split, data_path_list):
        assert (len(data_path_list) > 0), "Can not load {} data".format(split)
        for data_path in data_path_list:
            with open(data_path, 'r') as f:
                for line in f.readlines():
                    curr_data = json.loads(line)
                    graph = curr_data["graph"]
                    graph_index = len(self.id2kg[split])
                    self.id2kg[split][graph_index] = graph
                    for i in range(len(curr_data["all_labels"])):
                        self.dataset[split]["graph"].append(graph_index)
                        self.dataset[split]["task"].append([curr_data["all_tasks"][i]])
                        self.dataset[split]["label"].append(curr_data["all_labels"][i])

    def split_reset(self, split):
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
            indices = self.rng.choice(self.data_size, self.batch_size)
        else:
            start = self.batch_pointer
            end = min(start + self.batch_size, self.data_size)
            indices = np.arange(start, end)
            self.batch_pointer += self.batch_size
            if self.batch_pointer >= self.data_size:
                self.batch_pointer = 0
        graph, task, label = [], [], []
        for idx in indices:
            graph.append(self.id2kg[self.split][self.data["graph"][idx]])
            task.append(self.data["task"][idx])
            label.append(self.data["label"][idx])
        return graph, task, label