import os
import json
from os.path import join as pjoin
from tqdm import tqdm
import numpy as np
import gym
from glob import glob


class VAData(gym.Env):
    def __init__(self, config, data_path_dict):
        """
        :param data_path_dict: a dict, key in ["train", "valid", "test"], each value is a list of paths
        """
        self.rng = None
        self.config = config
        self.read_config()
        self.seed(self.random_seed)
        self.id2task = {
                'train':{},
                'valid':{},
                'test':{}       
        }
        self.task2id = {
                'train':{},
                'valid':{},
                'test':{}       
        }
        self.id2action = {
                'train':{},
                'valid':{},
                'test':{}       
        }
        self.action2id = {
                'train':{},
                'valid':{},
                'test':{}       
        }
        self.id_pairs = {
                'train':set(),
                'valid':set(),
                'test':set()       
        }
        self.dataset = {}
        for split in ["train", "valid", "test"]:
            print("----- Building {} dataset -----".format(split))
            self.dataset[split] = {
                "task": [],
                "action": [],
                "label": []
                }
            self.load_dataset(split, data_path_dict[split])
            print("----- {}: {} samples".format(split, len(self.dataset[split]["task"])))
            print("----- ----- ----- ----- -----")

        self.train_size = len(self.dataset["train"]["task"])
        self.valid_size = len(self.dataset["valid"]["task"])
        self.test_size = len(self.dataset["test"]["task"])
        self.batch_pointer = None
        self.data_size, self.batch_size, self.data = None, None, None
        self.split = "train"
        # self.get_id_statistics()

    def get_id_statistics(self):
        print("Tasks  |Tr {}|Ev {}|Te {}".format(
                                                len(self.task2id['train']),
                                                len(self.task2id['valid']),
                                                len(self.task2id['test']),
                                                ))
        print("Actions|Tr {}|Ev {}|Te {}".format(
                                                len(self.action2id['train']),
                                                len(self.action2id['valid']),
                                                len(self.action2id['test']),
                                                ))
        print("Pairs  |Tr {}|Ev {}|Te {}".format(
                                                len(self.id_pairs['train']),
                                                len(self.id_pairs['valid']),
                                                len(self.id_pairs['test']),
                                                ))

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
                    task = curr_data["task"]
                    all_actions = curr_data["all_actions"]
                    all_labels = curr_data["all_labels"]
                    if task in self.task2id[split]:
                        task_index = self.task2id[split][task]
                    else:
                        task_index = len(self.task2id[split])
                        self.task2id[split][task] = task_index
                        self.id2task[split][task_index] = task
                    all_action_indices = []
                    for action in all_actions:
                        if action in self.action2id[split]:
                            action_index = self.action2id[split][action]
                        else:
                            action_index = len(self.action2id[split])
                            self.action2id[split][action] = action_index
                            self.id2action[split][action_index] = action
                        all_action_indices.append(action_index)
                    for i in range(len(curr_data["all_labels"])):
                        curr_id_pair = (task_index, all_action_indices[i])
                        if curr_id_pair not in self.id_pairs[split]:
                            self.id_pairs[split].add(curr_id_pair)
                            self.dataset[split]['task'].append(task_index)
                            self.dataset[split]['action'].append(all_action_indices[i])
                            self.dataset[split]['label'].append(all_labels[i])

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
        task_batch, action_batch, label_batch = [], [], []
        for idx in indices:
            task_batch.append([self.id2task[self.split][self.data['task'][idx]]])
            action_batch.append([self.id2action[self.split][self.data['action'][idx]]])
            label_batch.append(self.data['label'][idx])
        return task_batch, action_batch, label_batch