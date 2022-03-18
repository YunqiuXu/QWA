import os
import random
import copy
import codecs
import spacy
from os.path import join as pjoin

import numpy as np
import torch
import torch.nn.functional as F
from textworld import EnvInfos

import dqn_memory_priortized_replay_buffer
from generic import to_np, to_pt, _words_to_ids, _word_to_id, pad_sequences, update_graph_triplets, preproc, max_len, ez_gather_dim_1
from generic import sort_target_commands, process_facts, serialize_facts, gen_graph_commands, process_fully_obs_facts
from generic import generate_labels_for_ap, generate_labels_for_sp, LinearSchedule
from layers import NegativeLogLoss, compute_mask, masked_mean

from model import KG_Manipulation
from model_AP import AP_KG_Manipulation
from model_VT import Task_Generator
from model_VA import Action_Validator

import kg_utils
import refine_action_utils
from focal_loss import FocalLoss


class Agent:
    def __init__(self, config):
        self.config = config
        self.load_config()
        self.mode = "train"

        self.task_net = Task_Generator(config=self.config, 
                                          word_vocab=self.word_vocab, 
                                          node_vocab=self.node_vocab, 
                                          relation_vocab=self.relation_vocab)
        if self.task != 'vt':
            print("Task {}: do not train the task_net".format(self.task))

        if self.task == 'ap':
            print("Use AP_KG_Manipulation for AP pretraining")
            self.online_net = AP_KG_Manipulation(config=self.config, 
                                          word_vocab=self.word_vocab, 
                                          node_vocab=self.node_vocab, 
                                          relation_vocab=self.relation_vocab)  
        else:
            print("Use KG_Manipulation")
            self.online_net = KG_Manipulation(config=self.config, 
                                              word_vocab=self.word_vocab, 
                                              node_vocab=self.node_vocab, 
                                              relation_vocab=self.relation_vocab)        
        if self.task not in {'rl','ap'}:
            print("Task {}: do not use the online net!".format(self.task))

        self.action_net = Action_Validator(config=self.config, 
                                          word_vocab=self.word_vocab, 
                                          node_vocab=self.node_vocab, 
                                          relation_vocab=self.relation_vocab)
        if self.task != 'va':
            print("Task {}: do not train the action_net".format(self.task))

        self.task_net.train()
        self.online_net.train()
        self.action_net.train()
        if self.use_cuda:
            self.task_net.cuda()
            self.online_net.cuda()
            self.action_net.cuda()

        # 3. Build target nets
        if self.task == "rl":
            print("Task rl: Target net")
            self.target_net = KG_Manipulation(config=self.config, 
                                              word_vocab=self.word_vocab, 
                                              node_vocab=self.node_vocab, 
                                              relation_vocab=self.relation_vocab)
            self.target_net.train()
            self.update_target_net()
            for param in self.target_net.parameters():
                param.requires_grad = False
            if self.use_cuda:
                self.target_net.cuda()

            if self.use_expert_task_net:
                print("Task rl: Use expert task net, do not load pre-trained VT model")
            else:
                print("Task rl: Load pre-trained VT model, do not train it!")
                self.load_pretrained_task_model(
                                            "{}/pretrainVT_task_model.pt".format(self.config['general']['VTptr_dir']),
                                            load_partial_graph=False)
                self.task_net.eval()
                for param in self.task_net.parameters():
                    param.requires_grad = False

            if self.use_TRA == 'expertTRA':
                print("Task rl: Use expert TRA, do not load pre-trained VA model")
            elif self.use_TRA == 'noTRA':
                print("Task rl: No TRA, do not load pre-trained VA model")
            else:
                assert self.use_TRA == 'pretrainedTRA'
                print("Task rl: Load pre-trained TRA, do not train it!")
                self.load_pretrained_action_model("{}/pretrainVA_action_model.pt".format(
                                            self.config['general']['VAptr_dir']),
                                            load_partial_graph=False)
                self.action_net.eval()
                for param in self.action_net.parameters():
                    param.requires_grad = False
        else:
            self.target_net = None
            print("Task {}: do not use target net / pre-trained task_net / pre-trained action_net!".format(self.task))

        if self.task in {'rl', 'ap'}:
            param_frozen_list = [] # should be changed into torch.nn.ParameterList()
            param_active_list = [] # should be changed into torch.nn.ParameterList()
            # Can only train part of the online net
            for k, v in self.online_net.named_parameters():
                keep_this = True
                for keyword in self.fix_parameters_keywords:
                    if keyword in k:
                        param_frozen_list.append(v)
                        keep_this = False
                        break
                if keep_this:
                    param_active_list.append(v)
            param_frozen_list = torch.nn.ParameterList(param_frozen_list)
            param_active_list = torch.nn.ParameterList(param_active_list)
            if self.step_rule == 'adam':
                self.optimizer = torch.optim.Adam([{'params': param_frozen_list, 'lr': 0.0},
                                                   {'params': param_active_list, 'lr': self.config['general']['training']['optimizer']['learning_rate']}],
                                                  lr=self.config['general']['training']['optimizer']['learning_rate'])
            elif self.step_rule == 'radam':
                from radam import RAdam
                self.optimizer = RAdam([{'params': param_frozen_list, 'lr': 0.0},
                                        {'params': param_active_list, 'lr': self.config['general']['training']['optimizer']['learning_rate']}],
                                       lr=self.config['general']['training']['optimizer']['learning_rate'])
            else:
                raise NotImplementedError
        else:
            assert self.task in {'vt', 'va'}

        if self.task == 'vt':
            param_frozen_list = [] # should be changed into torch.nn.ParameterList()
            param_active_list = [] # should be changed into torch.nn.ParameterList()
            # Train the whole task_net
            for k, v in self.task_net.named_parameters():
                keep_this = True
                if keep_this:
                    param_active_list.append(v)
            param_frozen_list = torch.nn.ParameterList(param_frozen_list)
            param_active_list = torch.nn.ParameterList(param_active_list)
            if self.step_rule == 'adam':
                self.task_optimizer = torch.optim.Adam([{'params': param_frozen_list, 'lr': 0.0},
                                                        {'params': param_active_list, 
                                                         'lr': self.config['general']['training']['optimizer']['learning_rate']}],
                                                          lr=self.config['general']['training']['optimizer']['learning_rate'])
            elif self.step_rule == 'radam':
                from radam import RAdam
                self.task_optimizer = RAdam([{'params': param_frozen_list, 'lr': 0.0},
                                             {'params': param_active_list, 
                                              'lr': self.config['general']['training']['optimizer']['learning_rate']}],
                                             lr=self.config['general']['training']['optimizer']['learning_rate'])
            else:
                raise NotImplementedError
        else:
            print("Task {}: no optimizer for task net!".format(self.task))

        if self.task == 'va':
            param_frozen_list = [] 
            param_active_list = [] 
            # Train the whole action_net
            for k, v in self.action_net.named_parameters():
                keep_this = True
                if keep_this:
                    param_active_list.append(v)
            param_frozen_list = torch.nn.ParameterList(param_frozen_list)
            param_active_list = torch.nn.ParameterList(param_active_list)
            if self.step_rule == 'adam':
                self.action_optimizer = torch.optim.Adam([{'params': param_frozen_list, 'lr': 0.0},
                                                          {'params': param_active_list, 
                                                           'lr': self.config['general']['training']['optimizer']['learning_rate']}],
                                                            lr=self.config['general']['training']['optimizer']['learning_rate'])
            elif self.step_rule == 'radam':
                from radam import RAdam
                self.action_optimizer = RAdam([{'params': param_frozen_list, 'lr': 0.0},
                                               {'params': param_active_list, 
                                              'lr': self.config['general']['training']['optimizer']['learning_rate']}],
                                              lr=self.config['general']['training']['optimizer']['learning_rate'])
            else:
                raise NotImplementedError
        else:
            print("Task {}: no optimizer for action net!".format(self.task))


    def load_config(self):
        self.real_valued_graph = self.config['general']['model']['real_valued_graph']
        self.task = self.config['general']['task']
        assert (self.task in {'rl', 'ap', 'vt','va'}), "Unsupported task {}".format(self.task)
        # word vocab
        self.word_vocab = []
        with codecs.open("./vocabularies/word_vocab.txt", mode='r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                self.word_vocab.append(line.strip())
        self.word2id = {}
        for i, w in enumerate(self.word_vocab):
            self.word2id[w] = i
        # node vocab
        self.node_vocab = []
        with codecs.open("./vocabularies/node_vocab.txt", mode='r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                self.node_vocab.append(line.strip().lower())
        self.node2id = {}
        for i, w in enumerate(self.node_vocab):
            self.node2id[w] = i
        # relation vocab
        self.relation_vocab = []
        with codecs.open("./vocabularies/relation_vocab.txt", mode='r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                self.relation_vocab.append(line.strip().lower())
        self.origin_relation_number = len(self.relation_vocab)
        # add reverse relations
        for i in range(self.origin_relation_number):
            self.relation_vocab.append(self.relation_vocab[i] + "_reverse")
        if not self.real_valued_graph:
            self.relation_vocab += ["self"]
        self.relation2id = {}
        for i, w in enumerate(self.relation_vocab):
            self.relation2id[w] = i
        self.step_rule = self.config['general']['training']['optimizer']['step_rule']
        self.init_learning_rate = self.config['general']['training']['optimizer']['learning_rate']
        self.clip_grad_norm = self.config['general']['training']['optimizer']['clip_grad_norm']
        self.learning_rate_warmup_until = self.config['general']['training']['optimizer']['learning_rate_warmup_until']
        self.fix_parameters_keywords = list(set(self.config['general']['training']['fix_parameters_keywords']))
        self.batch_size = self.config['general']['training']['batch_size']
        self.max_episode = self.config['general']['training']['max_episode']
        self.smoothing_eps = self.config['general']['training']['smoothing_eps']
        self.patience = self.config['general']['training']['patience']
        self.eval_g_belief = self.config['general']['evaluate']['g_belief']
        self.eval_batch_size = self.config['general']['evaluate']['batch_size']
        self.max_target_length = self.config['general']['evaluate']['max_target_length']
        # Set the random seed manually for reproducibility.
        self.random_seed = self.config['general']['random_seed']
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            if not self.config['general']['use_cuda']:
                print("WARNING: CUDA device detected but 'use_cuda: false' found in config.yaml")
                self.use_cuda = False
            else:
                torch.backends.cudnn.deterministic = True
                torch.cuda.manual_seed(self.random_seed)
                self.use_cuda = True
        else:
            self.use_cuda = False
        self.experiment_tag = self.config['general']['checkpoint']['experiment_tag']
        self.save_frequency = self.config['general']['checkpoint']['save_frequency']
        self.report_frequency = self.config['general']['checkpoint']['report_frequency']
        self.load_pretrained = self.config['general']['checkpoint']['load_pretrained']
        self.load_from_tag = self.config['general']['checkpoint']['load_from_tag']
        self.load_graph_generation_model_from_tag = self.config['general']['checkpoint']['load_graph_generation_model_from_tag']
        self.load_parameter_keywords = list(set(self.config['general']['checkpoint']['load_parameter_keywords']))
        self.nlp = spacy.load('en', disable=['ner', 'parser', 'tagger'])
        # AP specific
        self.ap_k_way_classification = self.config['ap']['k_way_classification']
        # RL specific
        self.fully_observable_graph = self.config['rl']['fully_observable_graph']
        # epsilon greedy
        self.epsilon_anneal_episodes = self.config['rl']['epsilon_greedy']['epsilon_anneal_episodes']
        self.epsilon_anneal_from = self.config['rl']['epsilon_greedy']['epsilon_anneal_from']
        self.epsilon_anneal_to = self.config['rl']['epsilon_greedy']['epsilon_anneal_to']
        self.epsilon = self.epsilon_anneal_from
        self.epsilon_scheduler = LinearSchedule(schedule_timesteps=self.epsilon_anneal_episodes, 
                                                initial_p=self.epsilon_anneal_from, 
                                                final_p=self.epsilon_anneal_to)
        self.noisy_net = self.config['rl']['epsilon_greedy']['noisy_net']
        if self.noisy_net:
            self.epsilon_anneal_episodes = -1
            self.epsilon = 0.0
        # drqn
        self.replay_sample_history_length = self.config['rl']['replay']['replay_sample_history_length']
        self.replay_sample_update_from = self.config['rl']['replay']['replay_sample_update_from']
        # replay buffer and updates
        self.buffer_reward_threshold = self.config['rl']['replay']['buffer_reward_threshold']
        self.prioritized_replay_beta = self.config['rl']['replay']['prioritized_replay_beta']
        self.beta_scheduler = LinearSchedule(schedule_timesteps=self.max_episode, 
                                             initial_p=self.prioritized_replay_beta, 
                                             final_p=1.0)
        self.accumulate_reward_from_final = self.config['rl']['replay']['accumulate_reward_from_final']
        self.prioritized_replay_eps = self.config['rl']['replay']['prioritized_replay_eps']
        self.count_reward_lambda = self.config['rl']['replay']['count_reward_lambda']
        self.discount_gamma_count_reward = self.config['rl']['replay']['discount_gamma_count_reward']
        self.graph_reward_lambda = self.config['rl']['replay']['graph_reward_lambda']
        self.graph_reward_type = self.config['rl']['replay']['graph_reward_type']
        self.discount_gamma_graph_reward = self.config['rl']['replay']['discount_gamma_graph_reward']
        self.discount_gamma_game_reward = self.config['rl']['replay']['discount_gamma_game_reward']
        self.replay_batch_size = self.config['rl']['replay']['replay_batch_size']
        self.dqn_memory = dqn_memory_priortized_replay_buffer.PrioritizedReplayMemory(
                            self.config['rl']['replay']['replay_memory_capacity'],
                            priority_fraction=self.config['rl']['replay']['replay_memory_priority_fraction'],
                            discount_gamma_game_reward=self.discount_gamma_game_reward,
                            discount_gamma_graph_reward=self.discount_gamma_graph_reward,
                            discount_gamma_count_reward=self.discount_gamma_count_reward,
                            accumulate_reward_from_final=self.accumulate_reward_from_final,
                            seed=self.config['general']['random_seed'])
        self.update_per_k_game_steps = self.config['rl']['replay']['update_per_k_game_steps']
        self.multi_step = self.config['rl']['replay']['multi_step']
        # input in rl training
        self.enable_recurrent_memory = self.config['rl']['model']['enable_recurrent_memory']
        self.enable_graph_input = self.config['rl']['model']['enable_graph_input']
        self.enable_text_input = self.config['rl']['model']['enable_text_input']
        assert (self.enable_graph_input) and (not self.enable_text_input), 'Do not allow textual obs'
        # rl train and eval
        self.max_nb_steps_per_episode = self.config['rl']['training']['max_nb_steps_per_episode']
        self.learn_start_from_this_episode = self.config['rl']['training']['learn_start_from_this_episode']
        self.target_net_update_frequency = self.config['rl']['training']['target_net_update_frequency']
        self.use_negative_reward = self.config['rl']['training']['use_negative_reward']
        self.eval_max_nb_steps_per_episode = self.config['rl']['evaluate']['max_nb_steps_per_episode']
        self.load_online_net = self.config['rl']['load_online_net']  # this is only used for rl
        assert self.load_online_net in {"apInit", "rdInit"}          # inited via ap / rd
        # Whether or not to use an expert task net, if False --> use our pre-trained task net
        self.use_expert_task_net = self.config['rl']['use_expert_task_net']
        self.use_TRA = self.config['rl']['use_TRA']
        assert self.use_TRA in {"expertTRA", "noTRA", "pretrainedTRA"}
        self.task_step_threshold = self.config['rl']['task_step_threshold'] 
        # vt specific
        self.load_task_net = self.config['vt']['load_task_net']
        self.vt_loss = self.config['vt']['vt_loss']
        assert self.vt_loss in {'ce','focal','focal_alt'}
        # va specific
        self.load_action_net = self.config['va']['load_action_net']
        self.va_loss = self.config['va']['va_loss']
        assert self.va_loss in {'ce','focal','focal_alt'}


    def train(self):
        """
        Tell the agent that it's training phase.
        RL / AP: only train the online net
        VT: only train the task net
        VA: only train the action net
        """
        self.mode = "train"
        self.online_net.train()
        if self.task == "vt":
            self.task_net.train()
        else:
            self.task_net.eval()
        if self.task == "va":
            self.action_net.train()
        else:
            self.action_net.eval()

    def eval(self):
        """
        Tell the agent that it's evaluation phase.
        """
        self.mode = "eval"
        self.online_net.eval()
        self.task_net.eval()
        self.action_net.eval()

    def update_target_net(self):
        if self.target_net is not None:
            self.target_net.load_state_dict(self.online_net.state_dict())

    def reset_noise(self):
        raise Exception("I do not use noise net!")
    
    def zero_noise(self):
        raise Exception("I do not use noise net!")

    def load_pretrained_action_model(self, load_from, load_partial_graph=True):
        """
        If load_partial_graph: only load some parts
        """
        print("Loading action model from: {}".format(load_from))
        try:
            if self.use_cuda:
                pretrained_dict = torch.load(load_from)
            else:
                pretrained_dict = torch.load(load_from, map_location='cpu')
            model_dict = self.action_net.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            if load_partial_graph and len(self.load_parameter_keywords) > 0:
                print("Only load following parts: {}".format(self.load_parameter_keywords))
                tmp_pretrained_dict = {}
                for k, v in pretrained_dict.items():
                    for keyword in self.load_parameter_keywords:
                        if keyword in k:
                            tmp_pretrained_dict[k] = v
                            break
                pretrained_dict = tmp_pretrained_dict
            else:
                print("Load all parts")
            model_dict.update(pretrained_dict)
            self.action_net.load_state_dict(model_dict)
        except:
            raise Exception("Failed to load checkpoint...")

    def load_pretrained_task_model(self, load_from, load_partial_graph=True):
        print("Loading task model from: {}".format(load_from))
        try:
            if self.use_cuda:
                pretrained_dict = torch.load(load_from)
            else:
                pretrained_dict = torch.load(load_from, map_location='cpu')
            model_dict = self.task_net.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            if load_partial_graph and len(self.load_parameter_keywords) > 0:
                print("Only load following parts: {}".format(self.load_parameter_keywords))
                tmp_pretrained_dict = {}
                for k, v in pretrained_dict.items():
                    for keyword in self.load_parameter_keywords:
                        if keyword in k:
                            tmp_pretrained_dict[k] = v
                            break
                pretrained_dict = tmp_pretrained_dict
            else:
                print("Load all parts")
            model_dict.update(pretrained_dict)
            self.task_net.load_state_dict(model_dict)
        except:
            raise Exception("Failed to load checkpoint...")

    def load_pretrained_model(self, load_from, load_partial_graph=True):
        print("Loading model from: {}".format(load_from))
        try:
            if self.use_cuda:
                pretrained_dict = torch.load(load_from)
            else:
                pretrained_dict = torch.load(load_from, map_location='cpu')
            model_dict = self.online_net.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            if load_partial_graph and len(self.load_parameter_keywords) > 0:
                print("Only load following parts: {}".format(self.load_parameter_keywords))
                tmp_pretrained_dict = {}
                for k, v in pretrained_dict.items():
                    for keyword in self.load_parameter_keywords:
                        if keyword in k:
                            tmp_pretrained_dict[k] = v
                            break
                pretrained_dict = tmp_pretrained_dict
            else:
                print("Load all parts")
            model_dict.update(pretrained_dict)
            self.online_net.load_state_dict(model_dict)
        except:
            raise Exception("Failed to load checkpoint...")

    def save_action_model_to_path(self, save_to):
        torch.save(self.action_net.state_dict(), save_to)
        print("Saved action_net's checkpoint to: {}".format(save_to))

    def save_task_model_to_path(self, save_to):
        torch.save(self.task_net.state_dict(), save_to)
        print("Saved task_net's checkpoint to: {}".format(save_to))

    def save_model_to_path(self, save_to):
        torch.save(self.online_net.state_dict(), save_to)
        print("Saved online_net's checkpoint to: {}".format(save_to))

    def select_additional_infos(self):
        """
        Returns what additional information should be made available at each game step.
        Requested information will be included within the `infos` dictionary
        passed to `CustomAgent.act()`. To request specific information, create a
        :py:class:`textworld.EnvInfos <textworld.envs.wrappers.filter.EnvInfos>`
        and set the appropriate attributes to `True`. The possible choices are:
        * `description`: text description of the current room, i.e. output of the `look` command;
        * `inventory`: text listing of the player's inventory, i.e. output of the `inventory` command;
        * `max_score`: maximum reachable score of the game;
        * `objective`: objective of the game described in text;
        * `entities`: names of all entities in the game;
        * `verbs`: verbs understood by the the game;
        * `command_templates`: templates for commands understood by the the game;
        * `admissible_commands`: all commands relevant to the current state;
        In addition to the standard information, game specific information
        can be requested by appending corresponding strings to the `extras`
        attribute. For this competition, the possible extras are:
        * `'recipe'`: description of the cookbook;
        * `'walkthrough'`: one possible solution to the game (not guaranteed to be optimal);
        Example:
            Here is an example of how to request information and retrieve it.
            >>> from textworld import EnvInfos
            >>> request_infos = EnvInfos(description=True, inventory=True, extras=["recipe"])
            ...
            >>> env = gym.make(env_id)
            >>> ob, infos = env.reset()
            >>> print(infos["description"])
            >>> print(infos["inventory"])
            >>> print(infos["extra.recipe"])
        Notes:
            The following information *won't* be available at test time:
            * 'walkthrough'
        """
        request_infos = EnvInfos()
        request_infos.admissible_commands = True
        request_infos.description = True
        request_infos.location = True
        request_infos.facts = True
        request_infos.last_action = True
        request_infos.game = True
        if self.use_negative_reward:
            request_infos.has_lost = True
            request_infos.has_won = True
        return request_infos

    def select_additional_infos_lite(self):
        """
        Returns what additional information should be made available at each game step.
        Requested information will be included within the `infos` dictionary
        passed to `CustomAgent.act()`. To request specific information, create a
        :py:class:`textworld.EnvInfos <textworld.envs.wrappers.filter.EnvInfos>`
        and set the appropriate attributes to `True`. The possible choices are:
        * `description`: text description of the current room, i.e. output of the `look` command;
        * `inventory`: text listing of the player's inventory, i.e. output of the `inventory` command;
        * `max_score`: maximum reachable score of the game;
        * `objective`: objective of the game described in text;
        * `entities`: names of all entities in the game;
        * `verbs`: verbs understood by the the game;
        * `command_templates`: templates for commands understood by the the game;
        * `admissible_commands`: all commands relevant to the current state;
        In addition to the standard information, game specific information
        can be requested by appending corresponding strings to the `extras`
        attribute. For this competition, the possible extras are:
        * `'recipe'`: description of the cookbook;
        * `'walkthrough'`: one possible solution to the game (not guaranteed to be optimal);
        Example:
            Here is an example of how to request information and retrieve it.
            >>> from textworld import EnvInfos
            >>> request_infos = EnvInfos(description=True, inventory=True, extras=["recipe"])
            ...
            >>> env = gym.make(env_id)
            >>> ob, infos = env.reset()
            >>> print(infos["description"])
            >>> print(infos["inventory"])
            >>> print(infos["extra.recipe"])
        Notes:
            The following information *won't* be available at test time:
            * 'walkthrough'
        """
        request_infos = EnvInfos()
        request_infos.admissible_commands = True
        request_infos.description = False
        request_infos.location = False
        request_infos.facts = False
        request_infos.last_action = False
        request_infos.game = True
        if self.use_negative_reward:
            request_infos.has_lost = True
            request_infos.has_won = True
        return request_infos

    def init(self, batch_size):
        if self.task_step_threshold == -1:
            self.task_step_threshold_list = None
        else:
            self.task_step_threshold_list = [0] * batch_size

    def choose_model(self, use_model="online"):
        if use_model == "online":
            model = self.online_net
        elif use_model == "target":
            model = self.target_net
        elif use_model == "task_net":
            model = self.task_net
        elif use_model == "action_net":
            model = self.action_net
        else:
            raise NotImplementedError
        return model

    def get_word_input(self, input_strings):
        word_list = [item.split() for item in input_strings]
        word_id_list = [_words_to_ids(tokens, self.word2id) for tokens in word_list]
        input_word = pad_sequences(word_id_list, maxlen=max_len(word_id_list)).astype('int32')
        input_word = to_pt(input_word, self.use_cuda)
        return input_word
    
    def get_graph_adjacency_matrix(self, triplets):
        adj = np.zeros((len(triplets), len(self.relation_vocab), len(self.node_vocab), len(self.node_vocab)), dtype="float32")
        for b in range(len(triplets)):
            node_exists = set()
            for t in triplets[b]:
                node1, node2, relation = t
                assert node1 in self.node_vocab, node1 + " is not in node vocab"
                assert node2 in self.node_vocab, node2 + " is not in node vocab"
                assert relation in self.relation_vocab, relation + " is not in relation vocab"
                node1_id, node2_id, relation_id = _word_to_id(node1, self.node2id), _word_to_id(node2, self.node2id), _word_to_id(relation, self.relation2id)
                adj[b][relation_id][node1_id][node2_id] = 1.0
                adj[b][relation_id + self.origin_relation_number][node2_id][node1_id] = 1.0
                node_exists.add(node1_id)
                node_exists.add(node2_id)
            for node_id in list(node_exists):
                adj[b, -1, node_id, node_id] = 1.0
        adj = to_pt(adj, self.use_cuda, type='float')
        return adj

    def get_graph_node_name_input(self):
        res = copy.copy(self.node_vocab)
        input_node_name = self.get_word_input(res)  # num_node x words
        return input_node_name

    def get_graph_relation_name_input(self):
        res = copy.copy(self.relation_vocab)
        res = [item.replace("_", " ") for item in res]
        input_relation_name = self.get_word_input(res)  # num_node x words
        return input_relation_name
    
    def get_action_candidate_list_input(self, action_candidate_list):
        batch_size = len(action_candidate_list)
        max_num_candidate = max_len(action_candidate_list)
        input_action_candidate_list = []
        for i in range(batch_size):
            word_level = self.get_word_input(action_candidate_list[i])
            input_action_candidate_list.append(word_level)
        max_word_num = max([item.size(1) for item in input_action_candidate_list])
        input_action_candidate = np.zeros((batch_size, max_num_candidate, max_word_num))
        input_action_candidate = to_pt(input_action_candidate, self.use_cuda, type="long")
        for i in range(batch_size):
            input_action_candidate[i, :input_action_candidate_list[i].size(0), :input_action_candidate_list[i].size(1)] = input_action_candidate_list[i]
        return input_action_candidate

    def encode_graph(self, graph_input, use_model):
        model = self.choose_model(use_model)
        input_node_name = self.get_graph_node_name_input()
        input_relation_name = self.get_graph_relation_name_input()
        if isinstance(graph_input, list):
            adjacency_matrix = self.get_graph_adjacency_matrix(graph_input)
        elif isinstance(graph_input, torch.Tensor):
            adjacency_matrix = graph_input
        else:
            raise NotImplementedError
        node_encoding_sequence, node_mask = model.encode_graph(input_node_name, input_relation_name, adjacency_matrix)
        return node_encoding_sequence, node_mask
    
    def encode_task(self, tasks, use_model):
        assert use_model in {"online", "target"}
        model = self.choose_model(use_model)
        input_tasks = self.get_word_input(tasks)        
        tasks_encoding_sequence, tasks_mask = model.encode_task(input_tasks)
        return tasks_encoding_sequence, tasks_mask

    def encode(self, observation_strings, graph_input, tasks, use_model):
        assert use_model in {"online", "target"}
        model = self.choose_model(use_model)
        # text obs
        assert (not self.enable_text_input), "Disable text input!"
        obs_encoding_sequence, obs_mask = None, None
        # graph obs
        assert (self.enable_graph_input), "Enable graph input!"
        node_encoding_sequence, node_mask = self.encode_graph(graph_input, use_model=use_model)
        # task
        tasks_encoding_sequence, tasks_mask = self.encode_task(tasks, use_model=use_model)
        return obs_encoding_sequence, obs_mask, node_encoding_sequence, node_mask, tasks_encoding_sequence, tasks_mask

    def va_scoring_paired(self, task_list, action_list, use_model=None):
        """
        :param task_list: a list of 1-lists, each 1-list contains a string
        :param action_list: a list of 1-lists, each 1-list contains a string
        :output va_scores: [batch, 2]
        """
        assert use_model == "action_net"
        model = self.choose_model(use_model)
        # 1. Get the task id list
        input_task_ids = self.get_action_candidate_list_input(task_list)
        # 2. Get the action id list
        input_action_ids = self.get_action_candidate_list_input(action_list)
        # 3. Get the scores (we do not use masks)
        va_scores = model.score_task_action_pair(input_task_ids, input_action_ids)
        return va_scores

    def task_scoring_paired(self, graph_input, task_list, use_model=None):
        assert use_model == "task_net"
        model = self.choose_model(use_model)
        input_node_name = self.get_graph_node_name_input()
        input_relation_name = self.get_graph_relation_name_input()
        if isinstance(graph_input, list):
            adjacency_matrix = self.get_graph_adjacency_matrix(graph_input)
        elif isinstance(graph_input, torch.Tensor):
            adjacency_matrix = graph_input
        else:
            raise NotImplementedError
        node_encoding_sequence, node_mask = model.encode_graph(input_node_name, 
                                                               input_relation_name, 
                                                               adjacency_matrix)
        input_task_candidate_list = self.get_action_candidate_list_input(task_list)
        assert input_task_candidate_list.shape[1] == 1, "There should be only 1 task per pair!"
        task_scores, task_masks = model.score_graph_task_pair(
                                                            input_task_candidate_list,
                                                            node_encoding_sequence, node_mask)
        return task_scores, task_masks

    def action_scoring(self, action_candidate_list, 
                      h_og=None, obs_mask=None, h_go=None, node_mask=None, h_tasks=None, tasks_mask=None, 
                      previous_h=None, previous_c=None, use_model=None):
        assert self.task == "rl"
        assert use_model in {"online", "target"}
        model = self.choose_model(use_model)
        input_action_candidate = self.get_action_candidate_list_input(action_candidate_list)
        action_scores, action_masks, new_h, new_c = model.score_actions(input_action_candidate, 
                                                                        h_og, obs_mask, 
                                                                        h_go, node_mask, 
                                                                        h_tasks, tasks_mask,     
                                                                        previous_h, previous_c)  
        return action_scores, action_masks, new_h, new_c

    def act_greedy(self, observation_strings, graph_input, action_candidate_list, tasks, previous_h=None, previous_c=None):
        assert self.task == "rl"
        with torch.no_grad():
            h_og, obs_mask, h_go, node_mask, h_tasks, tasks_mask = self.encode(observation_strings, 
                                                                               graph_input, 
                                                                               tasks, 
                                                                               use_model="online")
            action_scores, action_masks, new_h, new_c = self.action_scoring(action_candidate_list, 
                                                                            h_og, obs_mask, h_go, node_mask, h_tasks, tasks_mask, 
                                                                            previous_h, previous_c, use_model="online")
            action_indices_maxq = self.choose_maxQ_action(action_scores, action_masks)
            chosen_indices = action_indices_maxq
            chosen_indices = chosen_indices.astype(int)
            chosen_actions = [item[idx] for item, idx in zip(action_candidate_list, chosen_indices)]
            return chosen_actions, chosen_indices, new_h, new_c

    def act_random(self, observation_strings, graph_input, action_candidate_list, tasks, previous_h=None, previous_c=None):
        assert self.task == "rl"
        with torch.no_grad():
            h_og, obs_mask, h_go, node_mask, h_tasks, tasks_mask = self.encode(observation_strings, 
                                                                               graph_input,
                                                                               tasks, 
                                                                               use_model="online")
            action_scores, _, new_h, new_c = self.action_scoring(action_candidate_list, 
                                                                 h_og, obs_mask, h_go, node_mask, h_tasks, tasks_mask, 
                                                                 previous_h, previous_c, use_model="online")
            action_indices_random = self.choose_random_action(action_scores, action_candidate_list)
            chosen_indices = action_indices_random
            chosen_indices = chosen_indices.astype(int)
            chosen_actions = [item[idx] for item, idx in zip(action_candidate_list, chosen_indices)]
            return chosen_actions, chosen_indices, new_h, new_c

    def act(self, observation_strings, graph_input, action_candidate_list, tasks, previous_h=None, previous_c=None, random=False):
        assert self.task == "rl"
        with torch.no_grad():
            if self.mode == "eval":
                return self.act_greedy(observation_strings, graph_input, action_candidate_list, tasks, previous_h, previous_c)
            if random:
                return self.act_random(observation_strings, graph_input, action_candidate_list, tasks, previous_h, previous_c)
            batch_size = len(observation_strings)
            h_og, obs_mask, h_go, node_mask, h_tasks, tasks_mask = self.encode(observation_strings, 
                                                                               graph_input, 
                                                                               tasks, 
                                                                               use_model="online")
            action_scores, action_masks, new_h, new_c = self.action_scoring(action_candidate_list, 
                                                                            h_og, obs_mask, h_go, node_mask, h_tasks, tasks_mask, 
                                                                            previous_h, previous_c, use_model="online")
            action_indices_maxq = self.choose_maxQ_action(action_scores, action_masks)
            action_indices_random = self.choose_random_action(action_scores, action_candidate_list)
            rand_num = np.random.uniform(low=0.0, high=1.0, size=(batch_size,))
            less_than_epsilon = (rand_num < self.epsilon).astype("float32")  
            greater_than_epsilon = 1.0 - less_than_epsilon
            chosen_indices = less_than_epsilon * action_indices_random + greater_than_epsilon * action_indices_maxq
            chosen_indices = chosen_indices.astype(int)
            chosen_actions = [item[idx] for item, idx in zip(action_candidate_list, chosen_indices)]
            return chosen_actions, chosen_indices, new_h, new_c

    def choose_random_action(self, action_rank, action_unpadded=None):
        assert self.task == "rl"
        batch_size = action_rank.size(0)
        action_space_size = action_rank.size(1)
        if action_unpadded is None:
            indices = np.random.choice(action_space_size, batch_size)
        else:
            indices = []
            for j in range(batch_size):
                indices.append(np.random.choice(len(action_unpadded[j])))
            indices = np.array(indices)
        return indices

    def choose_maxQ_action(self, action_rank, action_mask=None):
        assert self.task == "rl"
        # minus the min value, so that all values are non-negative
        action_rank = action_rank - torch.min(action_rank, -1, keepdim=True)[0] + 1e-2  
        if action_mask is not None:
            assert action_mask.size() == action_rank.size(), (action_mask.size().shape, action_rank.size())
            action_rank = action_rank * action_mask
        action_indices = torch.argmax(action_rank, -1)  # batch
        return to_np(action_indices)

    def get_dqn_loss(self, episode_no):
        """
        Update neural model in agent. In this example we follow algorithm
        of updating model in dqn with replay memory.
        """
        assert self.task == "rl"
        if len(self.dqn_memory) < self.replay_batch_size:
            return None, None
        data = self.dqn_memory.sample(self.replay_batch_size, 
                                      beta=self.beta_scheduler.value(episode_no), 
                                      multi_step=self.multi_step)
        if data is None:
            return None, None
        obs_list, _, candidate_list, tasks, action_indices, graph_triplet_list, rewards, next_obs_list, _, next_candidate_list, next_graph_triplet_list, actual_indices, actual_ns, prior_weights = data
        h_og, obs_mask, h_go, node_mask, h_tasks, tasks_mask = self.encode(obs_list, 
                                                                           graph_triplet_list, 
                                                                           tasks, 
                                                                           use_model="online")
        action_scores, _, _, _ = self.action_scoring(candidate_list, 
                                                     h_og, obs_mask, h_go, node_mask, h_tasks, tasks_mask, 
                                                     None, None, use_model="online")
        action_indices = to_pt(action_indices, enable_cuda=self.use_cuda, type='long').unsqueeze(-1)
        q_value = ez_gather_dim_1(action_scores, action_indices).squeeze(1)  
        with torch.no_grad():
            h_og, obs_mask, h_go, node_mask, h_tasks, tasks_mask = self.encode(next_obs_list, 
                                                                               next_graph_triplet_list, 
                                                                               tasks, 
                                                                               use_model="online")
            next_action_scores, next_action_masks, _, _ = self.action_scoring(next_candidate_list, 
                                                                              h_og, obs_mask, h_go, node_mask, h_tasks, tasks_mask, 
                                                                              None, None, use_model="online")
            next_action_indices = self.choose_maxQ_action(next_action_scores, next_action_masks)  
            next_action_indices = to_pt(next_action_indices, enable_cuda=self.use_cuda, type='long').unsqueeze(-1)
            h_og, obs_mask, h_go, node_mask, h_tasks, tasks_mask = self.encode(next_obs_list, 
                                                                               next_graph_triplet_list, 
                                                                               tasks, 
                                                                               use_model="target")
            next_action_scores, next_action_masks, _, _ = self.action_scoring(next_candidate_list, 
                                                                              h_og, obs_mask, h_go, node_mask, h_tasks, tasks_mask, 
                                                                              None, None, use_model="target")
            next_q_value = ez_gather_dim_1(next_action_scores, next_action_indices).squeeze(1)  
            discount = to_pt((np.ones_like(actual_ns) * self.discount_gamma_game_reward) ** actual_ns, self.use_cuda, type="float")
        rewards = rewards + next_q_value * discount  
        loss = F.smooth_l1_loss(q_value, rewards, reduce=False)  
        prior_weights = to_pt(prior_weights, enable_cuda=self.use_cuda, type="float")
        loss = loss * prior_weights
        loss = torch.mean(loss)
        abs_td_error = np.abs(to_np(q_value - rewards))
        new_priorities = abs_td_error + self.prioritized_replay_eps
        self.dqn_memory.update_priorities(actual_indices, new_priorities)
        return loss, q_value

    def update_dqn(self, episode_no):
        assert self.task == "rl"
        if self.real_valued_graph:
            raise Exception("get_dqn_loss_with_real_graphs() should not be used")
            # dqn_loss, q_value = self.get_dqn_loss_with_real_graphs(episode_no)
        elif self.enable_recurrent_memory:
            raise Exception("get_drqn_loss() should not be used")
            # dqn_loss, q_value = self.get_drqn_loss(episode_no)
        else:
            dqn_loss, q_value = self.get_dqn_loss(episode_no)
        if dqn_loss is None:
            return None, None
        self.online_net.zero_grad()
        self.optimizer.zero_grad()
        dqn_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), self.clip_grad_norm)
        self.optimizer.step()
        return to_np(torch.mean(dqn_loss)), to_np(torch.mean(q_value))

    def update_task_candidate_list(self, current_triplets):
        assert self.task == "rl"
        batch_size = len(current_triplets)
        available_task_list = []
        ingredient_list_batch = [kg_utils.get_ingredients(t) for t in current_triplets]
        for i in range(batch_size):
            # Expert
            if self.use_expert_task_net:
                available_tasks = kg_utils.get_available_tasks(ingredient_list_batch[i], current_triplets[i])
            # Pretrained
            else:
                available_tasks = self.get_available_tasks_VT(ingredient_list_batch[i], current_triplets[i])
            # preprocess tasks to be like action candidates
            available_tasks_preproc = [preproc(item, tokenizer=self.nlp) for item in available_tasks]
            available_task_list.append(available_tasks_preproc)
        self.available_task_list = available_task_list

    def get_available_tasks_VT(self, ingredient_list, single_graph, k=3):
        """
        :param ingredient_list: a list of strings (ingredients only)
        :param single_graph: a list of triplets (a single graph)
        :param k: check the special condition when there's no 1s
        """
        assert self.task == "rl"
        assert not self.use_expert_task_net
        task_obj_candidates = ["knife", "meal"]
        task_obj_candidates += ingredient_list
        all_tasks = []
        for obj in task_obj_candidates:
            if obj == 'knife':
                verb_list = ['get']
            elif obj == 'meal':
                verb_list = ['make']
            else:
                verb_list = ['chop','dice','slice','fry','grill','roast','get']
            for verb in verb_list:
                task = "{} {}".format(verb, obj)
                all_tasks.append([task])
        expanded_graphs = [single_graph for _ in range(len(all_tasks))]
        with torch.no_grad():
            task_scores, _ = self.task_scoring_paired(expanded_graphs, all_tasks, use_model="task_net")
        task_ret = to_np(task_scores.log_softmax(dim=1))
        predictions = np.argmax(task_ret, -1).tolist()
        if np.sum(predictions) > 0:
            available_tasks = [all_tasks[i][0] for i in range(len(all_tasks)) if predictions[i] == 1]
        else:
            topk_indices = np.argsort(task_ret[:,1] - task_ret[:,0])[-k:]
            available_tasks = [all_tasks[i][0] for i in topk_indices]
        assert len(available_tasks) > 0
        return available_tasks

    def sample_tasks(self, curr_tasks):
        """
        Check whether current tasks are still available, if not, sample a new task
        Return two list of new tasks
        Note that at the beginning of an episode: "curr_tasks" should be a batch-list of None
        """
        assert self.task == "rl"
        batch_size = len(curr_tasks)
        task_verbs_list, task_objs_list = [], []
        for i in range(batch_size):
            curr_task = curr_tasks[i]
            curr_available_tasks = self.available_task_list[i]
            if curr_task in curr_available_tasks:
                if self.task_step_threshold == -1:
                    assert self.task_step_threshold_list is None
                    task_verb, task_obj = self._get_task_verb_obj(curr_task)
                else: 
                    assert self.task_step_threshold_list is not None
                    if self.task_step_threshold_list[i] >= self.task_step_threshold:
                        new_task = np.random.choice(curr_available_tasks)
                        task_verb, task_obj = self._get_task_verb_obj(new_task)
                        self.task_step_threshold_list[i] = 0
                    else:
                        task_verb, task_obj = self._get_task_verb_obj(curr_task)
                        self.task_step_threshold_list[i] += 1
            else:
                new_task = np.random.choice(curr_available_tasks)
                task_verb, task_obj = self._get_task_verb_obj(new_task)
                if self.task_step_threshold != -1:
                    self.task_step_threshold_list[i] = 0
            task_verbs_list.append(task_verb)
            task_objs_list.append(task_obj)
        return task_verbs_list, task_objs_list

    def _get_task_verb_obj(self, single_task):
        items = single_task.split(" ")
        return items[0], " ".join(items[1:])
    
    def refine_action_candidate(self, task_verbs, task_objs, action_candidate_list, dones):
        assert self.task == "rl"
        if self.use_TRA == 'noTRA':
            return action_candidate_list
        elif self.use_TRA == 'expertTRA':
            result_list = []
            batch_size = len(task_verbs)
            for i in range(batch_size):
                if dones[i]:
                    result_list.append(action_candidate_list[i])
                else:
                    result_list_single = refine_action_utils.refine_action_candidate_single(task_verbs[i],
                                                                                        task_objs[i],
                                                                                        action_candidate_list[i])
                    result_list.append(result_list_single)
        else:
            assert self.use_TRA == 'pretrainedTRA'
            result_list = []
            batch_size = len(task_verbs)
            for i in range(batch_size):
                if dones[i]:
                    result_list.append(action_candidate_list[i])
                else:
                    curr_task_verb = task_verbs[i]
                    curr_task_obj = task_objs[i]
                    curr_task = "{} {}".format(curr_task_verb, curr_task_obj)
                    curr_action_candidate = action_candidate_list[i]
                    if curr_task_obj == "meal":
                        result_list_single = []
                        for j in range(len(curr_action_candidate)):
                            action_ = curr_action_candidate[j]
                            verb = action_.split(" ")[0]
                            if verb in {'go', 'open'}:
                                result_list_single.append(action_)
                            elif ("meal" in action_) and (verb in {"take","prepare","eat"}):
                                result_list_single.append(action_)
                        if len(result_list_single) == 0:
                            result_list.append(curr_action_candidate)
                        else:
                            result_list.append(result_list_single)
                    else:
                        num_actions = len(curr_action_candidate)
                        task_input = [[curr_task] for _ in range(num_actions)]
                        action_input = [[single_action] for single_action in curr_action_candidate]
                        with torch.no_grad():
                            action_scores = self.va_scoring_paired(task_input, action_input, use_model="action_net")
                        action_ret = to_np(action_scores.log_softmax(dim=1))
                        predictions = np.argmax(action_ret, -1).tolist()
                        result_list_single = []
                        for j in range(len(curr_action_candidate)):
                            action_ = curr_action_candidate[j]
                            verb = action_.split(" ")[0]
                            if verb in {'go', 'open'}:
                                result_list_single.append(action_)
                            elif verb not in {'put','drop','close','examine','look'}:
                                if predictions[j] == 1:
                                    result_list_single.append(action_)
                        if len(result_list_single) == 0:
                            result_list.append(curr_action_candidate)
                        else:
                            result_list.append(result_list_single)
        return result_list


    def finish_of_episode(self, episode_no, batch_size):
        # Update target network
        if (episode_no + batch_size) % self.target_net_update_frequency <= episode_no % self.target_net_update_frequency:
            self.update_target_net()
        # Decay lambdas
        if episode_no < self.learn_start_from_this_episode:
            return
        # Update epsilon
        if episode_no < self.epsilon_anneal_episodes + self.learn_start_from_this_episode:
            self.epsilon = self.epsilon_scheduler.value(episode_no - self.learn_start_from_this_episode)
            self.epsilon = max(self.epsilon, 0.0)

    def get_game_info_at_certain_step_fully_observable(self, obs, infos):
        """
        Get all needed info from game engine for training.
        Arguments:
            obs: Previous command's feedback for each game.
            infos: Additional information for each game.
        """
        batch_size = len(obs)
        observation_strings = [preproc(item, tokenizer=self.nlp) for item in obs]
        action_candidate_list = []
        for b in range(batch_size):
            ac = [preproc(item, tokenizer=self.nlp) for item in infos["admissible_commands"][b]]
            action_candidate_list.append(ac)
        current_triplets = [] 
        for b in range(batch_size):
            new_f = set(process_fully_obs_facts(infos["game"][b], infos["facts"][b]))
            triplets = serialize_facts(new_f)
            current_triplets.append(triplets)
        return observation_strings, current_triplets, action_candidate_list, None, None

    def get_game_info_at_certain_step(self, obs, infos, prev_actions=None, prev_facts=None, return_gt_commands=False):
        """
        Get all needed info from game engine for training.
        Arguments:
            obs: Previous command's feedback for each game.
            infos: Additional information for each game.
        """
        if self.fully_observable_graph:
            return self.get_game_info_at_certain_step_fully_observable(obs, infos)
        else:
            raise Exception("Only use full graph!")
        batch_size = len(obs)
        observation_strings = [preproc(item, tokenizer=self.nlp) for item in obs]
        action_candidate_list = []
        for b in range(batch_size):
            ac = [preproc(item, tokenizer=self.nlp) for item in infos["admissible_commands"][b]]
            action_candidate_list.append(ac)
        new_facts = []
        current_triplets = []  
        commands_from_env = []  
        for b in range(batch_size):
            if prev_facts is None:
                new_f = process_facts(None, infos["game"][b], infos["facts"][b], None, None)
                prev_f = set()
            else:
                new_f = process_facts(prev_facts[b], infos["game"][b], infos["facts"][b], infos["last_action"][b], prev_actions[b])
                prev_f = prev_facts[b]
            new_facts.append(new_f)
            triplets = serialize_facts(new_f)
            current_triplets.append(triplets)
            target_commands = gen_graph_commands(new_f - prev_f, cmd="add") + gen_graph_commands(prev_f - new_f, cmd="delete")
            commands_from_env.append(target_commands)
        target_command_strings = []
        if return_gt_commands:
            target_command_strings = [" <sep> ".join(sort_target_commands(tgt_cmds)) for tgt_cmds in commands_from_env]
        return observation_strings, current_triplets, action_candidate_list, target_command_strings, new_facts

    def get_game_info_at_certain_step_lite(self, obs, infos):
        """
        Get all needed info from game engine for training.
        Arguments:
            obs: Previous command's feedback for each game.
            infos: Additional information for each game.
        """
        if self.fully_observable_graph:
            return self.get_game_info_at_certain_step_fully_observable(obs, infos)
        batch_size = len(obs)
        observation_strings = [preproc(item, tokenizer=self.nlp) for item in obs]
        action_candidate_list = []
        for b in range(batch_size):
            ac = [preproc(item, tokenizer=self.nlp) for item in infos["admissible_commands"][b]]
            action_candidate_list.append(ac)
        return observation_strings, action_candidate_list

    def update_knowledge_graph_triplets(self, triplets, prediction_strings):
        new_triplets = []
        for i in range(len(triplets)):
            predict_cmds = prediction_strings[i].split("<sep>")
            if predict_cmds[-1].endswith("<eos>"):
                predict_cmds[-1] = predict_cmds[-1][:-5].strip()
            else:
                predict_cmds = predict_cmds[:-1]
            if len(predict_cmds) == 0:
                new_triplets.append(triplets[i])
                continue
            predict_cmds = [" ".join(item.split()) for item in predict_cmds]
            predict_cmds = [item for item in predict_cmds if len(item) > 0]
            new_triplets.append(update_graph_triplets(triplets[i], predict_cmds, self.node_vocab, self.relation_vocab))
        return new_triplets

    def get_graph_rewards(self, prev_triplets, current_triplets):
        batch_size = len(current_triplets)
        if self.graph_reward_lambda == 0:
            return [0.0 for _ in current_triplets]
        if self.graph_reward_type == "triplets_increased":
            rewards = [float(len(c_triplet) - len(p_triplet)) for p_triplet, c_triplet in zip(prev_triplets, current_triplets)]
        elif self.graph_reward_type == "triplets_difference":
            rewards = []
            for b in range(batch_size):
                curr = current_triplets[b]
                prev = prev_triplets[b]
                curr = set(["|".join(item) for item in curr])
                prev = set(["|".join(item) for item in prev])
                diff_num = len(prev - curr) + len(curr - prev)
                rewards.append(float(diff_num))
        else:
            raise NotImplementedError
        rewards = [min(1.0, max(0.0, float(item) * self.graph_reward_lambda)) for item in rewards]
        return rewards

    def reset_binarized_counter(self, batch_size):
        self.binarized_counter_dict = [{} for _ in range(batch_size)]

    def get_binarized_count(self, observation_strings, update=True):
        batch_size = len(observation_strings)
        count_rewards = []
        for i in range(batch_size):
            concat_string = observation_strings[i]
            if concat_string not in self.binarized_counter_dict[i]:
                self.binarized_counter_dict[i][concat_string] = 0.0
            if update:
                self.binarized_counter_dict[i][concat_string] += 1.0
            r = self.binarized_counter_dict[i][concat_string]
            r = float(r == 1.0)
            count_rewards.append(r)
        return count_rewards

    def get_vt_logits(self, current_triplets, task_list, label_list):
        """
        :param curr_triplets: a batched list of graphs, each graph is a list of triplets, each triplet is a 3-list
        :param task_list: a batched list of 1-list, each 1-list contains a string (task)
        :param label_list: a batched list of 0/1s
        :output: task_loss, task_ret
        """
        assert self.task == 'vt'
        task_pt_labels = to_pt(np.array(label_list), self.use_cuda, type="long")
        task_scores, task_masks = self.task_scoring_paired(current_triplets, task_list, use_model="task_net")
        assert task_masks.shape[1] == 1
        if self.vt_loss == 'ce':
            task_loss = torch.nn.CrossEntropyLoss()(task_scores, task_pt_labels)
        elif self.vt_loss == 'focal':
            task_loss = FocalLoss(use_cuda=self.use_cuda)(task_scores, task_pt_labels)
        else:
            assert self.vt_loss == 'focal_alt'
            task_loss = FocalLoss(use_cuda=self.use_cuda, use_alter=True)(task_scores, task_pt_labels)
        task_ret = to_np(task_scores)
        return task_loss, task_ret

    def get_va_logits(self, task_list, action_list, label_list):
        """
        :param task_list: a list of 1-lists, each 1-list contains a string
        :param action_list: a list of 1-lists, each 1-list contains a string
        :param label_list: a list of 0/1s
        :output: action_loss, action_ret
        """
        assert self.task == 'va'
        action_pt_labels = to_pt(np.array(label_list), self.use_cuda, type="long")
        action_scores = self.va_scoring_paired(task_list, action_list, use_model="action_net")
        if self.va_loss == 'ce':
            action_loss = torch.nn.CrossEntropyLoss()(action_scores, action_pt_labels)
        elif self.va_loss == 'focal':
            action_loss = FocalLoss(use_cuda=self.use_cuda)(action_scores, action_pt_labels)
        else:
            assert self.va_loss == 'focal_alt'
            action_loss = FocalLoss(use_cuda=self.use_cuda, use_alter=True)(action_scores, action_pt_labels)
        action_ret = to_np(action_scores)
        return action_loss, action_ret

    def get_action_prediction_logits(self, current_triplets, previous_triplets, target_action, action_candidates):
        assert self.task == 'ap'
        h_g, node_mask = self.encode_graph(current_triplets, use_model="online")
        prev_h_g, prev_node_mask = self.encode_graph(previous_triplets, use_model="online")
        graph_mask = torch.gt(node_mask + prev_node_mask, 0.0).float()  
        labels, action_candidate_list = generate_labels_for_ap(target_action, 
                                                               action_candidates, 
                                                               k_way_classification=self.ap_k_way_classification)
        input_action_candidate = self.get_action_candidate_list_input(action_candidate_list)  
        batch_size = len(action_candidate_list)
        num_candidate, candidate_len = input_action_candidate.size(1), input_action_candidate.size(2)
        np_labels = pad_sequences(labels)
        pt_labels = to_pt(np_labels, self.use_cuda, type="long")
        cand_mask = np.zeros((batch_size, num_candidate), dtype="float32")
        for b in range(batch_size):
            cand_mask[b, :len(action_candidate_list[b])] = 1.0
        cand_mask = to_pt(cand_mask, self.use_cuda, type='float')
        attended_h_g = self.online_net.ap_attention_prj(self.online_net.ap_attention(h_g, prev_h_g, node_mask, prev_node_mask))
        ave_attended_h_g = masked_mean(attended_h_g, graph_mask, dim=1).unsqueeze(1)
        prev_attended_h_g = self.online_net.ap_attention_prj(self.online_net.ap_attention(prev_h_g, h_g, prev_node_mask, node_mask))
        ave_prev_attended_h_g = masked_mean(prev_attended_h_g, graph_mask, dim=1).unsqueeze(1)
        global_g = ave_attended_h_g.expand(-1, num_candidate, -1)  
        prev_global_g = ave_prev_attended_h_g.expand(-1, num_candidate, -1)  
        global_g = global_g * cand_mask.unsqueeze(-1)  
        prev_global_g = prev_global_g * cand_mask.unsqueeze(-1)  
        cand_encoding = []
        input_action_candidate = input_action_candidate.view(batch_size * num_candidate, candidate_len)
        tmp_batch_size = self.batch_size if self.mode == "train" else self.eval_batch_size
        n_tmp_batches = (input_action_candidate.size(0) + tmp_batch_size - 1) // tmp_batch_size
        for tmp_batch_id in range(n_tmp_batches):
            tmp_batch = input_action_candidate[tmp_batch_id * tmp_batch_size: (tmp_batch_id + 1) * tmp_batch_size]
            tmp_cand_encoding_sequence, tmp_cand_mask = self.online_net.encode_text_for_pretraining_tasks(tmp_batch)
            tmp_cand_encoding = masked_mean(tmp_cand_encoding_sequence, tmp_cand_mask, dim=1)
            cand_encoding.append(tmp_cand_encoding)
        cand_encoding = torch.cat(cand_encoding, 0)  
        cand_encoding = cand_encoding.view(batch_size, num_candidate, -1)  
        cand_encoding = cand_encoding * cand_mask.unsqueeze(-1)  
        apmlp_input = torch.cat([global_g, prev_global_g, cand_encoding], -1) 
        cand_mask_squared = torch.bmm(cand_mask.unsqueeze(-1), cand_mask.unsqueeze(1))  
        ap_ret, _ = self.online_net.ap_self_attention(apmlp_input, cand_mask_squared, apmlp_input, apmlp_input)  
        ap_ret = ap_ret * cand_mask.unsqueeze(-1)
        ap_ret = self.online_net.ap_linear_1(ap_ret) 
        ap_ret = torch.relu(ap_ret)
        ap_ret = ap_ret * cand_mask.unsqueeze(-1)
        ap_ret = self.online_net.ap_linear_2(ap_ret).squeeze(-1)  
        ap_ret = ap_ret * cand_mask
        ap_ret = ap_ret.masked_fill((1.0 - cand_mask).bool(), float('-inf'))
        loss = torch.nn.CrossEntropyLoss()(ap_ret, torch.argmax(pt_labels, -1))
        return loss, to_np(ap_ret), np_labels, action_candidate_list


