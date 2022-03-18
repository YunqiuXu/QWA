import datetime
import os
import time
import json
import math
import torch
import numpy as np
from os.path import join as pjoin
from glob import glob

from dataset_AP import APData
from agent import Agent
import generic
import evaluate


def train():
    print("===== 1. Load configs =====")
    time_1 = time.time()
    config = generic.load_config()
    output_dir = config['general']['output_dir']
    print("Output dir: {}".format(output_dir))

    print("===== 2. Init agent =====")
    agent = Agent(config)
    requested_infos = agent.select_additional_infos()

    print("===== 3. Build dataset as an env =====")
    env = APData(config)
    env.split_reset("train")

    json_file_name = agent.experiment_tag.replace(" ", "_")
    ave_train_loss = generic.HistoryScoreCache(capacity=500)
    episode_no = 0
    batch_no = 0
    best_eval_acc = 0.0   
    print("===== ===== ===== Start training ===== ===== =====")
    while(True):
        if episode_no > agent.max_episode:
            break
        agent.train()
        current_triplets, previous_triplets, target_action, action_candidates = env.get_batch()
        curr_batch_size = len(current_triplets)
        loss, _, _, _ = agent.get_action_prediction_logits(current_triplets, previous_triplets, target_action, action_candidates)
        
        # Update Model
        agent.online_net.zero_grad()
        agent.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.online_net.parameters(), agent.clip_grad_norm)
        agent.optimizer.step()
        loss = generic.to_np(loss)
        ave_train_loss.push(loss)
        
        # lr schedule
        if batch_no < agent.learning_rate_warmup_until:
            cr = agent.init_learning_rate / math.log2(agent.learning_rate_warmup_until)
            learning_rate = cr * math.log2(batch_no + 1)
        else:
            learning_rate = agent.init_learning_rate
        for param_group in agent.optimizer.param_groups:
            param_group['lr'] = learning_rate
        episode_no += curr_batch_size
        batch_no += 1
        if agent.report_frequency == 0 or (episode_no % agent.report_frequency > (episode_no - curr_batch_size) % agent.report_frequency):
            print("{} episodes finished".format(episode_no))
            continue

        # validating
        eval_acc, eval_loss = 0.0, 0.0
        
        if episode_no % agent.report_frequency <= (episode_no - curr_batch_size) % agent.report_frequency:
            print("===== ===== ===== Validating ===== ===== =====")
            eval_loss, eval_acc = evaluate.evaluate_action_prediction(env, agent, "valid")
            if eval_acc > best_eval_acc:
                best_eval_acc = eval_acc
                agent.save_model_to_path(output_dir + "/" + agent.experiment_tag + "_model.pt")
                agent.save_model_to_path(output_dir + "/" + agent.experiment_tag + "_model_{}m.pt".format(episode_no // 1000000))
                print("Saving best with valid acc: {:2.3f}".format(best_eval_acc))
            env.split_reset("train")
        time_2 = time.time()
        progress = "=== Epi: {:3d}|Time: {:.2f}m|TrL {:2.3f}|EvL {:2.3f}|EvAcc {:2.3f}"
        progress = progress.format(episode_no, (time_2 - time_1) / 60.,
                                   ave_train_loss.get_avg(),
                                   eval_loss, eval_acc)
        print(progress)
        _s = json.dumps({"Time":         "{:.2f}".format((time_2 - time_1) / 60.),
                         "TrL":          str(ave_train_loss.get_avg()),
                         "EvL":          str(eval_loss),
                         "EvAcc":        str(eval_acc)})
        with open(output_dir + "/" + json_file_name + '.json', 'a+') as outfile:
            outfile.write(_s + '\n')
            outfile.flush()

if __name__ == '__main__':
    train()
