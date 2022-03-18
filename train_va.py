import datetime
import os
import time
import json
import math
import torch
import numpy as np
from os.path import join as pjoin
from glob import glob

from dataset_VA import VAData
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
    assert agent.task == "va", "Wrong task {}, should be va!".format(agent.task)

    if agent.load_action_net == 'apInit':
        print("Init action net with AP pre-trained embeddings")
        ap_weight_path = config['general']['AP_ptr_dir']
        agent.load_pretrained_action_model(ap_weight_path, load_partial_graph=True)
    elif agent.load_action_net == 'rdInit':
        print("Init action net with random weights")
    else:
        raise Exception("Unsupported agent.load_action_net: {}".format(agent.load_action_net))

    print("===== 3. Build dataset as an env =====")
    data_path = config['va']['data_path']
    data_path_dict = {
                    "train": ["{}/VA_train.json".format(data_path)],
                    "valid": ["{}/VA_valid.json".format(data_path)],
                    "test": ["{}/VA_test.json".format(data_path)]
                 }
    env = VAData(config, data_path_dict)
    env.split_reset("train")

    json_file_name = agent.experiment_tag.replace(" ", "_")
    ave_train_action_loss = generic.HistoryScoreCache(capacity=500)
    episode_no = 0
    batch_no = 0
    best_total_eval_f1 = 0.0
    
    print("===== ===== ===== Start training ===== ===== =====")
    while(True):
        if episode_no > agent.max_episode:
            break
        agent.train()
        task, action, label = env.get_batch()
        curr_batch_size = len(task)
        action_loss, _ = agent.get_va_logits(task, action, label)
        agent.action_net.zero_grad()
        agent.action_optimizer.zero_grad()
        action_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.action_net.parameters(), agent.clip_grad_norm)
        agent.action_optimizer.step()
        ave_train_action_loss.push(generic.to_np(action_loss))
        if batch_no < agent.learning_rate_warmup_until:
            cr = agent.init_learning_rate / math.log2(agent.learning_rate_warmup_until)
            learning_rate = cr * math.log2(batch_no + 1)
        else:
            learning_rate = agent.init_learning_rate
        for param_group in agent.action_optimizer.param_groups:
            param_group['lr'] = learning_rate
        episode_no += curr_batch_size
        batch_no += 1

        if agent.report_frequency == 0 or (episode_no % agent.report_frequency > (episode_no - curr_batch_size) % agent.report_frequency):
            continue
        
        if episode_no % agent.report_frequency <= (episode_no - curr_batch_size) % agent.report_frequency:
            print("===== ===== ===== Validating ===== ===== =====")
            eval_action_loss, eval_action_accuracy, eval_action_precision, eval_action_recall, eval_action_f1 = evaluate.evaluate_va(env, agent, "valid")
            if eval_action_f1 > best_total_eval_f1:
                best_total_eval_f1 = eval_action_f1
                agent.save_action_model_to_path(output_dir + "/" + agent.experiment_tag + "_action_model_f1.pt")
                agent.save_action_model_to_path(output_dir + "/" + agent.experiment_tag + "_action_model_f1_{}m.pt".format(episode_no // 1000000))
                print("Save action net with best total f1 score: {:2.3f}".format(best_total_eval_f1))

            print("===== ===== ===== Testing ===== ===== =====")
            test_action_loss, test_action_accuracy, test_action_precision, test_action_recall, test_action_f1 = evaluate.evaluate_va(env, agent, "test")

            env.split_reset("train")

            # Record progress
            time_2 = time.time()
            progress1 = "=== Epi: {:3d}|Time: {:.2f}m|TrainLoss {:2.3f}".format(episode_no, 
                                                                                (time_2 - time_1) / 60., 
                                                                                ave_train_action_loss.get_avg())
            progress2 = "=== Valid|Loss {:2.3f}|Acc {:2.3f}|Precision {:2.3f}|Recall {:2.3f}|F1 {:2.3f}".format(
                                                                        eval_action_loss,
                                                                        eval_action_accuracy,
                                                                        eval_action_precision,
                                                                        eval_action_recall,
                                                                        eval_action_f1)
            progress3 = "=== Test |Loss {:2.3f}|Acc {:2.3f}|Precision {:2.3f}|Recall {:2.3f}|F1 {:2.3f}".format(
                                                                        test_action_loss,
                                                                        test_action_accuracy,
                                                                        test_action_precision,
                                                                        test_action_recall,
                                                                        test_action_f1)

            print(progress1)
            print(progress2)
            print(progress3)

            # Write into file
            _s = json.dumps({"Time":         "{:.2f}".format((time_2 - time_1) / 60.),
                             "TrL":          str(ave_train_action_loss.get_avg()),
                             "EvL":          str(eval_action_loss),
                             "EvAcc":        str(eval_action_accuracy),
                             "EvPrecision":  str(eval_action_precision),
                             "EvRecall":     str(eval_action_recall),
                             "EvF1":         str(eval_action_f1),
                             "TeL":          str(test_action_loss),
                             "TeAcc":        str(test_action_accuracy),
                             "TePrecision":  str(test_action_precision),
                             "TeRecall":     str(test_action_recall),
                             "TeF1":         str(test_action_f1)})
            with open(output_dir + "/" + json_file_name + '.json', 'a+') as outfile:
                print("Write log to: {}".format(output_dir + "/" + json_file_name + '.json'))
                outfile.write(_s + '\n')
                outfile.flush()

if __name__ == '__main__':
    train()




