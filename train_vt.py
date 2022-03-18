import datetime
import os
import time
import json
import math
import torch
import numpy as np
from os.path import join as pjoin
from glob import glob

from dataset_VT import VTData
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
    assert agent.task == "vt", "Wrong task {}, should be vt!".format(agent.task)

    if agent.load_task_net == 'apInit':
        print("Init task net with AP pre-trained embeddings")
        ap_weight_path = config['general']['AP_ptr_dir']
        agent.load_pretrained_task_model(ap_weight_path, load_partial_graph=True)
    elif agent.load_task_net == 'rdInit':
        print("Init task net with random weights")
    else:
        raise Exception("Unsupported agent.load_task_net: {}".format(agent.load_task_net))

    print("===== 3. Build dataset as an env =====")
    data_path = config['vt']['data_path']
    data_path_dict = {
                        "train": ["{}/VT_train.json".format(data_path)],
                        "valid": ["{}/VT_valid.json".format(data_path)],
                        "test": ["{}/VT_test.json".format(data_path)]
                     }
    env = VTData(config, data_path_dict)
    env.split_reset("train")

    json_file_name = agent.experiment_tag.replace(" ", "_")
    ave_train_task_loss = generic.HistoryScoreCache(capacity=500)
    episode_no = 0
    batch_no = 0
    best_total_eval_f1 = 0.0
    
    print("===== ===== ===== Start training ===== ===== =====")
    while(True):
        if episode_no > agent.max_episode:
            break
        agent.train()
        graph, task, label = env.get_batch()
        curr_batch_size = len(graph)
        task_loss, _ = agent.get_vt_logits(graph, task, label)
        agent.task_net.zero_grad()
        agent.task_optimizer.zero_grad()
        task_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.task_net.parameters(), agent.clip_grad_norm)
        agent.task_optimizer.step()
        ave_train_task_loss.push(generic.to_np(task_loss))
        if batch_no < agent.learning_rate_warmup_until:
            cr = agent.init_learning_rate / math.log2(agent.learning_rate_warmup_until)
            learning_rate = cr * math.log2(batch_no + 1)
        else:
            learning_rate = agent.init_learning_rate
        for param_group in agent.task_optimizer.param_groups:
            param_group['lr'] = learning_rate
        episode_no += curr_batch_size
        batch_no += 1

        if agent.report_frequency == 0 or (episode_no % agent.report_frequency > (episode_no - curr_batch_size) % agent.report_frequency):
            continue
        
        if episode_no % agent.report_frequency <= (episode_no - curr_batch_size) % agent.report_frequency:
            print("===== ===== ===== Validating ===== ===== =====")
            eval_task_loss, eval_task_accuracy, eval_task_precision, eval_task_recall, eval_task_f1 = evaluate.evaluate_vt(env, agent, "valid")
            if eval_task_f1 > best_total_eval_f1:
                best_total_eval_f1 = eval_task_f1
                agent.save_task_model_to_path(output_dir + "/" + agent.experiment_tag + "_task_model_f1.pt")
                agent.save_task_model_to_path(output_dir + "/" + agent.experiment_tag + "_task_model_f1_{}m.pt".format(episode_no // 1000000))
                print("Save task net with best total f1 score: {:2.3f}".format(best_total_eval_f1))

            print("===== ===== ===== Testing ===== ===== =====")
            test_task_loss, test_task_accuracy, test_task_precision, test_task_recall, test_task_f1 = evaluate.evaluate_vt(env, agent, "test")

            env.split_reset("train")

            # Record progress
            time_2 = time.time()
            progress1 = "=== Epi: {:3d}|Time: {:.2f}m|TrainLoss {:2.3f}".format(episode_no, 
                                                                                (time_2 - time_1) / 60., 
                                                                                ave_train_task_loss.get_avg())
            progress2 = "=== Valid|Loss {:2.3f}|Acc {:2.3f}|Precision {:2.3f}|Recall {:2.3f}|F1 {:2.3f}".format(
                                                                        eval_task_loss,
                                                                        eval_task_accuracy,
                                                                        eval_task_precision,
                                                                        eval_task_recall,
                                                                        eval_task_f1)
            progress3 = "=== Test |Loss {:2.3f}|Acc {:2.3f}|Precision {:2.3f}|Recall {:2.3f}|F1 {:2.3f}".format(
                                                                        test_task_loss,
                                                                        test_task_accuracy,
                                                                        test_task_precision,
                                                                        test_task_recall,
                                                                        test_task_f1)
            print(progress1)
            print(progress2)
            print(progress3)

            # write into file
            _s = json.dumps({"Time":         "{:.2f}".format((time_2 - time_1) / 60.),
                             "TrL":          str(ave_train_task_loss.get_avg()),
                             "EvL":          str(eval_task_loss),
                             "EvAcc":        str(eval_task_accuracy),
                             "EvPrecision":  str(eval_task_precision),
                             "EvRecall":     str(eval_task_recall),
                             "EvF1":         str(eval_task_f1),
                             "TeL":          str(test_task_loss),
                             "TeAcc":        str(test_task_accuracy),
                             "TePrecision":  str(test_task_precision),
                             "TeRecall":     str(test_task_recall),
                             "TeF1":         str(test_task_f1)})
            with open(output_dir + "/" + json_file_name + '.json', 'a+') as outfile:
                print("Write log to: {}".format(output_dir + "/" + json_file_name + '.json'))
                outfile.write(_s + '\n')
                outfile.flush()

if __name__ == '__main__':
    train()
