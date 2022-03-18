import numpy as np
import torch
import os
from generic import get_match_result, to_np, get_match_result_obs_gen
from sklearn.metrics import precision_score, recall_score, f1_score


def evaluate_action_prediction(env, agent, valid_test="valid", verbose=False):
    env.split_reset(valid_test)
    agent.eval()
    list_eval_acc, list_eval_loss = [], []
    counter = 0
    while(True):
        current_graph, previous_graph, target_action, action_choices = env.get_batch()
        with torch.no_grad():
            loss, ap_ret, np_labels, action_choices = agent.get_action_prediction_logits(current_graph, previous_graph, target_action, action_choices)
        loss = to_np(loss)
        pred = np.argmax(ap_ret, -1)  
        gt = np.argmax(np_labels, -1)  
        correct = (pred == gt).astype("float32").tolist()
        list_eval_acc += correct
        list_eval_loss += [loss]
        if env.batch_pointer == 0:
            break
    return np.mean(list_eval_loss), np.mean(list_eval_acc)

def evaluate_vt(env, agent, valid_test="valid"):
    agent.eval()
    env.split_reset(valid_test)
    task_loss_list = []
    task_label_list = []
    task_pred_list = []
    batch_counter = 0
    while(True):
        graph, task, label = env.get_batch()
        batch_counter += 1
        with torch.no_grad():
            task_loss, task_ret = agent.get_vt_logits(graph, task, label)
        task_loss_list += [to_np(task_loss)]
        task_label_list += label
        task_pred_list += np.argmax(task_ret, -1).tolist() # batched list
        if env.batch_pointer == 0:
            break
    assert len(task_label_list) == len(task_pred_list)
    total_accuracy_score = np.mean([i==j for (i,j) in zip(task_label_list, task_pred_list)])
    total_precision_score = precision_score(task_label_list, task_pred_list)
    total_recall_score = recall_score(task_label_list, task_pred_list)
    total_f1_score = f1_score(task_label_list, task_pred_list)
    return np.mean(task_loss_list), total_accuracy_score, total_precision_score, total_recall_score, total_f1_score

def evaluate_va(env, agent, valid_test="valid"):
    agent.eval()
    env.split_reset(valid_test)
    action_loss_list = []
    action_label_list = []
    action_pred_list = []
    batch_counter = 0
    while(True):
        task, action, label = env.get_batch()
        batch_counter += 1
        with torch.no_grad():
            action_loss, action_ret = agent.get_va_logits(task, action, label)
        action_loss_list += [to_np(action_loss)]
        action_label_list += label
        action_pred_list += np.argmax(action_ret, -1).tolist() # batched list
        if env.batch_pointer == 0:
            break
    assert len(action_label_list) == len(action_pred_list)
    total_accuracy_score = np.mean([i==j for (i,j) in zip(action_label_list, action_pred_list)])
    total_precision_score = precision_score(action_label_list, action_pred_list)
    total_recall_score = recall_score(action_label_list, action_pred_list)
    total_f1_score = f1_score(action_label_list, action_pred_list)
    return np.mean(action_loss_list), total_accuracy_score, total_precision_score, total_recall_score, total_f1_score

def evaluate(env, agent, num_games):
    if agent.fully_observable_graph:
        return evaluate_with_ground_truth_graph(env, agent, num_games)
    else:
        raise NotImplementedError

def evaluate_with_ground_truth_graph(env, agent, num_games):
    achieved_game_points = []
    total_game_steps = []
    game_name_list = []
    game_max_score_list = []
    game_id = 0
    while(True):
        if game_id >= num_games:
            break
        obs, infos = env.reset()
        for commands_ in infos["admissible_commands"]:
            for cmd_ in [cmd for cmd in commands_ if cmd != "examine cookbook" and cmd.split()[0] in ["examine", "look"]]:
                commands_.remove(cmd_)
        game_name_list += [game.metadata["uuid"].split("-")[-1] for game in infos["game"]]
        game_max_score_list += [game.max_score for game in infos["game"]]
        batch_size = len(obs)
        agent.eval()
        agent.init(batch_size) 
        chosen_actions, prev_step_dones = [], []
        for _ in range(batch_size):
            chosen_actions.append("restart")
            prev_step_dones.append(0.0)
        prev_h, prev_c = None, None
        observation_strings, current_triplets, action_candidate_list, _, _ = agent.get_game_info_at_certain_step(obs, infos, prev_actions=None, prev_facts=None)
        observation_strings = [item + " <sep> " + a for item, a in zip(observation_strings, chosen_actions)]
        still_running_mask = []
        final_scores = []
        agent.update_task_candidate_list(current_triplets)
        task_verbs_list, task_objs_list = agent.sample_tasks([None] * batch_size)

        curr_tasks = ["{} {}".format(vv,oo) for (vv,oo) in zip(task_verbs_list, task_objs_list)]
        action_candidate_list_refined = agent.refine_action_candidate(task_verbs_list, 
                                                                      task_objs_list,
                                                                      action_candidate_list,
                                                                      [False] * batch_size) # dones are all False at the beginning!
        for step_no in range(agent.eval_max_nb_steps_per_episode):
            chosen_actions, chosen_indices, prev_h, prev_c = agent.act_greedy(observation_strings, 
                                                                              current_triplets, 
                                                                              action_candidate_list_refined, 
                                                                              curr_tasks, 
                                                                              prev_h, prev_c)
            chosen_actions_before_parsing = []
            for curr_chosen_action in chosen_actions:
                if "frosted - glass door" in curr_chosen_action:
                    vvvv = curr_chosen_action.split(" ")[0]
                    chosen_actions_before_parsing.append("{} frosted-glass door".format(vvvv))
                else:
                    chosen_actions_before_parsing.append(curr_chosen_action)

            obs, scores, dones, infos = env.step(chosen_actions_before_parsing)
            for commands_ in infos["admissible_commands"]:
                for cmd_ in [cmd for cmd in commands_ if cmd != "examine cookbook" and cmd.split()[0] in ["examine", "look"]]:
                    commands_.remove(cmd_)

            observation_strings, current_triplets, action_candidate_list, _, _ = agent.get_game_info_at_certain_step(obs, infos, prev_actions=None, prev_facts=None)
            observation_strings = [item + " <sep> " + a for item, a in zip(observation_strings, chosen_actions)]
            agent.update_task_candidate_list(current_triplets)
            task_verbs_list, task_objs_list = agent.sample_tasks([None] * batch_size)
            curr_tasks = []
            for i in range(batch_size):
                if dones[i]:
                    curr_tasks.append("nothing")
                else:
                    curr_tasks.append("{} {}".format(task_verbs_list[i], task_objs_list[i]))
            action_candidate_list_refined = agent.refine_action_candidate(task_verbs_list, 
                                                                          task_objs_list, 
                                                                          action_candidate_list,
                                                                          dones)
            
            still_running = [1.0 - float(item) for item in prev_step_dones] 
            prev_step_dones = dones
            final_scores = scores
            still_running_mask.append(still_running)
            if np.sum(still_running) == 0:
                break
        achieved_game_points += final_scores
        still_running_mask = np.array(still_running_mask)
        total_game_steps += np.sum(still_running_mask, 0).tolist()
        game_id += batch_size
    achieved_game_points = np.array(achieved_game_points, dtype="float32")
    game_max_score_list = np.array(game_max_score_list, dtype="float32")
    normalized_game_points = achieved_game_points / game_max_score_list
    print_strings = []
    print_strings.append("EvLevel|Score: {:2.3f}|ScoreNorm: {:2.3f}|Steps: {:2.3f}".format(
                                                                                    np.mean(achieved_game_points), 
                                                                                    np.mean(normalized_game_points), 
                                                                                    np.mean(total_game_steps)))
    print_strings = "\n".join(print_strings)
    print(print_strings)
    return np.mean(achieved_game_points), np.mean(normalized_game_points), np.mean(total_game_steps), 0.0, print_strings




