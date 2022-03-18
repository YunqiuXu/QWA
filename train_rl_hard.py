import datetime
import os
import time
import copy
import json
import numpy as np

from agent import Agent
import generic
import evaluate
from dataset_RL import get_training_game_env_multiple_levels, get_evaluation_game_env
from generic import HistoryScoreCache, EpisodicCountingMemory


def train():
    print("===== 1. Load configs =====")
    time_1 = time.time()
    config = generic.load_config()
    output_dir = config['general']['output_dir']
    games_dir = config["general"]["games_dir"]
    print("Output dir: {}".format(output_dir))

    print("===== 2. Init agent =====")
    agent = Agent(config)
    requested_infos = agent.select_additional_infos()
    
    print("===== 3. Build envs =====")
    difficulty_level_list = [8, 9, 10]
    env, _ = get_training_game_env_multiple_levels(data_dir = games_dir + config['rl']['data_path'],
                                                   difficulty_level_list = difficulty_level_list,
                                                   training_size = config['rl']['training_size'],
                                                   requested_infos = requested_infos,
                                                   max_episode_steps = agent.max_nb_steps_per_episode, 
                                                   batch_size = agent.batch_size)
    eval_env_dict = {}
    for difficulty_level in difficulty_level_list:
        eval_title = "eval_level_{}".format(difficulty_level)
        eval_env, num_eval_game = get_evaluation_game_env(games_dir+config['rl']['data_path'],
                                                          difficulty_level,
                                                          requested_infos,
                                                          agent.eval_max_nb_steps_per_episode,
                                                          agent.eval_batch_size,
                                                          valid_or_test="valid")    
        eval_env_dict[eval_title] = {"eval_env": eval_env, "num_eval_game": num_eval_game}
    test_env_dict = {}
    for difficulty_level in difficulty_level_list:
        test_title = "test_level_{}".format(difficulty_level)
        test_env, num_test_game = get_evaluation_game_env(games_dir+config['rl']['data_path'],
                                                          difficulty_level,
                                                          requested_infos,
                                                          agent.eval_max_nb_steps_per_episode,
                                                          agent.eval_batch_size,
                                                          valid_or_test="test")    
        test_env_dict[test_title] = {"test_env": test_env, "num_test_game": num_test_game}

    step_in_total = 0
    episode_no = 0
    running_avg_game_points = HistoryScoreCache(capacity=500)
    running_avg_game_points_normalized = HistoryScoreCache(capacity=500)
    running_avg_graph_rewards = HistoryScoreCache(capacity=500)
    running_avg_count_rewards = HistoryScoreCache(capacity=500)
    running_avg_game_steps = HistoryScoreCache(capacity=500)
    running_avg_dqn_loss = HistoryScoreCache(capacity=500)
    running_avg_game_rewards = HistoryScoreCache(capacity=500)
    json_file_name = agent.experiment_tag.replace(" ", "_")
    best_train_performance_so_far, best_eval_performance_so_far = 0.0, 0.0
    prev_performance = 0.0

    print("===== 4. Load pre-trained online net!")
    assert agent.load_pretrained
    if agent.load_online_net == 'apInit':
        print("Init online net with AP pre-trained embeddings")
        ap_weight_path = config['general']['AP_ptr_dir']
        agent.load_pretrained_model(ap_weight_path, load_partial_graph=True)
    else:
        assert agent.load_online_net == 'rdInit'
        print("Init online net with random weights")
    agent.update_target_net()

    i_have_seen_these_states = EpisodicCountingMemory()
    i_am_patient = 0
    perfect_training = 0

    print("===== ===== ===== Start training ===== ===== =====")
    while(True):
        if episode_no > agent.max_episode:
            break
        np.random.seed(episode_no)
        env.seed(episode_no)
        obs, infos = env.reset()
        chosen_tasks_print = []           
        available_tasks_print = []
        # filter look and examine actions, this is applied for all variants
        for commands_ in infos["admissible_commands"]:
            for cmd_ in [cmd for cmd in commands_ if cmd != "examine cookbook" and cmd.split()[0] in ["examine", "look"]]:
                commands_.remove(cmd_)
        batch_size = len(obs)

        agent.train()
        agent.init(batch_size)
        game_name_list = [game.metadata["uuid"].split("-")[-1] for game in infos["game"]]
        game_max_score_list = [game.max_score for game in infos["game"]]
        i_have_seen_these_states.reset() 
        prev_triplets, chosen_actions, prev_game_facts = [], [], []
        prev_step_dones, prev_rewards = [], []
        for _ in range(batch_size):
            prev_triplets.append([])
            chosen_actions.append("restart")
            prev_game_facts.append(set())
            prev_step_dones.append(0.0)
            prev_rewards.append(0.0)
        prev_h, prev_c = None, None

        observation_strings, current_triplets, action_candidate_list, _, current_game_facts = agent.get_game_info_at_certain_step(obs, infos, prev_actions=chosen_actions, prev_facts=None)
        observation_for_counting = copy.copy(observation_strings)
        observation_strings = [item + " <sep> " + a for item, a in zip(observation_strings, chosen_actions)]
        i_have_seen_these_states.push(current_triplets) 

        agent.update_task_candidate_list(current_triplets)
        task_verbs_list, task_objs_list = agent.sample_tasks([None] * batch_size)
        curr_tasks = ["{} {}".format(vv,oo) for (vv,oo) in zip(task_verbs_list, task_objs_list)]
        available_tasks_print.append(agent.available_task_list[0])
        chosen_tasks_print.append(curr_tasks[0])

        # refine the actions
        action_candidate_list_refined = agent.refine_action_candidate(task_verbs_list, 
                                                                      task_objs_list,
                                                                      action_candidate_list,
                                                                      [False] * batch_size) # dones are all False at the beginning!
        
        if agent.count_reward_lambda > 0:
            agent.reset_binarized_counter(batch_size)
            _ = agent.get_binarized_count(observation_for_counting)

        # it requires to store sequences of transitions into memory with order,
        # so we use a cache to keep what agents returns, and push them into memory
        # altogether in the end of game.
        transition_cache = []
        still_running_mask = []
        game_rewards, game_points, graph_rewards, count_rewards = [], [], [], []
        print_actions = []
        act_randomly = False if agent.noisy_net else episode_no < agent.learn_start_from_this_episode

        for step_no in range(agent.max_nb_steps_per_episode):
            if agent.noisy_net:
                agent.reset_noise()
            new_chosen_actions, chosen_indices, prev_h, prev_c = agent.act(observation_strings, 
                                                                           current_triplets, 
                                                                           action_candidate_list_refined, 
                                                                           curr_tasks, 
                                                                           previous_h=prev_h, 
                                                                           previous_c=prev_c, 
                                                                           random=act_randomly)
            replay_info = [observation_strings, action_candidate_list_refined, curr_tasks, chosen_indices, current_triplets, chosen_actions] 
            transition_cache.append(replay_info)
            chosen_actions = new_chosen_actions

            # A special case: "frosted - glass door" -> "frosted-glass door"
            chosen_actions_before_parsing = []
            for curr_new_chosen_action in new_chosen_actions:
                if "frosted - glass door" in curr_new_chosen_action:
                    vvvv = curr_new_chosen_action.split(" ")[0]
                    chosen_actions_before_parsing.append("{} frosted-glass door".format(vvvv))
                else:
                    chosen_actions_before_parsing.append(curr_new_chosen_action)
            
            # Step and get feedback
            obs, scores, dones, infos = env.step(chosen_actions_before_parsing)
            for commands_ in infos["admissible_commands"]:
                for cmd_ in [cmd for cmd in commands_ if cmd != "examine cookbook" and cmd.split()[0] in ["examine", "look"]]:
                    commands_.remove(cmd_)
            prev_triplets = current_triplets
            prev_game_facts = current_game_facts
            observation_strings, current_triplets, action_candidate_list, _, current_game_facts = agent.get_game_info_at_certain_step(obs, infos, prev_actions=chosen_actions, prev_facts=prev_game_facts)
            observation_for_counting = copy.copy(observation_strings)
            observation_strings = [item + " <sep> " + a for item, a in zip(observation_strings, chosen_actions)]
                        
            agent.update_task_candidate_list(current_triplets)
            task_verbs_list, task_objs_list = agent.sample_tasks(curr_tasks)
            # If done, no task ("nothing")
            curr_tasks = []
            for i in range(batch_size):
                if dones[i]:
                    curr_tasks.append("nothing")
                else:
                    curr_tasks.append("{} {}".format(task_verbs_list[i], task_objs_list[i]))
            
            # Record the printing cases
            available_tasks_print.append(agent.available_task_list[0])
            chosen_tasks_print.append(curr_tasks[0])

            action_candidate_list_refined = agent.refine_action_candidate(task_verbs_list, 
                                                                          task_objs_list, 
                                                                          action_candidate_list,
                                                                          dones)

            has_not_seen = i_have_seen_these_states.has_not_seen(current_triplets)
            i_have_seen_these_states.push(current_triplets)  # update init triplets into memory
            if agent.noisy_net and step_in_total % agent.update_per_k_game_steps == 0:
                agent.reset_noise()
            if episode_no >= agent.learn_start_from_this_episode and step_in_total % agent.update_per_k_game_steps == 0:
                dqn_loss, _ = agent.update_dqn(episode_no)
                if dqn_loss is not None:
                    running_avg_dqn_loss.push(dqn_loss)
            if step_no == agent.max_nb_steps_per_episode - 1:
                # terminate the game because DQN requires one extra step
                dones = [True for _ in dones]

            step_in_total += 1
            still_running = [1.0 - float(item) for item in prev_step_dones]  # list of float
            prev_step_dones = dones
            step_rewards = [float(curr) - float(prev) for curr, prev in zip(scores, prev_rewards)]  # list of float
            game_points.append(copy.copy(step_rewards))

            # Compute rewards
            if agent.use_negative_reward:
                step_rewards = [-1.0 if _lost else r for r, _lost in zip(step_rewards, infos["has_lost"])]  
                step_rewards = [5.0 if _won else r for r, _won in zip(step_rewards, infos["has_won"])] 
            
            prev_rewards = scores
            if agent.fully_observable_graph:
                step_graph_rewards = [0.0 for _ in range(batch_size)]
            else:
                step_graph_rewards = agent.get_graph_rewards(prev_triplets, current_triplets)  
                step_graph_rewards = [r * float(m) for r, m in zip (step_graph_rewards, has_not_seen)]
            if agent.count_reward_lambda > 0:
                step_revisit_counting_rewards = agent.get_binarized_count(observation_for_counting, update=True)
                step_revisit_counting_rewards = [r * agent.count_reward_lambda for r in step_revisit_counting_rewards]
            else:
                step_revisit_counting_rewards = [0.0 for _ in range(batch_size)]
            still_running_mask.append(still_running)
            game_rewards.append(step_rewards)
            graph_rewards.append(step_graph_rewards)
            count_rewards.append(step_revisit_counting_rewards)
            print_actions.append(chosen_actions_before_parsing[0] if still_running[0] else "--")
            # if all ended, break
            if np.sum(still_running) == 0:
                break

        # Build rewards (list -> np -> pt), all with shape [step, batch]
        still_running_mask_np = np.array(still_running_mask)
        game_rewards_np = np.array(game_rewards) * still_running_mask_np    # step x batch
        game_points_np = np.array(game_points) * still_running_mask_np      # step x batch
        graph_rewards_np = np.array(graph_rewards) * still_running_mask_np  # step x batch
        count_rewards_np = np.array(count_rewards) * still_running_mask_np  # step x batch
        if agent.graph_reward_lambda > 0.0:
            graph_rewards_pt = generic.to_pt(graph_rewards_np, enable_cuda=agent.use_cuda, type='float')
        else:
            graph_rewards_pt = generic.to_pt(np.zeros_like(graph_rewards_np), enable_cuda=agent.use_cuda, type='float')
        if agent.count_reward_lambda > 0.0:
            count_rewards_pt = generic.to_pt(count_rewards_np, enable_cuda=agent.use_cuda, type='float') 
        else:
            count_rewards_pt = generic.to_pt(np.zeros_like(count_rewards_np), enable_cuda=agent.use_cuda, type='float')
        command_rewards_pt = generic.to_pt(game_rewards_np, enable_cuda=agent.use_cuda, type='float') 

        # push experience into replay buffer (dqn)
        avg_rewards_in_buffer = agent.dqn_memory.avg_rewards()
        for b in range(game_rewards_np.shape[1]):
            if still_running_mask_np.shape[0] == agent.max_nb_steps_per_episode and still_running_mask_np[-1][b] != 0:
                # need to pad one transition
                _need_pad = True
                tmp_game_rewards = game_rewards_np[:, b].tolist() + [0.0]
            else:
                _need_pad = False
                tmp_game_rewards = game_rewards_np[:, b]
            if np.mean(tmp_game_rewards) < avg_rewards_in_buffer * agent.buffer_reward_threshold:
                continue
            for i in range(game_rewards_np.shape[0]):
                observation_strings, action_candidate_list, tasks, chosen_indices, _triplets, prev_action_strings = transition_cache[i]
                is_final = True
                if still_running_mask_np[i][b] != 0:
                    is_final = False
                agent.dqn_memory.add(observation_strings[b], 
                                     prev_action_strings[b], 
                                     action_candidate_list[b], 
                                     tasks[b],
                                     chosen_indices[b], 
                                     _triplets[b], 
                                     command_rewards_pt[i][b], 
                                     graph_rewards_pt[i][b], 
                                     count_rewards_pt[i][b], 
                                     is_final)
                if still_running_mask_np[i][b] == 0:
                    break
            if _need_pad:
                observation_strings, action_candidate_list, tasks, chosen_indices, _triplets, prev_action_strings = transition_cache[-1]
                agent.dqn_memory.add(observation_strings[b], 
                                     prev_action_strings[b], 
                                     action_candidate_list[b], 
                                     tasks[b],
                                     chosen_indices[b], 
                                     _triplets[b], 
                                     command_rewards_pt[-1][b] * 0.0, 
                                     graph_rewards_pt[-1][b] * 0.0, 
                                     count_rewards_pt[-1][b] * 0.0, 
                                     True)

        for b in range(batch_size):
            running_avg_game_points.push(np.sum(game_points_np, 0)[b])
            game_max_score_np = np.array(game_max_score_list, dtype="float32")
            running_avg_game_points_normalized.push((np.sum(game_points_np, 0) / game_max_score_np)[b])
            running_avg_game_steps.push(np.sum(still_running_mask_np, 0)[b])
            running_avg_game_rewards.push(np.sum(game_rewards_np, 0)[b])
            running_avg_graph_rewards.push(np.sum(graph_rewards_np, 0)[b])
            running_avg_count_rewards.push(np.sum(count_rewards_np, 0)[b])

        # finish game
        agent.finish_of_episode(episode_no, batch_size)
        episode_no += batch_size
        if episode_no < agent.learn_start_from_this_episode:
            continue
        if agent.report_frequency == 0 or (episode_no % agent.report_frequency > (episode_no - batch_size) % agent.report_frequency):
            print("{} episodes finished".format(episode_no))
            continue
            
        time_2 = time.time()
        progress_train = "Train|Epi: {:3d}|Time: {:.2f}m|L_DQN: {:2.3f}|Score: {:2.3f}|ScoreNorm: {:2.3f} \nRew: {:2.3f}|RewGraph: {:2.3f}|RewCount: {:2.3f}|Steps: {:2.3f}"
        progress_train = progress_train.format(episode_no, 
                                              (time_2 - time_1) / 60.,
                                               running_avg_dqn_loss.get_avg(), 
                                               running_avg_game_points.get_avg(), 
                                               running_avg_game_points_normalized.get_avg(), 
                                               running_avg_game_rewards.get_avg(), 
                                               running_avg_graph_rewards.get_avg(), 
                                               running_avg_count_rewards.get_avg(), 
                                               running_avg_game_steps.get_avg())
        print(progress_train)

        # Print actions and tasks
        print(game_name_list[0] + ":    " + " | ".join(print_actions))
        print("\nTasks for env0:")
        for pppp in range(len(chosen_tasks_print)):
            print("{:25}|{}".format(chosen_tasks_print[pppp], available_tasks_print[pppp]))

        # Validate
        print("===== ===== ===== Validating ===== ===== =====")
        curr_train_performance = running_avg_game_points_normalized.get_avg()
        eval_performance_dict = {}
        eval_game_points_normalized_list = []
        for difficulty_level in difficulty_level_list:
            eval_title = "eval_level_{}".format(difficulty_level)
            eval_env = eval_env_dict[eval_title]["eval_env"]
            num_eval_game = eval_env_dict[eval_title]["num_eval_game"]
            eval_game_points, eval_game_points_normalized, eval_game_step, _, detailed_scores = evaluate.evaluate(
                                                                                                            eval_env, 
                                                                                                            agent, 
                                                                                                            num_eval_game)
            eval_performance_dict[eval_title] = {"eval_game_points":eval_game_points, 
                                                "eval_game_points_normalized":eval_game_points_normalized,
                                                "eval_game_step":eval_game_step,
                                                "detailed_scores":detailed_scores}
            eval_game_points_normalized_list.append(eval_game_points_normalized)

        print("===== ===== ===== Testing ===== ===== =====")
        test_performance_dict = {}
        for difficulty_level in difficulty_level_list:
            test_title = "test_level_{}".format(difficulty_level)
            test_env = test_env_dict[test_title]["test_env"]
            num_test_game = test_env_dict[test_title]["num_test_game"]
            test_game_points, test_game_points_normalized, test_game_step, _, test_detailed_scores = evaluate.evaluate(
                                                                                                                test_env, 
                                                                                                                agent, 
                                                                                                                num_test_game)
            test_performance_dict[test_title] = {"test_game_points":test_game_points, 
                                                 "test_game_points_normalized":test_game_points_normalized,
                                                 "test_game_step":test_game_step,
                                                 "test_detailed_scores":test_detailed_scores}

        # Check whether to restore model
        curr_eval_performance = np.mean(eval_game_points_normalized_list)
        curr_performance = curr_eval_performance
        if curr_eval_performance > best_eval_performance_so_far:
            best_eval_performance_so_far = curr_eval_performance
            agent.save_model_to_path(output_dir + "/" + agent.experiment_tag + "_model.pt")
        elif curr_eval_performance == best_eval_performance_so_far:
            if curr_eval_performance > 0.0:
                agent.save_model_to_path(output_dir + "/" + agent.experiment_tag + "_model.pt")
            else:
                if curr_train_performance >= best_train_performance_so_far:
                    agent.save_model_to_path(output_dir + "/" + agent.experiment_tag + "_model.pt")        

        # Update best train performance
        if curr_train_performance >= best_train_performance_so_far:
            best_train_performance_so_far = curr_train_performance
        if prev_performance <= curr_performance:
            i_am_patient = 0
        else:
            i_am_patient += 1
        prev_performance = curr_performance
        # if patient >= patience, resume from checkpoint
        if agent.patience > 0 and i_am_patient >= agent.patience:
            if os.path.exists(output_dir + "/" + agent.experiment_tag + "_model.pt"):
                print('No patience, reload from: {}'.format(output_dir + "/" + agent.experiment_tag + "_model.pt"))
                agent.load_pretrained_model(output_dir + "/" + agent.experiment_tag + "_model.pt", load_partial_graph=False)
                agent.update_target_net()
                i_am_patient = 0
        if running_avg_game_points_normalized.get_avg() >= 0.95:
            perfect_training += 1
        else:
            perfect_training = 0
        
        # write into file
        _s = json.dumps({"Time": "{:.2f}".format((time_2 - time_1) / 60.), # str(time_2 - time_1).rsplit(".")[0],
                         "L_DQN":       str(running_avg_dqn_loss.get_avg()),
                         "TrScore":     str(running_avg_game_points.get_avg()),
                         "TrScoreNorm": str(running_avg_game_points_normalized.get_avg()),
                         "TrRew":       str(running_avg_game_rewards.get_avg()),
                         "TrRewGraph":  str(running_avg_graph_rewards.get_avg()),
                         "TrRewCount":  str(running_avg_count_rewards.get_avg()),
                         "TrSteps":     str(running_avg_game_steps.get_avg()),
                         # validation
                         "EvScoreL8":     str(eval_performance_dict["eval_level_8"]["eval_game_points"]),
                         "EvScoreNormL8": str(eval_performance_dict["eval_level_8"]["eval_game_points_normalized"]),
                         "EvStepsL8":     str(eval_performance_dict["eval_level_8"]["eval_game_step"]),
                         "EvScoreL9":     str(eval_performance_dict["eval_level_9"]["eval_game_points"]),
                         "EvScoreNormL9": str(eval_performance_dict["eval_level_9"]["eval_game_points_normalized"]),
                         "EvStepsL9":     str(eval_performance_dict["eval_level_9"]["eval_game_step"]),
                         "EvScoreL10":     str(eval_performance_dict["eval_level_10"]["eval_game_points"]),
                         "EvScoreNormL10": str(eval_performance_dict["eval_level_10"]["eval_game_points_normalized"]),
                         "EvStepsL10":     str(eval_performance_dict["eval_level_10"]["eval_game_step"]),
                         # test
                         "TeScoreL8":     str(test_performance_dict["test_level_8"]["test_game_points"]),
                         "TeScoreNormL8": str(test_performance_dict["test_level_8"]["test_game_points_normalized"]),
                         "TeStepsL8":     str(test_performance_dict["test_level_8"]["test_game_step"]),
                         "TeScoreL9":     str(test_performance_dict["test_level_9"]["test_game_points"]),
                         "TeScoreNormL9": str(test_performance_dict["test_level_9"]["test_game_points_normalized"]),
                         "TeStepsL9":     str(test_performance_dict["test_level_9"]["test_game_step"]),
                         "TeScoreL10":     str(test_performance_dict["test_level_10"]["test_game_points"]),
                         "TeScoreNormL10": str(test_performance_dict["test_level_10"]["test_game_points_normalized"]),
                         "TeStepsL10":     str(test_performance_dict["test_level_10"]["test_game_step"]),
                         })
        with open(output_dir + "/" + json_file_name + '.json', 'a+') as outfile:
            print("Write log to: {}".format(output_dir + "/" + json_file_name + '.json'))
            outfile.write(_s + '\n')
            outfile.flush()

        if curr_performance == 1.0 and curr_train_performance >= 0.95:
            break
        if perfect_training >= 3:
            break


if __name__ == '__main__':
    train()
