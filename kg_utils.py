import numpy as np

def get_task_verb_candidates(triplets_full):
    return ['chop', 'dice', 'fry', 'get', 'grill', 'make', 'roast', 'slice']

def get_task_obj_candidates(triplets_full):
    task_obj_candidates = get_ingredients(triplets_full)
    task_obj_candidates.append('knife')
    task_obj_candidates.append('meal')
    return task_obj_candidates

def get_ingredients(triplets_full_0):
    result = []
    for triplet in triplets_full_0:
        if triplet[1] == 'cookbook':
            assert(triplet[2] == 'part_of'), "Strange triplet: {}".format(triplet)
            result.append(triplet[0])
    return result

def _check_ing_existance(ingredient, triplets_full):
    for triplet in triplets_full:
        if (triplet[0] == ingredient) and (triplet[2] in {'in', 'on', 'at'}):
            return True
    return False

def _check_ing_collection(ingredient, triplets_full):
    if [ingredient, "player", "in"] in triplets_full:
        return True
    else:
        return False

def _get_ing_req_status(ingredient, triplets_full):
    req_result_part1_temp = [] 
    req_result_part2_temp = [] 
    status_result = set()
    for triplet in triplets_full:
        if (triplet[0] == ingredient):
            if (triplet[2] == 'needs'):
                if triplet[1] in {'chopped', 'diced', 'sliced'}:
                    req_result_part1_temp.append(triplet[1])
                else:
                    req_result_part2_temp.append(triplet[1])
            if (triplet[2] == 'is'):
                status_result.add(triplet[1])
    req_result_part1 = []
    req_result_part2 = []
    for req in req_result_part1_temp:
        if req not in status_result:
            req_result_part1.append(req)
    for req in req_result_part2_temp:
        if req not in status_result:
            req_result_part2.append(req)
    return req_result_part1, req_result_part2, status_result

def get_available_tasks(ingredients, triplets_full, test_mode=False):
    ['chop', 'dice', 'fry', 'get', 'grill', 'make', 'roast', 'slice']
    req2verb = {
                'chopped':'chop',
                'diced':'dice',
                'fried':'fry',
                'grilled':'grill',
                'roasted':'roast',
                'sliced':'slice'
    }
    available_tasks = set()
    for ingredient in ingredients:
        if _check_ing_existance(ingredient, triplets_full):
            req_result_part1, req_result_part2, status_result = _get_ing_req_status(ingredient, triplets_full)
            if test_mode:
                print("Ing {}|Status: {}|Req1: {}|Req2: {}".format(ingredient, status_result, req_result_part1, req_result_part2))
            if not _check_ing_collection(ingredient, triplets_full):
                generated_task = "get {}".format(ingredient)
                available_tasks.add(generated_task)
            else:
                if len(req_result_part1) > 0:  
                    assert len(req_result_part1) <= 1, "At most one in req_result_part1, got {}".format(req_result_part1)
                    curr_req = req_result_part1[0]
                    if not _check_ing_collection("knife",triplets_full):
                        generated_task = "get knife"
                        available_tasks.add(generated_task)
                    else:
                        generated_task = "{} {}".format(req2verb[curr_req], ingredient)
                        available_tasks.add(generated_task)
                if len(req_result_part2) > 0:  
                    assert len(req_result_part2) <= 1, "At most one in req_result_part2, got {}".format(req_result_part2)
                    for curr_req in req_result_part2:
                        generated_task = "{} {}".format(req2verb[curr_req], ingredient)
                        available_tasks.add(generated_task)
    if len(available_tasks) == 0:
        available_tasks.add("make meal")
    return sorted(list(available_tasks))
