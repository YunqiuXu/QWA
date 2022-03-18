import numpy as np

def refine_action_candidate_single(task_verb, task_obj, action_candidates):
    if task_verb == "chop":
        return refine_chop(task_obj, action_candidates)
    elif task_verb == "dice":
        return refine_dice(task_obj, action_candidates)
    elif task_verb == "fry":
        return refine_fry(task_obj, action_candidates)
    elif task_verb == "get":
        return refine_get(task_obj, action_candidates)
    elif task_verb == "grill":
        return refine_grill(task_obj, action_candidates)
    elif task_verb == "make":
        return refine_make(task_obj, action_candidates)
    elif task_verb == "roast":
        return refine_roast(task_obj, action_candidates)
    elif task_verb == "slice":
        return refine_slice(task_obj, action_candidates)
    else:
        raise Exception("unrecognized verb: {}".format(task_verb))

def refine_chop(task_obj, action_candidates):
    result = []
    for action in action_candidates:
        verb = action.split(" ")[0]
        if verb in {"go","open"}:
            result.append(action)
        elif "chop {} with knife".format(task_obj) == action.lower():
            result.append(action)
    if len(result) == 0:
        return action_candidates
    return result

def refine_dice(task_obj, action_candidates):
    result = []
    for action in action_candidates:
        verb = action.split(" ")[0]
        if verb in {"go","open"}:
            result.append(action)
        elif "dice {} with knife".format(task_obj) == action.lower():
            result.append(action)
    if len(result) == 0:
        return action_candidates
    return result

def refine_fry(task_obj, action_candidates):
    result = []
    for action in action_candidates:
        verb = action.split(" ")[0]
        if verb in {"go","open"}:
            result.append(action)
        elif "cook {} with stove".format(task_obj) == action.lower():
            result.append(action)
    if len(result) == 0:
        return action_candidates
    return result

def refine_get(task_obj, action_candidates):
    result = []
    for action in action_candidates:
        verb = action.split(" ")[0]
        if verb in {"go","open"}:
            result.append(action)
        elif verb == 'take':
            target_action = "take {}".format(task_obj)
            partial_action = action.split(" from ")[0]
            if target_action == partial_action.lower():
                result.append(action)
    if len(result) == 0:
        return action_candidates
    return result


def refine_grill(task_obj, action_candidates):
    result = []
    for action in action_candidates:
        verb = action.split(" ")[0]
        if verb in {"go","open"}:
            result.append(action)
        elif "cook {} with bbq".format(task_obj) == action.lower():
            result.append(action)
    if len(result) == 0:
        return action_candidates
    return result

def refine_make(task_obj, action_candidates):
    result = []
    for action in action_candidates:
        verb = action.split(" ")[0]
        if verb in {"go","open", "prepare", "eat"}:
            result.append(action)
    if len(result) == 0:
        return action_candidates
    return result

def refine_roast(task_obj, action_candidates):
    result = []
    for action in action_candidates:
        verb = action.split(" ")[0]
        if verb in {"go","open"}:
            result.append(action)
        elif "cook {} with oven".format(task_obj) == action.lower():
            result.append(action)
    if len(result) == 0:
        return action_candidates
    return result

def refine_slice(task_obj, action_candidates):
    result = []
    for action in action_candidates:
        verb = action.split(" ")[0]
        if verb in {"go","open"}:
            result.append(action)
        elif "slice {} with knife".format(task_obj) == action.lower():
            result.append(action)
    if len(result) == 0:
        return action_candidates
    return result





