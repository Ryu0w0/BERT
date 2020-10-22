import pickle
import json


def get_json_parameters(load_root_path, load_key):
    assert load_key is not None, "Please specify load_key."
    json_open = open(f"{load_root_path}/{load_key}.json", "r")
    parameters = json.load(json_open)
    return parameters


def save_as_pickle(target, save_key, save_root_path="./output"):
    with open(f"{save_root_path}/{save_key}.pickle", "wb") as f:
        pickle.dump(target, f)
