import numpy as np
import click
import os
import re
import humanfriendly as hf
import uuid
from collections import defaultdict
from tqdm import tqdm


def extract_distance(s):
    # Regular expression to match floating-point numbers
    regex = (
        r"(\d+k)_profile_(\d+)_distance(\d+\.\d+)_degree_(\d+)_ges_(\d+)_round_(\d+)"
    )
    match = re.search(regex, s)
    if match:
        return {
            "bandwidth": hf.parse_size(match.group(1)),
            "profile": int(match.group(2)),
            "distance": float(match.group(3)),
            "degree": int(match.group(4)),
            "gesture": int(match.group(5)),
            "round": int(match.group(6)),
        }
    return None


class RadarIndex:
    def __init__(self, data_root) -> None:
        self.data_root = data_root
        self.generate_index()

    def generate_index(self):
        file_list = os.listdir(self.data_root)

        self.id_list = []
        self.id_dict = {}

        self.degree_dict = defaultdict(list)
        self.gesture_dict = defaultdict(list)
        self.round_dict = defaultdict(list)
        self.profile_dict = defaultdict(list)

        for file in tqdm(file_list, desc="Generating index"):
            state_dict = extract_distance(file)

            assert state_dict is not None
            assert file is not None

            state_dict["file"] = file
            id = str(uuid.uuid4())
            self.id_dict[id] = state_dict
            self.id_list.append(id)

            self.degree_dict[state_dict["degree"]].append(id)
            self.gesture_dict[state_dict["gesture"]].append(id)
            self.round_dict[state_dict["round"]].append(id)
            self.profile_dict[state_dict["profile"]].append(id)

    @property
    def cats(self):
        return ["g1", "g2", "g3", "g4", "g5"]


if __name__ == "__main__":
    RadarIndex("dataset")
