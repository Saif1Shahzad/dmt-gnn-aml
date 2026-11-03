import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    elif str(device).startswith("mps"):
        torch.mps.synchronize()
