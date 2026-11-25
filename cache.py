"""Hashes that tell whether a saved result was made from the same data and settings."""
import hashlib
import json
import os


def array_hash(a):
    return hashlib.sha256(a.tobytes()).hexdigest()


def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def frames_hash(paths):
    """Hash of the image files' names, sizes and modification times (the images are not read)."""
    h = hashlib.sha256()
    for path in paths:
        st = os.stat(path)
        h.update(f"{os.path.basename(path)}\0{st.st_size}\0{st.st_mtime_ns}\n".encode())
    return h.hexdigest()


def read_json(path):
    """The saved dict, or {} if there is none (or it is unreadable)."""
    try:
        with open(path) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def matches(saved, key):
    """Whether ``saved`` holds every entry of ``key``."""
    return all(k in saved and saved[k] == v for k, v in key.items())
