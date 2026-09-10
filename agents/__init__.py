"""
Loads MUST_backend/.env into the process environment the first time any agent
module is imported (fastapp1.py imports agents.auth at the top, so this runs
before anything reads a setting). Standard library only - no new dependency.

Variables already set in the environment win over the file, so a real
deployment's secrets (e.g. Hugging Face Space secrets) are never overridden
by a stray local .env.
"""
import os


def _load_env_file() -> None:
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            if key and key not in os.environ:
                os.environ[key] = value


_load_env_file()
