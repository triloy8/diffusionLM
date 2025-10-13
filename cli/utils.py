import argparse
import json
import sys
from typing import Callable, Optional, Type

from config import asdict_pretty


def add_config_args(parser: argparse.ArgumentParser, *, type_: Type = str) -> None:
    parser.add_argument("--config", type=type_, required=True, help="Path to TOML config")
    parser.add_argument("--print-config", action="store_true", help="Print resolved config and exit")


def load_config_or_print(load_fn: Callable[[str], object], config_path: str, print_flag: bool):
    cfg = load_fn(config_path)
    if print_flag:
        print(json.dumps(asdict_pretty(cfg), indent=2))
        return None
    return cfg


def die(msg: str, code: int = 2) -> None:
    print(msg, file=sys.stderr)
    sys.exit(code)
