"""Runner modules for Halo_dev workflows."""

from typing import Callable, Dict, List, Optional

from .finewiki_runner import main as finewiki_main

RunnerFunc = Callable[[Optional[List[str]]], None]

RUNNERS: Dict[str, RunnerFunc] = {
    "finewiki": finewiki_main,
}

__all__ = ["RUNNERS"]
