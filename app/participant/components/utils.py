from __future__ import annotations
import os
from pathlib import Path
import importlib

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

_CONFIGS_DIR = str(Path(__file__).parent.parent.parent.parent / "configs")

# Singleton class to serve as global access point to the config
class Config:
    _cfg: DictConfig = None

    @classmethod
    def get_cfg(cls) -> DictConfig:
        if cls._cfg is None:
            with initialize_config_dir(config_dir=_CONFIGS_DIR, version_base=None):
                cls._cfg = compose(config_name="config")
        return cls._cfg


def _instantiate(cfg_node: DictConfig, module: str) -> object:
    cls = getattr(importlib.import_module(module), cfg_node.class_name)
    return cls(cfg_node)

def read_system_prompt(class_name: str) -> str:
    cfg = Config.get_cfg()
    path_to_md = Path(cfg.paths.path_to_prompts) / f"{class_name}.md"
    return path_to_md.read_text()