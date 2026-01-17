from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping

import yaml

from halo_dev.models import Node
from halo_dev.node_processors import regex_param_extractor

_DEFAULT_TEMPLATE = Path("templates/tpch_trident.yaml")
_PARAM_NODE_ID = "rule_param_extractor"
_CACHED_CONFIG: Dict[str, Any] | None = None


def load_tpch_param_config(template_path: Path | None = None) -> Dict[str, Any]:
    global _CACHED_CONFIG
    if template_path is None and _CACHED_CONFIG is not None:
        return _CACHED_CONFIG
    path = template_path or _DEFAULT_TEMPLATE
    if not path.exists():
        raise FileNotFoundError(f"TPC-H template not found: {path}")
    data = yaml.safe_load(path.read_text())
    graph = data.get("graph") if isinstance(data, dict) else {}
    nodes = graph.get("nodes") if isinstance(graph, dict) else []
    config: Dict[str, Any] = {}
    if isinstance(nodes, list):
        for node in nodes:
            if not isinstance(node, Mapping):
                continue
            if node.get("id") != _PARAM_NODE_ID:
                continue
            raw_cfg = node.get("config")
            if isinstance(raw_cfg, Mapping):
                config = dict(raw_cfg)
            break
    if template_path is None:
        _CACHED_CONFIG = config
    return config


def extract_tpch_params(user_query: str, template_path: Path | None = None) -> Dict[str, Any]:
    config = load_tpch_param_config(template_path)
    node = Node(
        id=_PARAM_NODE_ID,
        type="processor",
        inputs=("user_query",),
        outputs=("tpch_params",),
        raw={},
    )
    output = regex_param_extractor({"user_query": user_query}, config, node)
    params = output.get("tpch_params")
    return dict(params) if isinstance(params, Mapping) else {}
