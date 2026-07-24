"""eDifFIQA model wrapper with config mapping."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Tuple

import torch
from torch import nn
from torchvision.transforms import Compose

# All variants: (config_yaml, weights_pth) relative to their variant dir
EDIFFIQA_CONF = {
    "ediffiqaL": ("configs/ediffiqaL_config.yaml", "ediffiqaL.pth"),
    "ediffiqaM": ("configs/ediffiqaM_config.yaml", "ediffiqaM.pth"),
    "ediffiqaS": ("configs/ediffiqaS_config.yaml", "ediffiqaS.pth"),
    "ediffiqaT": ("configs/ediffiqaT_config.yaml", "ediffiqaT.pth"),
}


class eDifFIQA(nn.Module):
    """eDifFIQA: FR backbone + MLP quality regression head."""

    def __init__(self, backbone_model: nn.Module, quality_head: nn.Module, return_feat: bool = False):
        super().__init__()
        self.base_model = backbone_model
        self.mlp = quality_head
        self.return_feat = return_feat

    def forward(self, x):
        feat = self.base_model(x)
        pred = self.mlp(feat)
        if self.return_feat:
            return feat, pred
        return pred


def _parse_yaml(path: str) -> dict:
    """Minimal YAML loader (only for nested dicts/lists/scalars)."""
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def _load_model_module(module_path: str, weights_path: str | None = None):
    """Dynamically import and instantiate a model class."""
    module_name, func_name = module_path.rsplit(".", 1)
    mod = importlib.import_module(module_name)
    cls_or_func = getattr(mod, func_name)
    instance = cls_or_func()
    if weights_path and Path(weights_path).exists():
        instance.load_state_dict(torch.load(weights_path, map_location="cpu"))
    return instance


def _build_transformation(trans_args: dict) -> Compose:
    """Build torchvision.Compose from config transformation entries."""
    transforms = []
    idx = 1
    while True:
        key = f"trans_{idx}"
        if key not in trans_args:
            break
        entry = trans_args[key]
        module_path = entry["module"]
        params = entry.get("params", {})
        mod_name, func_name = module_path.rsplit(".", 1)
        mod = importlib.import_module(mod_name)
        fn = getattr(mod, func_name)
        transforms.append(fn(**params))
        idx += 1
    return Compose(transforms)


def load_ediffiqa(variant: str, weights_dir: Path, device: str = "cpu") -> Tuple[eDifFIQA, Compose]:
    """Load a pre-trained eDifFIQA model from `weights_dir/variant/`.

    Args:
        variant: One of "ediffiqaL", "ediffiqaM", "ediffiqaS", "ediffiqaT".
        weights_dir: Path to deid/evaluation/weights/<variant>/.
        device: Torch device string.

    Returns:
        (model, transform) ready for inference.
    """
    # Ensure model/ dir is importable as "model.xxx"
    # Insert the PARENT of model/ so that `import model.iresnet` works
    weights_parent = str(weights_dir.parent.resolve())
    if weights_parent not in sys.path:
        sys.path.insert(0, weights_parent)

    if variant not in EDIFFIQA_CONF:
        raise ValueError(f"Unknown variant {variant!r}. Expected one of: {list(EDIFFIQA_CONF.keys())}")

    config_rel, weights_rel = EDIFFIQA_CONF[variant]
    config_path = weights_dir / config_rel
    weights_path = weights_dir / weights_rel

    # Parse config
    cfg = _parse_yaml(str(config_path))

    # Build backbone
    base_cfg = cfg["base_model"]
    backbone_weights = None
    if base_cfg.get("weights"):
        backbone_weights = str(weights_dir / base_cfg["weights"])
    backbone = _load_model_module(base_cfg["module"], backbone_weights)
    transform = _build_transformation(base_cfg["transformations"])

    # Build MLP head
    mlp_cfg = cfg["mlp"]
    mlp_mod_name, mlp_fn_name = mlp_cfg["module"].rsplit(".", 1)
    head_cls = importlib.import_module(mlp_mod_name)
    head_func = getattr(head_cls, mlp_fn_name)
    head = head_func(**mlp_cfg["params"])

    # Wrap in eDifFIQA
    ediffiqa_mod_name, ediffiqa_fn_name = cfg["ediffiqa"]["module"].rsplit(".", 1)
    wrap_cls = importlib.import_module(ediffiqa_mod_name)
    wrap_func = getattr(wrap_cls, ediffiqa_fn_name)
    model = wrap_func(
        backbone_model=backbone,
        quality_head=head,
        **cfg["ediffiqa"]["params"],
    )

    # Load full model weights (overwrites backbone + head with trained params)
    model.load_state_dict(torch.load(str(weights_path), map_location=device))
    model.to(device).eval()
    return model, transform
