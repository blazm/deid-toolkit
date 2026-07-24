from __future__ import annotations

import configparser
import json
import logging
from pathlib import Path
from typing import Any, Optional

import yaml

from deid.config.models import Settings

logger = logging.getLogger(__name__)


def _get_builtin_scripts(package: str) -> set[str]:
    """List script stems bundled in a deid.* package directory."""
    from importlib import resources

    pkg = resources.files(package)
    return {
        p.stem for p in pkg.iterdir()
        if p.suffix == ".py" and p.is_file()
        and p.stem not in {"__init__", "utils"}
    }


def _get_builtin_envs(package: str):
    """Yield Traversable objects for bundled .yml environment files."""
    from importlib import resources

    pkg = resources.files(package)
    yield from (p for p in pkg.iterdir() if p.suffix == ".yml" and p.is_file())

# Fallback file paths
_DEFAULT_CONFIG_YAML = Path("root_dir/deid-config.yaml")
_DEFAULT_CONFIG_INI = Path("config.ini")
_DEFAULT_PIPELINE_YML = Path("root_dir/pipeline.yml")


class ConfigLoader:
    """Unified configuration loader.

    Priority:
    1. ``root_dir/deid-config.yaml`` (if it exists)
    2. ``config.ini`` + ``root_dir/pipeline.yml`` (legacy fallback)
    """

    def __init__(self, config_yaml_path: Optional[Path] = None) -> None:
        self._yaml_path = config_yaml_path or _DEFAULT_CONFIG_YAML
        self._ini_path = _DEFAULT_CONFIG_INI
        self._pipeline_path = _DEFAULT_PIPELINE_YML
        self._settings: Optional[Settings] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def settings(self) -> Settings:
        if self._settings is None:
            self._settings = self._load()
        return self._settings

    def load_datasets(self) -> list[str]:
        """Auto-discover original dataset directories."""
        original = self.settings.datasets_path / "original"
        if not original.is_dir():
            return []
        return sorted(d.name for d in original.iterdir() if d.is_dir())

    def load_aligned_datasets(self) -> list[str]:
        aligned = self.settings.datasets_path / "aligned"
        if not aligned.is_dir():
            return []
        return sorted(d.name for d in aligned.iterdir() if d.is_dir())

    def load_techniques(self) -> list[str]:
        techniques_dir = self.settings.techniques_path
        user_techs: set[str] = set()
        if techniques_dir.is_dir():
            user_techs = {p.stem for p in techniques_dir.glob("*.py") if p.is_file()}
        # Union with built-in techniques from the package
        user_techs |= set(_get_builtin_scripts("deid.techniques"))
        # Validation is always available as a hardcoded technique
        user_techs.add("validation")
        return sorted(user_techs)

    def load_evaluations(self) -> list[str]:
        eval_dir = self.settings.evaluation_path
        user_evals: set[str] = set()
        if eval_dir.is_dir():
            user_evals = {
                p.stem for p in eval_dir.glob("*")
                if p.is_file() and p.suffix in {".py", ".sh"}
            }
        # Union with built-in evaluations from the package
        user_evals |= set(_get_builtin_scripts("deid.evaluation"))
        # Filter out legacy/broken scripts — superseded by better alternatives
        _deprecated_evals = frozenset({
            "adaface_iv",       # redundant: adaface_optimized is faster + caches embeddings
            "vggface",          # broken .t7 loader; use deepface_vggface instead
            "vggface_optimized",# broken .t7 loader; use deepface_vggface instead
        })
        user_evals -= _deprecated_evals
        return sorted(user_evals)

    def resolve_eval_alias(self, name: str) -> str:
        """Resolve an evaluation alias to its canonical name.

        Maps user-friendly names (e.g. "adaface") to the actual script name
        ("adaface_optimized"). Returns the input unchanged if no alias exists.
        """
        _aliases = {
            "adaface": "adaface_optimized",
            "fid": "FID",
        }
        return _aliases.get(name, name)

    def list_evaluations_grouped(self) -> list[tuple[str, list[str]]]:
        """Return evaluations grouped by category for display.

        Returns: list of (category_name, [eval_names]) tuples.
        Unknown/custom evals are placed under "Other".
        """
        # Category mapping — expandable as new scripts are added
        _categories = {
            "Verification": frozenset({
                "arcface", "adaface_optimized", "adaface_iv",
                "swinface", "deepface_vggface",
                "vggface", "vggface_optimized",
            }),
            "Image Quality": frozenset({
                "ssim", "lpips", "mse", "fid", "FID",
                "pytorchFid", "ediffiqa",
            }),
            "Data Utility": frozenset({
                "dan", "deepface_gender", "deepface_GD",
                "deepface_expression", "deepface_age", "deepface_race",
                "hsemotion", "restnet18_GD",
            }),
        }

        evals = set(self.load_evaluations())  # flat, already filtered
        grouped: dict[str, list[str]] = {}
        for category, members in _categories.items():
            found = sorted(evals & members)
            if found:
                grouped[category] = found
        other = sorted(evals - set().union(*_categories.values()))
        if other:
            grouped["Other"] = other

        # Reorder: Verification first, then Image Quality, Data Utility, Other
        _order = ["Verification", "Image Quality", "Data Utility", "Other"]
        return [(cat, grouped[cat]) for cat in _order if cat in grouped]

    def load_environments(self) -> dict[str, str]:
        env_dir = self.settings.environments_path
        user_envs: dict[str, str] = {}
        if env_dir.is_dir():
            for p in env_dir.glob("*.yml"):
                user_envs[p.stem] = p.stem
        # Union with built-in environments from the package
        for p in _get_builtin_envs("deid.environments"):
            user_envs[p.stem] = p.stem
        # Merge with settings (settings takes precedence)
        user_envs.update(self.settings.environments)
        return user_envs

    def load_visualizations(self) -> list[str]:
        viz_dir = self.settings.visualization_path
        if not viz_dir.is_dir():
            return list(self.settings.visualization.selections)
        return sorted(p.stem for p in viz_dir.glob("*.py") if p.is_file())

    def list_results(self) -> dict[str, dict[str, dict[str, Path]]]:
        """Return {dataset: {technique: {metric: csv_path}}}."""
        results_dir = self.settings.result_path
        if not results_dir.is_dir():
            return {}

        mapping: dict[str, dict[str, dict[str, Path]]] = {}
        # Walk {technique}/{dataset}/*.csv and index by {dataset}/{technique}/{metric}
        for tech_dir in sorted(results_dir.iterdir()):
            if not tech_dir.is_dir() or tech_dir.name == "__pycache__":
                continue
            tech_name = tech_dir.name
            for ds_dir in sorted(tech_dir.iterdir()):
                if not ds_dir.is_dir():
                    continue
                ds_name = ds_dir.name
                for csv_path in sorted(ds_dir.glob("*.csv")):
                    mapping.setdefault(ds_name, {}).setdefault(tech_name, {})[csv_path.stem] = csv_path
        return mapping

    def get_result_manifest(self) -> Optional[dict]:
        """Load the results manifest (pipeline run metadata)."""
        manifest = self.settings.result_path / "manifest.json"
        if manifest.exists():
            with open(manifest) as f:
                return json.load(f)
        return None

    # ------------------------------------------------------------------
    # Internal loading
    # ------------------------------------------------------------------
    def _load(self) -> Settings:
        if self._yaml_path.exists():
            return self._load_yaml()
        return self._load_legacy()

    def _load_yaml(self) -> Settings:
        with open(self._yaml_path) as f:
            data: dict[str, Any] = yaml.safe_load(f) or {}

        # Build Settings from YAML data
        kw: dict[str, Any] = {}
        kw["root_dir"] = data.get("root_dir", "root_dir")
        kw["result_dir"] = data.get("result_dir", "results")
        kw["logs_dir"] = data.get("logs_dir", "logs")

        ds = data.get("datasets")
        if ds:
            kw["datasets"] = DatasetSelection(**ds) if isinstance(ds, dict) else DatasetSelection(selected=ds)

        tech = data.get("techniques")
        if tech and isinstance(tech, dict):
            kw["techniques"] = TechniqueSelection(
                selected=tech.get("selected", []),
                args=tech.get("args", {}),
            )
        elif tech:
            kw["techniques"] = TechniqueSelection(selected=tech if isinstance(tech, list) else [])

        ev = data.get("evaluation")
        if ev:
            kw["evaluation"] = EvaluationSelection(selected=ev if isinstance(ev, list) else ev.get("selected", []))

        if data.get("environments"):
            kw["environments"] = data["environments"]
        if data.get("visualization"):
            vs = data["visualization"]
            kw["visualization"] = (
                VisualizationSetting(selections=vs)
                if isinstance(vs, list)
                else VisualizationSetting(**vs)
            )
        if data.get("dataset_renames"):
            kw["dataset_renames"] = data["dataset_renames"]
        if data.get("technique_renames"):
            kw["technique_renames"] = data["technique_renames"]
        if data.get("evaluation_renames"):
            kw["evaluation_renames"] = data["evaluation_renames"]

        return Settings(**kw)

    def _load_legacy(self) -> Settings:
        """Fallback: build Settings from config.ini + pipeline.yml."""
        if not self._ini_path.exists():
            logger.warning("No config.ini found, returning defaults")
            return Settings()

        ini = configparser.ConfigParser()
        ini.read(self._ini_path)

        # pipeline.yml for renames
        renames: dict[str, dict[str, str]] = {
            "dataset_renames": {},
            "technique_renames": {},
            "evaluation_renames": {},
        }
        if self._pipeline_path.exists():
            with open(self._pipeline_path) as f:
                pipeline = yaml.safe_load(f) or {}
            for section, key in [
                ("datasets", "rename"),
                ("techniques", "rename"),
                ("evaluations", "rename"),
            ]:
                items = pipeline.get(section, {}) or {}
                for name, cfg in items.items():
                    if isinstance(cfg, dict) and "rename" in cfg:
                        renames[f"{section}_renames"][name] = cfg["rename"]

        return Settings(
            root_dir=ini.get("settings", "root_dir", fallback="root_dir"),
            result_dir=ini.get("settings", "result_dir", fallback="results"),
            logs_dir=ini.get("settings", "logs_dir", fallback="logs"),
            datasets={"selected": ini.get("selection", "datasets", fallback="").split()},
            techniques={
                "selected": ini.get("selection", "techniques", fallback="").split(),
                "args": {},
            },
            evaluation={"selected": ini.get("selection", "evaluation", fallback="").split()},
            environments=dict(ini.items("Available Environments")) if ini.has_section("Available Environments") else {},
            visualization={
                "selections": []
                if not ini.has_section("Available Visualizations")
                else dict(ini.items("Available Visualizations"))
            },
            **renames,
        )


from deid.config.models import DatasetSelection, EvaluationSelection, TechniqueSelection, VisualizationSetting  # noqa: E402
