from __future__ import annotations

import configparser
import logging
from pathlib import Path
from typing import Optional

import yaml

from deid.config.loader import ConfigLoader

logger = logging.getLogger(__name__)


class Migrator:
    """Migrate ``config.ini`` + ``pipeline.yml`` -> ``root_dir/deid-config.yaml``."""

    def __init__(self, config_yaml_path: Optional[Path] = None) -> None:
        self._yaml_path = config_yaml_path or Path("root_dir/deid-config.yaml")
        self._loader = ConfigLoader()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------
    def migrate(self, dry_run: bool = True) -> Optional[dict]:
        """Run migration.  ``dry_run=True`` prints summary without writing."""
        ini = configparser.ConfigParser()
        if not self._loader._ini_path.exists():
            logger.warning("No config.ini found -- nothing to migrate")
            return None

        ini.read(self._loader._ini_path)

        pipeline: dict = {}
        if self._loader._pipeline_path.exists():
            with open(self._loader._pipeline_path) as f:
                pipeline = yaml.safe_load(f) or {}

        mapping = self._build_mapping(ini, pipeline)

        if dry_run:
            self._print_summary(mapping)
            return mapping

        self._yaml_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._yaml_path, "w") as f:
            yaml.dump(mapping, f, default_flow_style=False, sort_keys=False)

        logger.info("Wrote migration to %s", self._yaml_path)
        return mapping

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    @staticmethod
    def _build_mapping(ini: configparser.ConfigParser, pipeline: dict) -> dict:
        mapping: dict = {
            "root_dir": "root_dir",
            "result_dir": "results",
            "logs_dir": "logs",
        }

        # Selections
        mapping["datasets"] = {
            "selected": ini.get("selection", "datasets", fallback="").split()
        }
        mapping["techniques"] = {
            "selected": ini.get("selection", "techniques", fallback="").split(),
            "args": {},
        }
        mapping["evaluation"] = {
            "selected": ini.get("selection", "evaluation", fallback="").split()
        }

        # Environments
        if ini.has_section("Available Environments"):
            mapping["environments"] = dict(ini.items("Available Environments"))

        # Visualizations
        if ini.has_section("Available Visualizations"):
            mapping["visualization"] = dict(ini.items("Available Visualizations"))

        # Renames from pipeline.yml
        for section, key, target in [
            ("datasets", "rename", "dataset_renames"),
            ("techniques", "rename", "technique_renames"),
            ("evaluations", "rename", "evaluation_renames"),
        ]:
            items = pipeline.get(section, {}) or {}
            renames: dict = {}
            for name, cfg in items.items():
                if isinstance(cfg, dict) and "rename" in cfg:
                    renames[name] = cfg["rename"]
            if renames:
                mapping[target] = renames

        # Technique args from pipeline.yml
        tech_items = pipeline.get("techniques", {}) or {}
        args_map: dict = {}
        for name, cfg in tech_items.items():
            if isinstance(cfg, dict) and "args" in cfg:
                args_map[name] = cfg["args"]
        if args_map:
            mapping["techniques"]["args"] = args_map

        return mapping

    @staticmethod
    def _print_summary(mapping: dict) -> None:
        print("=== Migration Summary ===")
        print(f"Output: {Path('root_dir/deid-config.yaml')}")
        print()

        datasets = mapping.get("datasets", {}).get("selected", [])
        print(f"Datasets ({len(datasets)}): {', '.join(datasets) or '(none)'}")

        techniques = mapping.get("techniques", {}).get("selected", [])
        print(f"Techniques ({len(techniques)}): {', '.join(techniques) or '(none)'}")

        evals = mapping.get("evaluation", {}).get("selected", [])
        print(f"Evaluations ({len(evals)}): {', '.join(evals) or '(none)'}")

        envs = mapping.get("environments", {})
        if envs:
            print(f"Environments ({len(envs)}): {', '.join(envs)}")

        vis = mapping.get("visualization", {})
        if isinstance(vis, dict) and vis:
            print(f"Visualizations ({len(vis)} sections): {', '.join(vis)}")

        renames = mapping.get("technique_renames", {})
        if renames:
            print(f"\nTechnique renames: {renames}")
        renames = mapping.get("dataset_renames", {})
        if renames:
            print(f"Dataset renames: {renames}")
        renames = mapping.get("evaluation_renames", {})
        if renames:
            print(f"Evaluation renames: {renames}")

        print("\nRun ``deid config migrate --yes`` to write the file.")
