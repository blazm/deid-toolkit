from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatasetSelection(BaseModel):
    selected: list[str] = Field(default_factory=list, alias="selected")


class TechniqueSelection(BaseModel):
    selected: list[str] = Field(default_factory=list, alias="selected")
    args: dict[str, str] = Field(default_factory=dict)


class EvaluationSelection(BaseModel):
    selected: list[str] = Field(default_factory=list, alias="selected")


class VisualizationSetting(BaseModel):
    selections: list[str] = Field(default_factory=list, alias="selections")


class Settings(BaseSettings):
    """Unified toolkit configuration.

    Loaded from ``deid-config.yaml`` (preferred) or fallen back to
    ``config.ini`` + ``pipeline.yml`` when the YAML is missing.
    """

    model_config = SettingsConfigDict(
        env_prefix="DEID_",
        env_file=".env",
    )

    root_dir: str = "root_dir"
    result_dir: str = "results"
    logs_dir: str = "logs"

    datasets: DatasetSelection = Field(default_factory=DatasetSelection)
    techniques: TechniqueSelection = Field(default_factory=TechniqueSelection)
    evaluation: EvaluationSelection = Field(default_factory=EvaluationSelection)

    # Mapped from [Available Environments] / [Available Visualizations]
    environments: dict[str, str] = Field(default_factory=dict)
    visualization: VisualizationSetting = Field(default_factory=VisualizationSetting)

    # Merged from pipeline.yml rename mappings
    dataset_renames: dict[str, str] = Field(default_factory=dict)
    technique_renames: dict[str, str] = Field(default_factory=dict)
    evaluation_renames: dict[str, str] = Field(default_factory=dict)

    @property
    def root_path(self) -> Path:
        return Path(self.root_dir)

    @property
    def result_path(self) -> Path:
        return self.root_path / self.result_dir

    @property
    def logs_path(self) -> Path:
        return self.root_path / self.logs_dir

    @property
    def datasets_path(self) -> Path:
        return self.root_path / "datasets"

    @property
    def techniques_path(self) -> Path:
        return self.root_path / "techniques"

    @property
    def evaluation_path(self) -> Path:
        return self.root_path / "evaluation"

    @property
    def environments_path(self) -> Path:
        return self.root_path / "environments"

    @property
    def visualization_path(self) -> Path:
        return self.root_path / "visualization"

    @property
    def aligned_path(self) -> Path:
        return self.datasets_path / "aligned"

    @property
    def original_path(self) -> Path:
        return self.datasets_path / "original"

    @property
    def deid_path(self) -> Path:
        return self.datasets_path / "deidentified"
