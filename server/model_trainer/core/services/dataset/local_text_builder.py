from __future__ import annotations

from dataclasses import dataclass

from ...contracts.dataset import DatasetBuilder, DatasetConfig
from ...services.training.dataset_builder import split_corpus_files


@dataclass
class LocalTextDatasetBuilder(DatasetBuilder):
    def split(self: LocalTextDatasetBuilder, cfg: DatasetConfig) -> tuple[list[str], list[str]]:
        return split_corpus_files(cfg)
