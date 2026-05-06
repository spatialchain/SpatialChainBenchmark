from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


CanonicalRecord = dict[str, Any]


class DatasetAdapter(ABC):
    name: str

    @abstractmethod
    def adapt_record(self, record: dict[str, Any], idx: int) -> CanonicalRecord:
        """
        Converte um registro do dataset de origem para o formato canônico
        usado internamente pelo pipeline.
        """
        raise NotImplementedError
