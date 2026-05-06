from spatial_eval.datasets.adapters import CurrentFormatAdapter, ImageQuestionAnswerAdapter, TestsetImageIdAdapter
from spatial_eval.datasets.base import CanonicalRecord, DatasetAdapter


_ADAPTERS: dict[str, DatasetAdapter] = {
    CurrentFormatAdapter.name: CurrentFormatAdapter(),
    ImageQuestionAnswerAdapter.name: ImageQuestionAnswerAdapter(),
    TestsetImageIdAdapter.name: TestsetImageIdAdapter(),
}


def available_adapters() -> list[str]:
    return sorted(_ADAPTERS.keys())


def get_adapter(name: str) -> DatasetAdapter:
    key = (name or "").strip().lower()
    if key not in _ADAPTERS:
        raise ValueError(f"Unknown dataset adapter: {name}. Available: {', '.join(available_adapters())}")
    return _ADAPTERS[key]


def adapt_records(records: list[dict], adapter_name: str) -> list[CanonicalRecord]:
    adapter = get_adapter(adapter_name)
    converted: list[CanonicalRecord] = []
    for idx, record in enumerate(records):
        converted.append(adapter.adapt_record(record, idx=idx))
    return converted


__all__ = [
    "CanonicalRecord",
    "DatasetAdapter",
    "CurrentFormatAdapter",
    "ImageQuestionAnswerAdapter",
    "TestsetImageIdAdapter",
    "available_adapters",
    "get_adapter",
    "adapt_records",
]
