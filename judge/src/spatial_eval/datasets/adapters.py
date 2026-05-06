from __future__ import annotations

from pathlib import Path
from typing import Any

from spatial_eval.datasets.base import CanonicalRecord, DatasetAdapter


class CurrentFormatAdapter(DatasetAdapter):
    """
    Adapter para o formato já usado no projeto (compatível com batch_1/sample).
    """

    name = "current"

    def adapt_record(self, record: dict[str, Any], idx: int) -> CanonicalRecord:
        question = str(record.get("question", "")).strip()
        if not question:
            raise ValueError("CurrentFormatAdapter: missing required field 'question'.")

        row_idx = record.get("row_idx", idx)
        question_type = record.get("type")
        if question_type is None:
            question_type = record.get("question_type")

        custom_id = record.get("custom_id")
        if custom_id is None:
            custom_id = f"row_{row_idx}"

        canonical: CanonicalRecord = {
            "row_idx": row_idx,
            "custom_id": str(custom_id),
            "question": question,
            "ground_truth": record.get("ground_truth"),
            "type": question_type,
            "scene_graph": record.get("scene_graph"),
            "image_path": record.get("image_path"),
        }

        # Mantemos campos extras para providers de replay/depuração.
        canonical.update(record)
        # Garante que os campos canônicos tenham prioridade final.
        canonical["row_idx"] = row_idx
        canonical["custom_id"] = str(custom_id)
        canonical["question"] = question
        canonical["ground_truth"] = record.get("ground_truth")
        canonical["type"] = question_type
        canonical["scene_graph"] = record.get("scene_graph")
        canonical["image_path"] = record.get("image_path")
        return canonical


class ImageQuestionAnswerAdapter(DatasetAdapter):
    """
    Exemplo de adapter para datasets com campos:
    - image_path (ou image)
    - question
    - answer (opcional)
    """

    name = "image-qa"

    def adapt_record(self, record: dict[str, Any], idx: int) -> CanonicalRecord:
        question = str(record.get("question", "")).strip()
        if not question:
            raise ValueError("ImageQuestionAnswerAdapter: missing required field 'question'.")

        image_path = record.get("image_path")
        if image_path is None:
            image_path = record.get("image")

        row_idx = record.get("row_idx", idx)
        custom_id = record.get("custom_id", f"row_{row_idx}")

        canonical: CanonicalRecord = {
            "row_idx": row_idx,
            "custom_id": str(custom_id),
            "question": question,
            "ground_truth": record.get("ground_truth", record.get("answer")),
            "type": record.get("type", record.get("question_type", "image-qa")),
            "scene_graph": record.get("scene_graph"),
            "image_path": image_path,
        }
        canonical.update(record)
        canonical["row_idx"] = row_idx
        canonical["custom_id"] = str(custom_id)
        canonical["question"] = question
        canonical["image_path"] = image_path
        return canonical


class TestsetImageIdAdapter(DatasetAdapter):
    """
    Adapter para datasets com `imageID`/`imageId` e perguntas em `question_x`.

    Espera receber `__images_dir` no record quando a imagem não vem em `image_path`.
    """

    name = "testset-image-qa"

    def adapt_record(self, record: dict[str, Any], idx: int) -> CanonicalRecord:
        question = str(record.get("question_x", record.get("question", ""))).strip()
        if not question:
            raise ValueError("TestsetImageIdAdapter: missing required field 'question_x'/'question'.")

        row_idx = record.get("row_idx", idx)
        custom_id = record.get("custom_id", f"row_{row_idx}")
        question_type = record.get("type", record.get("question_type", "image-qa"))

        image_path = self._resolve_image_path(record)
        canonical: CanonicalRecord = {
            "row_idx": row_idx,
            "custom_id": str(custom_id),
            "question": question,
            "ground_truth": record.get("ground_truth", record.get("answer")),
            "type": question_type,
            "scene_graph": record.get("scene_graph"),
            "image_path": image_path,
        }
        canonical.update(record)
        canonical["row_idx"] = row_idx
        canonical["custom_id"] = str(custom_id)
        canonical["question"] = question
        canonical["ground_truth"] = record.get("ground_truth", record.get("answer"))
        canonical["type"] = question_type
        canonical["scene_graph"] = record.get("scene_graph")
        canonical["image_path"] = image_path
        return canonical

    def _resolve_image_path(self, record: dict[str, Any]) -> str:
        raw_image_path = record.get("image_path") or record.get("image")
        if raw_image_path:
            candidate = Path(str(raw_image_path))
            if candidate.exists():
                return str(candidate)

        image_id = record.get("imageID", record.get("imageId", record.get("image_id")))
        if image_id is None:
            raise ValueError("TestsetImageIdAdapter: missing required field 'imageID'/'imageId'.")
        image_id_text = str(image_id).strip()
        if not image_id_text:
            raise ValueError("TestsetImageIdAdapter: image ID cannot be empty.")

        images_dir = record.get("__images_dir")
        if not images_dir:
            raise ValueError("TestsetImageIdAdapter: missing images directory context '__images_dir'.")
        images_dir_path = Path(str(images_dir))
        if not images_dir_path.exists():
            raise ValueError(f"TestsetImageIdAdapter: images directory not found: {images_dir_path}")

        matches = sorted(images_dir_path.glob(f"{image_id_text}.*"))
        if not matches:
            raise ValueError(
                f"TestsetImageIdAdapter: image file not found for imageID={image_id_text} in {images_dir_path}"
            )
        return str(matches[0])
