from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from spatial_eval.datasets import available_adapters
from spatial_eval.pipeline import (
    run_dataset_conversion,
    run_evaluation,
    run_generation,
    run_image_question_generation,
    run_image_question_generation_batch,
    run_judging,
)
from spatial_eval.prompt_store import default_prompt_path, load_prompt_from_file
from spatial_eval.providers import (
    LLMGenerationProvider,
    LLMJudgeProvider,
    OllamaGenerationProvider,
    OllamaJudgeProvider,
    ReplayGenerationProvider,
    RuleBasedJudgeProvider,
)


def _read_optional_prompt(path: str | None) -> str | None:
    if not path:
        return None
    return load_prompt_from_file(path)


def _default_model_for_backend(backend: str) -> str:
    backend = backend.lower()
    if backend == "openai":
        return "gpt-4o-mini"
    if backend == "gemini":
        return "gemini-3-flash-preview"
    if backend == "anthropic":
        return "claude-4-5-haiku"
    if backend == "ollama":
        return "llama3.2"
    raise ValueError(f"Unsupported backend: {backend}")


def _build_generation_provider(args: argparse.Namespace):
    if args.generation_provider == "replay":
        return ReplayGenerationProvider()
    backend = args.generation_backend
    model = args.generation_model or _default_model_for_backend(backend)
    if backend == "ollama":
        return OllamaGenerationProvider(
            model=model,
            base_url=args.ollama_base_url,
            env_path=args.env_file,
            timeout_s=args.timeout_s,
            max_retries=args.llm_retries,
            retry_wait_s=args.retry_wait_s,
            temperature=args.generation_temperature,
            max_tokens=args.generation_max_tokens,
            keep_alive=args.ollama_keep_alive,
        )
    return LLMGenerationProvider(
        backend=backend,
        model=model,
        env_path=args.env_file,
        timeout_s=args.timeout_s,
        max_retries=args.llm_retries,
        retry_wait_s=args.retry_wait_s,
        temperature=args.generation_temperature,
        max_tokens=args.generation_max_tokens,
    )


def _build_judge_provider(args: argparse.Namespace):
    if args.judge_provider == "rule-based":
        return RuleBasedJudgeProvider()
    backend = args.judge_backend
    model = args.judge_model or _default_model_for_backend(backend)
    if backend == "ollama":
        return OllamaJudgeProvider(
            model=model,
            base_url=args.ollama_base_url,
            env_path=args.env_file,
            timeout_s=args.timeout_s,
            max_retries=args.llm_retries,
            retry_wait_s=args.retry_wait_s,
            temperature=args.judge_temperature,
            max_tokens=args.judge_max_tokens,
            keep_alive=args.ollama_keep_alive,
            judge_evidence=args.judge_evidence,
        )
    return LLMJudgeProvider(
        backend=backend,
        model=model,
        env_path=args.env_file,
        timeout_s=args.timeout_s,
        max_retries=args.llm_retries,
        retry_wait_s=args.retry_wait_s,
        temperature=args.judge_temperature,
        max_tokens=args.judge_max_tokens,
        judge_evidence=args.judge_evidence,
    )


def _add_common_io_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-dir", default="reports", help="Diretório de saída")
    parser.add_argument("--env-file", default=".env", help="Arquivo .env com chaves de API")
    parser.add_argument("--timeout-s", type=int, default=120, help="Timeout de chamadas LLM (segundos)")
    parser.add_argument("--llm-retries", type=int, default=1, help="Quantidade de retries em timeout/erro transitório.")
    parser.add_argument("--retry-wait-s", type=float, default=2.0, help="Espera (segundos) entre retries de LLM.")
    parser.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        default=True,
        help="Retoma execução usando registros já salvos quando disponível.",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Ignora registros salvos e reprocessa do zero.",
    )
    parser.add_argument("--ollama-base-url", default="http://localhost:11434", help="URL base da API local do Ollama.")
    parser.add_argument("--ollama-keep-alive", default=None, help="Valor keep_alive opcional do Ollama (ex.: 5m).")


def _add_sampling_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--sample-per-type",
        type=int,
        default=0,
        help="Amostras por tipo (default: 0 = sem limite por tipo).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Número máximo de amostras (default: sem limite).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed para amostragem")


def _add_dataset_adapter_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset-adapter",
        choices=tuple(available_adapters()),
        default="current",
        help="Adapter de conversão do dataset para o formato canônico.",
    )


def _add_images_dir_arg(parser: argparse.ArgumentParser, *, required: bool = False) -> None:
    parser.add_argument(
        "--images-dir",
        required=required,
        default=None,
        help="Diretório base das imagens do dataset (usado por adapters com imageID).",
    )


def _add_generation_provider_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--generation-provider",
        choices=("replay", "llm"),
        default="replay",
        help="Provider do gerador.",
    )
    parser.add_argument(
        "--generation-backend",
        choices=("openai", "gemini", "anthropic", "ollama"),
        default="openai",
        help="Backend LLM do gerador (quando generation-provider=llm).",
    )
    parser.add_argument("--generation-model", default=None, help="Modelo do gerador.")
    parser.add_argument("--generation-temperature", type=float, default=0.0, help="Temperatura do gerador.")
    parser.add_argument("--generation-max-tokens", type=int, default=1024, help="Max tokens do gerador.")


def _add_judge_provider_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--judge-provider",
        choices=("rule-based", "llm"),
        default="rule-based",
        help="Provider do juiz.",
    )
    parser.add_argument(
        "--judge-backend",
        choices=("openai", "gemini", "anthropic", "ollama"),
        default="openai",
        help="Backend LLM do juiz (quando judge-provider=llm).",
    )
    parser.add_argument("--judge-model", default=None, help="Modelo do juiz.")
    parser.add_argument("--judge-temperature", type=float, default=0.0, help="Temperatura do juiz.")
    parser.add_argument("--judge-max-tokens", type=int, default=1024, help="Max tokens do juiz.")


def _add_score_justification_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--score-justification",
        dest="score_justification",
        action="store_true",
        default=False,
        help="Inclui justificativa textual na saída do judge.",
    )
    parser.add_argument(
        "--no-score-justification",
        dest="score_justification",
        action="store_false",
        help="Não solicita justificativa textual ao judge.",
    )


def _add_judge_evidence_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--judge-evidence",
        choices=("image", "img_graph", "both"),
        default="img_graph",
        help="Evidência usada pelo judge: imagem, grafo, ou ambos.",
    )


def _add_judge_assets_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--judge-images-dir",
        default=None,
        help="Diretório base de imagens usado apenas na fase de judge.",
    )
    parser.add_argument(
        "--judge-img-graph-file",
        default=None,
        help="Arquivo JSON com mapeamento imageId -> scene graph para a fase de judge.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Spatial Reasoning Validator CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    convert_parser = subparsers.add_parser(
        "convert-dataset",
        help="Converte dataset para o formato canônico interno.",
    )
    convert_parser.add_argument(
        "--dataset",
        required=True,
        help="Caminho do dataset de entrada (.jsonl ou .json).",
    )
    convert_parser.add_argument(
        "--output-file",
        required=True,
        help="Caminho do JSONL convertido de saída.",
    )
    _add_dataset_adapter_arg(convert_parser)

    generate_parser = subparsers.add_parser("generate", help="Executa apenas a etapa de geração.")
    generate_parser.add_argument("--dataset", required=True, help="Caminho para .jsonl ou .json")
    generate_parser.add_argument(
        "--generator-prompt",
        default=str(default_prompt_path("generator")),
        help="Arquivo de prompt do gerador.",
    )
    _add_common_io_args(generate_parser)
    _add_sampling_args(generate_parser)
    _add_dataset_adapter_arg(generate_parser)
    _add_images_dir_arg(generate_parser, required=False)
    _add_generation_provider_args(generate_parser)

    generate_iq_parser = subparsers.add_parser(
        "generate-iq",
        help="Geração single-shot com input: imagem + pergunta.",
    )
    generate_iq_parser.add_argument("--image-path", required=True, help="Caminho da imagem.")
    generate_iq_parser.add_argument("--question", required=True, help="Pergunta sobre a imagem.")
    generate_iq_parser.add_argument(
        "--generator-prompt",
        default=str(default_prompt_path("generator_vision")),
        help="Arquivo de prompt do gerador vision.",
    )
    generate_iq_parser.add_argument(
        "--output-file",
        default=None,
        help="Arquivo JSON de saída (opcional).",
    )
    _add_common_io_args(generate_iq_parser)
    generate_iq_parser.add_argument(
        "--generation-provider",
        choices=("llm",),
        default="llm",
        help="Provider do gerador para imagem+pergunta.",
    )
    generate_iq_parser.add_argument(
        "--generation-backend",
        choices=("openai", "gemini", "anthropic", "ollama"),
        default="gemini",
        help="Backend LLM do gerador.",
    )
    generate_iq_parser.add_argument("--generation-model", default=None, help="Modelo do gerador.")
    generate_iq_parser.add_argument("--generation-temperature", type=float, default=0.0, help="Temperatura do gerador.")
    generate_iq_parser.add_argument("--generation-max-tokens", type=int, default=1024, help="Max tokens do gerador.")

    generate_iq_batch_parser = subparsers.add_parser(
        "generate-iq-batch",
        help="Geração em lote para dataset imagem+pergunta (imageID + pasta de imagens).",
    )
    generate_iq_batch_parser.add_argument("--dataset", required=True, help="Caminho para .jsonl ou .json")
    _add_images_dir_arg(generate_iq_batch_parser, required=True)
    generate_iq_batch_parser.add_argument(
        "--generator-prompt",
        default=str(default_prompt_path("generator_vision")),
        help="Arquivo de prompt do gerador vision.",
    )
    _add_common_io_args(generate_iq_batch_parser)
    _add_sampling_args(generate_iq_batch_parser)
    _add_dataset_adapter_arg(generate_iq_batch_parser)
    generate_iq_batch_parser.set_defaults(dataset_adapter="testset-image-qa")
    generate_iq_batch_parser.add_argument(
        "--generation-provider",
        choices=("llm",),
        default="llm",
        help="Provider do gerador para imagem+pergunta em lote.",
    )
    generate_iq_batch_parser.add_argument(
        "--generation-backend",
        choices=("openai", "gemini", "anthropic", "ollama"),
        default="gemini",
        help="Backend LLM do gerador.",
    )
    generate_iq_batch_parser.add_argument("--generation-model", default=None, help="Modelo do gerador.")
    generate_iq_batch_parser.add_argument(
        "--generation-temperature", type=float, default=0.0, help="Temperatura do gerador."
    )
    generate_iq_batch_parser.add_argument(
        "--generation-max-tokens", type=int, default=1024, help="Max tokens do gerador."
    )

    judge_parser = subparsers.add_parser("judge", help="Executa apenas a etapa de julgamento.")
    judge_parser.add_argument(
        "--generation-records",
        required=True,
        help="JSONL produzido pelo comando generate.",
    )
    judge_parser.add_argument(
        "--judge-prompt",
        default=str(default_prompt_path("judge")),
        help="Arquivo de prompt do juiz.",
    )
    judge_parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Número máximo de amostras para julgar (default: sem limite).",
    )
    _add_common_io_args(judge_parser)
    _add_judge_provider_args(judge_parser)
    _add_score_justification_args(judge_parser)
    _add_judge_evidence_args(judge_parser)
    _add_judge_assets_args(judge_parser)

    eval_parser = subparsers.add_parser("evaluate", help="Executa avaliação em lote.")
    eval_parser.add_argument("--dataset", required=True, help="Caminho para .jsonl ou .json")
    eval_parser.add_argument(
        "--generator-prompt",
        default=str(default_prompt_path("generator")),
        help="Arquivo de prompt do gerador.",
    )
    eval_parser.add_argument(
        "--judge-prompt",
        default=str(default_prompt_path("judge")),
        help="Arquivo de prompt do juiz.",
    )
    _add_common_io_args(eval_parser)
    _add_sampling_args(eval_parser)
    _add_dataset_adapter_arg(eval_parser)
    _add_images_dir_arg(eval_parser, required=False)
    _add_generation_provider_args(eval_parser)
    _add_judge_provider_args(eval_parser)
    _add_score_justification_args(eval_parser)
    _add_judge_evidence_args(eval_parser)
    _add_judge_assets_args(eval_parser)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "convert-dataset":
        result = run_dataset_conversion(
            dataset_path=args.dataset,
            output_path=args.output_file,
            dataset_adapter=args.dataset_adapter,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "generate":
        generation_provider = _build_generation_provider(args)
        result = run_generation(
            dataset_path=args.dataset,
            output_dir=args.output_dir,
            generation_provider=generation_provider,
            dataset_adapter=args.dataset_adapter,
            sample_per_type=args.sample_per_type,
            max_samples=args.max_samples,
            seed=args.seed,
            generator_prompt_template=_read_optional_prompt(args.generator_prompt),
            images_dir=args.images_dir,
            resume=args.resume,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "judge":
        judge_provider = _build_judge_provider(args)
        print(f"[judge] Iniciando julgamento de registros: {args.generation_records}")
        judge_start = time.perf_counter()
        result = run_judging(
            generation_records_path=args.generation_records,
            output_dir=args.output_dir,
            judge_provider=judge_provider,
            judge_prompt_template=_read_optional_prompt(args.judge_prompt),
            score_justification=args.score_justification,
            judge_evidence=args.judge_evidence,
            max_samples=args.max_samples,
            judge_images_dir=args.judge_images_dir,
            judge_img_graph_file=args.judge_img_graph_file,
            resume=args.resume,
            progress_callback=lambda idx, total, custom_id: print(
                f"[judge] [{idx}/{total}] Julgando custom_id={custom_id}"
            ),
        )
        judge_elapsed = time.perf_counter() - judge_start
        total_judged = result.get("metrics", {}).get("total")
        if total_judged is None:
            total_judged = "?"
        print(f"[judge] Finalizado: {total_judged} amostras em {judge_elapsed:.2f}s")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "generate-iq":
        generation_provider = _build_generation_provider(args)
        output_file = args.output_file
        if output_file is None:
            output_file = str(Path(args.output_dir) / "generate_iq_output.json")
        result = run_image_question_generation(
            image_path=args.image_path,
            question=args.question,
            generation_provider=generation_provider,
            prompt_template=_read_optional_prompt(args.generator_prompt),
            output_path=output_file,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "generate-iq-batch":
        generation_provider = _build_generation_provider(args)
        result = run_image_question_generation_batch(
            dataset_path=args.dataset,
            images_dir=args.images_dir,
            output_dir=args.output_dir,
            generation_provider=generation_provider,
            dataset_adapter=args.dataset_adapter,
            sample_per_type=args.sample_per_type,
            max_samples=args.max_samples,
            seed=args.seed,
            prompt_template=_read_optional_prompt(args.generator_prompt),
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "evaluate":
        generation_provider = _build_generation_provider(args)
        judge_provider = _build_judge_provider(args)
        result = run_evaluation(
            dataset_path=args.dataset,
            output_dir=args.output_dir,
            generation_provider=generation_provider,
            judge_provider=judge_provider,
            dataset_adapter=args.dataset_adapter,
            sample_per_type=args.sample_per_type,
            max_samples=args.max_samples,
            seed=args.seed,
            generator_prompt_template=_read_optional_prompt(args.generator_prompt),
            judge_prompt_template=_read_optional_prompt(args.judge_prompt),
            images_dir=args.images_dir,
            score_justification=args.score_justification,
            judge_evidence=args.judge_evidence,
            judge_images_dir=args.judge_images_dir,
            judge_img_graph_file=args.judge_img_graph_file,
            resume=args.resume,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
