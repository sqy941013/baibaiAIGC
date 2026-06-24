"""Deai (降AIGC / de-AI-flavor) API server.

Exposes a simple HTTP API for the two-round AIGC reduction pipeline
built on baibaiAIGC's existing scripts.  Designed to run as a
standalone Docker service in a docker-compose stack.
"""
from __future__ import annotations

import json
import os
import queue
import re
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

from flask import Flask, Response, jsonify, request

# ---------------------------------------------------------------------------
# Bootstrap: make the scripts/ directory importable
# ---------------------------------------------------------------------------
SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from markdown_chunks import detect_markdown_blocks, process_text_blocks  # noqa: E402
from aigc_records import ROOT_DIR  # noqa: E402
from aigc_round_service import PROMPT_PROFILES, run_round  # noqa: E402
from llm_client import llm_completion, read_api_config  # noqa: E402
from app_config import get_app_config_path, load_app_config  # noqa: E402

app = Flask(__name__)

DEFAULT_LLM_TIMEOUT_SECONDS = None
DEFAULT_HEARTBEAT_INTERVAL_SECONDS = 15
DEFAULT_LLM_PROBE_INTERVAL_SECONDS = 60
DEFAULT_LLM_PROBE_TIMEOUT_SECONDS = 10
DEFAULT_LLM_PROBE_FAILURE_THRESHOLD = 3
_JOBS: dict[str, dict[str, Any]] = {}
_JOBS_LOCK = threading.Lock()


def _expected_prompt_paths() -> list[Path]:
    """Absolute paths of every prompt file declared in PROMPT_PROFILES."""
    seen: dict[str, Path] = {}
    for prompts in PROMPT_PROFILES.values():
        for relative in prompts.values():
            absolute = (ROOT_DIR / relative).resolve()
            seen[str(absolute)] = absolute
    return list(seen.values())


def _missing_prompt_paths() -> list[Path]:
    return [path for path in _expected_prompt_paths() if not path.is_file()]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_api_config(payload: dict) -> tuple[str | None, str | None, str | None, str | None]:
    """Resolve API credentials with fallback chain.

    Priority order:
    1. Container-side environment variables, *if* they fully configure a
       provider (any of ``MINIMAX_*`` / ``BAIBAIAIGC_*`` / ``OPENAI_*``
       covering api_key + model + base_url). When the deai container is
       set up with its own LLM provider, payload-forwarded credentials
       from the caller are ignored, so polish always runs against the
       operator's chosen provider regardless of what the upstream
       service uses.
    2. Otherwise, request body parameters (``payload``) — the upstream
       service may forward its own credentials to keep deai polish on
       the same model as the inline path. Missing fields fall back to
       the partial env values resolved by :func:`read_api_config`.
    3. ``~/.baibaiaigc/config.json`` via :func:`load_app_config`
       (maps ``baseUrl`` / ``apiKey`` / ``model`` / ``apiType``).
    """
    env_api_key, env_model, env_base_url, env_api_type = read_api_config(
        None, None, None, None,
    )
    if env_api_key and env_model and env_base_url:
        return env_api_key, env_model, env_base_url, env_api_type

    api_key = payload.get("api_key") or payload.get("apiKey") or env_api_key
    model = payload.get("model") or env_model
    base_url = payload.get("base_url") or payload.get("baseUrl") or env_base_url
    api_type = payload.get("api_type") or payload.get("apiType") or env_api_type

    if not (api_key and model and base_url) and get_app_config_path().exists():
        try:
            cfg = load_app_config()
        except Exception:
            cfg = {}
        api_key = api_key or (cfg.get("apiKey") or None)
        model = model or (cfg.get("model") or None)
        base_url = base_url or (cfg.get("baseUrl") or None)
        api_type = api_type or (cfg.get("apiType") or None)

    return api_key, model, base_url, api_type


def _build_transform(
    api_key: str,
    model: str,
    base_url: str,
    api_type: str | None,
    temperature: float,
    timeout: int | None,
    progress_callback: Any | None = None,
    heartbeat_interval: int = DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
):
    """Return a transform(chunk_text, prompt_input, round_number, chunk_id) -> str."""
    def transform(chunk_text: str, prompt_input: str, round_number: int, chunk_id: str) -> str:
        if progress_callback is None:
            return llm_completion(
                prompt_input,
                model=model,
                api_key=api_key,
                base_url=base_url,
                api_type=api_type,
                temperature=temperature,
                timeout=timeout,
            )

        progress_callback({
            "phase": "llm-request-start",
            "round": round_number,
            "chunkId": chunk_id,
            "timeout": timeout or 0,
            "model": model,
        })
        result_queue: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=1)
        started_at = time.time()

        def emit_stream_progress(event: dict[str, object]) -> None:
            progress_event = {key: value for key, value in event.items() if key != "delta"}
            progress_callback({
                **progress_event,
                "phase": "llm-stream-chunk",
                "round": round_number,
                "chunkId": chunk_id,
                "elapsedSeconds": progress_event.get(
                    "elapsedSeconds",
                    round(time.time() - started_at, 1),
                ),
                "timeout": timeout or 0,
                "model": model,
            })

        def call_llm() -> None:
            try:
                result_queue.put((
                    "ok",
                    llm_completion(
                        prompt_input,
                        model=model,
                        api_key=api_key,
                        base_url=base_url,
                        api_type=api_type,
                        temperature=temperature,
                        timeout=timeout,
                        stream_callback=emit_stream_progress,
                    ),
                ))
            except Exception as exc:  # noqa: BLE001 - propagate original error to caller.
                result_queue.put(("error", exc))

        worker = threading.Thread(target=call_llm, daemon=True)
        worker.start()
        last_probe_at = 0.0
        consecutive_probe_failures = 0
        last_probe: dict[str, Any] | None = None
        probe_enabled = _llm_probe_enabled()
        probe_interval = _llm_probe_interval()
        probe_timeout = _llm_probe_timeout()
        probe_failure_threshold = _llm_probe_failure_threshold()
        while True:
            try:
                status, value = result_queue.get(timeout=max(1, heartbeat_interval))
                break
            except queue.Empty:
                now = time.time()
                elapsed_seconds = round(now - started_at, 1)
                event: dict[str, Any] = {
                    "phase": "llm-request-waiting",
                    "round": round_number,
                    "chunkId": chunk_id,
                    "elapsedSeconds": elapsed_seconds,
                    "timeout": timeout or 0,
                    "model": model,
                }
                if probe_enabled and (now - last_probe_at) >= probe_interval:
                    last_probe_at = now
                    last_probe = _probe_llm_api(base_url, timeout=probe_timeout)
                    if last_probe.get("ok"):
                        consecutive_probe_failures = 0
                    else:
                        consecutive_probe_failures += 1
                    event["llmProbe"] = last_probe
                    event["llmProbeFailures"] = consecutive_probe_failures
                elif last_probe is not None:
                    event["llmProbe"] = last_probe
                    event["llmProbeFailures"] = consecutive_probe_failures

                progress_callback(event)

                if (
                    probe_enabled
                    and consecutive_probe_failures >= probe_failure_threshold
                ):
                    detail = last_probe.get("detail") if last_probe else ""
                    error_message = (
                        "LLM API liveness probe failed "
                        f"{consecutive_probe_failures} consecutive times while waiting "
                        f"for round {round_number} chunk {chunk_id}: "
                        f"status={(last_probe or {}).get('status')}; detail={detail or 'n/a'}"
                    )
                    progress_callback({
                        "phase": "llm-api-unreachable",
                        "round": round_number,
                        "chunkId": chunk_id,
                        "elapsedSeconds": elapsed_seconds,
                        "llmProbe": last_probe,
                        "llmProbeFailures": consecutive_probe_failures,
                        "error": error_message,
                    })
                    raise RuntimeError(error_message)

        if status == "error":
            progress_callback({
                "phase": "llm-request-error",
                "round": round_number,
                "chunkId": chunk_id,
                "elapsedSeconds": round(time.time() - started_at, 1),
                "error": str(value),
            })
            raise value

        progress_callback({
            "phase": "llm-request-complete",
            "round": round_number,
            "chunkId": chunk_id,
            "elapsedSeconds": round(time.time() - started_at, 1),
        })
        return str(value)
    return transform


def _markdown_structural_markers(text: str) -> dict[str, int]:
    """Count key Markdown structural markers in *text*."""
    return {
        "tables": len(re.findall(r"^\|", text, re.MULTILINE)),
        "images": len(re.findall(r"!\[", text)),
        "links": len(re.findall(r"(?<!\!)\[", text)),
        "code_blocks": len(re.findall(r"```", text)) // 2,
        "inline_code": len(re.findall(r"`(?![`])", text)) // 2,
        "headings": len(re.findall(r"^#{1,6}\s", text, re.MULTILINE)),
        "bold": len(re.findall(r"\*\*", text)) // 2,
        "italic": len(re.findall(r"(?<!\*)\*(?!\*)", text)) // 2,
        "list_items": len(re.findall(r"^\s*[-*+]\s", text, re.MULTILINE))
                    + len(re.findall(r"^\s*\d+\.\s", text, re.MULTILINE)),
        "blockquotes": len(re.findall(r"^>", text, re.MULTILINE)),
        "horizontal_rules": len(re.findall(r"^---+$", text, re.MULTILINE))
                           + len(re.findall(r"^\*\*\*+$", text, re.MULTILINE)),
    }


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw = os.environ.get(name, "").strip()
    if raw:
        try:
            return max(minimum, int(raw))
        except ValueError:
            app.logger.warning("Invalid %s=%r, using default %s", name, raw, default)
    return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw not in {"0", "false", "no", "off", "disabled", "disable"}


def _optional_timeout_value(value: Any, default: int | None) -> int | None:
    raw = "" if value is None else str(value).strip().lower()
    if raw in {"0", "false", "no", "none", "off", "disabled", "disable"}:
        return None
    if raw:
        try:
            return max(1, int(raw))
        except (TypeError, ValueError):
            return default
    return default


def _payload_timeout(payload: dict) -> int | None:
    raw = payload.get("timeout")
    if raw in (None, ""):
        return _optional_timeout_value(
            os.environ.get("DEAI_LLM_TIMEOUT"),
            DEFAULT_LLM_TIMEOUT_SECONDS,
        )
    return _optional_timeout_value(
        raw,
        _optional_timeout_value(os.environ.get("DEAI_LLM_TIMEOUT"), DEFAULT_LLM_TIMEOUT_SECONDS),
    )


def _heartbeat_interval() -> int:
    return _env_int("DEAI_HEARTBEAT_INTERVAL_SECONDS", DEFAULT_HEARTBEAT_INTERVAL_SECONDS)


def _llm_probe_enabled() -> bool:
    return _env_bool("DEAI_LLM_PROBE_ENABLED", True)


def _llm_probe_interval() -> int:
    return _env_int("DEAI_LLM_PROBE_INTERVAL_SECONDS", DEFAULT_LLM_PROBE_INTERVAL_SECONDS)


def _llm_probe_timeout() -> int:
    return _env_int("DEAI_LLM_PROBE_TIMEOUT_SECONDS", DEFAULT_LLM_PROBE_TIMEOUT_SECONDS)


def _llm_probe_failure_threshold() -> int:
    return _env_int("DEAI_LLM_PROBE_FAILURE_THRESHOLD", DEFAULT_LLM_PROBE_FAILURE_THRESHOLD)


def _probe_llm_api(base_url: str, *, timeout: int) -> dict[str, Any]:
    """Return low-cost liveness for the configured upstream LLM endpoint.

    Streaming chunk progress refreshes the job heartbeat when the provider is
    actively returning data. When no stream data has arrived for a heartbeat
    interval, this probe checks whether the configured provider base URL is
    still reachable. Any HTTP response below 500 means the endpoint is
    reachable; 5xx and transport errors are treated as probe failures.
    """

    probe_url = (base_url or "").strip().rstrip("/")
    if not probe_url:
        return {
            "ok": False,
            "status": "missing_base_url",
            "checked_at": _now(),
        }

    request_obj = urllib.request.Request(
        probe_url,
        headers={"User-Agent": "ai-doc-gen-deai-liveness/1.0"},
        method="GET",
    )
    checked_at = _now()
    try:
        with urllib.request.urlopen(request_obj, timeout=timeout) as response:
            status_code = int(getattr(response, "status", 200) or 200)
        return {
            "ok": status_code < 500,
            "status": status_code,
            "checked_at": checked_at,
            "url": probe_url,
        }
    except urllib.error.HTTPError as exc:
        status_code = int(getattr(exc, "code", 0) or 0)
        return {
            "ok": 0 < status_code < 500,
            "status": status_code or "http_error",
            "checked_at": checked_at,
            "url": probe_url,
            "detail": str(exc.reason or exc),
        }
    except Exception as exc:  # noqa: BLE001 - probe failure should be reported, not crash formatting.
        return {
            "ok": False,
            "status": "transport_error",
            "checked_at": checked_at,
            "url": probe_url,
            "detail": str(exc),
        }


def _now() -> float:
    return time.time()


def _update_job(job_id: str, **fields: Any) -> None:
    now = _now()
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            return
        job.update(fields)
        job["updated_at"] = now
        if fields.get("heartbeat", True):
            job["last_heartbeat_at"] = now
        job.pop("heartbeat", None)


def _job_snapshot(job_id: str) -> dict[str, Any] | None:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            return None
        data = dict(job)
    if data.get("status") != "completed":
        data.pop("content", None)
    return data


def _sse_payload(event: str, data: dict[str, Any]) -> str:
    return (
        f"event: {event}\n"
        f"data: {json.dumps(data, ensure_ascii=False, separators=(',', ':'))}\n\n"
    )


def _job_stream_interval() -> float:
    raw = os.environ.get("DEAI_JOB_STREAM_INTERVAL_SECONDS", "").strip()
    if raw:
        try:
            return max(0.25, float(raw))
        except ValueError:
            app.logger.warning("Invalid DEAI_JOB_STREAM_INTERVAL_SECONDS=%r, using default 1", raw)
    return 1.0


def _iter_job_events(job_id: str):
    """Yield Server-Sent Events for a job until it reaches a terminal state."""

    interval = _job_stream_interval()
    last_revision: tuple[object, object, object, object] | None = None
    while True:
        job = _job_snapshot(job_id)
        if job is None:
            yield _sse_payload("error", {"error": "job not found", "job_id": job_id})
            return

        revision = (
            job.get("status"),
            job.get("phase"),
            job.get("updated_at"),
            job.get("last_heartbeat_at"),
        )
        if revision != last_revision:
            last_revision = revision
            yield _sse_payload("job", job)

        status = str(job.get("status") or "").strip().lower()
        if status == "completed":
            yield _sse_payload("completed", job)
            return
        if status == "failed":
            yield _sse_payload("failed", job)
            return

        yield ": heartbeat\n\n"
        time.sleep(interval)


def _progress_callback_for_job(job_id: str):
    def capture(event: dict[str, Any]) -> None:
        phase = str(event.get("phase") or "running")
        _update_job(
            job_id,
            status="running",
            phase=phase,
            progress=event,
            heartbeat=True,
        )

    return capture


def _run_deai_job(job_id: str, payload: dict[str, Any]) -> None:
    _update_job(job_id, status="running", phase="resolving-config", heartbeat=True)
    content = payload["content"]
    rounds = int(payload.get("rounds", 2))
    prompt_profile = payload.get("prompt_profile", "cn")
    temperature = float(payload.get("temperature", 0.7))
    chunk_limit = int(payload.get("chunk_limit", 850))
    timeout = _payload_timeout(payload)
    dry_run = bool(payload.get("dry_run"))

    if prompt_profile == "en" and rounds > 1:
        rounds = 1

    try:
        if dry_run:
            result = process_deai(
                content,
                rounds=rounds,
                prompt_profile=prompt_profile,
                api_key="",
                model="",
                base_url="",
                dry_run=True,
                progress_callback=_progress_callback_for_job(job_id),
            )
        else:
            api_key, model, base_url, api_type = _resolve_api_config(payload)
            if not (api_key and model and base_url):
                raise ValueError(
                    "API mode requires api_key, model, and base_url "
                    "(via request body, environment variables, or ~/.baibaiaigc/config.json)."
                )
            result = process_deai(
                content,
                rounds=rounds,
                prompt_profile=prompt_profile,
                api_key=api_key,
                model=model,
                base_url=base_url,
                api_type=api_type,
                temperature=temperature,
                chunk_limit=chunk_limit,
                timeout=timeout,
                progress_callback=_progress_callback_for_job(job_id),
            )
    except Exception as exc:  # noqa: BLE001 - expose concise status to caller, log full traceback.
        app.logger.exception("deai job %s failed", job_id)
        _update_job(
            job_id,
            status="failed",
            phase="failed",
            error=str(exc),
            heartbeat=True,
        )
        return

    _update_job(
        job_id,
        status="completed",
        phase="completed",
        content=result.get("content"),
        result={
            key: value
            for key, value in result.items()
            if key != "content"
        },
        heartbeat=True,
    )


def _validate_payload(payload: dict | None) -> tuple[dict[str, Any] | None, tuple[Response, int] | None]:
    if not payload or "content" not in payload:
        return None, (jsonify({"error": "content is required"}), 400)
    content = payload["content"]
    if not isinstance(content, str) or not content.strip():
        return None, (jsonify({"error": "content must be a non-empty string"}), 400)
    return dict(payload), None


# ---------------------------------------------------------------------------
# Core processing logic
# ---------------------------------------------------------------------------

def _looks_like_markdown(text: str) -> bool:
    """Quick heuristic: does *text* contain Markdown structural elements?"""
    indicators = [
        re.search(r"^\|", text, re.MULTILINE),          # tables
        re.search(r"!\[", text),                          # images
        re.search(r"^```", text, re.MULTILINE),           # code blocks
        re.search(r"^> ", text, re.MULTILINE),            # blockquotes
        re.search(r"^\s*[-*+]\s", text, re.MULTILINE),    # unordered lists
        re.search(r"^\s*\d+\.\s", text, re.MULTILINE),    # ordered lists
    ]
    return sum(1 for m in indicators if m) >= 1


def process_deai(
    content: str,
    *,
    rounds: int = 2,
    prompt_profile: str = "cn",
    api_key: str,
    model: str,
    base_url: str,
    api_type: str | None = None,
    temperature: float = 0.7,
    chunk_limit: int = 850,
    timeout: int | None = None,
    dry_run: bool = False,
    progress_callback: Any | None = None,
) -> dict[str, Any]:
    """Run the deai pipeline on *content* and return result dict.

    For Markdown content (detected by structural elements), uses block-aware
    processing that preserves tables, code blocks, lists, blockquotes, etc.
    For plain text, falls back to the standard chunking pipeline.
    """
    before_markers = _markdown_structural_markers(content)
    use_markdown_blocks = _looks_like_markdown(content)

    round_results: list[dict] = []
    current_text = content

    if progress_callback is not None:
        progress_callback({
            "phase": "started",
            "rounds": rounds,
            "markdownMode": use_markdown_blocks,
            "inputLength": len(content),
        })

    for round_num in range(1, rounds + 1):
        if progress_callback is not None:
            progress_callback({
                "phase": "round-start",
                "round": round_num,
                "rounds": rounds,
                "markdownMode": use_markdown_blocks,
                "inputLength": len(current_text),
            })
        if dry_run:
            transform = lambda chunk_text, *_: chunk_text  # noqa: E731
        else:
            transform = _build_transform(
                api_key,
                model,
                base_url,
                api_type,
                temperature,
                timeout,
                progress_callback=progress_callback,
            )

        if use_markdown_blocks:
            blocks = detect_markdown_blocks(current_text)
            if dry_run:
                # In dry-run mode, just reconstruct without processing text blocks
                parts = []
                for block in blocks:
                    if block.preserved:
                        parts.append(block.text)
                    else:
                        parts.append(block.text)  # identity in dry-run
                current_text = "\n".join(parts)
                round_results.append({
                    "round": round_num,
                    "mode": "markdown_blocks_dry",
                    "blocks_total": len(blocks),
                    "preserved_blocks": sum(1 for b in blocks if b.preserved),
                })
            else:
                current_text = process_text_blocks(
                    blocks,
                    transform=transform,
                    prompt_profile=prompt_profile,
                    round_number=round_num,
                    chunk_limit=chunk_limit,
                    progress_callback=progress_callback,
                )
                round_results.append({
                    "round": round_num,
                    "mode": "markdown_blocks",
                    "blocks_total": len(blocks),
                    "preserved_blocks": sum(1 for b in blocks if b.preserved),
                })
        else:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp = Path(tmpdir)
                doc_id = "deai_api_input"
                input_path = tmp / "input.txt"
                output_path = tmp / f"round{round_num}.txt"
                manifest_path = tmp / f"round{round_num}_manifest.json"

                input_path.write_text(current_text, encoding="utf-8")

                result = run_round(
                    doc_id=doc_id,
                    round_number=round_num,
                    input_path=input_path,
                    output_path=output_path,
                    manifest_path=manifest_path,
                    transform=transform,
                    prompt_profile=prompt_profile,
                    chunk_limit=chunk_limit,
                    progress_callback=progress_callback,
                )
                round_results.append(result)
                current_text = output_path.read_text(encoding="utf-8")

        if progress_callback is not None:
            progress_callback({
                "phase": "round-complete",
                "round": round_num,
                "rounds": rounds,
                "outputLength": len(current_text),
            })

    after_markers = _markdown_structural_markers(current_text)

    result = {
        "content": current_text,
        "rounds_executed": rounds,
        "round_details": round_results,
        "dry_run": dry_run,
        "markdown_mode": use_markdown_blocks,
        "input_markers": before_markers,
        "output_markers": after_markers,
        "markdown_integrity": before_markers == after_markers,
        "input_length": len(content),
        "output_length": len(current_text),
    }
    if progress_callback is not None:
        progress_callback({
            "phase": "completed",
            "rounds": rounds,
            "outputLength": len(current_text),
            "markdownIntegrity": result["markdown_integrity"],
        })
    return result


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health() -> tuple[Response, int]:
    missing = _missing_prompt_paths()
    if missing:
        return jsonify({
            "service": "deai",
            "status": "unhealthy",
            "reason": "prompt files missing",
            "missing_prompts": [str(p.relative_to(ROOT_DIR)) for p in missing],
        }), 503
    return jsonify({"status": "ok", "service": "deai"}), 200


@app.route("/api/deai/process", methods=["POST"])
def deai_process() -> tuple[Response, int]:
    """Process text through the deai pipeline.

    Request JSON body:
    {
        "content": "...",           // required: text to process
        "rounds": 2,                // optional: 1 or 2, default 2
        "prompt_profile": "cn",     // optional: "cn" or "en", default "cn"
        "api_key": "...",           // optional (env / ~/.baibaiaigc/config.json fallback)
        "model": "...",             // optional (env / ~/.baibaiaigc/config.json fallback)
        "base_url": "...",          // optional (env / ~/.baibaiaigc/config.json fallback)
        "api_type": null,           // optional: "chat_completions" or "responses"
        "temperature": 0.7,         // optional
        "chunk_limit": 850,         // optional
        "timeout": 120,             // optional, per-chunk LLM timeout
        "dry_run": false            // optional, skip LLM for testing
    }
    """
    payload, error_response = _validate_payload(request.get_json(silent=True))
    if error_response is not None:
        return error_response
    assert payload is not None

    content = payload["content"]

    rounds = int(payload.get("rounds", 2))
    prompt_profile = payload.get("prompt_profile", "cn")
    temperature = float(payload.get("temperature", 0.7))
    chunk_limit = int(payload.get("chunk_limit", 850))
    timeout = _payload_timeout(payload)
    dry_run = bool(payload.get("dry_run"))

    # en profile only supports 1 round
    if prompt_profile == "en" and rounds > 1:
        rounds = 1

    if dry_run:
        result = process_deai(
            content,
            rounds=rounds,
            prompt_profile=prompt_profile,
            api_key="",
            model="",
            base_url="",
            dry_run=True,
        )
        return jsonify(result), 200

    api_key, model, base_url, api_type = _resolve_api_config(payload)
    if not (api_key and model and base_url):
        return jsonify({
            "error": "API mode requires api_key, model, and base_url "
                     "(via request body, environment variables, or "
                     "~/.baibaiaigc/config.json). "
                     "Use dry_run=true to skip LLM calls.",
        }), 400

    try:
        result = process_deai(
            content,
            rounds=rounds,
            prompt_profile=prompt_profile,
            api_key=api_key,
            model=model,
            base_url=base_url,
            api_type=api_type,
            temperature=temperature,
            chunk_limit=chunk_limit,
            timeout=timeout,
        )
        return jsonify(result), 200
    except Exception:
        # Log the full traceback server-side but do not leak internal details
        # (stack frames, file paths, etc.) to the caller.
        app.logger.exception("deai_process failed")
        return jsonify({"error": "internal error processing deai request"}), 500


@app.route("/api/deai/jobs", methods=["POST"])
def deai_create_job() -> tuple[Response, int]:
    """Create an asynchronous deai job.

    Long document polish runs can take hours. This endpoint returns quickly
    and exposes progress/heartbeat through ``GET /api/deai/jobs/<job_id>``.
    """
    payload, error_response = _validate_payload(request.get_json(silent=True))
    if error_response is not None:
        return error_response
    assert payload is not None

    job_id = uuid.uuid4().hex
    now = _now()
    with _JOBS_LOCK:
        _JOBS[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "phase": "queued",
            "created_at": now,
            "updated_at": now,
            "last_heartbeat_at": now,
            "input_length": len(payload["content"]),
            "rounds": int(payload.get("rounds", 2)),
            "prompt_profile": payload.get("prompt_profile", "cn"),
        }

    worker = threading.Thread(target=_run_deai_job, args=(job_id, payload), daemon=True)
    worker.start()
    return jsonify({
        "job_id": job_id,
        "status": "queued",
        "status_url": f"/api/deai/jobs/{job_id}",
        "stream_url": f"/api/deai/jobs/{job_id}/stream",
    }), 202


@app.route("/api/deai/jobs/<job_id>", methods=["GET"])
def deai_get_job(job_id: str) -> tuple[Response, int]:
    job = _job_snapshot(job_id)
    if job is None:
        return jsonify({"error": "job not found"}), 404
    return jsonify(job), 200


@app.route("/api/deai/jobs/<job_id>/stream", methods=["GET"])
def deai_stream_job(job_id: str) -> Response:
    """Stream job progress with Server-Sent Events.

    The polling status endpoint remains the compatibility contract. This
    stream endpoint lets callers keep one read open and receive every
    heartbeat/progress update produced by the LLM streaming callback.
    """

    return Response(
        _iter_job_events(job_id),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    missing = _missing_prompt_paths()
    if missing:
        rendered = ", ".join(str(p.relative_to(ROOT_DIR)) for p in missing)
        print(
            f"Deai API server cannot start: missing prompt files: {rendered}",
            file=sys.stderr,
        )
        sys.exit(1)
    port = int(os.environ.get("PORT", 8000))
    print(f"Deai API server starting on port {port}")
    app.run(host="0.0.0.0", port=port, threaded=True)


if __name__ == "__main__":
    main()
