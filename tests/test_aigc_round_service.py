from __future__ import annotations

import json
import shutil
import sys
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
TEMP_ROOT = ROOT_DIR / ".tmp_tests"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from aigc_round_service import (
    INLINE_BOLD_TOLERANCE,
    RoundPausedError,
    RoundStoppedError,
    build_progress_path,
    build_stop_request_path,
    detect_disallowed_answer_style_pattern,
    detect_introduced_block_markdown,
    detect_prefixed_wrapper,
    detect_suffixed_wrapper,
    detect_wrapped_chat_answer,
    get_prompt_mapping,
    request_stop,
    run_round,
    validate_chunk_output,
)


class ValidateChunkOutputTests(unittest.TestCase):
    def test_disallowed_explanation_prefix_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "contains disallowed answer-style pattern"):
            validate_chunk_output("这是改写后的内容", "说明：这是改写后的内容", "p0_c0")

    def test_disallowed_answer_prefix_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "contains disallowed answer-style pattern"):
            validate_chunk_output("这是改写后的内容", "修改后：这是改写后的内容", "p0_c0")

    def test_input_prefix_inheritance_is_allowed(self) -> None:
        validate_chunk_output("说明：实验结果如下。", "说明：实验结果如下，但是表达更自然。", "p0_c0")
        validate_chunk_output("修改后：系统配置如下。", "修改后：系统配置如下，并补充了说明。", "p0_c1")
        validate_chunk_output("改写后：这是示例文本。", "改写后：这是示例文本，并给出后续描述。", "p0_c2")

    def test_mid_sentence_reference_is_allowed(self) -> None:
        validate_chunk_output("原文", "系统返回“改写后：”字段作为标识。", "p0_c0")
        validate_chunk_output("原文", "标签“说明：”用于提示，不代表回答前缀。", "p0_c1")
        validate_chunk_output("原文", "正文中部出现修改后：这种标签时默认放行。", "p0_c2")

    def test_added_invitation_suffix_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "contains disallowed answer-style pattern"):
            validate_chunk_output("这是原始正文。", "这是原始正文。如果你愿意，我也可以继续帮你调整。", "p0_c0")

    def test_original_invitation_content_is_allowed(self) -> None:
        validate_chunk_output("如果你愿意，这句话本身就是原文。", "如果你愿意，这句话本身就是原文，并略作润色。", "p0_c0")


class RelaxedValidationTests(unittest.TestCase):
    def test_inline_bold_within_tolerance_is_allowed(self) -> None:
        input_text = "本文档介绍系统的核心模块以及它们之间的依赖关系。"
        output_text = "本文档介绍系统的**核心模块**以及它们之间的**依赖关系**。"
        validate_chunk_output(input_text, output_text, "p0_c0")

    def test_inline_bold_when_input_already_has_emphasis(self) -> None:
        input_text = "**关键术语**指的是出现在词汇表中的条目。"
        output_text = "**关键术语**指的是**词汇表**中明确收录的**条目**。"
        validate_chunk_output(input_text, output_text, "p0_c0")

    def test_excessive_inline_bold_is_rejected(self) -> None:
        input_text = "简短一句话。"
        output_text = "".join(f"**关键{i}**" for i in range(INLINE_BOLD_TOLERANCE + 3))
        with self.assertRaisesRegex(ValueError, "introduced excessive inline emphasis"):
            validate_chunk_output(input_text, output_text, "p0_c0")

    def test_block_heading_is_rejected_when_not_in_input(self) -> None:
        input_text = "本节描述系统行为。"
        output_text = "## 标题\n\n本节描述系统行为，更自然一些。"
        with self.assertRaisesRegex(ValueError, "introduced block-level markdown formatting"):
            validate_chunk_output(input_text, output_text, "p0_c0")

    def test_block_heading_allowed_when_input_already_has_heading_marker(self) -> None:
        input_text = "## 系统行为\n\n本节描述系统行为。"
        output_text = "## 系统行为\n\n本节描述系统行为，并补充了说明。"
        validate_chunk_output(input_text, output_text, "p0_c0")

    def test_block_blockquote_is_rejected_when_not_in_input(self) -> None:
        input_text = "这只是一段普通的正文，没有任何引用结构。"
        output_text = "> 这只是一段普通的正文，没有任何引用结构。"
        with self.assertRaisesRegex(ValueError, "introduced block-level markdown formatting"):
            validate_chunk_output(input_text, output_text, "p0_c0")

    def test_inline_greater_than_sign_is_allowed(self) -> None:
        input_text = "性能指标 p99 延迟。"
        output_text = "性能指标 p99 延迟 > 200ms 时触发告警。"
        validate_chunk_output(input_text, output_text, "p0_c0")

    def test_bullet_bold_prefix_is_rejected_when_not_in_input(self) -> None:
        input_text = "支持两种模式：默认模式和高级模式。"
        output_text = "支持两种模式。\n- **默认模式**：开箱即用\n- **高级模式**：可调参数"
        with self.assertRaisesRegex(ValueError, "introduced block-level markdown formatting"):
            validate_chunk_output(input_text, output_text, "p0_c0")

    def test_short_input_modest_expansion_is_allowed(self) -> None:
        input_text = "他来了。"
        output_text = "他终于在傍晚时分赶到了现场，比预定的时间晚了大约半小时。"
        validate_chunk_output(input_text, output_text, "p0_c0")

    def test_extreme_expansion_is_still_rejected(self) -> None:
        input_text = "他来了。"
        output_text = "他来了。" + "随后场面陷入混乱，所有人都在等待解释。" * 80
        with self.assertRaisesRegex(ValueError, "expanded abnormally"):
            validate_chunk_output(input_text, output_text, "p0_c0")

    def test_long_prose_within_threex_is_allowed(self) -> None:
        input_text = "本系统采用模块化设计，包含数据接入、特征工程、模型训练、推理服务、监控告警等组件，能够支持端到端的机器学习生命周期管理。" * 4
        output_text = "本平台采用模块化架构，覆盖数据接入、特征工程、模型训练、推理服务和监控告警等环节，可支撑完整的机器学习生命周期管理工作。" * 4
        validate_chunk_output(input_text, output_text, "p0_c0")

    def test_detect_introduced_block_markdown_returns_first_new_marker(self) -> None:
        self.assertIsNone(detect_introduced_block_markdown("普通正文", "改写后的普通正文。"))
        self.assertIsNone(detect_introduced_block_markdown("## 已有标题\n正文", "## 已有标题\n改写后的正文"))
        self.assertEqual(
            detect_introduced_block_markdown("普通正文", "## 新标题\n改写后的正文"),
            "## ",
        )
        self.assertEqual(
            detect_introduced_block_markdown("普通正文", "> 改写后的正文"),
            "> ",
        )


class DetectAnswerStylePatternTests(unittest.TestCase):
    def test_detects_new_prefixed_wrapper_only_when_body_aligns(self) -> None:
        self.assertEqual(detect_prefixed_wrapper("这是新增前缀", "说明：这是新增前缀"), "说明：")
        self.assertEqual(detect_prefixed_wrapper("这是新增前缀", "修改后：这是新增前缀"), "修改后：")
        self.assertEqual(detect_prefixed_wrapper("这是新增前缀", "改写后：这是新增前缀"), "改写后：")

    def test_detects_new_suffix_wrapper_when_body_aligns(self) -> None:
        self.assertEqual(
            detect_suffixed_wrapper("这是原始正文。", "这是原始正文。如果你愿意，我也可以继续帮你调整。"),
            "如果你愿意",
        )
        self.assertEqual(
            detect_suffixed_wrapper("这是原始正文。", "这是原始正文。请把需要修改的内容继续发我。"),
            "请把需要",
        )

    def test_ignores_original_or_mid_sentence_content(self) -> None:
        self.assertIsNone(detect_disallowed_answer_style_pattern("说明：实验结果如下", "说明：实验结果如下，并补充解释"))
        self.assertIsNone(detect_disallowed_answer_style_pattern("如果你愿意，这句话本身就是原文。", "如果你愿意，这句话本身就是原文，并补充解释。"))
        self.assertIsNone(detect_disallowed_answer_style_pattern("普通正文", "系统返回“改写后：”字段"))
        self.assertIsNone(detect_disallowed_answer_style_pattern("普通正文", "正文中部出现修改后：标签"))

    def test_detects_combined_wrapped_answer(self) -> None:
        self.assertEqual(
            detect_wrapped_chat_answer(
                "这是原始正文。",
                "说明：这是原始正文。如果你愿意，我也可以继续帮你调整。",
            ),
            "说明： ... 如果你愿意",
        )


class PromptMappingTests(unittest.TestCase):
    def test_prompt_mapping_uses_lowercase_distribution_paths(self) -> None:
        self.assertEqual(
            get_prompt_mapping("cn"),
            {
                1: "prompts/baibaiaigc1.md",
                2: "prompts/baibaiaigc2.md",
            },
        )
        self.assertEqual(
            get_prompt_mapping("en"),
            {
                1: "prompts/baibaiaigc-en.md",
            },
        )


class RunRoundRetryTests(unittest.TestCase):
    def make_temp_dir(self) -> Path:
        TEMP_ROOT.mkdir(exist_ok=True)
        temp_dir = TEMP_ROOT / f"case_{uuid.uuid4().hex}"
        temp_dir.mkdir(parents=True, exist_ok=False)
        self.addCleanup(shutil.rmtree, temp_dir, True)
        return temp_dir

    def test_answer_style_failure_retries_once_and_succeeds(self) -> None:
        temp_path = self.make_temp_dir()
        input_path = temp_path / "input.txt"
        output_path = temp_path / "output.txt"
        manifest_path = temp_path / "manifest.json"
        input_path.write_text("这是改写后的正文。", encoding="utf-8")

        prompts: list[str] = []
        responses = iter(["说明：这是改写后的正文。", "这是改写后的正文。"])

        def transform(_: str, prompt_input: str, __: int, ___: str) -> str:
            prompts.append(prompt_input)
            return next(responses)

        with patch("aigc_round_service.update_round", return_value={"ok": True}):
            result = run_round(
                doc_id="tests/retry-success.txt",
                round_number=1,
                input_path=input_path,
                output_path=output_path,
                manifest_path=manifest_path,
                transform=transform,
            )

        self.assertEqual(result["completed_chunk_count"], 1)
        self.assertEqual(output_path.read_text(encoding="utf-8"), "这是改写后的正文。")
        self.assertEqual(len(prompts), 2)
        self.assertIn("Do not output phrases like", prompts[0])
        self.assertIn("[RETRY OUTPUT CONTRACT]", prompts[1])
        self.assertIn("Do not add any answer-style prefix", prompts[1])

    def test_inherited_input_prefix_does_not_retry(self) -> None:
        temp_path = self.make_temp_dir()
        input_path = temp_path / "input.txt"
        output_path = temp_path / "output.txt"
        manifest_path = temp_path / "manifest.json"
        input_path.write_text("说明：实验结果如下。", encoding="utf-8")

        prompts: list[str] = []

        def transform(_: str, prompt_input: str, __: int, ___: str) -> str:
            prompts.append(prompt_input)
            return "说明：实验结果如下，并作了更自然的表达。"

        with patch("aigc_round_service.update_round", return_value={"ok": True}):
            result = run_round(
                doc_id="tests/inherited-prefix.txt",
                round_number=1,
                input_path=input_path,
                output_path=output_path,
                manifest_path=manifest_path,
                transform=transform,
            )

        self.assertEqual(result["completed_chunk_count"], 1)
        self.assertEqual(len(prompts), 1)
        self.assertNotIn("[RETRY OUTPUT CONTRACT]", prompts[0])

    def test_inherited_original_invitation_does_not_retry(self) -> None:
        temp_path = self.make_temp_dir()
        input_path = temp_path / "input.txt"
        output_path = temp_path / "output.txt"
        manifest_path = temp_path / "manifest.json"
        input_path.write_text("如果你愿意，这句话本身就是原文。", encoding="utf-8")

        prompts: list[str] = []

        def transform(_: str, prompt_input: str, __: int, ___: str) -> str:
            prompts.append(prompt_input)
            return "如果你愿意，这句话本身就是原文，并稍作润色。"

        with patch("aigc_round_service.update_round", return_value={"ok": True}):
            result = run_round(
                doc_id="tests/inherited-invitation.txt",
                round_number=1,
                input_path=input_path,
                output_path=output_path,
                manifest_path=manifest_path,
                transform=transform,
            )

        self.assertEqual(result["completed_chunk_count"], 1)
        self.assertEqual(len(prompts), 1)

    def test_second_answer_style_failure_pauses_with_same_error_shape(self) -> None:
        temp_path = self.make_temp_dir()
        input_path = temp_path / "input.txt"
        output_path = temp_path / "output.txt"
        manifest_path = temp_path / "manifest.json"
        progress_path = build_progress_path(manifest_path)
        input_path.write_text("仍然是回答腔", encoding="utf-8")

        call_count = 0

        def transform(_: str, __: str, ___: int, ____: str) -> str:
            nonlocal call_count
            call_count += 1
            return "说明：仍然是回答腔"

        with patch("aigc_round_service.update_round", return_value={"ok": True}):
            with self.assertRaisesRegex(RoundPausedError, "contains disallowed answer-style pattern"):
                run_round(
                    doc_id="tests/retry-fail.txt",
                    round_number=1,
                    input_path=input_path,
                    output_path=output_path,
                    manifest_path=manifest_path,
                    transform=transform,
                )

        self.assertEqual(call_count, 2)
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        self.assertEqual(progress["status"], "paused")
        self.assertIn("contains disallowed answer-style pattern", progress["last_error"])

    def test_non_answer_style_failure_does_not_retry(self) -> None:
        temp_path = self.make_temp_dir()
        input_path = temp_path / "input.txt"
        output_path = temp_path / "output.txt"
        manifest_path = temp_path / "manifest.json"
        input_path.write_text("这是原始正文。", encoding="utf-8")

        call_count = 0

        def transform(_: str, __: str, ___: int, ____: str) -> str:
            nonlocal call_count
            call_count += 1
            raise RuntimeError("network error")

        with patch("aigc_round_service.update_round", return_value={"ok": True}):
            with self.assertRaisesRegex(RoundPausedError, "network error"):
                run_round(
                    doc_id="tests/network-error.txt",
                    round_number=1,
                    input_path=input_path,
                    output_path=output_path,
                    manifest_path=manifest_path,
                    transform=transform,
                )

        self.assertEqual(call_count, 1)

    def test_user_requested_stop_marks_progress_stopped(self) -> None:
        temp_path = self.make_temp_dir()
        input_path = temp_path / "input.txt"
        output_path = temp_path / "output.txt"
        manifest_path = temp_path / "manifest.json"
        progress_path = build_progress_path(manifest_path)
        stop_path = build_stop_request_path(manifest_path)
        input_path.write_text("第一段。\n\n第二段。", encoding="utf-8")

        call_count = 0

        def transform(chunk_text: str, __: str, ___: int, ____: str) -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                request_stop(progress_path)
            return f"{chunk_text} 已改写"

        with patch("aigc_round_service.update_round", return_value={"ok": True}):
            with self.assertRaisesRegex(RoundStoppedError, "用户手动停止"):
                run_round(
                    doc_id="tests/stopped.txt",
                    round_number=1,
                    input_path=input_path,
                    output_path=output_path,
                    manifest_path=manifest_path,
                    transform=transform,
                    chunk_limit=3,
                )

        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        self.assertEqual(progress["status"], "stopped")
        self.assertEqual(progress["completed_chunks"], 1)
        self.assertEqual(progress["stop_reason"], "用户手动停止，保留当前进度，可继续执行当前轮。")
        self.assertFalse(progress["stop_requested"])
        self.assertFalse(stop_path.exists())

    def test_resume_after_stop_uses_saved_progress(self) -> None:
        temp_path = self.make_temp_dir()
        input_path = temp_path / "input.txt"
        output_path = temp_path / "output.txt"
        manifest_path = temp_path / "manifest.json"
        input_path.write_text("第一段。\n\n第二段。", encoding="utf-8")

        first_call_count = 0

        def stop_after_first_chunk(chunk_text: str, __: str, ___: int, ____: str) -> str:
            nonlocal first_call_count
            first_call_count += 1
            if first_call_count == 1:
                request_stop(build_progress_path(manifest_path))
            return f"{chunk_text} 已改写"

        with patch("aigc_round_service.update_round", return_value={"ok": True}):
            with self.assertRaises(RoundStoppedError):
                run_round(
                    doc_id="tests/resume-after-stop.txt",
                    round_number=1,
                    input_path=input_path,
                    output_path=output_path,
                    manifest_path=manifest_path,
                    transform=stop_after_first_chunk,
                    chunk_limit=3,
                )

        resumed_call_count = 0

        def resume_transform(chunk_text: str, __: str, ___: int, ____: str) -> str:
            nonlocal resumed_call_count
            resumed_call_count += 1
            return f"{chunk_text} 已改写"

        with patch("aigc_round_service.update_round", return_value={"ok": True}):
            result = run_round(
                doc_id="tests/resume-after-stop.txt",
                round_number=1,
                input_path=input_path,
                output_path=output_path,
                manifest_path=manifest_path,
                transform=resume_transform,
                chunk_limit=3,
            )

        self.assertTrue(result["resumed"])
        self.assertEqual(resumed_call_count, result["completed_chunk_count"] - 1)
        output_text = output_path.read_text(encoding="utf-8")
        self.assertIn("第一段", output_text)
        self.assertIn("第二段", output_text)
        self.assertIn("已改写", output_text)


if __name__ == "__main__":
    unittest.main()
