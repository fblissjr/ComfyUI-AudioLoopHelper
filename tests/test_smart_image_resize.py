"""Tests for LTXSmartImageResize.

Last updated: 2026-05-07

Adaptive multi-stage lanczos downscaling. Standard `lanczos` (kernel
radius 3) integrates ~6 input samples per output pixel; at reduction
ratios >2x linear, the kernel sees too few samples and aliases. The
fix is staged downscaling — at most 2x reduction per pass keeps each
pass within the kernel's clean range.

Pure-function planner (`_compute_resize_stages`) tested directly +
behavioral tests on the node's `execute` against synthetic IMAGE
tensors.

The node lives in `nodes.py` and is registered via the extension's
node list.
"""

from __future__ import annotations

import math

import pytest
import torch

from nodes import (  # type: ignore[import-not-found]
    LTXSmartImageResize,
    _compute_resize_stages,
    _crop_to_aspect,
)


def _img(h: int, w: int, b: int = 1) -> torch.Tensor:
    """ComfyUI IMAGE shape: [B, H, W, C] in float32 [0,1]."""
    return torch.rand(b, h, w, 3, dtype=torch.float32)


# ---------- Planner: pure-function tests ----------

class TestComputeResizeStages:
    def test_no_op_when_source_equals_target(self):
        stages = _compute_resize_stages(832, 448, 832, 448)
        # Empty list = no resize work needed (or single-stage identity is also acceptable;
        # contract: caller treats len==0 as pass-through).
        assert stages == []

    def test_single_stage_for_small_reduction(self):
        # 1024x576 -> 832x448: ratio max(1024/832, 576/448) = 1.286, well below 2x
        stages = _compute_resize_stages(1024, 576, 832, 448)
        assert stages == [(832, 448)]

    def test_single_stage_for_2x_or_below(self):
        # exactly 2x reduction is still within lanczos kernel's clean range
        stages = _compute_resize_stages(1664, 896, 832, 448)
        assert stages == [(832, 448)]

    def test_two_stages_for_user_2752x1536_case(self):
        # The actual reported case: 2752x1536 -> 832x448, ratio 3.31x linear
        stages = _compute_resize_stages(2752, 1536, 832, 448)
        assert len(stages) == 2, f"expected 2 stages, got {stages}"
        # Last stage is exactly the target
        assert stages[-1] == (832, 448)
        # Each stage reduces by at most 2x linear vs the previous
        prev_w, prev_h = 2752, 1536
        for sw, sh in stages:
            assert prev_w / sw <= 2.0 + 1e-6, f"stage {sw}x{sh} reduces > 2x from {prev_w}x{prev_h}"
            assert prev_h / sh <= 2.0 + 1e-6
            prev_w, prev_h = sw, sh

    def test_three_stages_for_4k(self):
        # 3840x2160 -> 832x448, ratio max(3840/832, 2160/448) = max(4.62, 4.82) = 4.82
        stages = _compute_resize_stages(3840, 2160, 832, 448)
        assert len(stages) == 3, f"expected 3 stages for 4K source, got {stages}"
        assert stages[-1] == (832, 448)
        prev_w, prev_h = 3840, 2160
        for sw, sh in stages:
            assert prev_w / sw <= 2.0 + 1e-6
            assert prev_h / sh <= 2.0 + 1e-6
            prev_w, prev_h = sw, sh

    def test_upscale_is_single_stage(self):
        # Upscaling from 512x288 -> 832x448. No aliasing risk; staging would just blur.
        stages = _compute_resize_stages(512, 288, 832, 448)
        assert stages == [(832, 448)]

    def test_stage_count_formula(self):
        # ceil(log2(ratio)) stages for ratio > 2x
        cases = [
            (1024, 576, 832, 448, 1),     # ratio 1.29 -> 1
            (1664, 896, 832, 448, 1),     # ratio 2.0 -> 1
            (1665, 897, 832, 448, 2),     # ratio just over 2 -> 2 (ceil(log2(2.001)) == 2)
            (2752, 1536, 832, 448, 2),    # ratio 3.31 -> 2
            (3328, 1792, 832, 448, 2),    # ratio 4.0 -> 2
            (3329, 1793, 832, 448, 3),    # ratio just over 4 -> 3
            (3840, 2160, 832, 448, 3),    # 4K -> 3
        ]
        for sw, sh, tw, th, expected_stages in cases:
            stages = _compute_resize_stages(sw, sh, tw, th)
            assert len(stages) == expected_stages, (
                f"src {sw}x{sh} -> tgt {tw}x{th}: expected {expected_stages} stages, "
                f"got {len(stages)}: {stages}"
            )


# ---------- Behavioral tests on the node ----------

class TestLTXSmartImageResizeExecute:
    def test_output_dims_match_target(self):
        img = _img(1536, 2752)
        out = LTXSmartImageResize.execute(image=img, width=832, height=448)
        # ComfyUI returns NodeOutput; unwrap
        out_img = out[0]
        assert out_img.shape == (1, 448, 832, 3), f"unexpected shape {out_img.shape}"

    def test_passthrough_when_source_equals_target(self):
        img = _img(448, 832)
        out = LTXSmartImageResize.execute(image=img, width=832, height=448)
        out_img = out[0]
        assert out_img.shape == (1, 448, 832, 3)
        # Pass-through: no-stage path returns the input tensor (no reallocation)
        # Acceptable equivalence: pixel-exact match
        assert torch.equal(out_img, img)

    def test_preserves_batch_dim(self):
        img = _img(1536, 2752, b=4)
        out = LTXSmartImageResize.execute(image=img, width=832, height=448)
        out_img = out[0]
        assert out_img.shape == (4, 448, 832, 3)

    def test_output_dtype_float32(self):
        img = _img(1536, 2752)
        out = LTXSmartImageResize.execute(image=img, width=832, height=448)
        out_img = out[0]
        assert out_img.dtype == torch.float32

    def test_upscale_path(self):
        img = _img(288, 512)
        out = LTXSmartImageResize.execute(image=img, width=832, height=448)
        out_img = out[0]
        assert out_img.shape == (1, 448, 832, 3)


class TestCropToAspect:
    def test_no_op_when_aspect_matches(self):
        img = _img(448, 832)  # aspect 832:448 = 1.857
        out = _crop_to_aspect(img, 832, 448, "top")
        assert torch.equal(out, img)

    def test_crops_width_when_source_too_wide(self):
        # Source 1000x500 (aspect 2.0), target 832x448 (aspect 1.857)
        # Source is too wide -> crop width to 500 * 1.857 = 928
        img = _img(500, 1000)
        out = _crop_to_aspect(img, 832, 448, "center")
        assert out.shape[1] == 500  # height preserved
        # Cropped width within rounding tolerance of target aspect
        cropped_w = out.shape[2]
        assert abs(cropped_w / out.shape[1] - 832 / 448) < 0.01

    def test_crops_height_when_source_too_tall(self):
        # Source 800x800 (aspect 1.0), target 832x448 (aspect 1.857)
        # Source is too tall -> crop height
        img = _img(800, 800)
        out = _crop_to_aspect(img, 832, 448, "top")
        assert out.shape[2] == 800  # width preserved
        cropped_h = out.shape[1]
        assert abs(out.shape[2] / cropped_h - 832 / 448) < 0.01

    def test_smart_resize_with_keep_proportion_handles_aspect_mismatch(self):
        # Square 1024x1024 source -> 832x448 target should be cropped+resized,
        # not stretched.
        img = _img(1024, 1024)
        out = LTXSmartImageResize.execute(
            image=img, width=832, height=448,
            keep_proportion=True, crop_position="center",
        )[0]
        assert out.shape == (1, 448, 832, 3)


class TestSchema:
    def test_schema_has_required_inputs(self):
        # AST-walk approach since define_schema returns _Passthrough stubs
        # without ComfyUI loaded. Inspect the source file directly.
        import ast
        from pathlib import Path

        path = Path(__file__).resolve().parent.parent / "nodes.py"
        src = path.read_text()
        # Locate class body
        if "class LTXSmartImageResize" not in src:
            pytest.fail("LTXSmartImageResize class not found")
        cls_start = src.index("class LTXSmartImageResize")
        next_cls_offset = src.find("\nclass ", cls_start + len("class LTXSmartImageResize"))
        cls_body = src[cls_start:next_cls_offset if next_cls_offset != -1 else None]

        # Required input names appear in the class body
        for required in ("image", "width", "height"):
            assert f'"{required}"' in cls_body or f"'{required}'" in cls_body, (
                f"LTXSmartImageResize schema missing input {required!r}"
            )
