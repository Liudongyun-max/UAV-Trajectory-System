from __future__ import annotations


def scaled_intrinsics(native_fx: float, native_fy: float, native_cx: float, native_cy: float, scaling_ratio: float) -> dict[str, float]:
    return {
        "fx": native_fx * scaling_ratio,
        "fy": native_fy * scaling_ratio,
        "cx": native_cx * scaling_ratio,
        "cy": native_cy * scaling_ratio,
    }


def validate_4k_to_2k_profile() -> dict:
    native_width = 3840
    native_height = 2160
    output_width = 2560
    output_height = 1440
    ratio_w = output_width / native_width
    ratio_h = output_height / native_height
    return {
        "render_native_4k": True,
        "output_2k": True,
        "native_resolution": [native_width, native_height],
        "output_resolution": [output_width, output_height],
        "ratio_w": ratio_w,
        "ratio_h": ratio_h,
        "ratio_consistent": abs(ratio_w - ratio_h) < 1e-6,
        "method": "area",
    }

