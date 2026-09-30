"""一键运行全部测试：python tests/run_all.py

也可以用 pytest 跑（若环境里装了 pytest）：pytest tests/
"""
from __future__ import annotations

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SUITES = [
    ("加速层", "test_accel.py"),
    ("RAW 读取层", "test_raw_io.py"),
    ("显示管线与统计", "test_display_stats.py"),
    ("缺陷检测算法", "test_defects.py"),
    ("批量分析 / 黑电平", "test_batch.py"),
    ("导出层", "test_export.py"),
    ("旧算法回归 (bad pixel)", "test_algorithm.py"),
    ("旧算法回归 (bayer 坏点)", "test_bad_pixel_logic.py"),
    ("旧算法回归 (坏线)", "test_bad_line_logic.py"),
    ("无头 UI 集成", "test_ui_smoke.py"),
]

# 这些套件在两种后端下都要通过（有/没有 OpenCV 的代码路径不同）
DUAL_BACKEND_SUITES = ["test_accel.py", "test_display_stats.py", "test_defects.py",
                       "test_batch.py", "test_ui_smoke.py"]


def _run_suite(title, script, env, extra_env=None):
    path = os.path.join(HERE, script)
    if not os.path.exists(path):
        print(f"!! 缺少 {script}")
        return False
    print(f"\n{'=' * 70}\n>> {title}  ({script})\n{'=' * 70}")
    run_env = dict(env)
    run_env.update(extra_env or {})
    proc = subprocess.run([sys.executable, path], cwd=ROOT, env=run_env)
    return proc.returncode == 0


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="运行全部测试")
    ap.add_argument("--no-cv2", action="store_true",
                    help="强制禁用 OpenCV，只跑 numpy 回退路径")
    ap.add_argument("--single-backend", action="store_true",
                    help="只跑当前后端，不做双后端复跑")
    args = ap.parse_args()

    env = dict(os.environ)
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    env["PYTHONPATH"] = ROOT + os.pathsep + env.get("PYTHONPATH", "")
    if args.no_cv2:
        env["RAWV2_NO_CV2"] = "1"

    failed, total = [], 0
    for title, script in SUITES:
        total += 1
        if not _run_suite(title, script, env):
            failed.append(script)

    if not args.single_backend and not args.no_cv2:
        print(f"\n{'#' * 70}\n# 复跑：强制 numpy 回退（RAWV2_NO_CV2=1）"
              f"—— 没装 OpenCV 的用户走的就是这些路径\n{'#' * 70}")
        for title, script in SUITES:
            if script not in DUAL_BACKEND_SUITES:
                continue
            total += 1
            if not _run_suite(f"{title} [numpy 回退]", script, env, {"RAWV2_NO_CV2": "1"}):
                failed.append(f"{script} (numpy 回退)")

    print(f"\n{'=' * 70}")
    if failed:
        print(f"结果：{total - len(failed)}/{total} 套通过，失败：{failed}")
        return 1
    print(f"结果：全部 {total} 套测试通过 ✔")
    return 0


if __name__ == "__main__":
    sys.exit(main())
