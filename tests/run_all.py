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
    ("RAW 读取层", "test_raw_io.py"),
    ("显示管线与统计", "test_display_stats.py"),
    ("缺陷检测算法", "test_defects.py"),
    ("导出层", "test_export.py"),
    ("旧算法回归 (bad pixel)", "test_algorithm.py"),
    ("旧算法回归 (bayer 坏点)", "test_bad_pixel_logic.py"),
    ("旧算法回归 (坏线)", "test_bad_line_logic.py"),
    ("无头 UI 集成", "test_ui_smoke.py"),
]


def main() -> int:
    env = dict(os.environ)
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    env["PYTHONPATH"] = ROOT + os.pathsep + env.get("PYTHONPATH", "")
    failed = []
    for title, script in SUITES:
        path = os.path.join(HERE, script)
        if not os.path.exists(path):
            print(f"!! 缺少 {script}")
            failed.append(script)
            continue
        print(f"\n{'=' * 70}\n>> {title}  ({script})\n{'=' * 70}")
        proc = subprocess.run([sys.executable, path], cwd=ROOT, env=env)
        if proc.returncode != 0:
            failed.append(script)
    print(f"\n{'=' * 70}")
    if failed:
        print(f"结果：{len(SUITES) - len(failed)}/{len(SUITES)} 套通过，失败：{failed}")
        return 1
    print(f"结果：全部 {len(SUITES)} 套测试通过 ✔")
    return 0


if __name__ == "__main__":
    sys.exit(main())
