"""测试公用小工具：断言收集 + 统一的 main 运行器。"""
from __future__ import annotations

import os
import sys
import traceback

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

_FAILURES = []
_CHECKS = 0


def check(cond, msg: str):
    global _CHECKS
    _CHECKS += 1
    if not cond:
        _FAILURES.append(msg)
        print(f"  FAIL: {msg}")
    return bool(cond)


def approx(a, b, tol=1e-6) -> bool:
    return abs(float(a) - float(b)) <= tol


def check_close(a, b, tol=1e-6, msg=""):
    ok = approx(a, b, tol)
    return check(ok, f"{msg} 期望 {b}，实际 {a} (tol={tol})")


def run(module_globals, title: str) -> int:
    """运行当前模块里所有 test_* 函数。"""
    global _FAILURES, _CHECKS
    _FAILURES, _CHECKS = [], 0
    tests = [(name, fn) for name, fn in sorted(module_globals.items())
             if name.startswith("test_") and callable(fn)]
    print(f"=== {title} ({len(tests)} 项) ===")
    for name, fn in tests:
        print(f"[{name}]")
        try:
            fn()
        except Exception:
            _FAILURES.append(f"{name} 抛异常")
            traceback.print_exc()
    print(f"--- {title}: {_CHECKS - len(_FAILURES)}/{_CHECKS} 断言通过, "
          f"{len(_FAILURES)} 项失败 ---")
    for f in _FAILURES:
        print(f"  失败: {f}")
    return 1 if _FAILURES else 0
