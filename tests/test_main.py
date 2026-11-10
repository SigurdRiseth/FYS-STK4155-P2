"""Smoke test of the package entry point.

LLM-assisted
------------
Tool: Claude (claude-opus-5-5, Claude Code CLI, November 2026)
Role: Wrote the test; checked by the author.
"""

import pytest

from fys_stk4155_p2 import main


def test_main_prints_greeting(capsys: pytest.CaptureFixture[str]) -> None:
    main()
    captured = capsys.readouterr()
    assert captured.out == "Hello from fys-stk4155-p2!\n"
