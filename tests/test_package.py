from __future__ import annotations

import importlib.metadata
import importlib.resources
import subprocess
import sys

import quartet


def test_root_api_and_version_come_from_installed_metadata() -> None:
    assert quartet.__version__ == importlib.metadata.version("quartet") == "0.6.0.dev0"
    assert set(quartet.__all__) == {
        "DEFAULT_CONFIGS",
        "DEFAULT_WORKLOADS",
        "HardwareProfile",
        "KernelConfig",
        "Measurement",
        "Workload",
        "__version__",
        "read_jsonl",
        "write_jsonl",
    }
    assert all(hasattr(quartet, name) for name in quartet.__all__)


def test_css_and_typing_marker_are_packaged_resources() -> None:
    package = importlib.resources.files("quartet")
    tooling = importlib.resources.files("quartet.tooling")
    css = tooling.joinpath("report.css").read_text(encoding="utf-8")
    assert ".control-panel" in css
    assert package.joinpath("py.typed").is_file()


def test_root_import_keeps_replay_and_report_lazy() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys, quartet; "
                "assert 'quartet.replay' not in sys.modules; "
                "assert 'quartet.tooling.report' not in sys.modules; "
                "assert 'quartet.tooling.verification' not in sys.modules"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout == ""
