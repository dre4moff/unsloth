"""A minimal desktop wheel must not reach setup without its required zoo."""

import os
from pathlib import Path
import re
import subprocess

import pytest


@pytest.mark.parametrize(
    "package,wheel,no_torch,installs_zoo",
    [
        ("unsloth", "/bundle with spaces/backend.whl", "false", True),
        ("unsloth", "/bundle/backend.whl", "true", True),
        ("unsloth", "", "false", False),
        ("custom-package", "/bundle/backend.whl", "false", False),
    ],
)
def test_bundled_core_handoff_installs_zoo_without_changing_other_install_modes(
    package, wheel, no_torch, installs_zoo
):
    source = (Path(__file__).resolve().parents[3] / "install.sh").read_text()
    helper = re.search(r"^_install_bundled_backend_zoo\(\) \{\n.*?^\}", source, re.M | re.S)
    assert helper is not None
    # Execute the actual shell helper; only the external package installer is
    # replaced. Empty CLI-only wheel metadata reproduces the missing-zoo case.
    script = "set -eu\nrun_install_cmd_retry() { printf '%s\\0' \"$@\"; }\n"
    script += "_build_unsloth_torch_overrides() { :; }\n"
    script += helper.group(0) + "\n_install_bundled_backend_zoo\n"
    result = subprocess.run(
        ["sh", "-c", script], check=True, capture_output=True,
        env={**os.environ, "PACKAGE_NAME": package,
             "UNSLOTH_DESKTOP_BACKEND_WHEEL": wheel, "SKIP_TORCH": no_torch,
             "_VENV_PY": "/venv with spaces/bin/python", "_UNSLOTH_TORCH_OVERRIDES": ""},
    )
    args = result.stdout.decode().strip("\0").split("\0") if result.stdout else []
    if installs_zoo:
        assert "unsloth-zoo>=2026.10.5" in args
        assert args[args.index("--python") + 1] == "/venv with spaces/bin/python"
        assert ("--no-deps" in args) == (no_torch == "true")
    else:
        assert args == []
