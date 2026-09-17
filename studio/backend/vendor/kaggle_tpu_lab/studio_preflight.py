"""MIT-licensed checks prepended to the unmodified upstream serving kernel.

No model/runtime implementation lives here. Fail before installing gigabytes
when Kaggle silently assigned a CPU, then verify JAX's actual TPU backend.
"""
import glob as _studio_glob
import json as _studio_json
import subprocess as _studio_subprocess
import time as _studio_time
import urllib.request as _studio_request


def studio_preflight(publish):
    publish("preflight")
    for attempt in range(3):
        devices = _studio_glob.glob("/dev/accel[0-9]*") + _studio_glob.glob("/dev/vfio/[0-9]*")
        if devices:
            break
        if attempt < 2:
            _studio_time.sleep(5)
    else:
        publish("failed", step="tpu-devices", retryable=True,
                tail="No TPU device nodes were attached to the Kaggle VM.")
        raise SystemExit(1)

    for attempt in range(3):
        try:
            for url in ("https://pypi.org/simple/vllm-tpu/", "https://download.pytorch.org/whl/cpu/"):
                request = _studio_request.Request(url, method="HEAD")
                with _studio_request.urlopen(request, timeout=10):
                    pass
            return
        except Exception:
            if attempt < 2:
                publish("network-retry", attempt=attempt + 1)
                _studio_time.sleep(5)
    publish("failed", step="network", retryable=True,
            tail="Package repositories are unreachable from the Kaggle VM.")
    raise SystemExit(1)


def studio_validate_tpu(python, publish):
    # Explicit backend selection prevents JAX's silent CPU fallback. The probe
    # runs in a child and releases libtpu before upstream starts vLLM.
    code = ("import json, jax\n"
            "devices = jax.devices('tpu')\n"
            "print(json.dumps({'count':len(devices), 'platforms':[d.platform for d in devices]}))")
    try:
        result = _studio_subprocess.run([python, "-c", code], capture_output=True, text=True, timeout=90)
        data = _studio_json.loads(result.stdout.strip().splitlines()[-1]) if result.returncode == 0 else {}
        valid = data.get("count") == 8 and data.get("platforms") == ["tpu"] * 8
    except (OSError, ValueError, IndexError, _studio_subprocess.TimeoutExpired):
        valid = False
    if not valid:
        publish("failed", step="tpu-devices", retryable=True,
                tail="The installed JAX runtime cannot initialize eight TPU devices.")
        raise SystemExit(1)
    publish("tpu-verified", devices=8)
