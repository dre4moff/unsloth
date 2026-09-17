"""MIT-licensed Studio launcher adapter; serving remains in upstream's kernel.

Structured progress and Kaggle log fallback work even when ntfy is unreachable.
Only fixed diagnostic messages cross the launcher/UI boundary.
"""
import json
import os
import re
import tempfile
import urllib.request

JSON_MODE = False


def emit(state, message, **fields):
    if JSON_MODE:
        print(json.dumps({"status": state, "message": message, **fields}), flush=True)


def diagnostic(text, step=None):
    # Upstream documents GCE metadata DNS warnings as harmless on Kaggle.
    # They must not be confused with a failure reaching package/tunnel hosts.
    text = "\n".join(line for line in str(text).lower().splitlines()
                     if "metadata.google.internal" not in line and "unable to poll the tpu gce metadata" not in line)
    if step == "network" or any(s in text for s in ("temporary failure in name resolution", "name or service not known", "network is unreachable")):
        return "Kaggle cannot reach the Internet (DNS/network failure). Check Internet access in the Kaggle notebook and account verification, then retry Start."
    if any(s in text for s in ("unauthorized", "not authenticated", "invalid token", "expired token")) or re.search(r"\b(?:http(?: error)?|status(?: code)?)[: =]+401\b", text):
        return "Kaggle rejected the API token. Generate a new token in Kaggle Settings > API and update this connection."
    if "quota" in text or re.search(r"\b(?:http(?: error)?|status(?: code)?)[: =]+429\b", text):
        return "Kaggle TPU quota or capacity is unavailable. Check the account quota and retry when a TPU is available."
    if "permission" in text or "phone verif" in text or re.search(r"\b(?:http(?: error)?|status(?: code)?)[: =]+403\b", text):
        return "Kaggle denied access. Check account verification and permission to use TPUs and Internet."
    if "no module named" in text and "kaggle" in text:
        return "The Kaggle CLI is missing from the Studio runtime. Restart Studio to repair its dependencies."
    if step == "tpu-devices" or "insufficient devices for 2d mesh" in text or "unexpected worker hostname" in text:
        return "Kaggle did not expose the eight TPU devices required by this model. Studio requested TPU v5e-8; retry Start to provision a fresh TPU session."
    if "out of memory" in text or "resource_exhausted" in text:
        return "The Kaggle TPU ran out of memory while loading or compiling the model. Reduce the context length or parallel sequences, then retry Start."
    if "timed out while allocating the interactive tpu session" in text:
        return "Kaggle did not finish allocating the interactive TPU session within Studio's wait window. The request may still be pending on Kaggle; use Reconnect before starting another session."
    if step == "health-timeout":
        return "The Kaggle model did not become healthy within the TPU compilation deadline. Retry Start to create a fresh session."
    if step == "server":
        return "The Kaggle model server stopped while loading weights or compiling TPU graphs. The Python runtime was installed; retry Start to create a fresh session."
    if step in ("install", "runtime-check") or "no matching distribution" in text:
        return "The Kaggle Python runtime failed to install or validate. Retry Start to create a fresh session."
    if step == "tunnel" or text.strip() == "tunnel":
        return "The Cloudflare tunnel is unavailable. The Kaggle session may still be running; use Reconnect."
    return "The Kaggle launcher failed. Check the notebook logs and account status, then retry."


def save_state(path, state):
    if JSON_MODE:
        # READY is recoverable from the kernel events; Studio stores the
        # inference key encrypted. No plaintext key is needed for reattachment.
        state = {key: value for key, value in state.items() if key != "api_key"}
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(state, f)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def failure_context(logs, step):
    marker = "PHASE server-launch" if step in ("server", "health-timeout") else "PHASE install"
    return logs.rsplit(marker, 1)[-1][-20000:]


def render(ev, logs=""):
    phase = ev.get("phase")
    if phase == "ready":
        if not ev.get("endpoint"):
            emit("DISCONNECTED", diagnostic("tunnel"))
            return
        emit("READY", "Kaggle model loaded; verifying the endpoint.",
             base_url=ev["endpoint"], api_key=ev.get("api_key"),
             model=ev.get("model"), context_length=ev.get("max_model_len"))
    elif phase in ("failed", "stopped", "tunnel-failed"):
        step = ev.get("step") or ("tunnel" if phase == "tunnel-failed" else "server")
        emit("ERROR", diagnostic(failure_context(logs, step) + "\n" + str(ev.get("tail", "")), step=step))
    elif phase == "auto-shutdown":
        emit("STOPPED", "The Kaggle keepalive window ended.")
    elif phase == "heartbeat":
        # A heartbeat alone is not proof that the public tunnel is reachable.
        emit("HEARTBEAT", "The Kaggle model is serving.")
    else:
        messages = {
            "preflight": "Checking that Kaggle assigned a TPU and Internet access.",
            "network-retry": "Waiting for Kaggle package downloads to become reachable; retrying automatically.",
            "tpu-verified": "All eight TPU devices verified; preparing the model.",
            "install": "Installing the Kaggle model runtime.",
            "installed": "Kaggle runtime installed; preparing model weights.",
            "weights-mounted": "Model weights attached on Kaggle.",
            "weights-download": "Kaggle is downloading the model weights.",
            "cache-restored": "Kaggle compilation cache restored.",
            "server-launch": "Loading the model and compiling TPU graphs.",
            "compiling": "Compiling TPU graphs; this can take 20–35 minutes.",
            "tunnel-url": "Tunnel reserved; waiting for the model to finish loading.",
            "serving": "Model loaded; preparing the endpoint.",
            "network-unavailable": "Kaggle reports a DNS/network failure. Check Internet access in the notebook; Stop ends this attempt.",
        }
        if phase in messages:
            message = messages[phase]
            if phase == "compiling" and isinstance(ev.get("elapsed_s"), (int, float)):
                message = f"Compiling TPU graphs: {max(0, int(ev['elapsed_s'])) // 60} minutes elapsed; typically 20–35 minutes."
            emit("LOADING", message)


def kernel_logs(kaggle, kernel, expected_run=None):
    result = kaggle("kernels", "logs", kernel)
    if result.returncode:
        return [], ""
    raw = result.stdout or ""
    if not raw.strip():
        # The one-shot command returns only *persisted* logs on Kaggle 2.2.
        # A running session needs its live stream to diagnose ntfy/network loss.
        result = kaggle("kernels", "logs", kernel, "--follow", timeout=5)
        raw = result.stdout or ""
    try:
        entries = json.loads(raw)
        raw = "".join(e.get("data", "") for e in entries if isinstance(e, dict))
    except (ValueError, TypeError):
        pass
    if expected_run:
        marker = "STUDIO_RUN " + expected_run
        # Kaggle can return the preceding version's persisted logs while the
        # new version provisions. Never consume its old failed/READY events.
        if marker not in raw:
            result = kaggle("kernels", "logs", kernel, "--follow", timeout=5)
            raw = result.stdout or ""
        if marker not in raw:
            return [], ""
        raw = raw.rsplit(marker, 1)[-1]
    events = []
    for line in raw.splitlines():
        match = re.search(r"PHASE ([\w-]+)\s*(.*)$", line)
        if match:
            try:
                extra = json.loads(match[2]) if match[2].strip() else {}
                events.append({**extra, "phase": match[1]})
            except (ValueError, TypeError):
                continue
    if "temporary failure in name resolution" in raw.lower() and not any(e.get("phase") == "failed" for e in events):
        events.append({"phase": "network-unavailable"})
    return events, raw


def access_denied(text):
    text = str(text).lower()
    return any(marker in text for marker in ("unauthorized", "not authenticated", "invalid token", "expired token", "permission", "phone verif")) or bool(re.search(r"\b(?:http(?: error)?|status(?: code)?)[: =]+40[13]\b", text))


def status(kaggle, kernel):
    result = kaggle("kernels", "status", kernel)
    raw = (result.stdout or "") + (result.stderr or "")
    if result.returncode:
        if access_denied(raw):
            emit("ERROR", diagnostic(raw))
        else:
            emit("DISCONNECTED", "Kaggle status is temporarily unreachable. Use Reconnect; the existing notebook may still be queued or running.")
        return "UNKNOWN"
    match = re.search(r"KernelWorkerStatus\.(\w+)", raw)
    return match[1] if match else "UNKNOWN"


def terminal_status(value, logs=""):
    if value == "ERROR":
        # Installation is mentioned in every successful startup. Diagnose the
        # final failure, never a keyword from the entire historical log.
        failures = re.findall(r"PHASE failed\s+(\{[^\n]*\})", logs)
        failure = {}
        if failures:
            try:
                failure = json.loads(failures[-1])
            except ValueError:
                pass
        step = failure.get("step")
        emit("ERROR", diagnostic(failure_context(logs, step) + "\n" + str(failure.get("tail", "")), step=step), session_ended=True)
        return True
    if value in ("COMPLETE", "CANCELACKNOWLEDGED", "CANCELED", "CANCELLED"):
        emit("STOPPED", "The Kaggle session ended.")
        return True
    return False


def health(ready):
    """Never announce old READY events as live without probing the tunnel."""
    url = ready.get("endpoint", "")
    if not re.fullmatch(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com/v1", url):
        return False
    req = urllib.request.Request(url + "/models", headers={"Authorization": "Bearer " + ready.get("api_key", "")})
    try:
        with urllib.request.urlopen(req, timeout=12) as response:
            models = json.loads(response.read(1024 * 1024)).get("data", [])
            return any(m.get("id") == ready.get("model") for m in models)
    except Exception:
        return False
