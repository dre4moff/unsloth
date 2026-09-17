#!/usr/bin/env python3
"""
kaggle-tpu-lab launcher — serve Qwen3.8-27B on a free Kaggle TPU from your terminal.

    python launch.py serve                 # push the kernel and watch it come up
    python launch.py serve --reasoning-effort medium --mtp 3
    python launch.py status                # one-shot status + recent events
    python launch.py stop                  # kill the TPU session

Requires the Kaggle CLI, authenticated:  pip install kaggle   (see README).
Only the Python standard library is used here.
"""
import argparse
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import uuid
from pathlib import Path

import studio_bridge as bridge

HERE = Path(__file__).resolve().parent
KERNEL_SRC = HERE / "kernel" / "serve_qwen38.py"
STATE_FILE = Path(
    os.environ.get("UNSLOTH_KAGGLE_TPU_STATE_FILE", "").strip()
    or (Path.home() / ".kaggle-tpu-lab.json")
)

WEIGHTS_DATASET = "rahim3/qwen3-8-27b-bf16"
ENV_DATASET = "rahim3/qwen38-tpu-env-v5e8"   # XLA compile cache + cloudflared + manifest


def studio_interactive():
    """Lazy import: the standalone upstream-style CLI remains stdlib-only."""
    import studio_interactive as interactive
    return interactive

# Friendly one-liners for each phase the kernel publishes.
PHASE_TEXT = {
    "install":            "Building the Python runtime with uv (~30 s)...",
    "installed":          "Runtime ready.",
    "mtp-patch-applied":  "MTP state-rollback patch applied.",
    "mtp-patch-failed":   "MTP patch did not apply — speculative decoding disabled for safety.",
    "cache-restored":     None,  # rendered below (depends on config coverage)
    "cache-missing":      "No compile cache found — cold compile, add ~10 min.",
    "weights-mounted":    "Weights found mounted (no download needed).",
    "weights-download":   "Downloading weights from Hugging Face (~5 min)...",
    "weights-downloaded": "Weights downloaded.",
    "server-launch":      "Starting vLLM — loading 55 GB of weights, then TPU graph compile...",
    "tunnel-url":         None,
    "compiling":          None,  # rendered with elapsed time below
    "serving":            "Server is HEALTHY.",
    "benchmark":          None,
    "ready":              None,
    "heartbeat":          None,
    "failed":             None,
    "auto-shutdown":      "Keepalive window ended — kernel shut down cleanly.",
    "stopped":            "Server exited unexpectedly.",
}


def kaggle(*args, capture=True, timeout=45):
    cmd = [sys.executable, "-m", "kaggle", *args]
    if args[:2] in (("kernels", "logs"), ("kernels", "status")):
        timeout = min(timeout, 15)
    try:
        r = subprocess.run(cmd, capture_output=capture, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        if args[:2] != ("kernels", "logs") or "--follow" not in args:
            if args[:2] in (("kernels", "logs"), ("kernels", "status")):
                return subprocess.CompletedProcess(cmd, 124, stdout="", stderr="Kaggle status/log request timed out.")
            raise
        # Snapshot the live stream, which otherwise waits until the TPU exits.
        # run() has already killed and reaped this short-lived CLI process.
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        r = subprocess.CompletedProcess(cmd, 0, stdout=output, stderr="")
    return r


def say(msg):
    if bridge.JSON_MODE:
        return
    print(time.strftime("[%H:%M] "), msg, flush=True)


def check_auth():
    r = kaggle("kernels", "list", "-m", "--page-size", "1")
    if r.returncode != 0:
        sys.exit("Kaggle CLI is not working or not authenticated.\n"
                 "Install with `pip install kaggle`, then put your API token in place\n"
                 "(https://www.kaggle.com/settings -> Create New Token).\n\n"
                 f"Error was:\n{(r.stderr or r.stdout).strip()}")


def kaggle_username(cli_arg):
    if cli_arg:
        return cli_arg
    r = kaggle("config", "view")
    m = re.search(r"username[:=]\s*(\S+)", (r.stdout or "") + (r.stderr or ""))
    if m and m.group(1) not in ("None", "-"):
        return m.group(1).strip("'\"")
    sys.exit("Could not detect your Kaggle username — pass it with --user <name>.")


def prepare_kernel(cfg):
    src = KERNEL_SRC.read_text()
    src, count = re.subn(r"^CFG = None  # __LAUNCHER_CONFIG__.*$", f"CFG = {cfg!r}", src, count=1, flags=re.M)
    if count != 1:
        raise RuntimeError("The bundled upstream kernel config marker changed.")
    if bridge.JSON_MODE:
        checks = (HERE / "studio_preflight.py").read_text()
        for anchor, addition in (
            ("# ---------------- 1. runtime ----------------", "studio_preflight(publish)\n\n"),
            ('publish("installed", secs=int(time.time() - t), via=runtime)', "studio_validate_tpu(PY, publish)\n"),
        ):
            if src.count(anchor) != 1:
                raise RuntimeError("The bundled upstream kernel preflight marker changed.")
            src = src.replace(anchor, addition + anchor, 1)
        src = checks + "\nprint('STUDIO_RUN ' + " + repr(cfg["ntfy_topic"]) + ", flush=True)\n" + src
    return src


def cmd_serve(args):
    # Retry one failed allocation automatically, using the same private notebook.
    # Never relaunch a live/unknown session or retry credentials/quota failures.
    for attempt in range(2):
        failure = submit_serve(args, attempt)
        if not (bridge.JSON_MODE and attempt == 0 and isinstance(failure, dict) and failure.get("retryable") is True):
            if isinstance(failure, dict):
                render_event(failure)
            return
        state = load_state()
        if state.get("interactive"):
            # The interactive helper already cancelled the failed session. A
            # fresh request can therefore start immediately without polling the
            # unrelated committed-kernel status used by the upstream batch path.
            bridge.emit("PROVISIONING", "Kaggle did not provide a usable interactive TPU session. Retrying automatically (attempt 2 of 2).")
            continue
        kernel = state["kernel"]
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            value = bridge.status(kaggle, kernel)
            if value in ("ERROR", "COMPLETE", "CANCELACKNOWLEDGED", "CANCELED", "CANCELLED"):
                break
            if value == "UNKNOWN":
                render_event(failure)
                return
            time.sleep(5)
        else:
            render_event(failure)
            return
        bridge.emit("PROVISIONING", "Kaggle did not provide a usable TPU session. Retrying automatically (attempt 2 of 2).")


def submit_serve(args, attempt=0):
    bridge.emit("STARTING", "Validating the saved Kaggle token.")
    check_auth()
    user = kaggle_username(args.user)
    slug = args.slug
    topic = "ktl-" + uuid.uuid4().hex[:20]
    api_key = "sk-" + secrets.token_hex(16)

    cfg = {
        "ntfy_topic": topic,
        "api_key": api_key,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "mtp_tokens": args.mtp,
        "reasoning_effort_default": args.reasoning_effort,
        "keepalive_min": args.keepalive_min,
        "weights_dataset": args.weights_dataset,
    }
    if args.no_tools:
        cfg["tool_call_parser"] = ""
    if args.text_only:
        cfg["text_only"] = True
    if args.verbose:
        cfg["verbose"] = True
    if args.fast_start:
        cfg["fast_start"] = True

    src = prepare_kernel(cfg)

    # Submit the serving code with the allocation request, as upstream does.
    # A saved batch version starts autonomously when capacity becomes available;
    # an interactive bootstrap depends on this local process surviving the queue.
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "serve_qwen38.py").write_text(src)
        (td / "kernel-metadata.json").write_text(json.dumps({
            "id": f"{user}/{slug}",
            "title": slug,
            "code_file": "serve_qwen38.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": "true",
            "enable_gpu": "false",
            "enable_tpu": "true",
            "enable_internet": "true",
            "machine_shape": "TpuV5E8",
            "dataset_sources": [args.weights_dataset, ENV_DATASET],
            "competition_sources": [], "kernel_sources": [], "model_sources": [],
        }, indent=1))
        bridge.emit("PROVISIONING", "Submitting the private Kaggle TPU notebook.")
        say(f"Pushing kernel {user}/{slug} (TPU v5e-8)...")
        # Record ownership before the network request: closing Studio while
        # Kaggle accepts the push must not leave an untracked TPU session.
        bridge.save_state(STATE_FILE,
            {"kernel": f"{user}/{slug}", "topic": topic, "api_key": api_key, "started_at": int(time.time()),
             "run_marker": bridge.JSON_MODE, "attempt": attempt})
        r = kaggle("kernels", "push", "-p", str(td), "--accelerator", "TpuV5E8")
        out = (r.stdout or "") + (r.stderr or "")
        if "successfully pushed" not in out:
            sys.exit(f"Push failed:\n{out.strip()}")
        for line in out.splitlines():
            if "not valid dataset sources" in line:
                say(f"WARNING: {line.strip()} — the kernel will still run, "
                    "but may need to download weights / compile cold.")

    bridge.save_state(STATE_FILE,
        {"kernel": f"{user}/{slug}", "topic": topic, "api_key": api_key, "started_at": int(time.time()),
         "run_marker": bridge.JSON_MODE, "attempt": attempt})
    say("Pushed. Kaggle takes a few minutes to provision the TPU and attach the "
        "datasets; the endpoint is usually live ~22 min after the kernel starts.")
    say("Watching progress (Ctrl-C is safe — the server keeps running; "
        "`python launch.py status` re-attaches, `... stop` kills it).")
    return watch(f"{user}/{slug}", topic, run_marker=bridge.JSON_MODE, defer_retryable=attempt == 0)


def read_events(topic, since):
    try:
        with urllib.request.urlopen(
                f"https://ntfy.sh/{topic}/json?poll=1&since={since}", timeout=15) as r:
            body = r.read().decode()
    except Exception:
        return []
    events = []
    for line in body.splitlines():
        try:
            e = json.loads(line)
        except Exception:
            continue
        if not isinstance(e, dict) or e.get("event") != "message":
            continue
        try:
            payload = json.loads(e.get("message", "{}"))
            if isinstance(payload, dict) and isinstance(e.get("time"), (int, float)):
                events.append((e["time"], payload))
        except Exception:
            continue
    return events


def render_event(ev, logs=""):
    if bridge.JSON_MODE:
        bridge.render(ev, logs)
        return
    phase = ev.get("phase", "?")
    if phase == "compiling":
        say(f"Loading / compiling... {ev.get('elapsed_s', 0) // 60} min elapsed "
            "(typically ~20 min with the env dataset, ~35 min without)")
    elif phase == "cache-restored":
        if ev.get("covers_this_config", True):
            say("XLA compile cache restored for this exact config — fast start.")
        else:
            say("XLA compile cache restored, but not for this config — its graphs "
                "compile cold (add ~10 min).")
    elif phase == "tunnel-url":
        say(f"Endpoint URL reserved: {ev.get('endpoint')}  (not live yet — wait for the banner)")
    elif phase == "serving":
        say(f"Server is HEALTHY after {ev.get('startup_secs', 0) // 60} min.")
    elif phase == "benchmark":
        say(f"Quick benchmark: {ev.get('decode_tok_s', '?')} tok/s single-stream decode "
            f"(sanity: {ev.get('sanity', '')!r})")
    elif phase == "ready":
        print("\n" + "=" * 66)
        print("  YOUR ENDPOINT IS LIVE")
        print(f"  base URL : {ev['endpoint']}")
        print(f"  API key  : {ev['api_key']}")
        print(f"  model    : {ev['model']}   (context: {ev.get('max_model_len', '?')})")
        print("=" * 66)
        print("""
Try it:
  curl $BASE/chat/completions -H "Authorization: Bearer $KEY" \\
    -H "Content-Type: application/json" -d '{
      "model": "qwen3.8-27b",
      "messages": [{"role": "user", "content": "Hello!"}],
      "chat_template_kwargs": {"reasoning_effort": "low"}
    }'

See the README for hooking this into Claude Code, Codex CLI, opencode, etc.
""")
        say(f"The kernel keeps serving for up to {ev.get('keepalive_min', '?')} min. "
            "Ctrl-C here does NOT stop it; use `python launch.py stop`.")
    elif phase == "heartbeat":
        say(f"Still serving ({ev.get('up_min', '?')} min up) — {ev.get('endpoint', '')}")
    elif phase == "failed":
        say(f"FAILED at step {ev.get('step', '?')}.")
        if ev.get("tail"):
            print("--- last server output ---")
            print(ev["tail"])
        say("Full log: `python launch.py status` after the kernel exits, or the "
            "kernel page on kaggle.com.")
    else:
        text = PHASE_TEXT.get(phase)
        say(text if text else f"{phase} {json.dumps({k: v for k, v in ev.items() if k != 'phase'})}")


def watch(kernel, topic, *, run_marker=False, defer_retryable=False):
    since = int(time.time()) - 24 * 3600
    seen_events = set()
    last_status = None
    seen_boot = False
    unavailable_reads = 0
    queued_since = time.time()
    if STATE_FILE.exists():
        saved = load_state()
        if saved.get("topic") == topic:
            queued_since = saved.get("started_at", queued_since)
    try:
        while True:
            events = read_events(topic, since)
            # ntfy cannot report a network failure inside Kaggle. The official
            # Kaggle logs remain reachable from the Mac in that situation.
            log_events, log_text = bridge.kernel_logs(kaggle, kernel, expected_run=topic if run_marker else None) if bridge.JSON_MODE else ([], "")
            events += [(since, ev) for ev in log_events]
            for ts, ev in events:
                identity = json.dumps(ev, sort_keys=True)
                if identity in seen_events:
                    continue
                seen_events.add(identity)
                since = max(since, ts)
                seen_boot = True
                if ev.get("phase") == "failed" and ev.get("retryable") is True and defer_retryable:
                    bridge.emit("PROVISIONING", "Kaggle session failed its initial checks; preparing an automatic retry.")
                    return ev
                render_event(ev, log_text)
                if ev.get("phase") in ("failed", "auto-shutdown", "stopped"):
                    # render_event already reports the failure's exact step and
                    # tail. A second diagnosis of the full startup log used to
                    # overwrite server/compile errors with an install error.
                    return
            # Keep the cursor on the last received event; do not skip delayed events.
            r = kaggle("kernels", "status", kernel)
            out = (r.stdout or "") + (r.stderr or "")
            m = re.search(r'"KernelWorkerStatus\.(\w+)"', out)
            status = m.group(1) if m else "UNKNOWN"
            if status == "UNKNOWN":
                unavailable_reads += 1
                if bridge.access_denied(out):
                    bridge.emit("ERROR", bridge.diagnostic(out))
                    return
                if unavailable_reads >= 5:
                    bridge.emit("DISCONNECTED", "Kaggle status is temporarily unreachable. The notebook may still be queued or running; use Reconnect to resume monitoring without starting another TPU.")
                    return
                bridge.emit("MONITOR_RETRY", "Kaggle status is temporarily unreachable; retrying automatically. The existing notebook has not been stopped.")
                time.sleep(15)
                continue
            unavailable_reads = 0
            if status == "QUEUED":
                minutes = max(0, int(time.time() - queued_since)) // 60
                bridge.emit("PROVISIONING", f"Waiting for Kaggle TPU v5e-8: {minutes} minutes in queue. Model installation has not started; Kaggle controls allocation. Stop cancels this request.")
            if status != last_status:
                if status == "QUEUED":
                    say("Kaggle: queued — waiting for a TPU v5e-8 slot...")
                elif status == "RUNNING" and not seen_boot:
                    bridge.emit("PROVISIONING", "Kaggle is attaching datasets and preparing the TPU.")
                    say("Kaggle: provisioning the VM and attaching datasets "
                        "(a few minutes)...")
                elif status in ("ERROR", "CANCELACKNOWLEDGED", "COMPLETE"):
                    if status == "ERROR" and not seen_boot and not log_text:
                        bridge.emit("ERROR", "Kaggle ended the request before the model started and returned no execution logs. Check TPU v5e-8 availability in the Kaggle editor, then retry Start.", session_ended=True)
                    else:
                        bridge.terminal_status(status, log_text)
                    say(f"Kernel finished with status {status}.")
                    return
                last_status = status
            time.sleep(30)
    except KeyboardInterrupt:
        say("Detached. The kernel keeps running — `python launch.py status` to "
            "re-attach, `python launch.py stop` to kill it.")


def cmd_build_env(args):
    """Maintainer flow. When the kernel finishes:
        kaggle kernels output <user>/<slug> -p bundle_out
        then create/version the dataset from bundle_out/bundle (see README)."""
    bridge.emit("STARTING", "Validating the saved Kaggle token.")
    check_auth()
    user = kaggle_username(args.user)
    topic = "ktl-" + uuid.uuid4().hex[:20]
    cfg = {"build_bundle": True, "ntfy_topic": topic, "weights_dataset": args.weights_dataset}
    src = KERNEL_SRC.read_text()
    src, n = re.subn(r"^CFG = None  # __LAUNCHER_CONFIG__.*$",
                     f"CFG = {cfg!r}", src, count=1, flags=re.M)
    if n != 1:
        sys.exit("kernel/serve_qwen38.py is missing the __LAUNCHER_CONFIG__ line")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "build_env.py").write_text(src)
        (td / "kernel-metadata.json").write_text(json.dumps({
            "id": f"{user}/{args.slug}", "title": args.slug, "code_file": "build_env.py",
            "language": "python", "kernel_type": "script", "is_private": "true",
            "enable_gpu": "false", "enable_tpu": "true", "enable_internet": "true",
            "dataset_sources": [args.weights_dataset],
            "competition_sources": [], "kernel_sources": [], "model_sources": [],
        }, indent=1))
        r = kaggle("kernels", "push", "-p", str(td))
        out = (r.stdout or "") + (r.stderr or "")
        if "successfully pushed" not in out:
            sys.exit(f"Push failed:\n{out.strip()}")
    bridge.save_state(STATE_FILE, {"kernel": f"{user}/{args.slug}", "topic": topic,
                                      "api_key": ""})
    say(f"Pushed {user}/{args.slug}. It serves each config once (~1.5 h total) and "
        "leaves xla_cache.tar / cloudflared / manifest.json in its output.")
    watch(f"{user}/{args.slug}", topic)


def load_state():
    if not STATE_FILE.exists():
        sys.exit("No launch state found — run `python launch.py serve` first.")
    return json.loads(STATE_FILE.read_text())


def interactive_events(st):
    events = [ev for _, ev in read_events(st["topic"], int(time.time()) - 24 * 3600)]
    ready = next((ev for ev in reversed(events) if ev.get("phase") == "ready"), None)
    return events, ready


def interactive_status(st, *, follow=False):
    interactive = studio_interactive()
    operation_name = str(st.get("operation_name") or "")
    try:
        session = interactive.restore_session(operation_name) if operation_name else None
    except interactive.InteractiveSessionClosed:
        bridge.emit("ERROR", "Kaggle closed the previous interactive session. Start creates an automatic server run.", session_ended=True)
        return
    events, ready = interactive_events(st)
    if ready and bridge.health(ready):
        render_event(ready)
    elif session and interactive.session_alive(session):
        bridge.emit("PROVISIONING", "Reconnected to the interactive Kaggle TPU session.")
        if events:
            render_event(events[-1])
    elif operation_name and not st.get("stopped"):
        elapsed = max(0, int(time.time()) - int(st.get("started_at") or time.time()))
        minutes = elapsed // 60
        bridge.emit("PROVISIONING", f"The previous interactive allocation is still pending: {minutes} minutes elapsed. Stop it, then Start to use automatic queued execution.")
    else:
        bridge.emit("DISCONNECTED", "The interactive Kaggle session is unavailable. Retry Reconnect or Start a new session.")
        return

    if not follow:
        return
    seen = {json.dumps(ev, sort_keys=True) for ev in events}
    while True:
        time.sleep(10)
        current, _ = interactive_events(st)
        for event in current:
            identity = json.dumps(event, sort_keys=True)
            if identity in seen:
                continue
            seen.add(identity)
            render_event(event)
            if event.get("phase") in ("failed", "auto-shutdown", "stopped"):
                return
        try:
            session = interactive.restore_session(operation_name) if operation_name else None
        except interactive.InteractiveSessionClosed:
            bridge.emit("ERROR", "Kaggle closed the previous interactive session. Start creates an automatic server run.", session_ended=True)
            return
        if session is None:
            continue
        if not interactive.session_alive(session):
            bridge.emit("DISCONNECTED", "The interactive Kaggle session is no longer reachable. Retry Reconnect before starting another TPU.")
            return


def cmd_status(args):
    if not STATE_FILE.exists():
        bridge.emit("STOPPED", "No Kaggle session has been started for this connection.")
        if bridge.JSON_MODE:
            return
    st = load_state()
    if st.get("stopped"):
        bridge.emit("STOPPED", "The Kaggle TPU session is stopped.")
        return
    if bridge.JSON_MODE and st.get("interactive"):
        interactive_status(st, follow=bool(args.follow))
        return
    value = bridge.status(kaggle, st["kernel"])
    if value == "UNKNOWN":
        return
    log_events, log_text = bridge.kernel_logs(kaggle, st["kernel"], expected_run=st["topic"] if st.get("run_marker") else None)
    if bridge.terminal_status(value, log_text):
        return
    events = [ev for _, ev in read_events(st["topic"], int(time.time()) - 24 * 3600)]
    events = log_events + events
    ready = next((ev for ev in reversed(events) if ev.get("phase") == "ready"), None)
    if ready and value == "RUNNING":
        if bridge.health(ready):
            render_event(ready)
        else:
            bridge.emit("DISCONNECTED", "The Kaggle session is running but its tunnel is unavailable. Use Reconnect; no new TPU was started.")
    elif value in ("RUNNING", "QUEUED"):
        if value == "QUEUED":
            minutes = max(0, int(time.time()) - int(st.get("started_at") or time.time())) // 60
            bridge.emit("PROVISIONING", f"Kaggle TPU v5e-8 is queued: {minutes} minutes elapsed. The saved server will start automatically when Kaggle assigns capacity.")
        else:
            bridge.emit("PROVISIONING", "Reconnected to the running Kaggle session.")
        if events:
            render_event(events[-1], log_text)
    else:
        bridge.emit("DISCONNECTED", "Kaggle session status is unavailable; retry Reconnect.")
    if args.follow and value in ("RUNNING", "QUEUED"):
        watch(st["kernel"], st["topic"], run_marker=bool(st.get("run_marker")))


def cmd_stop(args):
    if not STATE_FILE.exists():
        bridge.emit("STOPPED", "No Kaggle session has been started.")
        return
    st = load_state()
    if st.get("stopped"):
        bridge.emit("STOPPED", "The Kaggle TPU session is already stopped.")
        return
    if bridge.JSON_MODE and st.get("interactive"):
        session_id = int(st.get("kernel_session_id") or 0)
        if session_id:
            studio_interactive().cancel_session(session_id)
        bridge.save_state(STATE_FILE, {**st, "stopped": True})
        bridge.emit("STOPPED", "The interactive Kaggle TPU session is stopped; its private bootstrap notebook was preserved for the next Start.")
        return
    say(f"Deleting kernel {st['kernel']} (terminates the TPU session)...")
    p = subprocess.run([sys.executable, "-m", "kaggle", "kernels", "delete",
                        st["kernel"], "--yes"], capture_output=True, text=True, timeout=45)
    if p.returncode:
        raise RuntimeError((p.stdout + p.stderr).strip())
    bridge.save_state(STATE_FILE, {**st, "stopped": True})
    bridge.emit("STOPPED", "The Kaggle TPU session is stopped.")
    say((p.stdout + p.stderr).strip() or "done")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="push the serving kernel and watch it come up")
    s.add_argument("--user", help="Kaggle username (auto-detected if possible)")
    s.add_argument("--slug", default="qwen38-tpu-serve", help="kernel name")
    s.add_argument("--max-model-len", type=int, default=262144,
                   help="context length (default: native 262k; use 131072 with "
                        "--max-num-seqs 16 for max multi-stream throughput)")
    s.add_argument("--max-num-seqs", type=int, default=4)
    s.add_argument("--mtp", type=int, default=3,
                   help="MTP speculative tokens (0 disables). +34%% decode in our A/B test; made "
                        "lossless by the bundled GDN state-rollback patch "
                        "(verified 12/12 greedy exact-match)")
    s.add_argument("--reasoning-effort", default="xhigh",
                   choices=["xhigh", "medium", "low"],
                   help="server-side default; clients can still override per request")
    s.add_argument("--keepalive-min", type=int, default=480,
                   help="auto-shutdown after this many minutes of serving")
    s.add_argument("--weights-dataset", default=WEIGHTS_DATASET)
    s.add_argument("--no-tools", action="store_true",
                   help="disable tool-calling support")
    s.add_argument("--text-only", action="store_true",
                   help="skip the vision tower: ~8 min faster start, image inputs "
                        "then error out")
    s.add_argument("--verbose", action="store_true",
                   help="show every vLLM log line in the kernel log")
    s.add_argument("--fast-start", action="store_true",
                   help="skip TPU graph precompile: endpoint live in ~4 min (with the env "
                        "dataset), common request shapes are warmed right after; an "
                        "unusual request shape stalls ~1 min the first time")
    s.set_defaults(fn=cmd_serve)

    s = sub.add_parser("build-env", help="(maintainers) push a kernel that builds the "
                       "env dataset: venv + XLA cache + cloudflared")
    s.add_argument("--user", help="Kaggle username (auto-detected if possible)")
    s.add_argument("--slug", default="qwen38-env-bundle")
    s.add_argument("--weights-dataset", default=WEIGHTS_DATASET)
    s.set_defaults(fn=cmd_build_env)

    s = sub.add_parser("status", help="show current kernel status + recent events")
    s.add_argument("--follow", "-f", action="store_true", help="keep watching")
    s.set_defaults(fn=cmd_status)

    s = sub.add_parser("stop", help="terminate the TPU session")
    s.set_defaults(fn=cmd_stop)

    for parser in (s for s in sub.choices.values()):
        parser.add_argument("--json", action="store_true", help="Machine-readable Studio lifecycle events")
    args = ap.parse_args()
    bridge.JSON_MODE = args.json
    try:
        args.fn(args)
    except (Exception, SystemExit) as exc:
        if not bridge.JSON_MODE:
            raise
        bridge.emit("ERROR", bridge.diagnostic(str(exc)))
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
