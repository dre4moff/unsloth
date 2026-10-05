#!/usr/bin/env python3
"""phone-push.py - copy files/directories into Backburner's Documents over the USB link, without devicectl.

  scripts/phone-push.py PHONE_IP LOCAL_PATH [REMOTE_NAME]

Serves exactly the files being pushed (nothing else from LOCAL_PATH's parent) over HTTP on the Mac's address facing the
phone (link-local only, a random port, for the duration of the copy) and has Backburner download every file with its `fetch` command (ANE command port 50061).
A directory (e.g. a .mlmodelc) is copied file by file. REMOTE_NAME defaults to LOCAL_PATH's basename.
Why: `devicectl device copy` needs the developer disk image mounted, which fails on iOS 27.2 with Xcode 27 at times.
"""
import http.server, json, os, socket, sys, threading, time, urllib.parse

phone, local = sys.argv[1], os.path.abspath(sys.argv[2])
remote = sys.argv[3] if len(sys.argv) > 3 else os.path.basename(local)
root = os.path.dirname(local)

# the Mac's address on the interface that reaches the phone
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.connect((phone, 50061)); me = s.getsockname()[0]; s.close()

files = [local] if os.path.isfile(local) else [os.path.join(d, f) for d, _, fs in os.walk(local) for f in fs]
# serve exactly these files and nothing else in `root` (it can be a big folder, e.g. ~/Models or your home)
allowed = {"/" + urllib.parse.quote(os.path.relpath(p, root)) for p in files}

class H(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **k): super().__init__(*a, directory=root, **k)
    def log_message(self, *a): pass
    def list_directory(self, path): self.send_error(404); return None
    def send_head(self):
        if self.path not in allowed:
            self.send_error(404); return None
        return super().send_head()
srv = http.server.ThreadingHTTPServer((me, 0), H)
port = srv.server_address[1]
threading.Thread(target=srv.serve_forever, daemon=True).start()

c = socket.create_connection((phone, 50061), timeout=900); f = c.makefile("rw")
t0, total = time.time(), 0
for p in sorted(files):
    rel = os.path.relpath(p, root)
    dst = remote + rel[len(os.path.basename(local)):]
    url = f"http://{me}:{port}/" + urllib.parse.quote(rel)
    f.write(f"fetch {url} {dst}\n"); f.flush()
    r = json.loads(f.readline())
    if r.get("error") or r.get("bytes") != os.path.getsize(p):
        sys.exit(f"FAILED {rel}: {r}")
    total += r["bytes"]
dt = time.time() - t0
print(f"pushed {len(files)} files, {total/1e6:.0f} MB to Documents/{remote} in {dt:.1f} s ({total/1e6/max(dt,1e-9):.0f} MB/s)")
srv.shutdown()
