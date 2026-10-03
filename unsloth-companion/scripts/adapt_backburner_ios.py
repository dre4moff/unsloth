#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate the lifecycle adapter; never edit the vendored inference algorithms.

Only socket binding/cancellation, owned-thread joins, storage isolation, and
activation-time environment defaults differ from StayLameBro's pinned sources.
"""
from pathlib import Path
import hashlib
import json
import shutil
import sys

repo = Path(__file__).resolve().parents[2]
vendor = repo / "studio/backend/vendor/backburner"
out = Path(sys.argv[1]).resolve()
engine = Path(sys.argv[2]).resolve()
meta = json.loads((vendor / "UPSTREAM.json").read_text())
for rel, digest in meta["files"].items():
    if hashlib.sha256((vendor / rel).read_bytes()).hexdigest() != digest:
        raise SystemExit(f"Vendored upstream source changed: {rel}")
shutil.copytree(vendor, out, dirs_exist_ok=True)
for rel in ["tools/split-prefill/ane-ffn.h", "src/llama-ext.h"]:
    dst = out / "llama.cpp" / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(engine / rel, dst)

# All three services are compiled inside one private framework. Mac and phone
# keep the original little-endian wire messages and all compute kernels.
bridge = out / "ios/Backburner/Sidecar/RPCBridge.mm"
text = bridge.read_text()
text = text.replace('#include "../../../llama.cpp/tools/split-prefill/tail-server.h"',
                    '#include "../../../Lifecycle.h"\n#include "../../../llama.cpp/tools/split-prefill/tail-server.h"')
text = text.replace('__attribute__((constructor)) static void sidecar_env_defaults()',
                    'static void sidecar_env_defaults()')
text = text.replace('@"Documents"', '@"Documents/Backburner"')
text = text.replace('@"Documents/anekv"', '@"Documents/Backburner/anekv"')
text = text.replace('if (strncmp(p->ifa_name, "lo", 2) == 0) {',
                    'if (strncmp(p->ifa_name, "lo", 2) == 0 || strcmp(p->ifa_name, "en0") == 0 || strncmp(p->ifa_name, "awdl", 4) == 0 || strncmp(p->ifa_name, "llw", 3) == 0 || strncmp(p->ifa_name, "pdp_ip", 6) == 0) {')
text = text.replace('static NSMutableDictionary *ane_mem() {',
                    'static NSMutableDictionary *ane_mem() {')
text = text.replace('    return r;\n}\nstatic MLComputeUnits',
                    f'    r[@"device_model"] = [SidecarRPC deviceModel];\n    r[@"engine_commit"] = @"{meta["engine_commit"]}";\n    r[@"source_sha256"] = bb_source_sha;\n    return r;\n}}\nstatic MLComputeUnits')
# Join the prewarm with its command worker, so it cannot run after teardown.
text = text.replace('            dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{',
                    '            {')
text = text.replace('                g_mac_activating = false;\n            });',
                    '                g_mac_activating = false;\n            }')
text = text.replace('if (async) dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{ prewarm(); });',
                    'if (async) { if (bb_tail_prewarm.joinable()) bb_tail_prewarm.join(); bb_tail_prewarm = std::thread(prewarm); }')
text = text.replace('    std::string err = srv.serve(port);\n    g_tail_status',
                    '    std::string err = srv.serve(port);\n    if (bb_tail_prewarm.joinable()) bb_tail_prewarm.join();\n    g_tail_status')
text = text.replace('return [d URLByAppendingPathComponent:f];',
                    'return [[d URLByAppendingPathComponent:@"Backburner"] URLByAppendingPathComponent:f];')
# Original defaults are applied only after the subagent runtime has unloaded.
# Remove the generic GPU RPC service; the original default serve.sh uses only
# split prefill, phone-held KV, and the ANE command service.
start = text.index('+ (NSString *)startHost:')
end = text.index('\n@end', start)
text = text[:start] + text[end:]
start = text.index('static int connect_local_v4(')
end = text.index('+ (NSString *)deviceModel', start)
text = text[:start] + text[end:]
text = text.replace('while (true) {\n        int fd = accept(srv, nullptr, nullptr);',
                    'while (bb_active) {\n        int fd = accept(srv, nullptr, nullptr);')
text = text.replace('if (fd < 0) {   // survive a USB replug',
                    'if (fd < 0) { if (!bb_active) break; // survive a USB replug')
text = text.replace('return nullptr;\n    }\n    fprintf(stderr, "ane bench on',
                    'close(srv); return nullptr;\n    }\n    fprintf(stderr, "ane bench on')
text = text.replace('    static dispatch_once_t once;\n    dispatch_once(&once, ^{\n        std::thread([port] {',
                    '    bb_workers.emplace_back([port] {')
text = text.replace('            std::string err = srv.serve(port);\n            std::lock_guard<std::mutex> lk(g_pa_status.mu);',
                    '            std::string err = srv.serve(port);\n            while (!bb_tail_done) usleep(1000);\n            g_pa_ane.store(nullptr);\n            srv.releaseStorage(); srv.set_page_engine(nullptr);\n            delete ane_owner; delete eng;\n            std::lock_guard<std::mutex> lk(g_pa_status.mu);')
text = text.replace('            pa::server srv(', '            pa::ane_engine *ane_owner = nullptr;\n            pa::server srv(')
text = text.replace('                    srv.set_page_engine(ane);', '                    ane_owner = ane;\n                    srv.set_page_engine(ane);')
text = text.replace('        }).detach();\n    });', '        });')
start = text.index('+ (void)startANEBenchPort:')
text = text[:start] + '''+ (void)startANEBenchPort:(int)port {
    bb_workers.emplace_back([port] { ane_server_main((void *)(intptr_t)port); });
}
+ (void)beginServices {
    std::lock_guard<std::mutex> guard(bb_lifecycle_mu);
    if (bb_active) return;
    NSString *ip = [self cableAddress];
    if (ip.length == 0) return;
    inet_pton(AF_INET, ip.UTF8String, &bb_address);
    bb_save_env(); sidecar_env_defaults();
    NSString *sourcePath = [NSHomeDirectory() stringByAppendingPathComponent:@"Documents/Backburner/source.json"];
    NSData *sourceData = [NSData dataWithContentsOfFile:sourcePath];
    NSDictionary *source = sourceData ? [NSJSONSerialization JSONObjectWithData:sourceData options:0 error:nil] : nil;
    bb_source_sha = [source isKindOfClass:[NSDictionary class]] && [source[@"sha256"] isKindOfClass:[NSString class]] ? source[@"sha256"] : @"";
    bb_active = true; bb_tail_done = false;
    NSString *tail = [NSHomeDirectory() stringByAppendingPathComponent:@"Documents/Backburner/tail.gguf"];
    bb_workers.emplace_back([tail] { [SidecarRPC startTailPort:50060 modelPath:tail]; bb_tail_done = true; });
    [self startPhoneAttnPort:50062];
    [self startANEBenchPort:50061];
}
+ (void)endServices {
    std::lock_guard<std::mutex> guard(bb_lifecycle_mu);
    bb_active = false;
    { std::lock_guard<std::mutex> lk(bb_socket_mu);
      for (int fd : bb_sockets) ::shutdown(fd, SHUT_RDWR); }
    for (auto &worker : bb_workers) if (worker.joinable()) worker.join();
    bb_workers.clear();
    g_ane.clear();
    spt::ane_ffn_close(g_tail_ane); g_tail_ane = nullptr; g_tail_ane_n = 0; g_tail_ane_on = false;
    g_tail_status.set("offline");
    { std::lock_guard<std::mutex> lk(g_mac_mu); g_mac_phase.clear(); }
    ggml_backend_metal_residency_release_now();
    bb_restore_env();
}
+ (BOOL)servicesRunning { return bb_active; }
@end
'''
# Both initial loading and ongoing sessions observe shutdown. Socket shutdown
# interrupts accept/recv without freeing a model while a compute call uses it.
for rel in ['llama.cpp/tools/split-prefill/tail-server.h', 'phone-attn/phone-attn.h']:
    path = out / rel
    data = path.read_text()
    data = data.replace('int fd = accept(srv, nullptr, nullptr);', 'int fd = accept(srv, nullptr, nullptr);')
    data = data.replace('if (fd < 0) {   // a USB replug', 'if (fd < 0) { if (!bb_active) break; // a USB replug')
    data = data.replace('while (true) {\n            int fd = accept(srv', 'while (bb_active) {\n            int fd = accept(srv')
    data = data.replace('for (;;) {\n            int fd = ::accept(srv', 'while (bb_active) {\n            int fd = ::accept(srv')
    if 'tail-server' in rel:
        data = data.replace('            return true;\n        };\n        model_', '            return bb_active.load();\n        };\n        model_')
    else:
        data = data.replace('    void set_page_engine(page_engine * pe)',
                            '    ~server() { drop_all(); }\n    void releaseStorage() { drop_all(); }\n    void set_page_engine(page_engine * pe)')
    path.write_text(data)
bridge.write_text(text)
header = out / 'ios/Backburner/Sidecar/RPCBridge.h'
header.write_text(header.read_text().replace('@interface SidecarRPC : NSObject',
    '@interface SidecarRPC : NSObject\n+ (void)beginServices;\n+ (void)endServices;\n+ (BOOL)servicesRunning;'))
(out / 'Lifecycle.h').write_text(r'''#pragma once
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <poll.h>
#include <fcntl.h>
#include <atomic>
#include <mutex>
#include <set>
#include <vector>
#include <thread>
#include <map>
#include <string>
#include <cstdlib>
static std::atomic<bool> bb_active{false};
static std::atomic<bool> bb_tail_done{true};
static NSString *bb_source_sha = @"";
static std::mutex bb_socket_mu, bb_lifecycle_mu;
static std::set<int> bb_sockets;
static std::vector<std::thread> bb_workers;
static std::thread bb_tail_prewarm;
static in_addr bb_address;
static std::map<std::string, std::pair<bool,std::string>> bb_env;
static void bb_save_env() {
    for (const char *k : {"GGML_METAL_FA_PREFILL_GQA", "GGML_METAL_RESIDENCY_KEEP_ALIVE_S", "GGML_METAL_FA_PREFILL_NA"}) {
        const char *v = getenv(k); bb_env[k] = {v != nullptr, v ? v : ""};
    }
}
static void bb_restore_env() {
    for (const auto &kv : bb_env) {
        if (kv.second.first) setenv(kv.first.c_str(), kv.second.second.c_str(), 1);
        else unsetenv(kv.first.c_str());
    }
    bb_env.clear();
}
static int bb_setenv(const char *key, const char *value, int overwrite) {
    if (!bb_env.count(key)) { const char *old = getenv(key); bb_env[key] = {old != nullptr, old ? old : ""}; }
    return ::setenv(key, value, overwrite);
}
static int bb_socket(int domain, int type, int protocol) {
    std::lock_guard<std::mutex> lk(bb_socket_mu);
    if (!bb_active) { errno = ECANCELED; return -1; }
    int fd = ::socket(domain,type,protocol); if (fd >= 0) bb_sockets.insert(fd); return fd;
}
static int bb_accept(int srv, sockaddr *addr, socklen_t *len) {
    // Darwin shutdown(SHUT_RDWR) does not wake accept() on a listening socket.
    // Poll the nonblocking listener, while established clients retain the
    // original blocking protocol and are interrupted by shutdown().
    while (bb_active) {
        pollfd pending{srv, POLLIN, 0};
        int ready = ::poll(&pending,1,100);
        if (!bb_active) break;
        if (ready == 0 || (ready < 0 && errno == EINTR)) continue;
        if (ready < 0) return -1;
        int fd = ::accept(srv,addr,len);
        if (fd < 0 && (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR)) continue;
        std::lock_guard<std::mutex> lk(bb_socket_mu);
        if (fd >= 0) {
            if (!bb_active) { ::close(fd); errno = ECANCELED; return -1; }
            // Darwin inherits O_NONBLOCK from the listener. The unchanged
            // upstream recv/send framing requires a blocking client socket.
            ::fcntl(fd,F_SETFL,::fcntl(fd,F_GETFL) & ~O_NONBLOCK);
            bb_sockets.insert(fd);
        }
        return fd;
    }
    errno = ECANCELED; return -1;
}
static int bb_listen(int fd, int backlog) {
    int result = ::listen(fd,backlog);
    if (result == 0) ::fcntl(fd,F_SETFL,::fcntl(fd,F_GETFL) | O_NONBLOCK);
    return result;
}
static int bb_close(int fd) {
    std::lock_guard<std::mutex> lk(bb_socket_mu);
    bb_sockets.erase(fd); return ::close(fd);
}
static int bb_bind(int fd, const sockaddr *addr, socklen_t len) {
    if (addr->sa_family == AF_INET) {
        sockaddr_in wired = *(const sockaddr_in *)addr;
        wired.sin_addr = bb_address;
        return ::bind(fd,(sockaddr *)&wired,len);
    }
    errno = EAFNOSUPPORT; return -1;
}
#define socket bb_socket
#define accept bb_accept
#define listen bb_listen
#define close bb_close
#define bind bb_bind
#define setenv bb_setenv
''')
print(out)
