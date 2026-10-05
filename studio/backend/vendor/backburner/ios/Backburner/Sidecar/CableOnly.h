// CableOnly.h - who may talk to the app's servers: only the Mac at the other end of the USB cable.
//
// The app's listeners (ggml RPC :50052, prefill tail :50060, control :50061, phone attention :50062) speak protocols with no
// authentication: anything that can connect can load models, write files into Documents, or read and write GPU memory through
// ggml RPC. They used to accept connections from every network the phone is on, Wi-Fi included (GitHub issue #2).
//
// The rule: a connection is served only if it arrived on a WIRED interface (SIOCGIFFUNCTIONALTYPE; the USB link to the Mac)
// with both ends link-local (169.254/16, or fe80::/10 on that interface), or on loopback (in-app hops). Wi-Fi, cellular,
// AWDL, VPN and every routable address are refused before a single byte is read. The checks below are plain functions so
// tests/security/cable-policy-test.cpp exercises them on the Mac.
//
// Escape hatch, for a cable that reports another interface type: Documents/env.txt BB_CABLE_IF_TYPES=2,0 (functional types,
// net/if.h IFRTYPE_FUNCTIONAL_*). env.txt can only be written over the cable (devicectl, or :50061 fetch from the Mac).
#pragma once

#include <arpa/inet.h>
#include <net/if.h>
#include <netinet/in.h>
#include <ifaddrs.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <sys/sockio.h>
#include <unistd.h>

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

namespace bb {

enum class verdict { allow, deny };

inline bool v4_loopback(uint32_t a)   { return (a >> 24) == 127; }               // host byte order
inline bool v4_link_local(uint32_t a) { return (a >> 16) == 0xA9FE; }            // 169.254/16
inline bool v6_loopback(const in6_addr & a)   { return IN6_IS_ADDR_LOOPBACK(&a); }
inline bool v6_link_local(const in6_addr & a) { return IN6_IS_ADDR_LINKLOCAL(&a); }
inline bool v6_v4mapped(const in6_addr & a)   { return IN6_IS_ADDR_V4MAPPED(&a); }
inline uint32_t v6_mapped_v4(const in6_addr & a) {
    uint32_t v; memcpy(&v, &a.s6_addr[12], 4); return ntohl(v);
}

// interface functional types that count as the cable (default: WIRED only)
inline bool if_type_allowed(uint32_t type, const char * env_list) {
    if (!env_list || !*env_list) return type == IFRTYPE_FUNCTIONAL_WIRED;
    for (const char * p = env_list; *p; ) {
        char * end = nullptr;
        const long v = strtol(p, &end, 10);
        if (end == p) { p++; continue; }
        if (v >= 0 && (uint32_t) v == type) return true;
        p = end;
    }
    return false;
}

// The decision, from the two endpoints and the functional type of the local interface (~0u if unknown).
// Pure: no system calls, so the tests can feed it any combination.
inline verdict decide(const sockaddr * local, const sockaddr * peer, uint32_t if_type, const char * env_types, std::string & why) {
    if (!local || !peer || local->sa_family != peer->sa_family) { why = "unknown address family"; return verdict::deny; }
    uint32_t l4 = 0, p4 = 0;
    bool v4 = false;
    if (local->sa_family == AF_INET) {
        l4 = ntohl(((const sockaddr_in *) local)->sin_addr.s_addr);
        p4 = ntohl(((const sockaddr_in *) peer)->sin_addr.s_addr);
        v4 = true;
    } else if (local->sa_family == AF_INET6) {
        const in6_addr & la = ((const sockaddr_in6 *) local)->sin6_addr;
        const in6_addr & pa = ((const sockaddr_in6 *) peer)->sin6_addr;
        if (v6_v4mapped(la) && v6_v4mapped(pa)) {
            l4 = v6_mapped_v4(la); p4 = v6_mapped_v4(pa); v4 = true;
        } else {
            if (v6_loopback(la) && v6_loopback(pa)) { why = "loopback"; return verdict::allow; }
            if (!v6_link_local(la) || !v6_link_local(pa)) { why = "not link-local IPv6"; return verdict::deny; }
        }
    } else {
        why = "unknown address family"; return verdict::deny;
    }
    if (v4) {
        if (v4_loopback(l4) && v4_loopback(p4)) { why = "loopback"; return verdict::allow; }
        if (!v4_link_local(l4) || !v4_link_local(p4)) { why = "not link-local IPv4"; return verdict::deny; }
    }
    if (!if_type_allowed(if_type, env_types)) {
        why = "interface type " + (if_type == ~0u ? std::string("unknown") : std::to_string(if_type)) + " is not the cable";
        return verdict::deny;
    }
    why = "cable";
    return verdict::allow;
}

// functional type of interface `name` (~0u if the ioctl fails)
inline uint32_t if_functional_type(const char * name) {
    int s = socket(AF_INET, SOCK_DGRAM, 0);
    if (s < 0) return ~0u;
    ifreq ifr = {};
    strlcpy(ifr.ifr_name, name, sizeof ifr.ifr_name);
    const uint32_t t = ioctl(s, SIOCGIFFUNCTIONALTYPE, &ifr) == 0 ? ifr.ifr_ifru.ifru_functional_type : ~0u;
    close(s);
    return t;
}

// the interface that owns local address `a` (IPv6: its scope id names it directly)
inline std::string if_of_local(const sockaddr * a) {
    if (a->sa_family == AF_INET6) {
        const auto * a6 = (const sockaddr_in6 *) a;
        if (!v6_v4mapped(a6->sin6_addr) && a6->sin6_scope_id) {
            char n[IF_NAMESIZE] = {};
            if (if_indextoname(a6->sin6_scope_id, n)) return n;
        }
    }
    uint32_t want4 = 0;
    const bool v4 = a->sa_family == AF_INET || v6_v4mapped(((const sockaddr_in6 *) a)->sin6_addr);
    if (v4) want4 = a->sa_family == AF_INET ? ((const sockaddr_in *) a)->sin_addr.s_addr : htonl(v6_mapped_v4(((const sockaddr_in6 *) a)->sin6_addr));
    std::string found;
    ifaddrs * ifs = nullptr;
    if (getifaddrs(&ifs) != 0) return found;
    for (ifaddrs * p = ifs; p && found.empty(); p = p->ifa_next) {
        if (!p->ifa_addr) continue;
        if (v4 && p->ifa_addr->sa_family == AF_INET && ((sockaddr_in *) p->ifa_addr)->sin_addr.s_addr == want4) found = p->ifa_name;
        if (!v4 && p->ifa_addr->sa_family == AF_INET6 &&
            !memcmp(&((sockaddr_in6 *) p->ifa_addr)->sin6_addr, &((const sockaddr_in6 *) a)->sin6_addr, sizeof(in6_addr))) found = p->ifa_name;
    }
    freeifaddrs(ifs);
    return found;
}

// what the gate has refused, for the control port's `mem` report (scripts/check-phone-exposure.sh reads it)
struct gate_stats {
    std::atomic<uint64_t> allowed { 0 }, denied { 0 };
    std::mutex mu;
    std::string last_denied, cable_if;
    uint32_t cable_if_type = ~0u;
};
inline gate_stats & stats() { static gate_stats s; return s; }

// The accept filter for every listener in the app: call right after accept(), before reading anything.
inline bool cable_accept(int fd, std::string & why) {
    sockaddr_storage l = {}, p = {};
    socklen_t ll = sizeof l, pl = sizeof p;
    if (getsockname(fd, (sockaddr *) &l, &ll) != 0 || getpeername(fd, (sockaddr *) &p, &pl) != 0) {
        why = "no address"; stats().denied++; return false;
    }
    const std::string ifn = if_of_local((sockaddr *) &l);
    const uint32_t type = ifn.empty() ? ~0u : if_functional_type(ifn.c_str());
    const bool ok = decide((sockaddr *) &l, (sockaddr *) &p, type, getenv("BB_CABLE_IF_TYPES"), why) == verdict::allow;
    char peer[INET6_ADDRSTRLEN] = "?";
    if (p.ss_family == AF_INET)  inet_ntop(AF_INET,  &((sockaddr_in *)  &p)->sin_addr,  peer, sizeof peer);
    if (p.ss_family == AF_INET6) inet_ntop(AF_INET6, &((sockaddr_in6 *) &p)->sin6_addr, peer, sizeof peer);
    auto & s = stats();
    if (ok) {
        s.allowed++;
        if (why == "cable") { std::lock_guard<std::mutex> lk(s.mu); s.cable_if = ifn; s.cable_if_type = type; }
    } else {
        s.denied++;
        std::lock_guard<std::mutex> lk(s.mu);
        s.last_denied = std::string(peer) + " on " + (ifn.empty() ? "?" : ifn) + ": " + why;
    }
    why = std::string(peer) + " on " + (ifn.empty() ? "?" : ifn) + ": " + why;
    return ok;
}

// ---- inputs that name files or URLs (the control port's fetch / load / plan / tailane) ----

// A path under Documents: relative, no "..", no empty or hidden components, only [A-Za-z0-9._-] and '/'.
inline bool safe_doc_path(const std::string & s) {
    if (s.empty() || s.size() > 512 || s[0] == '/') return false;
    size_t i = 0;
    while (i <= s.size()) {
        const size_t j = s.find('/', i);
        const std::string c = s.substr(i, (j == std::string::npos ? s.size() : j) - i);
        if (c.empty() || c[0] == '.') return false;   // also rejects "." and ".."
        for (char ch : c) {
            if (!(isalnum((unsigned char) ch) || ch == '.' || ch == '_' || ch == '-')) return false;
        }
        if (j == std::string::npos) break;
        i = j + 1;
    }
    return true;
}

// fetch may only download from the Mac over the cable: http://169.254.A.B[:PORT]/path
inline bool fetch_url_ok(const std::string & u) {
    const std::string pre = "http://";
    if (u.compare(0, pre.size(), pre) != 0) return false;
    const size_t host_end = u.find_first_of(":/", pre.size());
    const std::string host = u.substr(pre.size(), host_end == std::string::npos ? std::string::npos : host_end - pre.size());
    in_addr a = {};
    if (inet_pton(AF_INET, host.c_str(), &a) != 1) return false;   // a literal address only: no names, no userinfo (@)
    if (!v4_link_local(ntohl(a.s_addr))) return false;
    if (host_end != std::string::npos && u[host_end] == ':') {
        size_t k = host_end + 1;
        if (k >= u.size() || !isdigit((unsigned char) u[k])) return false;
        while (k < u.size() && isdigit((unsigned char) u[k])) k++;
        if (k < u.size() && u[k] != '/') return false;
    }
    return u.find('@') == std::string::npos;
}

} // namespace bb
