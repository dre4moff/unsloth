#import "ios/Backburner/Sidecar/RPCBridge.h"
#import <Foundation/Foundation.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <cstdio>
#include <cstring>
#include <cassert>
#include <cstdlib>
#include <thread>
#include <chrono>
#include <signal.h>
#include "../../studio/backend/vendor/backburner/phone-attn/phone-attn.h"
static int connectPort(int port) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in a = {}; a.sin_family = AF_INET; a.sin_port = htons(port); a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    timeval timeout{2,0}; setsockopt(fd,SOL_SOCKET,SO_RCVTIMEO,&timeout,sizeof(timeout));
    if (connect(fd,(sockaddr *)&a,sizeof(a))) { close(fd); return -1; } return fd;
}
int main() { @autoreleasepool {
    // Match llama-server's process policy for the original Mac client.
    signal(SIGPIPE,SIG_IGN);
    setenv("PA_REPLY_TIMEOUT_S","5",1);
    setenv("GGML_METAL_FA_PREFILL_GQA", "preserved-test-value", 1);
    unsetenv("GGML_METAL_FA_PREFILL_NA");
    for (int cycle = 0; cycle < 3; ++cycle) {
        fprintf(stderr,"begin cycle %d\n",cycle+1); [SidecarRPC beginServices]; assert([SidecarRPC servicesRunning]);
        int cmd = -1;
        for (int n = 0; n < 100 && cmd < 0; ++n) { cmd = connectPort(50061); if(cmd < 0) usleep(100000); }
        assert(cmd >= 0); send(cmd,"mem\n",4,0); char reply[65536]={}; assert(recv(cmd,reply,sizeof(reply)-1,0)>0);
        assert(strstr(reply,"engine_commit")); close(cmd);
        int attn = -1;
        for(int n=0;n<100 && attn<0;++n) { attn=connectPort(50062); if(attn<0) usleep(100000); }
        assert(attn>=0);
        struct { uint32_t magic,kind; uint64_t bytes; } hello{0x4E544150,1,0};
        assert(send(attn,&hello,sizeof(hello),0)==sizeof(hello));
        char response[88]; size_t done=0; while(done<sizeof(response)) { ssize_t n=recv(attn,response+done,sizeof(response)-done,0); assert(n>0); done+=n; }
        assert(*(uint32_t *)response==0x4E544150 && *(uint32_t *)(response+16)==4);
        close(attn);
        // Exercise the actual adapted phone service: session save must return
        // byte-identical keys/values, and epoch reset must discard old rows.
        pa::client client; assert(client.connect("127.0.0.1",50062));
        pa::hello_rep greeting{}; assert(client.hello(greeting) && greeting.version==4);
        pa::config_req config{}; config.n_layer=1; config.n_head_kv=1;
        config.rs=512; config.hb=512; config.sme_workers=1;
        assert(client.config(config));
        std::vector<uint16_t> keys(2*256,0x3c00), values(2*256,0x4000), restored(2*256);
        assert(client.append(0,0,2,keys.data(),values.data(),512));
        assert(client.fetch(0,0,2,0,restored.data(),512) && restored==keys);
        assert(client.fetch(0,0,2,1,restored.data(),512) && restored==values);
        assert(client.truncate(0));
        assert(!client.fetch(0,0,2,0,restored.data(),512));
        assert(client.config(config));
        std::fill(keys.begin(),keys.end(),0x4200);
        assert(client.append(0,0,2,keys.data(),values.data(),512));
        assert(client.fetch(0,0,2,0,restored.data(),512) && restored==keys);
        int tail = -1;
        for(int n=0;n<1000 && tail<0;++n) { tail=connectPort(50060); if(tail<0) usleep(100000); }
        assert(tail>=0);
        int idleCommand = connectPort(50061); assert(idleCommand>=0);
        // Leave all services with real persistent/idle clients during stop.
        fprintf(stderr,"stop cycle %d\n",cycle+1); auto t0=std::chrono::steady_clock::now(); [SidecarRPC endServices];
        assert(std::chrono::steady_clock::now()-t0 < std::chrono::seconds(5)); close(tail); close(idleCommand);
        assert(![SidecarRPC servicesRunning]);
        for(int port: {50060,50061,50062}) assert(connectPort(port)<0);
        assert(!strcmp(getenv("GGML_METAL_FA_PREFILL_GQA"),"preserved-test-value"));
        assert(getenv("GGML_METAL_FA_PREFILL_NA")==nullptr);
        printf("cycle %d: HELLO v4 + FETCH K/V + epoch reset + mem + persistent-client stop + closed ports + restored environment OK\n",cycle+1);
    }
    // A connected phone which sends no reply must not keep generation stuck.
    int listener=socket(AF_INET,SOCK_STREAM,0); assert(listener>=0);
    sockaddr_in address{}; address.sin_family=AF_INET; address.sin_addr.s_addr=htonl(INADDR_LOOPBACK);
    assert(bind(listener,(sockaddr *)&address,sizeof(address))==0 && listen(listener,1)==0);
    socklen_t length=sizeof(address); assert(getsockname(listener,(sockaddr *)&address,&length)==0);
    std::thread frozen([listener] {
        int fd=accept(listener,nullptr,nullptr); assert(fd>=0);
        char request[16]; assert(recv(fd,request,sizeof(request),MSG_WAITALL)==sizeof(request));
        char byte; while(recv(fd,&byte,1,0)>0) {} close(fd); close(listener);
    });
    pa::client frozenClient; assert(frozenClient.connect("127.0.0.1",ntohs(address.sin_port)));
    auto started=std::chrono::steady_clock::now(); pa::hello_rep reply{};
    assert(!frozenClient.hello(reply));
    auto elapsed=std::chrono::steady_clock::now()-started;
    assert(elapsed>=std::chrono::seconds(4) && elapsed<std::chrono::seconds(10));
    assert(frozenClient.last_err.find("did not answer")!=std::string::npos);
    frozenClient.abort_io(); frozen.join(); unsetenv("PA_REPLY_TIMEOUT_S");
    printf("silent phone: bounded reply timeout + interrupted socket OK\n");
} }
