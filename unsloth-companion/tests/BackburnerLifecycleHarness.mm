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
static int connectPort(int port) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in a = {}; a.sin_family = AF_INET; a.sin_port = htons(port); a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    timeval timeout{2,0}; setsockopt(fd,SOL_SOCKET,SO_RCVTIMEO,&timeout,sizeof(timeout));
    if (connect(fd,(sockaddr *)&a,sizeof(a))) { close(fd); return -1; } return fd;
}
int main() { @autoreleasepool {
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
        assert(*(uint32_t *)response==0x4E544150 && *(uint32_t *)(response+16)==3);
        int tail = -1;
        for(int n=0;n<1000 && tail<0;++n) { tail=connectPort(50060); if(tail<0) usleep(100000); }
        assert(tail>=0);
        int idleCommand = connectPort(50061); assert(idleCommand>=0);
        // Leave all services with real persistent/idle clients during stop.
        fprintf(stderr,"stop cycle %d\n",cycle+1); auto t0=std::chrono::steady_clock::now(); [SidecarRPC endServices];
        assert(std::chrono::steady_clock::now()-t0 < std::chrono::seconds(5)); close(attn); close(tail); close(idleCommand);
        assert(![SidecarRPC servicesRunning]);
        for(int port: {50060,50061,50062}) assert(connectPort(port)<0);
        assert(!strcmp(getenv("GGML_METAL_FA_PREFILL_GQA"),"preserved-test-value"));
        assert(getenv("GGML_METAL_FA_PREFILL_NA")==nullptr);
        printf("cycle %d: HELLO v3 + mem + persistent-client stop + closed ports + restored environment OK\n",cycle+1);
    }
} }
