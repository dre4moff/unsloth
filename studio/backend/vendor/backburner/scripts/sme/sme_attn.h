// sme_attn.h - Swift/ObjC-callable entry points of sme_attn.c (add to the app's bridging header).
// Compile sme_attn.c into the target with: -mcpu=apple-a18 -DSME_BENCH_MAIN -DSME_BENCH_NO_MAIN
#pragma once
#ifdef __cplusplus
extern "C" {
#endif
int sme2_available(void);                 // 1 = FEAT_SME2 and 512-bit streaming vectors; call before anything else
int sme_bench(int argc, char **argv);     // same arguments as the macOS CLI (argv[0] ignored); prints to stdout
#ifdef __cplusplus
}
#endif
