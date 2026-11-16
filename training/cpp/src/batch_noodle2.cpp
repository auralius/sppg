// Host-side Noodle2 INT8 inference on batches of 256-sample quantized windows.
// Protocol: input file raw uint8/int8 [N,256], output file raw int8 [N,2,256].
#include <Noodle2.h>
#include "peaknet_model_flash.h"
#include "peaknet_golden.h"
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>

int main(int argc, char **argv) {
    if (argc!=3) { std::fprintf(stderr,"usage: batch_noodle2 INPUT.int8 OUTPUT.int8\n");return 2; }
    FILE *fi=std::fopen(argv[1],"rb");
    FILE *fo=std::fopen(argv[2],"wb");
    if (!fi || !fo) { std::fprintf(stderr,"cannot open IO files\n");return 2; }
    static peaknet_ppg::Model m;
    if (m.bind_flash()!=noodle2::OK) return 3;
    static noodle2::NoodleGraph g;
    if (peaknet_ppg::build_graph(g,m)!=noodle2::OK) return 4;
    int8_t in[256],out[512];
    size_t n=0;
    while (std::fread(in,1,sizeof(in),fi)==sizeof(in)) {
        if (g.set_input(in)!=noodle2::OK || g.run(0,0,4)!=noodle2::OK) return 5;
        for (int j=0;j<512;++j){
            if (g.output().get(j,out[j])!=noodle2::OK) return 6;
        }
        if (std::fwrite(out,1,sizeof(out),fo)!=sizeof(out)) return 7;
        ++n;
    }
    std::fclose(fi);std::fclose(fo);
    std::fprintf(stderr,"Noodle2 batches=%zu arena=%zu parameters=%u tile=4\n",
                 n,(size_t)g.arena_bytes(),(unsigned)peaknet_ppg::parameter_bytes);
    return 0;
}
