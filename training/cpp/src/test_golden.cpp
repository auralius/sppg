#include <Noodle2.h>
#include "peaknet_model_flash.h"
#include "peaknet_golden.h"
#include <cstdio>
#include <cstdlib>
#include <cstdint>

int main(){
    static peaknet_ppg::Model model;
    if(model.bind_flash()!=noodle2::OK){std::puts("BIND_FLASH_FAIL");return 1;}
    static noodle2::NoodleGraph graph;
    if(peaknet_ppg::build_graph(graph,model)!=noodle2::OK){
        std::printf("GRAPH_COMPILE_FAIL status=%d\n",int(graph.status()));return 2;
    }
    std::printf("PEAKNET_PC graph_nodes=%u arena=%lu naive=%lu model_bytes=%lu\n",
        unsigned(graph.node_count()),(unsigned long)graph.arena_bytes(),
        (unsigned long)graph.naive_activation_bytes(),(unsigned long)peaknet_ppg::parameter_bytes);
    for(int i=0;i<peaknet_golden::COUNT;i++){
        for(int tiled=0;tiled<2;tiled++){
            int tile=tiled ? 4 : 0;
            if(graph.set_input(peaknet_golden::input[i])!=noodle2::OK) return 3;
            const noodle2::Status status=graph.run(0,0,uint8_t(tile));
            if(status!=noodle2::OK){std::printf("GRAPH_RUN_FAIL %d %d status=%d\n",i,tile,int(status));return 4;}
            if(graph.output().size()!=512){std::puts("OUTPUT_SHAPE_FAIL");return 5;}
            for(unsigned j=0;j<512;j++){
                int8_t val=0;
                if(graph.output().get(j,val)!=noodle2::OK){std::puts("READ_FAIL");return 6;}
                if(val !=peaknet_golden::output[i][j]){
                    std::printf("INTEGER_MISMATCH vector=%d tile=%d index=%u C++=%d Python=%d\n",
                                i,tile,j,int(val),int(peaknet_golden::output[i][j]));return 7;
                }
            }
        }
        std::printf("VECTOR_%d_DIRECT_AND_TILE_PASS\n",i);
    }
    std::printf("PEAKNET_NOODLE2_INT8_LOGITS_EXACT_PASS vectors=%d (direct and tiled)\n",peaknet_golden::COUNT);
    return 0;
}
