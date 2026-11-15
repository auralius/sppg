#pragma once
// Host/ESP32-shared preprocessing and candidate extraction.
// Preprocessing exactly follows tools/ppg_synth.py real_windows():
// negative RAW IR, per-window z score using population standard deviation,
// then round half away from zero into symmetric INT8 (zero_point 0).
// This header deliberately contains no Arduino or Noodle2 dependencies.
#include <stdint.h>
#include <stddef.h>
#include <math.h>

namespace peaknet_live {
static const size_t WINDOW = 256;
static const uint16_t HOP = 128;
static const uint8_t FIRST_DECODE = 96;
static const uint8_t LAST_DECODE_EXCLUSIVE = 224;
static const int8_t MIN_LOGIT = 0; // sigmoid(logit)>=0.5; experimental, not calibrated
static const uint16_t MIN_SEPARATION = 35; // 350 ms, same class

struct Candidate {
    uint32_t sample_idx;
    uint8_t channel; // 0=systolic-class, 1=diastolic-class, both UNVALIDATED
    int8_t logit;
};

inline void quantize_ir(const uint32_t* ir, float input_scale,
                        int8_t* dst, float* standard_deviation = 0) {
    double mean=0.0;
    for(size_t i=0;i<WINDOW;i++) mean += double(ir[i]);
    mean /= double(WINDOW);
    double variance=0.0;
    for(size_t i=0;i<WINDOW;i++) {
        const double delta=double(ir[i])-mean;
        variance += delta*delta;
    }
    const double stddev=sqrt(variance/double(WINDOW));
    if(standard_deviation) *standard_deviation=float(stddev);
    const double denominator=stddev+1.e-9;
    for(size_t i=0;i<WINDOW;i++) {
        // Python normalizes to float32 before quantizing into INT8.
        const float z=float((mean-double(ir[i]))/denominator);
        const double scaled=double(z)/double(input_scale);
        const double rounded=copysign(floor(fabs(scaled)+0.5),scaled);
        dst[i]=int8_t(rounded>127.0?127:(rounded< -128.0?-128:int(rounded)));
    }
}

// Independent local maxima over the central 128 samples. Same region from
// consecutive 128-hop windows never overlaps. We purposely exclude edge
// context and do NOT claim these are physiologically verified landmarks.
inline size_t extract_candidates(const int8_t* output, uint32_t start_sample,
                                 Candidate* result, size_t capacity) {
    size_t count=0;
    for(uint8_t c=0;c<2;c++) {
        int last=-1000;
        for(int i=FIRST_DECODE;i<LAST_DECODE_EXCLUSIVE;i++) {
            const int8_t y=output[size_t(c)*WINDOW+i];
            if(y<MIN_LOGIT || y<output[size_t(c)*WINDOW+i-1] ||
               y<=output[size_t(c)*WINDOW+i+1]) continue;
            if(count>0 && last>=0 && (i-last)<MIN_SEPARATION) {
                if(y>result[count-1].logit) {
                    result[count-1].sample_idx=start_sample+uint32_t(i);
                    result[count-1].logit=y;
                    last=i;
                }
                continue;
            }
            if(count==capacity) return count;
            result[count++]={start_sample+uint32_t(i),c,y};
            last=i;
        }
    }
    return count;
}
} // namespace peaknet_live
