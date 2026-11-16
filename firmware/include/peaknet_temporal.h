#pragma once
// Cross-class temporal postprocessing for tentative PeakNet landmarks.
// Runs AFTER INT8 inference; does not alter either neural-network output.
// Raw events and rejected events can be inspected independently.
#include <stddef.h>
#include <stdint.h>
#include "peaknet_live.h"

namespace peaknet_temporal {

inline uint32_t distance(uint32_t a, uint32_t b) {
    return a > b ? a - b : b - a;
}

// Keep all S candidates. Suppress D candidates within 'guard' samples
// of ANY S candidate. Both earlier and later S are considered; raw events
// need not be sorted by time. guard=0 disables postprocessing completely.
// A predicted landmark is never relocated or synthesized.
inline size_t exclude_near_s(const peaknet_live::Candidate *raw,
                             size_t rawCount,
                             peaknet_live::Candidate *kept,
                             size_t capacity,
                             uint32_t guard,
                             size_t *removedD = nullptr) {
    size_t out = 0;
    size_t removed = 0;
    for (size_t i = 0; i < rawCount; ++i) {
        bool reject = false;
        if (guard > 0 && raw[i].channel == 1) {
            for (size_t j = 0; j < rawCount; ++j) {
                if (raw[j].channel == 0 &&
                    distance(raw[i].sample_idx, raw[j].sample_idx) <= guard) {
                    reject = true;
                    break;
                }
            }
        }
        if (reject) {
            ++removed;
        } else if (out < capacity) {
            kept[out++] = raw[i];
        }
    }
    if (removedD) *removedD = removed;
    return out;
}

// Original v15 rolling-event merge: same-class peaks only. Events may
// arrive in class order, not timestamp order. Keeping this behavior unchanged
// makes the temporal guard the only decoder change.
inline void append_merged(peaknet_live::Candidate *events, size_t &eventCount,
                          size_t capacity, const peaknet_live::Candidate &candidate) {
    for (size_t j = eventCount; j > 0; --j) {
        peaknet_live::Candidate &old = events[j - 1];
        if (old.channel != candidate.channel) continue;
        if (candidate.sample_idx >= old.sample_idx &&
            candidate.sample_idx - old.sample_idx < peaknet_live::MIN_SEPARATION) {
            if (candidate.logit > old.logit) old = candidate;
            return;
        }
        break;
    }
    if (capacity == 0) return;
    if (eventCount == capacity) {
        for (size_t j = 1; j < eventCount; ++j) events[j - 1] = events[j];
        --eventCount;
    }
    events[eventCount++] = candidate;
}

} // namespace peaknet_temporal
