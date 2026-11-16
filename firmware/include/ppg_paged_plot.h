#pragma once
// Single-threaded (Arduino loop/core 1) fixed-duration waveform pages.
// Ingested samples never mutate the currently visible page; publication happens
// only from the outer loop, not during nested FIFO service in TFT drawing.
// This header is platform-independent so we can unit-test page boundaries.
#include <stddef.h>
#include <stdint.h>
#include <string.h>

template <size_t N>
class PPGPagedPlot {
public:
    static_assert(N > 1 && N <= 65535, "Invalid PPG page length");

    void push(float sample, uint32_t sampleIndex) {
        if (collectCount_ == 0) collectStart_ = sampleIndex;
        collecting_[collectCount_++] = sample;
        if (collectCount_ == N) {
            memcpy(pending_, collecting_, sizeof(pending_));
            pendingStart_ = collectStart_;
            pendingReady_ = true;
            collectCount_ = 0;
        }
    }

    // Call outside of drawWaveform(). If FIFO polling occurs during rendering,
    // push() may stage another page but the visible page remains immutable.
    bool publish() {
        if (!pendingReady_) return false;
        memcpy(visible_, pending_, sizeof(visible_));
        visibleStart_ = pendingStart_;
        visibleReady_ = true;
        pendingReady_ = false;
        ++pageNumber_;
        return true;
    }

    void reset() {
        collectCount_ = 0;
        pendingReady_ = false;
        visibleReady_ = false;
        visibleStart_ = 0;
        collectStart_ = 0;
        pendingStart_ = 0;
        ++pageNumber_;
    }

    bool ready() const { return visibleReady_; }
    uint32_t firstSample() const { return visibleStart_; }
    uint32_t pageNumber() const { return pageNumber_; }
    const float *samples() const { return visible_; }
    size_t collectingSamples() const { return collectCount_; }

private:
    float collecting_[N] = {};
    float pending_[N] = {};
    float visible_[N] = {};
    size_t collectCount_ = 0;
    uint32_t collectStart_ = 0;
    uint32_t pendingStart_ = 0;
    uint32_t visibleStart_ = 0;
    uint32_t pageNumber_ = 0;
    bool pendingReady_ = false;
    bool visibleReady_ = false;
};
