#pragma once
#include <math.h>

// Generated from digital-filter.ipynb (SymPy + Tustin).
// Direct Form I biquad; 2nd-order Butterworth low-pass filter.
struct ButterworthLPF2 {
    float b0 = 0, b1 = 0, b2 = 0, a1 = 0, a2 = 0;
    float x1 = 0, x2 = 0, y1 = 0, y2 = 0;

    void reset() { x1 = x2 = y1 = y2 = 0.0f; }

    // Returns false if cutoff/sampling settings are invalid.
    bool configure(float fs_hz, float fc_hz) {
        if (fs_hz <= 0.0f || fc_hz <= 0.0f || fc_hz >= fs_hz/2.0f)
            return false;
        const float T_s = 1.0f / fs_hz;
        const float omega_c = (2.0f / T_s) *
            tanf(3.14159265358979323846f * fc_hz / fs_hz);
        const float q = T_s * omega_c;
        const float q2 = q*q;
        const float t0 = 2.8284271247461901*q;
        const float t1 = q2 + 4.0;
        const float t2 = 1.0/(t0 + t1);
        const float t3 = q2*t2;
        b0 = t3;
        b1 = 2.0*t3;
        b2 = t3;
        a1 = 2.0*t2*(q - 2.0)*(q + 2.0);
        a2 = t2*(-t0 + t1);
        reset();
        return true;
    }

    float step(float x0) {
        const float y0 = -a1*y1 - a2*y2 + b0*x0 + b1*x1 + b2*x2;
        x2 = x1; x1 = x0;
        y2 = y1; y1 = y0;
        return y0;
    }
};
