// MAX30102 PPG oscilloscope for classic ESP32 HMI 4MB/CH340/ST7789 170x320.
// Compile-time options: IR/RED and RAW/HPF/HPF+LPF (PlatformIO environments).
// Acquisition: both RED and IR, 100 Hz, 411 us LED pulses, no hardware averaging.
// Plot: RAW, 2nd-order Tustin Butterworth HPF, or HPF+LPF (unchanged).
// BOOT GPIO0: capture 10s+ raw RED+IR to RAM, then CSV to microSD.
// A pinned core-0 task runs Noodle2 INT8 PeakNet on inverted RAW IR windows.
// Live LCD/100Hz MAX30102 FIFO acquisition stays on Arduino core 1.

#include <Arduino.h>
#include <Wire.h>
#include <TFT_eSPI.h>
#include <MAX30105.h> // SparkFun's MAX3010x library also supports MAX30102.
#include <SPI.h>
#include <SD.h>
#include <math.h>
#include <string.h>
#include <Noodle2.h>
#include "peaknet_model_flash.h"
#include "peaknet_live.h"
#ifndef PPG_DIAG_MODE
#define PPG_DIAG_MODE 0
#endif
#if PPG_DIAG_MODE
#include "peaknet_golden.h" // frozen input/output: same computation in every test
#endif
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "butterworth_hpf2.h"
#include "butterworth_lpf2.h"

#if defined(PLOT_RED) && defined(PLOT_IR)
#error "Choose either PLOT_RED or PLOT_IR, not both"
#endif
#if !defined(PLOT_RED) && !defined(PLOT_IR)
#error "Choose a PlatformIO IR or RED environment"
#endif

// Diagnostic mode 0 = original live SPPG; 1 = core0 quiet; 2 = core0 sensor;
// 3 = core0 sensor + LCD; 4 = core1 quiet. Never changes Noodle2 kernels.
#if PPG_DIAG_MODE < 0 || PPG_DIAG_MODE > 4
#error "PPG_DIAG_MODE must be 0..4"
#endif

#ifndef PPG_HPF
#define PPG_HPF 1
#endif
#ifndef PPG_HPF_CUTOFF_HZ
#define PPG_HPF_CUTOFF_HZ 0.50f
#endif
#ifndef PPG_LPF
#define PPG_LPF 1
#endif
#ifndef PPG_LPF_CUTOFF_HZ
#define PPG_LPF_CUTOFF_HZ 6.0f
#endif
#if PPG_LPF && !PPG_HPF
#error "PPG_LPF requires PPG_HPF: use RAW, HPF, or HPF+LPF modes"
#endif
#ifndef SERIAL_RAW
#define SERIAL_RAW 1
#endif
#ifndef PPG_NN_CONV_TILE
#define PPG_NN_CONV_TILE NOODLE2_DEFAULT_CONV_TILE // Same kernel as graph.run() default: tiled 4x4
#endif
#ifndef PPG_ENABLE_NN
#define PPG_ENABLE_NN 1
#endif
#ifndef PPG_RECORD_SECONDS
#define PPG_RECORD_SECONDS 10
#endif
#if PPG_RECORD_SECONDS < 10
#error "PPG_RECORD_SECONDS must be at least 10 seconds"
#endif

namespace {
constexpr int PIN_SDA = 4;
constexpr int PIN_SCL = 5;
constexpr int PIN_LCD_BL = 21; // LCD backlight on classic ESP32 HMI board
constexpr int PIN_BOOT = 0;     // Physical BOOT button (active-low)
// microSD on HSPI: dedicated peripheral, independent of the TFT_eSPI bus.
constexpr int PIN_SD_SCK = 18;
constexpr int PIN_SD_MISO = 19;
constexpr int PIN_SD_MOSI = 23;
constexpr int PIN_SD_CS = 27;
constexpr uint32_t SD_SPI_HZ = 4000000;
constexpr uint8_t MAX30102_ADDRESS = 0x57;

// These arguments directly program MAX30102 registers via sensor.setup().
constexpr uint8_t LED_AMPLITUDE = 0x1F; // ~6.2 mA nominal per LED
constexpr uint8_t FIFO_AVERAGE = 1;   // no hardware averaging
constexpr uint8_t LED_MODE = 2;       // RED + IR each sample period
constexpr int SAMPLE_RATE_HZ = 100;
constexpr int LED_PULSE_WIDTH_US = 411; // 18-bit ADC resolution
constexpr int ADC_RANGE_NA = 8192;     // photodiode current full scale
constexpr unsigned long REFRESH_MS = 75; // ~13 FPS normal LCD redraw
constexpr unsigned long RECORD_REFRESH_MS = 150; // ~7 FPS during capture; prioritize FIFO
constexpr size_t DRAW_POLL_INTERVAL = 12; // Service sensor during long TFT redraws

constexpr int SCREEN_W = 320;          // landscape setRotation(1)
constexpr int SCREEN_H = 170;
constexpr int PLOT_X = 8;
constexpr int PLOT_Y = 50;
constexpr int PLOT_W = SCREEN_W - 16; // 304 samples = 3.04 s @ 100 Hz
constexpr int PLOT_H = 98;
constexpr float PI_F = 3.14159265358979323846f;
constexpr size_t NN_EVENT_MEMORY = 32;
constexpr uint32_t NN_TASK_STACK_BYTES = 8192;
constexpr uint32_t NN_LOG_EVERY = PPG_DIAG_MODE ? 1u : 6u;
constexpr bool DIAG_SENSOR_ON = (PPG_DIAG_MODE == 0 || PPG_DIAG_MODE == 2 || PPG_DIAG_MODE == 3);
constexpr bool DIAG_LCD_ON = (PPG_DIAG_MODE == 0 || PPG_DIAG_MODE == 3);
constexpr bool DIAG_SD_ON = (PPG_DIAG_MODE == 0 || PPG_DIAG_MODE == 3);
constexpr BaseType_t NN_WORKER_CORE = (PPG_DIAG_MODE == 4) ? 1 : 0;
constexpr uint32_t DIAG_PERIOD_MS = 4500; // no backlogs in our ~3.2s worst-case
constexpr uint32_t DIAG_JOBS = 5;
static uint32_t diagYieldUs = 0; // written and read on the NN task only

constexpr uint8_t NN_QUEUE_LEN = 1; // latest-window-wins, never block acquisition

// Tustin/Bilinear transform with digital cutoff prewarping:
// q = Ts*omega_prewarped = 2*tan(pi*fc/fs).
// The two generated filter classes are in include/*.h.

#if defined(PLOT_RED)
constexpr const char *CHANNEL_NAME = "RED";
constexpr uint16_t TRACE_COLOR = TFT_RED;
#else
constexpr const char *CHANNEL_NAME = "IR";
constexpr uint16_t TRACE_COLOR = TFT_CYAN;
#endif

TFT_eSPI tft;
MAX30105 sensor;
SPIClass sdSPI(HSPI);

// 1001 samples at 100Hz represent a nominal 10.00s from first to last.
// Capture only integer, unfiltered data. No SD writes during acquisition.
constexpr size_t RECORD_SAMPLE_COUNT =
    static_cast<size_t>(SAMPLE_RATE_HZ) * PPG_RECORD_SECONDS + 1;
static_assert(RECORD_SAMPLE_COUNT <= 5000, "Capture exceeds RAM budget; reduce recording length");
struct RawRecord {
    uint32_t red;
    uint32_t ir;
    uint32_t readElapsedUs; // Host FIFO read timestamp, NOT ADC conversion timestamp.
};
RawRecord recorded[RECORD_SAMPLE_COUNT];

enum class CaptureState : uint8_t { Idle, Recording, SavePending };
CaptureState captureState = CaptureState::Idle;
size_t recordedCount = 0;
uint32_t firstRecordUs = 0;
uint32_t longFifoBurstsDuringCapture = 0;
bool sdReady = false;
char recordingStatus[48] = {};
uint32_t recordingStatusUntilMs = 0;
bool bootLastRead = HIGH;
bool bootStableRead = HIGH;
uint32_t bootChangedAtMs = 0;
constexpr uint32_t BOOT_DEBOUNCE_MS = 40;

// 2nd-order Butterworth HPF generated from digital-filter-2.ipynb.
// The first raw sample establishes a fixed reference so that the
// float IIR biquad processes small AC values, not the large PPG DC offset.
// This also retains the original no-startup-spike behavior.
struct DCBlocker {
    ButterworthHPF2 filter;
    float baseline = 0.0f;
    bool initialized = false;

    void configure(float fs_hz, float fc_hz) {
        const float Ts = 1.0f / fs_hz;
        const float wc = (2.0f / Ts) * tanf(PI_F * fc_hz / fs_hz);
        filter.configure(Ts, wc);
        initialized = false;
        baseline = 0.0f;
    }

    float apply(uint32_t sample) {
        const float x = static_cast<float>(sample);
        if (!initialized) {
            initialized = true;
            baseline = x;
        }
        return filter.step(x - baseline);
    }
};

// 2nd-order Butterworth LPF generated from digital-filter.ipynb.
struct LowPass {
    ButterworthLPF2 filter;

    bool configure(float fs_hz, float fc_hz) {
        return filter.configure(fs_hz, fc_hz);
    }

    float apply(float x) { return filter.step(x); }
};

DCBlocker redBlocker;
DCBlocker irBlocker;
LowPass redLowPass;
LowPass irLowPass;
float history[PLOT_W] = {};
// Frozen frame snapshot: sensor polling during TFT drawing must not move the trace.
float plotSnapshot[PLOT_W] = {};
size_t writeIndex = 0;
size_t sampleCount = 0;
uint32_t latestRaw = 0;
float latestAC = 0.0f;
float latestFiltered = 0.0f;
uint32_t sampleIndex = 0;
uint32_t longFifoBursts = 0;
unsigned long lastRefresh = 0;
// Separate, latest-only inference stream. The task receives a 256-sample
// RAW IR snapshot. It never touches the I2C sensor, TFT, SD or Tustin filters.
struct NNWindow {
    uint32_t ir[peaknet_live::WINDOW];
    uint32_t startSample;
    uint32_t generation;
};
static uint32_t nnRawRing[peaknet_live::WINDOW] = {};
static uint16_t nnRingWrite = 0;
static uint32_t nnSamplesSinceReset = 0;
static QueueHandle_t nnQueue = nullptr;
static bool nnReady = false;
static noodle2::NoodleGraph nnGraph;
static peaknet_ppg::Model nnModel;
static portMUX_TYPE nnGuard = portMUX_INITIALIZER_UNLOCKED;
static volatile uint32_t nnGeneration = 0;
struct NNShared {
    uint32_t completed = 0;
    uint32_t enqueued = 0;
    uint32_t overwritten = 0;
    uint32_t inferenceUs = 0;
    uint32_t lastInputStart = 0;
    uint8_t candidatesSys = 0;
    uint8_t candidatesDia = 0;
    int errorCode = 0;
    size_t eventCount = 0;
    peaknet_live::Candidate events[NN_EVENT_MEMORY] = {};
};
static NNShared nnShared;

void nnInvalidate() {
#if PPG_ENABLE_NN
    if (!nnReady) return;
    portENTER_CRITICAL(&nnGuard);
    ++nnGeneration; // Any still-running window becomes stale.
    nnShared.eventCount = 0;
    nnShared.inferenceUs = 0;
    nnShared.completed = 0;
    nnShared.candidatesSys = nnShared.candidatesDia = 0;
    portEXIT_CRITICAL(&nnGuard);
    nnRingWrite = 0;
    nnSamplesSinceReset = 0;
    if(nnQueue) xQueueReset(nnQueue);
#endif
}

void nnOnRawIR(uint32_t rawIr, uint32_t globalSampleIndex) {
#if PPG_ENABLE_NN
#if PPG_DIAG_MODE
    // In tests A-D the queue is paced by loop(); the sensor still works in B/C.
    // For comparable NN workloads all four modes use golden input[0].
    (void)rawIr; (void)globalSampleIndex;
    return;
#endif
    if(!nnReady || !nnQueue) return;
    nnRawRing[nnRingWrite] = rawIr;
    nnRingWrite = uint16_t((nnRingWrite+1u) % peaknet_live::WINDOW);
    ++nnSamplesSinceReset;
    if(nnSamplesSinceReset < peaknet_live::WINDOW ||
      ((nnSamplesSinceReset-peaknet_live::WINDOW) % peaknet_live::HOP)!=0)
        return;
    NNWindow window = {};
    for(size_t i=0;i<peaknet_live::WINDOW;i++)
        window.ir[i]=nnRawRing[(nnRingWrite+i) % peaknet_live::WINDOW];
    window.startSample=globalSampleIndex-(peaknet_live::WINDOW-1);
    const bool overwriting=uxQueueMessagesWaiting(nnQueue)>0;
    portENTER_CRITICAL(&nnGuard);
    window.generation=nnGeneration;
    nnShared.enqueued++;
    if(overwriting) nnShared.overwritten++;
    portEXIT_CRITICAL(&nnGuard);
    // xQueueOverwrite on a queue of length one replaces stale queued work.
    (void)xQueueOverwrite(nnQueue,&window);
#else
    (void)rawIr;(void)globalSampleIndex;
#endif
}

uint32_t nnMicros() { return micros(); }

// Node boundaries are safe points to let core-0 IDLE0 run and feed TWDT.
// A yield only after graph.run() is too late on a ~3.16s inference.
void nnYieldBetweenNodes() {
#if PPG_DIAG_MODE
    const uint32_t start=micros();
#endif
    vTaskDelay(pdMS_TO_TICKS(10));
#if PPG_DIAG_MODE
    diagYieldUs += micros()-start; // includes any scheduling wait after the delay
#endif
}

void nnTask(void*) {
#if PPG_ENABLE_NN
    NNWindow window = {};
    int8_t input[peaknet_live::WINDOW] = {};
    int8_t output[2*peaknet_live::WINDOW] = {};
    uint32_t nodeUs[7] = {};
    uint32_t windows = 0;
    for(;;) {
        if(xQueueReceive(nnQueue,&window,portMAX_DELAY)!=pdTRUE) continue;
        // RAW IR is inverted and z-scored INDEPENDENTLY per window, exactly
        // as in the Python training data; do not feed Tustin display output.
        float rawStd=0;
#if PPG_DIAG_MODE
        // The exact same INT8 vector in A, B, C and D; no I2C/input variability.
        memcpy(input,peaknet_golden::input[0],sizeof(input));
#else
        peaknet_live::quantize_ir(window.ir,peaknet_ppg::input_scale,input,&rawStd);
        if(rawStd < 10.0f) continue; // No plausible contact signal; not physiology.
#endif
        noodle2::Status st=nnGraph.set_input(input);
        if(st!=noodle2::OK) {
            portENTER_CRITICAL(&nnGuard);nnShared.errorCode=int(st);portEXIT_CRITICAL(&nnGuard);
            continue;
        }
        memset(nodeUs,0,sizeof(nodeUs));
#if PPG_DIAG_MODE
        diagYieldUs=0;
#endif
        const uint32_t startUs=micros();
        st=nnGraph.run(nodeUs,nnMicros,PPG_NN_CONV_TILE,nullptr,nnYieldBetweenNodes);
        const uint32_t elapsedUs=micros()-startUs;
        if(st!=noodle2::OK) {
            portENTER_CRITICAL(&nnGuard);nnShared.errorCode=int(st);portEXIT_CRITICAL(&nnGuard);
            continue;
        }
        bool good=true;
        for(unsigned i=0;i<2*peaknet_live::WINDOW;i++)
            if(nnGraph.output().get(i,output[i])!=noodle2::OK){good=false;break;}
        if(!good){portENTER_CRITICAL(&nnGuard);nnShared.errorCode=-1;portEXIT_CRITICAL(&nnGuard);continue;}
        peaknet_live::Candidate found[12] = {};
        const size_t foundCount=peaknet_live::extract_candidates(
            output,window.startSample,found,sizeof(found)/sizeof(found[0]));
        uint8_t foundS=0,foundD=0;
        for(size_t i=0;i<foundCount;i++) (found[i].channel?foundD:foundS)++;
        bool accepted=false;
        uint32_t overwrittenAtWindow=0;
        portENTER_CRITICAL(&nnGuard);
        if(window.generation==nnGeneration) {
            nnShared.completed++;
            nnShared.inferenceUs=elapsedUs;
            nnShared.lastInputStart=window.startSample;
            nnShared.candidatesSys=foundS;
            nnShared.candidatesDia=foundD;
            nnShared.errorCode=0;
            overwrittenAtWindow=nnShared.overwritten;
            // Event memory is intentionally a small rolling log for the LCD.
            for(size_t i=0;i<foundCount;i++) {
                // Merge nearly coincident candidates across hop boundaries.
                // The two classes are handled independently.
                bool duplicate=false;
                for(size_t j=nnShared.eventCount;j>0;j--) {
                    peaknet_live::Candidate &old=nnShared.events[j-1];
                    if(old.channel!=found[i].channel) continue;
                    if(found[i].sample_idx>=old.sample_idx &&
                       found[i].sample_idx-old.sample_idx<peaknet_live::MIN_SEPARATION) {
                        if(found[i].logit>old.logit) old=found[i];
                        duplicate=true;
                    }
                    break;
                }
                if(duplicate) continue;
                if(nnShared.eventCount==NN_EVENT_MEMORY) {
                    memmove(nnShared.events,nnShared.events+1,
                        sizeof(nnShared.events[0])*(NN_EVENT_MEMORY-1));
                    nnShared.eventCount--;
                }
                nnShared.events[nnShared.eventCount++]=found[i];
            }
            accepted=true;
        }
        portEXIT_CRITICAL(&nnGuard);
        if(!accepted) continue;
        ++windows;
        if(windows==1 || windows%NN_LOG_EVERY==0) {
#if PPG_DIAG_MODE
            uint32_t layerSum=0;
            for(int i=1;i<=6;i++) layerSum+=nodeUs[i];
            unsigned mismatches=0;
            for(unsigned i=0;i<2*peaknet_live::WINDOW;i++)
                if(output[i]!=peaknet_golden::output[0][i]) ++mismatches;
            // Per-node times are WALL times (interrupts/preemption can occur
            // inside a layer). yield_us is wall time inside deliberate yields.
            // residual_us includes graph overhead and any other scheduling.
            const uint32_t residual=elapsedUs>layerSum+diagYieldUs ?
                elapsedUs-layerSum-diagYieldUs:0;
            Serial.printf("# DIAG mode=%d job=%lu core=%d mhz=%u tile=%u "
                "wall_us=%lu layer_us=%lu yield_us=%lu residual_us=%lu "
                "mismatch=%u drop=%lu stack_words=%lu "
                "L1=%lu L2=%lu L3=%lu L4=%lu L5=%lu L6=%lu\n",
                int(PPG_DIAG_MODE),(unsigned long)windows,int(xPortGetCoreID()),
                unsigned(ESP.getCpuFreqMHz()),unsigned(PPG_NN_CONV_TILE),
                (unsigned long)elapsedUs,
                (unsigned long)layerSum,(unsigned long)diagYieldUs,
                (unsigned long)residual,mismatches,
                (unsigned long)overwrittenAtWindow,
                (unsigned long)uxTaskGetStackHighWaterMark(nullptr),
                (unsigned long)nodeUs[1],(unsigned long)nodeUs[2],
                (unsigned long)nodeUs[3],(unsigned long)nodeUs[4],
                (unsigned long)nodeUs[5],(unsigned long)nodeUs[6]);
#else
            Serial.printf("# NN win=%lu t=%luus S?=%u D?=%u raw_std=%.1f tile=%u drop=%lu "
                "L1=%lu L2=%lu L3=%lu L4=%lu L5=%lu L6=%lu\n",
                (unsigned long)windows,(unsigned long)elapsedUs,
                unsigned(foundS),unsigned(foundD),double(rawStd),unsigned(PPG_NN_CONV_TILE),
                (unsigned long)overwrittenAtWindow,
                (unsigned long)nodeUs[1],(unsigned long)nodeUs[2],
                (unsigned long)nodeUs[3],(unsigned long)nodeUs[4],
                (unsigned long)nodeUs[5],(unsigned long)nodeUs[6]);
#endif
        }
#if PPG_DIAG_MODE
        if(windows==DIAG_JOBS) Serial.println("# DIAG DONE: collect five result lines, then flash next mode");
#endif
        vTaskDelay(pdMS_TO_TICKS(20)); // Also yield between whole windows.
    }
#else
    vTaskDelete(nullptr);
#endif
}

bool startNN() {
#if PPG_ENABLE_NN
    noodle2::Status st=nnModel.bind_flash();
    if(st==noodle2::OK) st=peaknet_ppg::build_graph(nnGraph,nnModel);
    if(st!=noodle2::OK) {
        Serial.printf("# NN init FAILED status=%u\n",unsigned(st));
        return false;
    }
    // One diagnostic pass in setup (core 1, no continuous LCD redraw).
    // Compare against the separate ~977ms benchmark and live task times.
    // Clear MAX30102 FIFO afterward: sampling occurred during this test.
    int8_t baseline[peaknet_live::WINDOW] = {};
    if(nnGraph.set_input(baseline)!=noodle2::OK) return false;
    const uint32_t baselineTick=micros();
    const noodle2::Status baselineStatus=nnGraph.run(nullptr, nullptr, PPG_NN_CONV_TILE);
    const uint32_t baselineUs=micros()-baselineTick;
    Serial.printf("# NN setup_baseline_us=%lu status=%d cpu_mhz=%u tile=%u (no continuous drawing)\n",
                  static_cast<unsigned long>(baselineUs),int(baselineStatus),
                  unsigned(ESP.getCpuFreqMHz()),unsigned(PPG_NN_CONV_TILE));
    if (DIAG_SENSOR_ON) sensor.clearFIFO();
    if(baselineStatus!=noodle2::OK) return false;

    nnQueue=xQueueCreate(NN_QUEUE_LEN,sizeof(NNWindow));
    if(!nnQueue){Serial.println("# NN queue allocation FAILED");return false;}
    // A-D: use an identical worker task, priority and model on each selected core.
    const BaseType_t ok=xTaskCreatePinnedToCore(nnTask,"peaknet",NN_TASK_STACK_BYTES,
                                               nullptr,1,nullptr,NN_WORKER_CORE);
    if(ok!=pdPASS) {
        Serial.println("# NN task allocation FAILED");
        vQueueDelete(nnQueue);nnQueue=nullptr;return false;
    }
    Serial.printf("# NN started core%d arena=%u params=%u nodes=%u hop=%u "
                  "input=%s tile=%u\n",int(NN_WORKER_CORE),unsigned(nnGraph.arena_bytes()),
                  unsigned(peaknet_ppg::parameter_bytes),unsigned(nnGraph.node_count()),
                  unsigned(peaknet_live::HOP),
                  PPG_DIAG_MODE ? "golden_fixed" : "-rawIR/zscore",
                  unsigned(PPG_NN_CONV_TILE));
    return true;
#else
    return false;
#endif
}


// Sensor FIFO servicing is also called *during* lengthy TFT waveform redraws.
void serviceSensor();
void drawCaptureStatus(const char *message);

void captureRawSample(uint32_t red, uint32_t ir) {
    if (captureState != CaptureState::Recording) return;
    RawRecord &rec = recorded[recordedCount];
    const uint32_t nowUs = micros();
    if (recordedCount == 0) firstRecordUs = nowUs;
    rec.red = red;
    rec.ir = ir;
    rec.readElapsedUs = nowUs - firstRecordUs;
    ++recordedCount;
    if (recordedCount == RECORD_SAMPLE_COUNT)
        captureState = CaptureState::SavePending;
}

void processSample(uint32_t red, uint32_t ir) {
    // Store original ADC readings before any HPF/LPF processing.
    captureRawSample(red, ir);
    nnOnRawIR(ir, sampleIndex);
    const float redAC = redBlocker.apply(red);
    const float irAC = irBlocker.apply(ir);
    // Always compute both filtered channels; display mode does not alter CSV.
    const float redBand = redLowPass.apply(redAC);
    const float irBand = irLowPass.apply(irAC);
    (void)redBand; (void)irBand; // Retain both filters even with serial CSV disabled.
#if defined(PLOT_RED)
    latestRaw = red;
    latestAC = redAC;
    latestFiltered = redBand;
#else
    latestRaw = ir;
    latestAC = irAC;
    latestFiltered = irBand;
#endif

#if PPG_LPF
    history[writeIndex] = -latestFiltered;
#elif PPG_HPF
    history[writeIndex] = -latestAC;
#else
    history[writeIndex] = -static_cast<float>(latestRaw);
#endif

    writeIndex = (writeIndex + 1) % PLOT_W;
    if (sampleCount < PLOT_W) ++sampleCount;

#if SERIAL_RAW
    // Avoid serial transmission while recording so 100Hz FIFO service stays prompt.
    if (captureState == CaptureState::Idle) {
    // The first three columns are compatible with the original CSV export.
    // time_ms is host read time; sample_idx provides the nominal 100-Hz axis.
    Serial.printf("%lu,%lu,%lu,%.2f,%.2f,%lu,%.2f,%.2f\n",
                  static_cast<unsigned long>(millis()),
                  static_cast<unsigned long>(red),
                  static_cast<unsigned long>(ir),
                  static_cast<double>(redAC),
                  static_cast<double>(irAC),
                  static_cast<unsigned long>(sampleIndex),
                  static_cast<double>(redBand),
                  static_cast<double>(irBand));
    }
#endif
    ++sampleIndex;
}

void snapshotPlotHistory() {
    const size_t start = (writeIndex + PLOT_W - sampleCount) % PLOT_W;
    for (size_t i = 0; i < sampleCount; ++i)
        plotSnapshot[i] = history[(start + i) % PLOT_W];
}

void drawHeader() {
    char title[44];
    tft.fillScreen(TFT_BLACK);
    tft.setTextFont(2);
    tft.setTextSize(1);
    tft.setTextColor(TFT_WHITE, TFT_BLACK);
    tft.drawString("SPPG + Noodle2", 8, 4);
    tft.setTextColor(TRACE_COLOR, TFT_BLACK);
#if PPG_LPF
    snprintf(title, sizeof(title), "%s HP %.1f LP %.0fHz", CHANNEL_NAME,
             static_cast<double>(PPG_HPF_CUTOFF_HZ),
             static_cast<double>(PPG_LPF_CUTOFF_HZ));
#elif PPG_HPF
    snprintf(title, sizeof(title), "%s HPF %.2fHz", CHANNEL_NAME,
             static_cast<double>(PPG_HPF_CUTOFF_HZ));
#else
    snprintf(title, sizeof(title), "%s RAW", CHANNEL_NAME);
#endif
    tft.drawRightString(title, SCREEN_W - 8, 4, 2);
    tft.drawFastHLine(8, 45, SCREEN_W - 16, TFT_DARKGREY);
}

void drawWaveform() {
    const size_t frameSamples = sampleCount;
    const uint32_t frameNextSampleIndex=sampleIndex; // freeze marker mapping with trace
    snapshotPlotHistory();
    const uint32_t frameRaw = latestRaw;
    const float frameAC = latestAC;
    const float frameFiltered = latestFiltered;
    char line[68];
    tft.fillRect(8, 24, SCREEN_W - 16, 18, TFT_BLACK);
    tft.setTextFont(2);
    tft.setTextColor(TRACE_COLOR, TFT_BLACK);
    snprintf(line, sizeof(line), "Raw:%lu", static_cast<unsigned long>(frameRaw));
    tft.drawString(line, 8, 25);
    tft.setTextColor(TFT_LIGHTGREY, TFT_BLACK);
    snprintf(line, sizeof(line), "AC:%+.0f LP:%+.0f",
             static_cast<double>(frameAC), static_cast<double>(frameFiltered));
    tft.drawRightString(line, SCREEN_W - 8, 25, 2);

    // Marker labels in older frames could protrude into the unused gutters.
    // Clear those thin margins too, so no stray pixels remain at the edges.
    tft.fillRect(0, PLOT_Y, PLOT_X, PLOT_H, TFT_BLACK);
    tft.fillRect(PLOT_X, PLOT_Y, PLOT_W, PLOT_H, TFT_BLACK);
    tft.fillRect(PLOT_X + PLOT_W, PLOT_Y,
                 SCREEN_W - (PLOT_X + PLOT_W), PLOT_H, TFT_BLACK);
    tft.drawFastHLine(PLOT_X, PLOT_Y, PLOT_W, TFT_DARKGREY);
    tft.drawFastHLine(PLOT_X, PLOT_Y + PLOT_H - 1, PLOT_W, TFT_DARKGREY);
    if (!frameSamples) return;

    float minimum = plotSnapshot[0];
    float maximum = minimum;
    for (size_t i = 1; i < frameSamples; ++i) {
        const float v = plotSnapshot[i];
        if (v < minimum) minimum = v;
        if (v > maximum) maximum = v;
    }

#if PPG_HPF
    // Symmetric Y-range, making the physical AC=0 level the center line.
    float amplitude = fmaxf(fabsf(minimum), fabsf(maximum));
    if (amplitude < 1.0f) amplitude = 1.0f;
    const float low = -1.1f * amplitude;
    const float high = 1.1f * amplitude;
#else
    // Auto-scale unaltered ADC samples: visualization only, no filtering.
    const float span = maximum - minimum;
    const float margin = fmaxf(span * 0.10f, 1.0f);
    const float low = minimum - margin;
    const float high = maximum + margin;
#endif
    const float range = high - low;
    tft.drawFastHLine(PLOT_X, PLOT_Y + PLOT_H / 2, PLOT_W, TFT_DARKGREY);

    const int xStart = PLOT_X + PLOT_W - static_cast<int>(frameSamples);
    int lastX = 0;
    int lastY = 0;
    for (size_t i = 0; i < frameSamples; ++i) {
        // Cooperative acquisition: no large unserviced SPI drawing interval.
        // MAX3010x library buffers just four FIFO records internally.
        if (i % DRAW_POLL_INTERVAL == 0) serviceSensor();
        const float v = plotSnapshot[i];
        const int x = xStart + static_cast<int>(i);
        float yNormalized = (v - low) / range;
        if (yNormalized < 0.0f) yNormalized = 0.0f;
        if (yNormalized > 1.0f) yNormalized = 1.0f;
        const int y = PLOT_Y + PLOT_H - 3 -
                      static_cast<int>(yNormalized * (PLOT_H - 5));
        if (i) tft.drawLine(lastX, lastY, x, y, TRACE_COLOR);
        lastX = x;
        lastY = y;
    }
    if (frameSamples == 1) tft.drawPixel(lastX, lastY, TRACE_COLOR);
    serviceSensor(); // Flush FIFO before updating the small footer.

#if PPG_ENABLE_NN && !PPG_DIAG_MODE
    // Prediction markers arrive after ~1s inference; S?/D? are tentative
    // classes, NOT validated systolic/diastolic landmarks on real PPG.
    peaknet_live::Candidate ev[NN_EVENT_MEMORY] = {};
    size_t nev=0;
    portENTER_CRITICAL(&nnGuard);
    nev=nnShared.eventCount;
    memcpy(ev,nnShared.events,nev*sizeof(ev[0]));
    portEXIT_CRITICAL(&nnGuard);
    const uint32_t frameStart=frameNextSampleIndex>=frameSamples ?
        frameNextSampleIndex-frameSamples : 0;
    for(size_t i=0;i<nev;i++) {
        if(ev[i].sample_idx<frameStart ||
           ev[i].sample_idx>=frameStart+frameSamples) continue;
        const int x=xStart+int(ev[i].sample_idx-frameStart);
        const bool dia=ev[i].channel!=0;
        const uint16_t col=dia?TFT_MAGENTA:TFT_YELLOW;
        const int yy=PLOT_Y+(dia?19:4);
        tft.drawFastVLine(x,yy,8,col);
        tft.setTextFont(1);
        tft.setTextColor(col,TFT_BLACK);
        // Font 1 is 6 pixels wide per character at text size 1.
        // Keep both characters within the plot, including at wraparound.
        constexpr int LABEL_W = 12;
        int labelX = x - 3;
        if (labelX < PLOT_X) labelX = PLOT_X;
        if (labelX > PLOT_X + PLOT_W - LABEL_W)
            labelX = PLOT_X + PLOT_W - LABEL_W;
        tft.drawString(dia?"D?":"S?",labelX,yy+8);
    }
#endif

    tft.fillRect(8, 153, SCREEN_W - 16, SCREEN_H - 153, TFT_BLACK);
    tft.setTextFont(1);
    tft.setTextColor(TFT_LIGHTGREY, TFT_BLACK);
#if PPG_LPF
    snprintf(line, sizeof(line), "HP+LP min:%+.0f max:%+.0f  ~3s",
             static_cast<double>(minimum), static_cast<double>(maximum));
#elif PPG_HPF
    snprintf(line, sizeof(line), "AC min:%+.0f max:%+.0f    ~3.0s",
             static_cast<double>(minimum), static_cast<double>(maximum));
#else
    snprintf(line, sizeof(line), "Raw min:%.0f max:%.0f   ~3.0s",
             static_cast<double>(minimum), static_cast<double>(maximum));
#endif
    if (captureState == CaptureState::Recording) {
        // The waveform keeps updating; status has priority over min/max text.
        const unsigned seconds = static_cast<unsigned>(
            recordedCount ? (recordedCount - 1) / SAMPLE_RATE_HZ : 0);
        snprintf(line, sizeof(line), "REC %u/%ds  %u/%u samples",
                 seconds, PPG_RECORD_SECONDS, static_cast<unsigned>(recordedCount),
                 static_cast<unsigned>(RECORD_SAMPLE_COUNT));
        tft.setTextColor(TFT_YELLOW, TFT_BLACK);
        tft.drawString(line, 8, 157);
    } else if (recordingStatus[0] &&
               static_cast<int32_t>(recordingStatusUntilMs - millis()) > 0) {
        tft.setTextColor(TFT_YELLOW, TFT_BLACK);
        tft.drawString(recordingStatus, 8, 157);
    } else {
#if PPG_ENABLE_NN
        uint32_t lastUs=0,completed=0,dropped=0;
        uint8_t sys=0,dia=0;
        int error=0;
        portENTER_CRITICAL(&nnGuard);
        lastUs=nnShared.inferenceUs;completed=nnShared.completed;
        dropped=nnShared.overwritten;
        sys=nnShared.candidatesSys;dia=nnShared.candidatesDia;
        error=nnShared.errorCode;
        portEXIT_CRITICAL(&nnGuard);
        if(error)
            snprintf(line,sizeof(line),"NN error=%d; see Serial",error);
        else if(completed) {
#if PPG_DIAG_MODE
            snprintf(line,sizeof(line),"DIAG %d NN %lums",int(PPG_DIAG_MODE),
                     (unsigned long)(lastUs/1000u));
#else
            snprintf(line,sizeof(line),"NN %lums S?%u D?%u drop%lu",
                (unsigned long)(lastUs/1000u),unsigned(sys),unsigned(dia),
                (unsigned long)dropped);
#endif
        }
        else
            snprintf(line,sizeof(line),"NN warming up... (IR)");
#endif
        tft.drawString(line, 8, 157);
    }
}

// Status during SD saving (the sensor is deliberately paused during SD I/O).
void drawCaptureStatus(const char *message) {
    tft.fillRect(8, 153, SCREEN_W - 16, SCREEN_H - 153, TFT_BLACK);
    tft.setTextFont(1);
    tft.setTextColor(TFT_YELLOW, TFT_BLACK);
    tft.drawString(message, 8, 157);
}

void setTemporaryStatus(const char *message) {
    snprintf(recordingStatus, sizeof(recordingStatus), "%s", message);
    recordingStatusUntilMs = millis() + 4000;
    drawCaptureStatus(recordingStatus);
}

void beginCapture() {
    if (!sdReady) {
        Serial.println("# BOOT ignored: no microSD detected; insert card and reboot");
        setTemporaryStatus("NO SD - insert card & reboot");
        return;
    }
    if (captureState != CaptureState::Idle) return;
    recordedCount = 0;
    longFifoBurstsDuringCapture = 0;
    recordingStatus[0] = '\0';
    captureState = CaptureState::Recording;
    lastRefresh = millis() - RECORD_REFRESH_MS; // draw the first live REC frame promptly
    Serial.printf("# REC begin: %d seconds, %u raw RED+IR samples\n",
                  PPG_RECORD_SECONDS, static_cast<unsigned>(RECORD_SAMPLE_COUNT));
    char msg[42];
    snprintf(msg, sizeof(msg), "REC 0s / %ds (RAM)", PPG_RECORD_SECONDS);
    drawCaptureStatus(msg);
}

void pollBootButton() {
    const bool level = digitalRead(PIN_BOOT);
    const uint32_t now = millis();
    if (level != bootLastRead) {
        bootLastRead = level;
        bootChangedAtMs = now;
    }
    if (now - bootChangedAtMs >= BOOT_DEBOUNCE_MS && level != bootStableRead) {
        bootStableRead = level;
        if (level == LOW) beginCapture(); // One capture per press, not per hold.
    }
}

// Called ONLY when acquisition has completed and the sensor is shut down.
// Return false if there is no free filename, card failure or short write.
bool writeRecordedCSV(char *writtenName, size_t nameCapacity) {
    char name[20] = {};
    bool freeName = false;
    for (unsigned n = 1; n <= 9999; ++n) {
        snprintf(name, sizeof(name), "/PPG%04u.CSV", n);
        if (!SD.exists(name)) { freeName = true; break; }
    }
    if (!freeName) return false;
    File out = SD.open(name, FILE_WRITE);
    if (!out) return false;

    // Each row is nominally 10ms apart; readElapsedUs measures host FIFO reads.
    // Header metadata preserves sensor settings and captures possible FIFO stalls.
    char line[116];
    bool ok = true;
    auto emit = [&](const char *str) {
        if (!ok) return;
        const size_t len = strlen(str);
        if (out.write(reinterpret_cast<const uint8_t *>(str), len) != len)
            ok = false;
    };
    emit("# sensor=MAX30102 channels=RED+IR raw_adc=1\n");
    snprintf(line, sizeof(line), "# fs_hz=%d pulse_width_us=%d led_amplitude=0x%02X fifo_average=%u adc_range_na=%d\n",
             SAMPLE_RATE_HZ, LED_PULSE_WIDTH_US, LED_AMPLITUDE, FIFO_AVERAGE, ADC_RANGE_NA);
    emit(line);
    snprintf(line, sizeof(line),
             "# display=%s hpf=%d hpf_hz=%.3f lpf=%d lpf_hz=%.3f\n",
             CHANNEL_NAME, PPG_HPF, static_cast<double>(PPG_HPF_CUTOFF_HZ),
             PPG_LPF, static_cast<double>(PPG_LPF_CUTOFF_HZ));
    emit(line);
    snprintf(line, sizeof(line), "# capture_seconds=%d samples=%u fifo_bursts_ge4=%lu\n",
             PPG_RECORD_SECONDS, static_cast<unsigned>(recordedCount),
             static_cast<unsigned long>(longFifoBurstsDuringCapture));
    emit(line);
    emit("# read_elapsed_us is the ESP32 FIFO read time, NOT a hardware ADC timestamp\n");
    emit("sample_idx,nominal_time_ms,read_elapsed_us,red,ir\n");
    for (size_t i = 0; ok && i < recordedCount; ++i) {
        snprintf(line, sizeof(line), "%u,%lu,%lu,%lu,%lu\n",
                 static_cast<unsigned>(i),
                 static_cast<unsigned long>((i * 1000UL) / SAMPLE_RATE_HZ),
                 static_cast<unsigned long>(recorded[i].readElapsedUs),
                 static_cast<unsigned long>(recorded[i].red),
                 static_cast<unsigned long>(recorded[i].ir));
        emit(line);
    }
    out.flush();
    out.close();
    if (!ok) { SD.remove(name); return false; }
    snprintf(writtenName, nameCapacity, "%s", name + 1); // Remove leading '/' for UI.
    return true;
}

void finishCaptureAndSave() {
    nnInvalidate(); // Never bridge a paused FIFO with an NN input window.
    // The SD card may block for hundreds of milliseconds. Keep the MAX30102
    // shut down until it finishes so its 32-slot FIFO cannot silently overrun.
    sensor.shutDown();
    while (sensor.available()) sensor.nextSample(); // Discard any uncaptured ring tail.
    drawCaptureStatus("Saving raw RED+IR to SD...");
    Serial.printf("# REC finished: %u samples, %.3fs host-read span\n",
                  static_cast<unsigned>(recordedCount),
                  static_cast<double>(recorded[recordedCount - 1].readElapsedUs / 1e6f));
    char filename[20] = {};
    const bool ok = writeRecordedCSV(filename, sizeof(filename));
    if (ok) {
        Serial.printf("# SD saved: /%s\n", filename);
        char status[48];
        snprintf(status, sizeof(status), "Saved %s", filename);
        setTemporaryStatus(status);
    } else {
        Serial.println("# SD WRITE ERROR; recording not saved");
        setTemporaryStatus("SD WRITE ERROR");
    }

    // A sensor pause creates a gap. Clear the FIFO and restart our existing
    // filters/plot to avoid treating that gap as a continuous 10ms step.
    sensor.clearFIFO();
    sensor.wakeUp();
    redBlocker.configure(SAMPLE_RATE_HZ, PPG_HPF_CUTOFF_HZ);
    irBlocker.configure(SAMPLE_RATE_HZ, PPG_HPF_CUTOFF_HZ);
    redLowPass.configure(SAMPLE_RATE_HZ, PPG_LPF_CUTOFF_HZ);
    irLowPass.configure(SAMPLE_RATE_HZ, PPG_LPF_CUTOFF_HZ);
    writeIndex = sampleCount = 0;
    latestRaw = 0;
    latestAC = latestFiltered = 0.0f;
    captureState = CaptureState::Idle;
}

void haltWithSensorError() {
    Serial.println("# ERROR MAX30102 not found at address 0x57");
    tft.fillRect(0, 24, SCREEN_W, SCREEN_H - 24, TFT_BLACK);
    tft.setTextFont(2);
    tft.setTextColor(TFT_RED, TFT_BLACK);
    tft.drawString("MAX30102 NOT FOUND", 8, 45);
    tft.setTextColor(TFT_WHITE, TFT_BLACK);
    tft.drawString("SDA: GPIO4  SCL: GPIO5", 8, 74);
    tft.drawString("Check wiring, then reset", 8, 98);
    while (true) delay(1000);
}

void printSensorRegisters() {
    // MAX30102 registers: FIFO_CONFIG, MODE_CONFIG, SPO2_CONFIG, RED_PA, IR_PA.
    const uint8_t fifo = sensor.readRegister8(MAX30102_ADDRESS, 0x08);
    const uint8_t mode = sensor.readRegister8(MAX30102_ADDRESS, 0x09);
    const uint8_t spo2 = sensor.readRegister8(MAX30102_ADDRESS, 0x0A);
    const uint8_t redPa = sensor.readRegister8(MAX30102_ADDRESS, 0x0C);
    const uint8_t irPa = sensor.readRegister8(MAX30102_ADDRESS, 0x0D);
    Serial.printf("# sensor FIFO=0x%02X MODE=0x%02X SPO2=0x%02X RED_PA=0x%02X IR_PA=0x%02X\n",
                  fifo, mode, spo2, redPa, irPa);
    // 0x47 = ADC range 8192nA (0x40), 100Hz (0x04), 411us (0x03).
    if ((spo2 & 0x7F) != 0x47 || (mode & 0x07) != 0x03 ||
        ((fifo >> 5) & 0x07) != 0 || redPa != LED_AMPLITUDE ||
        irPa != LED_AMPLITUDE) {
        Serial.println("# WARNING: sensor registers do not match requested settings");
    } else {
        Serial.println("# sensor settings verified: 100Hz, 411us, 18bit, 8192nA, AVG1, RED+IR");
    }
}
// The diagnostic workload is paced independently of the sensor so all four
// experiments receive the SAME single golden INT8 vector at the SAME rate.
void diagSubmitWindow() {
#if PPG_DIAG_MODE
    static uint32_t lastMs=millis();
    static uint32_t issued=0;
    if(!nnReady || !nnQueue || issued>=DIAG_JOBS) return;
    const uint32_t now=millis();
    if(now-lastMs<DIAG_PERIOD_MS) return;
    lastMs=now;
    NNWindow window={};
    window.startSample=issued*peaknet_live::WINDOW;
    portENTER_CRITICAL(&nnGuard);
    window.generation=nnGeneration;
    ++nnShared.enqueued;
    portEXIT_CRITICAL(&nnGuard);
    // No overwrite is expected at a 4.5s interval; verify drop=0 in logs.
    if(uxQueueMessagesWaiting(nnQueue)!=0) {
        portENTER_CRITICAL(&nnGuard);
        ++nnShared.overwritten;
        portEXIT_CRITICAL(&nnGuard);
    }
    xQueueOverwrite(nnQueue,&window);
    ++issued;
    Serial.printf("# DIAG submitted job=%lu/%lu sensor_samples=%lu fifo_bursts_ge4=%lu\n",
                  (unsigned long)issued,(unsigned long)DIAG_JOBS,
                  (unsigned long)sampleIndex,(unsigned long)longFifoBursts);
#endif
}
} // namespace

void setup() {
    Serial.begin(115200);
    delay(350);
    pinMode(PIN_BOOT, INPUT_PULLUP);
    bootLastRead = bootStableRead = digitalRead(PIN_BOOT);
    bootChangedAtMs = millis();
    // Hardware identity check: seller photographs sometimes mix ESP32 and ESP32-S3.
    Serial.printf("# chip=%s revision=%d flash=%u bytes, PSRAM=%u bytes\n",
                  ESP.getChipModel(), ESP.getChipRevision(),
                  static_cast<unsigned int>(ESP.getFlashChipSize()),
                  static_cast<unsigned int>(ESP.getPsramSize()));
    Serial.printf("# CPU_MHZ=%u (expect 240; confirm this in logs)\n",unsigned(ESP.getCpuFreqMHz()));
    Serial.println("# expected classic ESP32, 4 MB flash, LCD GPIO 14/13/15/2/21");

    if (DIAG_LCD_ON) {
        pinMode(PIN_LCD_BL, OUTPUT);
        digitalWrite(PIN_LCD_BL, HIGH);
        tft.init();
        tft.setRotation(1);
        tft.invertDisplay(true);
        drawHeader();
        tft.setTextColor(TFT_LIGHTGREY, TFT_BLACK);
        tft.drawString("Initializing sensor...", 8, 74);
    }
#if PPG_DIAG_MODE
    Serial.printf("# DIAG plan=%d core=%d sensor=%d lcd=%d sd=%d "
                  "fixed_golden=1 every_ms=%lu jobs=%lu\n",
                  PPG_DIAG_MODE,int(NN_WORKER_CORE),int(DIAG_SENSOR_ON),
                  int(DIAG_LCD_ON),int(DIAG_SD_ON),
                  (unsigned long)DIAG_PERIOD_MS,(unsigned long)DIAG_JOBS);
#endif

    // TFT_eSPI uses VSPI, whereas microSD uses dedicated HSPI. Neither
    // requires changing LCD pins or the existing MAX30102 I2C wiring.
    if (DIAG_SD_ON) {
        pinMode(PIN_SD_CS, OUTPUT);
        digitalWrite(PIN_SD_CS, HIGH);
        sdSPI.begin(PIN_SD_SCK, PIN_SD_MISO, PIN_SD_MOSI, PIN_SD_CS);
        sdReady = SD.begin(PIN_SD_CS, sdSPI, SD_SPI_HZ);
        Serial.printf("# SD card: %s (HSPI SCK18 MISO19 MOSI23 CS27)\n",
                      sdReady ? "READY" : "NOT FOUND");
        if (!sdReady && DIAG_LCD_ON)
            tft.drawString("SD unavailable; live plot only", 8, 95);
    }

    if (DIAG_SENSOR_ON) {
        Wire.begin(PIN_SDA, PIN_SCL);
        Wire.setClock(400000);
        if (!sensor.begin(Wire, I2C_SPEED_FAST)) {
            if(DIAG_LCD_ON) haltWithSensorError();
            Serial.println("# ERROR: MAX30102 missing (SDA4/SCL5)");
            while(true) delay(1000);
        }
        sensor.setup(LED_AMPLITUDE, FIFO_AVERAGE, LED_MODE,
                     SAMPLE_RATE_HZ, LED_PULSE_WIDTH_US, ADC_RANGE_NA);
        sensor.setPulseAmplitudeGreen(0); // Not used in MAX30102 mode.
    }
    // Configure both channels identically; one new sample per 10 ms FIFO record.
    redBlocker.configure(SAMPLE_RATE_HZ, PPG_HPF_CUTOFF_HZ);
    irBlocker.configure(SAMPLE_RATE_HZ, PPG_HPF_CUTOFF_HZ);
    if (!redLowPass.configure(SAMPLE_RATE_HZ, PPG_LPF_CUTOFF_HZ) ||
        !irLowPass.configure(SAMPLE_RATE_HZ, PPG_LPF_CUTOFF_HZ)) {
        Serial.println("# ERROR invalid LPF frequency");
        while (true) delay(1000);
    }
    Serial.printf("# filters TUSTIN_BUTTERWORTH2 fs=%dHz HPF=%.3fHz LPF=%.3fHz\n",
                  SAMPLE_RATE_HZ, static_cast<double>(PPG_HPF_CUTOFF_HZ),
                  static_cast<double>(PPG_LPF_CUTOFF_HZ));
    if (DIAG_SENSOR_ON) printSensorRegisters();
    if (DIAG_LCD_ON) {
        drawHeader();
        Serial.printf("# display %dx%d, selected %s, HPF %s %.3f Hz, LPF %s %.1f Hz\n",
                      tft.width(), tft.height(), CHANNEL_NAME,
                      PPG_HPF ? "ON" : "OFF", static_cast<double>(PPG_HPF_CUTOFF_HZ),
                      PPG_LPF ? "ON" : "OFF", static_cast<double>(PPG_LPF_CUTOFF_HZ));
    }
#if SERIAL_RAW
    Serial.println("time_ms,red,ir,hpf_red,hpf_ir,sample_idx,band_red,band_ir");
#endif
    lastRefresh = millis();
    Serial.printf("# RAM capture buffer: %u bytes; BOOT GPIO0 records %ds raw RED+IR\n",
                  static_cast<unsigned>(sizeof(recorded)), PPG_RECORD_SECONDS);
    nnReady=startNN();
    Serial.printf("# free_heap_after_nn=%u bytes\n",unsigned(ESP.getFreeHeap()));
}

// Poll MAX30102 frequently, including while the screen is being redrawn.
// Otherwise a long TFT_eSPI drawing burst can overflow SparkFun's four-slot ring.
namespace {
void serviceSensor() {
    if (captureState == CaptureState::SavePending) return;
    const uint16_t batch = sensor.check();
    if (batch >= 4) {
        ++longFifoBursts; // Possible lost samples in SparkFun's small library ring.
        if (captureState == CaptureState::Recording)
            ++longFifoBurstsDuringCapture;
    }
    while (sensor.available()) {
        const uint32_t red = sensor.getFIFORed();
        const uint32_t ir = sensor.getFIFOIR();
        sensor.nextSample();
        processSample(red, ir);
        if (captureState == CaptureState::SavePending) break;
    }
}
} // namespace

void loop() {
#if PPG_DIAG_MODE
    // Fixed-interval controlled experiments; BOOT/SD writing deliberately off.
    if (DIAG_SENSOR_ON) serviceSensor();
    diagSubmitWindow();
    if (DIAG_LCD_ON && millis()-lastRefresh>=REFRESH_MS) {
        lastRefresh=millis();
        drawWaveform(); // same full TFT waveform drawing as normal SPPG
    }
    delay(1);
    return;
#else
    pollBootButton();
    serviceSensor();
    if (captureState == CaptureState::SavePending) {
        finishCaptureAndSave();
        lastRefresh = millis();
        return;
    }

    // During capture draw ~7 FPS; normal mode retains the existing ~13 FPS.
    // The plot renders from a snapshot, while sensor polling is interleaved
    // between short bursts of TFT drawing (see drawWaveform()).
    const unsigned long now = millis();
    const unsigned long refreshInterval =
        captureState == CaptureState::Recording ? RECORD_REFRESH_MS : REFRESH_MS;
    if (now - lastRefresh >= refreshInterval) {
        lastRefresh = now;
        drawWaveform();
        // Serial diagnostics only after recording, avoiding serial I/O stalls.
        if (captureState == CaptureState::Idle && longFifoBursts) {
            Serial.printf("# WARNING FIFO bursts >=4: %lu (possible lost samples)\n",
                          static_cast<unsigned long>(longFifoBursts));
            longFifoBursts = 0;
        }
    }
    delay(1);
#endif // PPG_DIAG_MODE
}
