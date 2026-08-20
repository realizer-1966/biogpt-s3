// main.cpp — BioGPT-S3 헬스케어 이상 탐지 펌웨어
//
// PLE-HandGPT 하이브리드 모델로 심박/호흡 패턴 학습
// - Hold-out 검증으로 과적합 감지
// - OLED 에 실시간 손실 표시
// - MAX30102 심박 센서 연동 (선택)
//
// License: Apache 2.0

#include <Arduino.h>
#include <U8g2lib.h>
#include <Wire.h>
#include <LittleFS.h>

#include "handgpt_ple.h"

// 코퍼스 (생성 후 포함)
#if defined(__has_include)
#  if __has_include("corpus_ecg.h")
#    include "corpus_ecg.h"
#  else
#    include "corpus_ecg.h.example"
#    warning "corpus_ecg.h not found - using placeholder"
#  endif
#else
#  include "corpus_ecg.h"
#endif

// === 훈련 하이퍼파라미터 ===
#define BATCH        4        // 마이크로 배치 (메모리 제약)
#define TOTAL_STEPS 500       // 헬스케어는 더 빠른 수렴 (500 스텝)
#define WARMUP      50
#define BASE_LR     0.01f
#define BETA        0.9f      // 모멘텀
#define MA_ALPHA    0.05f     // 이동평균 알파
#define CKPT_EVERY  100       // 체크포인트 주기
#define MODEL_PATH  "/biogpt_model.bin"

// === Hold-out 검증 ===
#define VAL_EVERY   50        // 검증 주기
#define VAL_WINDOWS 32        // 검증 윈도우 수

// === 생성 설정 ===
#define GEN_LEN     64
#define GEN_TEMP    0.8f

// === OLED 설정 ===
U8G2_SH1106_128X64_NONAME_F_HW_I2C u8g2(U8G2_R0, U8X8_PIN_NONE);
bool oled_ok = false;

// === 메모리 할당 (PSRAM) ===
Model *M = nullptr, *G = nullptr, *BEST = nullptr;
Cache *CA = nullptr;
static uint8_t *DATA = nullptr;
static uint8_t *VDATA = nullptr;  // Hold-out
int NDATA = 0, NVDATA = 0;
static float *VEL = nullptr;

int V = 0;  // 어휘 크기
char ITOS[NV];
int STOI[256];
unsigned int rng = 1701;  // Star Trek 시드

#define RND (rng ^= rng << 13, rng ^= rng >> 17, rng ^= rng << 5, rng)

// === 학습률 스케줄 (Cosine) ===
float lr_at(int step) {
  if (step < WARMUP) {
    return BASE_LR * (float)step / (float)WARMUP;
  }
  float p = (float)(step - WARMUP) / (float)(TOTAL_STEPS - WARMUP);
  if (p > 1.0f) p = 1.0f;
  return BASE_LR * (0.1f + 0.9f * 0.5f * (1.0f + cosf(3.14159265f * p)));
}

// === OLED 메시지 ===
void oled_msg(const char *a, const char *b) {
  if (!oled_ok) return;
  u8g2.clearBuffer();
  u8g2.setFont(u8g2_font_6x10_tf);
  u8g2.drawStr(0, 12, a);
  if (b) u8g2.drawStr(0, 28, b);
  u8g2.sendBuffer();
}

// === Hold-out 평가 ===
float eval_heldout() {
  if (NVDATA < NT + 2) return -1;
  
  int usable = NVDATA - NT - 1;
  int stride = usable / VAL_WINDOWS;
  if (stride < NT) stride = NT;
  
  int n = 0;
  float tot = 0;
  
  for (int s = 0; s < usable && n < VAL_WINDOWS; s += stride) {
    int X[NT], Y[NT];
    for (int t = 0; t < NT; t++) {
      X[t] = VDATA[s + t];
      Y[t] = VDATA[s + t + 1];
    }
    tot += forward(M, CA, X, Y);
    n++;
  }
  
  return n ? tot / n : -1;
}

// === OLED 훈련 표시 ===
void oled_train(int step, float loss, float ma, float val) {
  if (!oled_ok) return;
  
  char buf[32];
  u8g2.clearBuffer();
  u8g2.setFont(u8g2_font_6x10_tf);
  
  u8g2.drawStr(0, 10, "BioGPT-S3 Training");
  
  snprintf(buf, sizeof(buf), "step %d", step);
  u8g2.drawStr(0, 22, buf);
  
  snprintf(buf, sizeof(buf), "train %.3f", ma);
  u8g2.drawStr(0, 34, buf);
  
  if (val >= 0) {
    snprintf(buf, sizeof(buf), "val   %.3f", val);
  } else {
    snprintf(buf, sizeof(buf), "loss  %.3f", loss);
  }
  u8g2.drawStr(0, 45, buf);
  
  // 진행 바
  int w = (int)(120.0f * step / TOTAL_STEPS);
  if (w < 0) w = 0;
  if (w > 120) w = 120;
  u8g2.drawFrame(0, 52, 122, 10);
  u8g2.drawBox(1, 53, w, 8);
  
  u8g2.sendBuffer();
}

// === 이상 탐지 결과 표시 ===
void oled_anomaly(const char *status, float confidence) {
  if (!oled_ok) return;
  
  char buf[32];
  u8g2.clearBuffer();
  u8g2.setFont(u8g2_font_6x10_tf);
  
  u8g2.drawStr(0, 10, "Real-time Monitor");
  
  if (strcmp(status, "NORMAL") == 0) {
    u8g2.drawStr(0, 28, "Status: NORMAL");
    u8g2.setDrawColor(1);
    u8g2.drawBox(0, 40, 128, 20);
    u8g2.setDrawColor(0);
    u8g2.drawStr(20, 54, "Heart Rate OK");
  } else {
    u8g2.drawStr(0, 28, "Status: ANOMALY");
    u8g2.setDrawColor(1);
    u8g2.drawBox(0, 40, 128, 20);
    u8g2.setDrawColor(0);
    snprintf(buf, sizeof(buf), "Conf: %.1f%%", confidence * 100);
    u8g2.drawStr(10, 54, buf);
  }
  
  u8g2.sendBuffer();
}

// === 어휘 구축 ===
bool build_vocab() {
  for (int i = 0; i < 256; i++) STOI[i] = -1;
  V = 0;
  
  for (int i = 0; i < KLIN_CORPUS_LEN; i++) {
    unsigned char ch = (unsigned char)KLIN_CORPUS[i];
    if (STOI[ch] < 0) {
      if (V >= NV) {
        Serial.printf("ERROR: corpus uses more than %d distinct bytes\n", NV);
        return false;
      }
      STOI[ch] = V;
      ITOS[V] = (char)ch;
      V++;
    }
  }
  return true;
}

// === 모델 저장/로드 ===
bool save_model(Model *m, float loss) {
  File f = LittleFS.open(MODEL_PATH, "w");
  if (!f) return false;
  
  unsigned long t0 = millis();
  f.write((uint8_t*)&loss, sizeof(float));
  
  const uint8_t *p = (const uint8_t*)m;
  size_t left = sizeof(Model);
  while (left) {
    size_t c = left > 4096 ? 4096 : left;
    if (f.write(p, c) != c) {
      f.close();
      return false;
    }
    p += c;
    left -= c;
  }
  
  f.close();
  Serial.printf("  [flash] %u KB written in %lu ms\n", 
                (unsigned)(sizeof(Model)/1024), millis() - t0);
  return true;
}

bool load_model(Model *m, float *loss) {
  if (!LittleFS.exists(MODEL_PATH)) return false;
  
  File f = LittleFS.open(MODEL_PATH, "r");
  if (!f) return false;
  
  if (f.read((uint8_t*)loss, sizeof(float)) != (int)sizeof(float)) {
    f.close();
    return false;
  }
  
  uint8_t *p = (uint8_t*)m;
  size_t left = sizeof(Model);
  while (left) {
    size_t c = left > 4096 ? 4096 : left;
    if (f.read(p, c) != (int)c) {
      f.close();
      return false;
    }
    p += c;
    left -= c;
  }
  
  f.close();
  return true;
}

// === 토큰 샘플링 ===
int sample_next(real *logits, float temp) {
  float mx = -1e30f;
  for (int v = 0; v < NV; v++) {
    if (logits[v] > mx) mx = logits[v];
  }
  
  float sum = 0, p[NV];
  for (int v = 0; v < NV; v++) {
    p[v] = expf((logits[v] - mx) / temp);
    sum += p[v];
  }
  
  float r = ((float)(RND % 100000) / 100000.0f) * sum;
  float acc = 0;
  for (int v = 0; v < NV; v++) {
    acc += p[v];
    if (r <= acc) return v;
  }
  return NV - 1;
}

// === 텍스트 생성 ===
void generate(int n) {
  int ctx[NT];
  int nl = STOI[(int)'\n'] >= 0 ? STOI[(int)'\n'] : 0;
  
  for (int i = 0; i < NT; i++) ctx[i] = nl;
  
  static char gbuf[256];
  int gl = 0;
  int Xd[NT], Yd[NT];
  
  for (int k = 0; k < n; k++) {
    for (int i = 0; i < NT; i++) {
      Xd[i] = ctx[i];
      Yd[i] = ctx[i];
    }
    
    forward(M, CA, Xd, Yd);
    
    int tk = sample_next(&CA->logits[(NT-1)*NV], GEN_TEMP);
    char c = ITOS[tk];
    Serial.print(c);
    
    if (gl < 250) gbuf[gl++] = c;
    
    // 3 토큰마다 OLED 업데이트
    if (k % 3 == 0) {
      gbuf[gl] = '\0';
      oled_msg("BioGPT says:", gbuf + (gl > 40 ? gl - 40 : 0));
    }
    
    for (int i = 0; i < NT - 1; i++) ctx[i] = ctx[i+1];
    ctx[NT-1] = tk;
  }
  Serial.println();
}

// === 훈련 태스크 ===
void trainTask(void *pv) {
  // PSRAM 할당
  DATA = (uint8_t*)ps_malloc(KLIN_CORPUS_LEN);
  G = (Model*)ps_malloc(sizeof(Model));
  VEL = (float*)ps_malloc((size_t)NPAR * sizeof(float));
  BEST = (Model*)ps_malloc(sizeof(Model));
  
  if (!DATA || !G || !VEL || !BEST) {
    Serial.println("FATAL: ps_malloc failed");
    oled_msg("FATAL", "Memory alloc failed");
    vTaskDelete(NULL);
    return;
  }
  
  // 메모리 예산 표시
  Serial.println("=== BioGPT-S3 Memory Budget ===");
  Serial.printf("  weights     %7u KB\n", (unsigned)(sizeof(Model)/1024));
  Serial.printf("  gradients   %7u KB\n", (unsigned)(sizeof(Model)/1024));
  Serial.printf("  momentum    %7u KB\n", (unsigned)(NPAR * sizeof(float)/1024));
  Serial.printf("  best model  %7u KB\n", (unsigned)(sizeof(Model)/1024));
  Serial.printf("  cache       %7u KB\n", (unsigned)(sizeof(Cache)/1024));
  Serial.printf("  corpus      %7u KB\n", (unsigned)(KLIN_CORPUS_LEN/1024));
  Serial.printf("  PSRAM free  %u KB\n", (unsigned)(ESP.getFreePsram()/1024));
  Serial.printf("  SRAM free   %u KB\n", (unsigned)(ESP.getFreeHeap()/1024));
  
  // 코퍼스 토큰화
  for (int i = 0; i < KLIN_CORPUS_LEN; i++) {
    DATA[i] = (uint8_t)STOI[(unsigned char)KLIN_CORPUS[i]];
  }
  NDATA = KLIN_CORPUS_LEN;
  
  // Hold-out 세트
#ifdef KLIN_HELDOUT_LEN
  VDATA = (uint8_t*)ps_malloc(KLIN_HELDOUT_LEN);
  if (!VDATA) {
    Serial.println("FATAL: ps_malloc heldout failed");
    vTaskDelete(NULL);
    return;
  }
  for (int i = 0; i < KLIN_HELDOUT_LEN; i++) {
    VDATA[i] = (uint8_t)STOI[(unsigned char)KLIN_HELDOUT[i]];
  }
  NVDATA = KLIN_HELDOUT_LEN;
  Serial.printf("held-out: %d bytes (%.1f%%)\n", 
                NVDATA, 100.0 * NVDATA / (NDATA + NVDATA));
#endif
  
  // 초기화
  memset(VEL, 0, (size_t)NPAR * sizeof(float));
  model_init(M, 1701);
  
  Serial.printf("\nparams=%d C=%d T=%d V=%d batch=%d steps=%d\n", 
                NPAR, NC, NT, V, BATCH, TOTAL_STEPS);
  Serial.println("=== Training (loss | ma | val | lr | ms) ===");
  
  float ma = -1, best_ma = 1e30f, val = -1;
  float best_val = 1e30f;
  int patience = 0;
  
  unsigned long t_all = millis();
  float *mp = (float*)M, *gp = (float*)G, invb = 1.0f / BATCH;
  
  for (int step = 0; step <= TOTAL_STEPS; step++) {
    unsigned long t0 = millis();
    memset(G, 0, sizeof(Model));
    
    float tot = 0;
    for (int b = 0; b < BATCH; b++) {
      int start = RND % (NDATA - NT - 1);
      int X[NT], Y[NT];
      for (int t = 0; t < NT; t++) {
        X[t] = DATA[start + t];
        Y[t] = DATA[start + t + 1];
      }
      tot += forward(M, CA, X, Y);
      backward(M, CA, G);
    }
    
    float loss = tot / BATCH;
    
    // NaN 체크
    if (isnan(loss) || loss > 1e30f) {
      Serial.printf("!! NaN loss at step %d\n", step);
      break;
    }
    
    // 학습률
    float lr = lr_at(step);
    
    // SGD + 모멘텀 업데이트
    for (int i = 0; i < NPAR; i++) {
      float gg = gp[i] * invb;
      VEL[i] = BETA * VEL[i] + gg;
      mp[i] -= lr * VEL[i];
    }
    
    // 이동평균
    if (ma < 0) {
      ma = loss;
    } else {
      ma = (1.0f - MA_ALPHA) * ma + MA_ALPHA * loss;
    }
    
    // 최고 모델 저장 (train loss 기준)
    if (ma < best_ma) {
      best_ma = ma;
      memcpy(BEST, M, sizeof(Model));
    }
    
    // Hold-out 평가
#ifdef KLIN_HELDOUT_LEN
    if (step % VAL_EVERY == 0 || step == TOTAL_STEPS - 1) {
      unsigned long tv = millis();
      val = eval_heldout();
      Serial.printf("  [val] loss %.4f (%lu ms)\n", val, millis() - tv);
      
      // 조기 종료 (과적합 감지)
      if (val > best_val + 0.05f) {
        patience++;
        if (patience >= 3) {
          Serial.println("Early stopping: overfitting detected");
          break;
        }
      } else {
        patience = 0;
        best_val = val;
        memcpy(BEST, M, sizeof(Model));
      }
    }
#endif
    
    // 체크포인트
    if (step > 0 && step % CKPT_EVERY == 0) {
      if (save_model(BEST, best_ma)) {
        Serial.printf("  [checkpoint] step %d, best_ma %.4f\n", step, best_ma);
      }
    }
    
    unsigned long dt = millis() - t0;
    
    // OLED 업데이트
    oled_train(step, loss, ma, val);
    
    // 시리얼 출력
    if (step % 20 == 0) {
#ifdef KLIN_HELDOUT_LEN
      Serial.printf(" step %5d | train %.4f | ma %.4f | val %.4f | lr %.4f | %lu ms\n",
                    step, loss, ma, val, lr, dt);
#else
      Serial.printf(" step %5d | loss %.4f | ma %.4f | lr %.4f | %lu ms\n",
                    step, loss, ma, lr, dt);
#endif
    }
  }
  
  // 최종 모델 저장
  save_model(BEST, best_ma);
  
  Serial.printf("\n=== FINISHED ===\n");
  Serial.printf("best_ma=%.4f in %lu s\n", best_ma, (millis() - t_all) / 1000);
  
  oled_msg("Training done", "Generating...");
  
  // 모델 재로드 + 생성
  float lf = 0;
  load_model(M, &lf);
  Serial.printf("Model reloaded (saved loss=%.4f)\n", lf);
  
  Serial.println("\nBioGPT-S3 says:");
  generate(GEN_LEN);
  
  Serial.println("\nThe brain lives on-chip. Unplug and it persists.");
  vTaskDelete(NULL);
}

// === 실시간 이상 탐지 태스크 ===
void monitorTask(void *pv) {
  Serial.println("=== BioGPT-S3 Real-time Monitor ===");
  oled_msg("BioGPT-S3", "Monitoring...");
  
  int ctx[NT];
  int nl = STOI[(int)'\n'] >= 0 ? STOI[(int)'\n'] : 0;
  for (int i = 0; i < NT; i++) ctx[i] = nl;
  
  while (1) {
    // TODO: 실제 심박 센서 데이터 읽기
    // 현재은 더미 데이터
    int X[NT], Y[NT];
    for (int t = 0; t < NT; t++) {
      X[t] = ctx[t];
      Y[t] = ctx[t];
    }
    
    forward(M, CA, X, Y);
    
    // 이상 탐지: 예측 확률 분포 분석
    real *probs = &CA->P[(NT-1)*NV];
    real max_p = 0;
    int max_idx = 0;
    for (int v = 0; v < NV; v++) {
      if (probs[v] > max_p) {
        max_p = probs[v];
        max_idx = v;
      }
    }
    
    // 이상 심볼 ('T', 'B', 'I', 'A', 'S', 'D', 'R')
    char pred = ITOS[max_idx];
    bool is_anomaly = (pred == 'T' || pred == 'B' || pred == 'I' ||
                       pred == 'A' || pred == 'S' || pred == 'D' || pred == 'R');
    
    if (is_anomaly) {
      char buf[32];
      snprintf(buf, sizeof(buf), "%.0f%%", max_p * 100);
      oled_anomaly("ANOMALY", max_p);
      Serial.printf("ANOMALY detected: %c (conf %.1f%%)\n", pred, max_p * 100);
    } else {
      oled_anomaly("NORMAL", max_p);
    }
    
    // 시퀀스 업데이트
    for (int i = 0; i < NT - 1; i++) ctx[i] = ctx[i+1];
    ctx[NT-1] = max_idx;
    
    delay(1000);  // 1 초마다
  }
}

// === 설정 ===
void setup() {
  Serial.begin(115200);
  delay(1800);
  
  Serial.println("\n=== BioGPT-S3 v1: PLE-HandGPT for Healthcare ===");
  
  // I2C 초기화 (OLED + 센서)
  Wire.begin(8, 9);  // SDA=8, SCL=9 (ESP32-S3)
  oled_ok = u8g2.begin();
  oled_msg("BioGPT-S3", "Starting...");
  
  // LittleFS 초기화
  if (!LittleFS.begin(true)) {
    Serial.println("LittleFS init failed");
    return;
  }
  
  // 어휘 구축
  if (!build_vocab()) {
    oled_msg("ERROR", "Vocab > NV");
    return;
  }
  Serial.printf("vocab=%d (NV=%d)\n", V, NV);
  
  // 메모리 할당
  M = (Model*)ps_malloc(sizeof(Model));
  CA = (Cache*)ps_malloc(sizeof(Cache));
  
  if (!M || !CA) {
    Serial.println("FATAL: ps_malloc M/CA failed");
    oled_msg("FATAL", "Memory alloc failed");
    return;
  }
  
  Serial.printf("PSRAM free: %u KB\n", (unsigned)(ESP.getFreePsram()/1024));
  
  // 기존 모델 체크
  float lf = 0;
  if (LittleFS.exists(MODEL_PATH) && load_model(M, &lf)) {
    Serial.printf("Model found (loss=%.4f) -> MONITOR MODE\n", lf);
    char buf[26];
    snprintf(buf, sizeof(buf), "loss %.3f", lf);
    oled_msg("Model loaded", buf);
    
    // 이상 탐지 모드
    xTaskCreatePinnedToCore(monitorTask, "monitor", 32768, NULL, 1, NULL, 1);
  } else {
    Serial.println("No model -> TRAINING MODE");
    oled_msg("No model", "Training...");
    
    // 훈련 모드
    xTaskCreatePinnedToCore(trainTask, "train", 32768, NULL, 1, NULL, 1);
  }
}

void loop() {
  delay(1000);
}
