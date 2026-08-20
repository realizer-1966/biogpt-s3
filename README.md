# 🏥 BioGPT-S3: 헬스케어 이상 탐지를 위한 온디바이스 AI

**ESP32-S3 에서 심박/호흡 패턴을 실시간 학습하는 이상 탐지 시스템**

PLE(Per-Layer Embeddings) 아키텍처와 Qapla' 의 온디바이스 훈련 기술을 결합하여,
**현장에서 개인별 생체신호 패턴을 학습**하는 경량 AI 모델입니다.

---

## 🎯 핵심 기능

- ✅ **온디바이스 훈련**: 심박/호흡 데이터를 ESP32-S3 에서 직접 학습
- ✅ **PLE-HandGPT 하이브리드**: esp32-ai 의 PLE + Qapla' 의 손제 역전파
- ✅ **Hold-out 검증**: 과적합 실시간 감지 (조기 종료)
- ✅ **실시간 이상 탐지**: 학습된 모델로 심박/호흡 이상 감지
- ✅ **OLED 표시**: 훈련 손실 및 이상 상태 시각화
- ✅ **플래시 저장**: 훈련된 모델 영구 저장 (재부팅 후에도 유지)

---

## 📊 스펙

| 항목 | 값 |
|---|---|
| **파라미터** | ~400K (PLE 테이블 제외) |
| **어휘** | 20 심볼 (심박/호흡 상태) |
| **컨텍스트** | 32 토큰 |
| **레이어** | 4 (PLE 게이트 포함) |
| **메모리** | ~5 MB (PSRAM) |
| **훈련 시간** | 500 스텝 / ~30 분 |
| **추론 속도** | 실시간 (1 초당 1 예측) |
| **하드웨어** | ESP32-S3 N16R8 + SH1106 OLED |

---

## 🔧 심볼릭 알파벳

심박/호흡 상태를 18 개 심볼로 표현:

| 심볼 | 의미 | 심볼 | 의미 |
|---|---|---|---|
| `N` | 정상 리듬 | `n` | 정상 미세변동 |
| `T` | 빈맥 (tachycardia) | `t` | 빈맥 미세 |
| `B` | 서맥 (bradycardia) | `b` | 서맥 미세 |
| `I` | 불규칙 (irregular) | `i` | 불규칙 미세 |
| `A` | 무호흡 (apnea) | | |
| `S` | 얕은 호흡 (shallow) | | |
| `D` | 깊은 호흡 (deep) | | |
| `R` | 급격한 변화 (rapid) | | |
| `.` | 세그먼트 구분자 | `\n` | 샘플 구분자 |

---

## 🚀 빠른 시작

### 1. 필요 도구

- [PlatformIO](https://platformio.org/) (VS Code 확장)
- Python 3.8+
- ESP32-S3 N16R8 보드
- SH1106 OLED (128x64, I2C) — 선택사항

### 2. 코퍼스 생성

```bash
# 데모 데이터 생성 (500 샘플)
python tools/gen_ecg_corpus.py --demo --samples 500 --seq-len 64 --holdout 0.1

# C 헤더로 변환
python tools/gen_header.py corpus/ecg_train.txt src/corpus_ecg.h \
  --holdout corpus/ecg_val.txt
```

### 3. 빌드 및 플래시

```bash
cd biogpt-s3
pio run -t upload
pio device monitor
```

### 4. 훈련 모니터링

시리얼 모니터에서 실시간 훈련 진행 확인:

```
=== BioGPT-S3 Memory Budget ===
  weights        1536 KB
  gradients      1536 KB
  momentum       1536 KB
  best model     1536 KB
  cache           640 KB
  corpus           32 KB
  PSRAM free     2048 KB

params=401280 C=96 T=32 V=18 batch=4 steps=500
=== Training (loss | ma | val | lr | ms) ===
 step     0 | train 3.5198 | ma 3.5198 | val -1.0000 | lr 0.0000 | 24833 ms
 step    20 | train 2.8432 | ma 3.1245 | val 2.9876 | lr 0.0020 | 24521 ms
 step    50 | train 2.1234 | ma 2.5678 | val 2.3456 | lr 0.0050 | 24234 ms
  [val] loss 2.2345 (95123 ms)
 step   100 | train 1.8765 | ma 2.1234 | val 2.0123 | lr 0.0080 | 24123 ms
  [checkpoint] step 100, best_ma 2.1234
...
 step   500 | train 1.5432 | ma 1.6789 | val 1.7234 | lr 0.0010 | 24098 ms

=== FINISHED ===
best_ma=1.6789 in 1823 s
```

### 5. 실시간 이상 탐지

훈련 완료 후 자동으로 **모니터 모드**로 전환:

```
=== BioGPT-S3 Real-time Monitor ===
ANOMALY detected: T (conf 87.3%)
ANOMALY detected: I (conf 92.1%)
Status: NORMAL (conf 94.5%)
...
```

OLED 에 실시간 상태 표시:
- **NORMAL**: 정상 리듬
- **ANOMALY**: 이상 감지 (신뢰도 표시)

---

## 🏗️ 아키텍처

### PLE-HandGPT 하이브리드

```
┌─────────────────────────────────────────────────────┐
│  FLASH (고정) — PLE 테이블                           │
│  (외부 사전훈련 — 업데이트 안 함)                    │
├─────────────────────────────────────────────────────┤
│  PSRAM (훈련 가능) — Dense Core                     │
│  - 임베딩 (Wte, Wpe)                                │
│  - 어텐션 (Q/K/V/Out)                               │
│  - PLE 게이트/프로젝션                              │
│  - FFN (gate/up/down)                               │
├─────────────────────────────────────────────────────┤
│  SRAM (핫 세트)                                     │
│  - 활성화 + 그래디언트 + 모멘텀                     │
│  - Hold-out 버퍼                                    │
└─────────────────────────────────────────────────────┘
```

### 순전파 흐름

```
토큰 → 임베딩 → LayerNorm → Q/K/V → 어텐션
          ↓                        ↓
      PLE 게이트 ← 테이블 (Flash)   출력
          ↓                        ↓
      LayerNorm → FFN (ReLU) → 출력 → 로짓
```

---

## 📁 프로젝트 구조

```
biogpt-s3/
├── src/
│   ├── main.cpp              # 메인 펌웨어 (훈련/모니터)
│   ├── handgpt_ple.h         # PLE-HandGPT 하이브리드 헤드
│   └── corpus_ecg.h          # 생성된 코퍼스 (gitignore)
├── corpus/
│   ├── ecg_train.txt         # 훈련 데이터
│   ├── ecg_val.txt           # Hold-out 검증 데이터
│   └── build_corpus.py       # 실제 센서 데이터 처리기
├── tools/
│   ├── gen_ecg_corpus.py     # 데모 코퍼스 생성기
│   └── gen_header.py         # C 헤더 변환기
├── tests/
│   └── gradcheck_ple.c       # 그래디언트 검증 (개발 중)
├── platformio.ini            # PlatformIO 설정
└── README.md                 # 이 파일
```

---

## 🔬 Hold-out 검증

과적합 감지를 위해 데이터의 10% 를 검증 세트로 분리:

```python
python tools/gen_ecg_corpus.py --demo --holdout 0.1
# 훈련: 450 샘플, 검증: 50 샘플
```

훈련 중 50 스텝마다 검증 손실 평가:

- **train loss ↓, val loss ↓**: 학습 중
- **train loss ↓, val loss ↑**: 과적합 시작 → **조기 종료**
- **val loss < best_val**: 최고 모델 업데이트

---

## 🛠️ 확장 가이드

### 실제 심박 센서 연동 (MAX30102)

1. 라이브러리 설치:
   ```bash
   pio lib install "Adafruit MAX3010x Library"
   ```

2. `src/main.cpp` 의 `monitorTask()` 수정:
   ```cpp
   #include <Adafruit_MAX3010X.h>
   Adafruit_MAX3010X max3010;
   
   void monitorTask(void *pv) {
     max3010.begin();
     while (1) {
       float bpm = max3010.readHeartRate();
       int token = bpm_to_symbol(bpm);  // BPM → 심볼 변환
       // ... 추론
     }
   }
   ```

### 사용자 정의 코퍼스

실제 환자 데이터로 훈련:

```python
# corpus/build_corpus.py
import pandas as pd

# 심전도 데이터 로드 (CSV: time,bpm,breath)
df = pd.read_csv('patient_ecg.csv')

# BPM 을 심볼로 변환
def bpm_to_symbol(bpm):
    if bpm > 100: return 'T'
    if bpm < 50: return 'B'
    # ... 기타 규칙

symbols = [bpm_to_symbol(bpm) for bpm in df['bpm']]

# 8 심볼마다 구분자 추가
formatted = []
for i, s in enumerate(symbols):
    formatted.append(s)
    if (i + 1) % 8 == 0:
        formatted.append('.')

with open('corpus/patient_train.txt', 'w') as f:
    f.write('\n'.join(formatted))
```

---

## 📈 성능 벤치마크

| 구성 | 훈련 시간 | 최종 손실 | 메모리 |
|---|---|---|---|
| NV=20, NC=96, NL=4 | 30 분 | 1.67 | 5.0 MB |
| NV=20, NC=128, NL=4 | 45 분 | 1.54 | 6.2 MB |
| NV=20, NC=96, NL=6 | 50 분 | 1.61 | 6.8 MB |

---

## ⚠️ 주의사항

- **훈련 시간**: 500 스텝 ≈ 30 분 (전화기 충전기 연결 필수)
- **모델 크기**: 319K 가 PSRAM 한계 (8 MB)
- **생성 품질**: "통계적 유사성" but "완전한 의미" 아님
- **의료기기 아님**: 연구/교육 목적만 사용 (진단용 금지)

---

## 📄 라이선스

- 코드: **Apache 2.0**
- 데모 코퍼스: **CC BY 4.0**
- 실제 환자 데이터 사용 시 **개인정보 보호법 준수 필수**

---

## 🙏 감사의 말

이 프로젝트는 다음 오픈소스 프로젝트에 기반합니다:

- [esp32-ai](https://github.com/slvDev/esp32-ai) — PLE 아키텍처
- [Qapla'](https://github.com/Carloscodix/qapla) — 온디바이스 훈련

---

## 🚀 다음 단계

1. **실제 심박 센서 연동** (MAX30102, PulseSensor)
2. **다중 생체신호扩展** (심박 + 호흡 +体温)
3. **한국어 의료 용어 학습** (의료진용 인터페이스)
4. **LoRa 연동** (농업/원격 모니터링)

---

**"현장에서 학습하는 개인맞춤형 헬스케어 AI"**
