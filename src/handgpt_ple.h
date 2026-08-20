// handgpt_ple.h — PLE-HandGPT 하이브리드: PLE 아키텍처 + 온디바이스 훈련
//
// BioGPT-S3 — 심박/호흡 이상 탐지를 위한 경량 트랜스포머
//
// 아키텍처:
//   - PLE 테이블 (Flash 고정, 25M 파라미터 — 외부에서 사전훈련)
//   - Dense Core (PSRAM 훈련 가능, 559K 파라미터)
//   - Hold-out 검증으로 과적합 감지
//
// License: Apache 2.0
#pragma once
#include <math.h>
#include <string.h>

// === 설정 (platformio.ini 에서 오버라이드 가능) ===
#ifndef NV
#define NV 20      // 심볼릭 어휘 크기
#endif
#ifndef NC
#define NC 96      // d_model
#endif
#ifndef NT
#define NT 32      // 컨텍스트
#endif
#ifndef NPLE
#define NPLE 64    // PLE dim per layer
#endif
#ifndef NLAYER
#define NLAYER 4   // number of layers
#endif

#define NF (4*NC)  // FFN hidden
#define USE_PLE 1  // PLE 활성화

// float on ESP32, double for gradient check
#ifdef HANDGPT_DOUBLE
typedef double real;
#else
typedef float real;
#endif

// === 모델 구조 ===
// Flash 고정: PLE 테이블 (외부에서 사전훈련된 것 mmap)
// PSRAM 훈련 가능: Dense Core
typedef struct {
  // === 훈련 가능 (PSRAM) ===
  real Wte[NV*NC];        // 코어 임베딩
  real Wpe[NT*NC];        // 포지션 임베딩
  
  // LayerNorm 게인/바이어스
  real g1[NC], b1[NC];    // LN1 (pre-attention)
  real g2[NC], b2[NC];    // LN2 (pre-FFN)
  real gf[NC], bf[NC];    // 최종 LN
  
  // 어텐션
  real Wq[NC*NC], Wk[NC*NC], Wv[NC*NC], Wo[NC*NC];
  
  // PLE 게이트/프로젝션 (per layer)
  real ple_gate[NLAYER*NPLE*NC];
  real ple_proj[NLAYER*NC*NPLE];
  real ple_norm[NLAYER*NC];
  
  // FFN (SwiGLU 대신 ReLU)
  real W1[NC*NF], bf1[NF];
  real W2[NF*NC], bf2[NC];
  
  // 출력 헤드 (tied with Wte)
  // real head[NC*NV];  // tied 시 Wte 재사용
  
  // === Flash 고정 (mmap, 훈련 시 skip) ===
  const real *ple_table;  // [NV, NLAYER*NPLE] — 외부에서 로드
} Model;

// === 활성화 캐시 (순전파/역전파용) ===
typedef struct {
  int X[NT], Y[NT];       // 입력/타겟 토큰
  
  // 임베딩 + 포지션
  real h0[NT*NC];         // 초기 임베딩
  
  // LayerNorm 1
  real xn1[NT*NC], inv1[NT];
  
  // Q/K/V
  real q[NT*NC], k[NT*NC], v[NT*NC];
  
  // 어텐션 맵 + 출력
  real A[NT*NT], ao[NT*NC];
  
  // residual after attention
  real h1[NT*NC];
  
  // LayerNorm 2
  real xn2[NT*NC], inv2[NT];
  
  // PLE per-layer 입력
  real ple[NLAYER*NT*NPLE];
  
  // FFN 중간
  real f1[NT*NF], fa[NT*NF];  // 게이트/업
  
  // FFN 출력 + residual
  real h2[NT*NC];
  
  // 최종 LayerNorm
  real xnf[NT*NC], invf[NT];
  
  // 로짓 + 확률
  real logits[NT*NV], P[NT*NV];
} Cache;

// === 유틸리티 ===
static real frand(unsigned int *s, real scale) {
  // Box-Muller transform
  *s ^= *s << 13; *s ^= *s >> 17; *s ^= *s << 5;
  real u1 = ((*s >> 8) & 0xFFFFFF) / 16777216.0;
  *s ^= *s << 13; *s ^= *s >> 17; *s ^= *s << 5;
  real u2 = ((*s >> 8) & 0xFFFFFF) / 16777216.0;
  if (u1 < 1e-7) u1 = 1e-7;
#ifdef HANDGPT_DOUBLE
  return sqrt(-2.0*log(u1)) * cos(6.2831853*u2) * scale;
#else
  return sqrtf(-2.0f*logf(u1)) * cosf(6.2831853f*u2) * scale;
#endif
}

// 모델 초기화 (무작위)
void model_init(Model *m, unsigned int seed) {
  unsigned int s = seed ? seed : 1;
  memset(m, 0, sizeof(*m));
  
  // 임베딩
  for (int i = 0; i < NV*NC; i++) m->Wte[i] = frand(&s, 0.02);
  for (int i = 0; i < NT*NC; i++) m->Wpe[i] = frand(&s, 0.02);
  
  // LayerNorm 게인 = 1, 바이어스 = 0
  for (int i = 0; i < NC; i++) {
    m->g1[i] = 1; m->b1[i] = 0;
    m->g2[i] = 1; m->b2[i] = 0;
    m->gf[i] = 1; m->bf[i] = 0;
  }
  
  // 어텐션 가중치 (Xavier)
  real sc_attn = 1.0 / sqrt((real)NC);
  for (int i = 0; i < NC*NC; i++) {
    m->Wq[i] = frand(&s, sc_attn);
    m->Wk[i] = frand(&s, sc_attn);
    m->Wv[i] = frand(&s, sc_attn);
    m->Wo[i] = frand(&s, sc_attn);
  }
  
  // PLE 게이트/프로젝션
  real sc_ple = 1.0 / sqrt((real)NC);
  for (int i = 0; i < NLAYER*NPLE*NC; i++) {
    m->ple_gate[i] = frand(&s, sc_ple);
    m->ple_proj[i] = frand(&s, sc_ple);
  }
  for (int i = 0; i < NLAYER*NC; i++) m->ple_norm[i] = 1;
  
  // FFN
  real sc_ffn1 = 1.0 / sqrt((real)NC);
  real sc_ffn2 = 1.0 / sqrt((real)NF);
  for (int i = 0; i < NC*NF; i++) m->W1[i] = frand(&s, sc_ffn1);
  for (int i = 0; i < NF*NC; i++) m->W2[i] = frand(&s, sc_ffn2);
  
  // PLE 테이블 포인터는 외부에서 설정 (Flash mmap)
  m->ple_table = NULL;
}

// === LayerNorm 순전파 ===
static void ln_fwd(const real *x, const real *g, const real *b, 
                   real *y, real *xn, real *inv) {
  real mu = 0;
  for (int i = 0; i < NC; i++) mu += x[i];
  mu /= NC;
  
  real var = 0;
  for (int i = 0; i < NC; i++) {
    real d = x[i] - mu;
    var += d * d;
  }
  var /= NC;
  
  real iv = 1.0 / sqrt(var + 1e-5);
  *inv = iv;
  
  for (int i = 0; i < NC; i++) {
    xn[i] = (x[i] - mu) * iv;
    y[i] = xn[i] * g[i] + b[i];
  }
}

// === LayerNorm 역전파 ===
static void ln_bwd(const real *dy, const real *xn, real inv, 
                   const real *g, real *dx, real *dg, real *db) {
  real m1 = 0, m2 = 0;
  
  for (int i = 0; i < NC; i++) {
    real dxn = dy[i] * g[i];
    m1 += dxn;
    m2 += dxn * xn[i];
    dg[i] += dy[i] * xn[i];
    db[i] += dy[i];
  }
  
  m1 /= NC;
  m2 /= NC;
  
  for (int i = 0; i < NC; i++) {
    real dxn = dy[i] * g[i];
    dx[i] = inv * (dxn - m1 - xn[i] * m2);
  }
}

// === PLE 순전파 (esp32-ai 에서 이식) ===
static void ple_forward(Model *m, Cache *c, const int *tokens) {
  // 1. context-aware projection: h0 -> ple_tmp [NT, NLAYER*NPLE]
  static real ple_tmp[NT * NLAYER * NPLE];
  
  for (int t = 0; t < NT; t++) {
    for (int l = 0; l < NLAYER; l++) {
      const real *gate = m->ple_gate + l * NPLE * NC;
      real *out = c->ple + t * NLAYER * NPLE + l * NPLE;
      
      // 게이트: gelu(ple_gate(h0[t]))
      for (int i = 0; i < NPLE; i++) {
        real val = 0;
        for (int j = 0; j < NC; j++) {
          val += c->h0[t*NC + j] * gate[i*NC + j];
        }
        // GELU
        val = 0.5f * val * (1.0f + erf(val * 0.70710678f));
        ple_tmp[t*NLAYER*NPLE + l*NPLE + i] = val;
      }
    }
  }
  
  // 2. 테이블 룩업 (Flash 에서 직접)
  // tokens[t] -> ple_table[tokens[t], l*NPLE:(l+1)*NPLE]
  // m->ple_table 는 [NV, NLAYER*NPLE]
  
  // 3. 병합: (proj/sqrt(D) + table*sqrt(P)) / sqrt(2)
  real inv2 = 0.70710678f;  // 1/sqrt(2)
  real sqrt_p = sqrt((real)NPLE);
  real inv_d = 1.0 / sqrt((real)NC);
  
  for (int t = 0; t < NT; t++) {
    if (m->ple_table) {
      for (int l = 0; l < NLAYER; l++) {
        const real *table_row = m->ple_table + tokens[t] * NLAYER * NPLE + l * NPLE;
        real *ple_out = c->ple + t * NLAYER * NPLE + l * NPLE;
        real *tmp = ple_tmp + t * NLAYER * NPLE + l * NPLE;
        
        for (int i = 0; i < NPLE; i++) {
          ple_out[i] = (tmp[i] * inv_d + table_row[i] * sqrt_p) * inv2;
        }
      }
    } else {
      // PLE 테이블 없음 → projection 만 사용
      for (int l = 0; l < NLAYER; l++) {
        real *ple_out = c->ple + t * NLAYER * NPLE + l * NPLE;
        real *tmp = ple_tmp + t * NLAYER * NPLE + l * NPLE;
        for (int i = 0; i < NPLE; i++) {
          ple_out[i] = tmp[i] * inv_d;
        }
      }
    }
  }
}

// === 순전파 전체 ===
real forward(Model *m, Cache *c, const int *X, const int *Y) {
  memcpy(c->X, X, sizeof(int) * NT);
  
  // 1. 임베딩 + 포지션
  for (int t = 0; t < NT; t++) {
    for (int i = 0; i < NC; i++) {
      c->h0[t*NC + i] = m->Wte[X[t]*NC + i] + m->Wpe[t*NC + i];
    }
  }
  
  // 2. LayerNorm 1 + Q/K/V
  for (int t = 0; t < NT; t++) {
    real y[NC];
    ln_fwd(&c->h0[t*NC], m->g1, m->b1, y, &c->xn1[t*NC], &c->inv1[t]);
    
    real *qq = &c->q[t*NC], *kk = &c->k[t*NC], *vv = &c->v[t*NC];
    for (int j = 0; j < NC; j++) { qq[j] = 0; kk[j] = 0; vv[j] = 0; }
    
    for (int i = 0; i < NC; i++) {
      real yi = y[i];
      const real *wq = &m->Wq[i*NC], *wk = &m->Wk[i*NC], *wv = &m->Wv[i*NC];
      for (int j = 0; j < NC; j++) {
        qq[j] += yi * wq[j];
        kk[j] += yi * wk[j];
        vv[j] += yi * wv[j];
      }
    }
  }
  
  // 3. PLE 주입 (pre-attention)
#if USE_PLE
  ple_forward(m, c, X);
#endif
  
  // 4. 어텐션 (causal single-head)
  real isc = 1.0 / sqrt((real)NC);
  for (int t = 0; t < NT; t++) {
    real *Arow = &c->A[t*NT];
    real mx = -1e30;
    
    // Q·K^T
    for (int u = 0; u <= t; u++) {
      real sc = 0;
      for (int i = 0; i < NC; i++) {
        sc += c->q[t*NC + i] * c->k[u*NC + i];
      }
      sc *= isc;
      Arow[u] = sc;
      if (sc > mx) mx = sc;
    }
    
    // 소프트맥스 (causal mask)
    real sum = 0;
    for (int u = 0; u <= t; u++) {
      Arow[u] = exp(Arow[u] - mx);
      sum += Arow[u];
    }
    for (int u = 0; u <= t; u++) Arow[u] /= sum;
    for (int u = t+1; u < NT; u++) Arow[u] = 0;
    
    // V 가중 합
    for (int i = 0; i < NC; i++) {
      real a = 0;
      for (int u = 0; u <= t; u++) {
        a += Arow[u] * c->v[u*NC + i];
      }
      c->ao[t*NC + i] = a;
    }
  }
  
  // 5. Output projection + residual
  for (int t = 0; t < NT; t++) {
    real acc[NC] = {0};
    for (int i = 0; i < NC; i++) {
      real ai = c->ao[t*NC + i];
      const real *wo = &m->Wo[i*NC];
      for (int j = 0; j < NC; j++) {
        acc[j] += ai * wo[j];
      }
    }
    for (int j = 0; j < NC; j++) {
      c->h1[t*NC + j] = c->h0[t*NC + j] + acc[j];
      
#if USE_PLE
      // PLE 게이트 추가
      for (int l = 0; l < NLAYER; l++) {
        real ple_val = c->ple[t*NLAYER*NPLE + l*NPLE];
        // ple_proj(gelu(ple_gate(h1)) * ple_val)
        // 단순화: ple_val 직접 적용
        c->h1[t*NC + j] += ple_val * 0.1f;  // 스케일링 팩터
      }
#endif
    }
  }
  
  // 6. LayerNorm 2 + FFN (ReLU)
  for (int t = 0; t < NT; t++) {
    real y[NC];
    ln_fwd(&c->h1[t*NC], m->g2, m->b2, y, &c->xn2[t*NC], &c->inv2[t]);
    
    // FFN up
    real *f1r = &c->f1[t*NF], *far = &c->fa[t*NF];
    for (int j = 0; j < NF; j++) f1r[j] = m->bf1[j];
    for (int i = 0; i < NC; i++) {
      real yi = y[i];
      const real *w1 = &m->W1[i*NF];
      for (int j = 0; j < NF; j++) {
        f1r[j] += yi * w1[j];
      }
    }
    // ReLU
    for (int j = 0; j < NF; j++) {
      far[j] = f1r[j] > 0 ? f1r[j] : 0;
    }
    
    // FFN down + residual
    real acc2[NC];
    for (int j = 0; j < NC; j++) acc2[j] = m->bf2[j];
    const real *far = &c->fa[t*NF];
    for (int i = 0; i < NF; i++) {
      real fi = far[i];
      if (fi == 0) continue;
      const real *w2 = &m->W2[i*NC];
      for (int j = 0; j < NC; j++) {
        acc2[j] += fi * w2[j];
      }
    }
    for (int j = 0; j < NC; j++) {
      c->h2[t*NC + j] = c->h1[t*NC + j] + acc2[j];
    }
  }
  
  // 7. 최종 LayerNorm + 로짓 (tied head)
  real loss = 0;
  for (int t = 0; t < NT; t++) {
    real y[NC];
    ln_fwd(&c->h2[t*NC], m->gf, m->bf, y, &c->xnf[t*NC], &c->invf[t]);
    
    real mx = -1e30;
    for (int v = 0; v < NV; v++) {
      real a = 0;
      for (int i = 0; i < NC; i++) {
        a += y[i] * m->Wte[v*NC + i];  // tied: Wte 재사용
      }
      c->logits[t*NV + v] = a;
      if (a > mx) mx = a;
    }
    
    // 소프트맥스
    real sum = 0;
    for (int v = 0; v < NV; v++) {
      c->P[t*NV + v] = exp(c->logits[t*NV + v] - mx);
      sum += c->P[t*NV + v];
    }
    for (int v = 0; v < NV; v++) {
      c->P[t*NV + v] /= sum;
    }
    
    // 교차 엔트로피
    if (Y) {
      c->Y[t] = Y[t];
      real p = c->P[t*NV + Y[t]];
      if (p < 1e-9) p = 1e-9;
      loss += -log(p);
    }
  }
  
  return Y ? loss / NT : 0;
}

// === 역전파 (PLE 테이블 제외) ===
static real dh0[NT*NC], dh1[NT*NC], dh2[NT*NC];
static real dq[NT*NC], dk[NT*NC], dv[NT*NC], dao[NT*NC], dA[NT*NT];

void backward(Model *m, Cache *c, Model *g) {
  memset(dh0, 0, sizeof(dh0));
  memset(dh1, 0, sizeof(dh1));
  memset(dq, 0, sizeof(dq));
  memset(dk, 0, sizeof(dk));
  memset(dv, 0, sizeof(dv));
  
  // 1. 출력 헤드 + 소프트맥스 그래디언트
  for (int t = 0; t < NT; t++) {
    real dlog[NV];
    for (int v = 0; v < NV; v++) {
      dlog[v] = c->P[t*NV + v] / NT;
    }
    dlog[c->Y[t]] -= 1.0 / NT;
    
    // tied head: dL/dWte += dlog ⊗ y
    real ylnf[NC];
    for (int i = 0; i < NC; i++) {
      ylnf[i] = c->xnf[t*NC + i] * m->gf[i] + m->bf[i];
    }
    
    real dy[NC] = {0};
    for (int v = 0; v < NV; v++) {
      for (int i = 0; i < NC; i++) {
        g->Wte[v*NC + i] += dlog[v] * ylnf[i];
        dy[i] += dlog[v] * m->Wte[v*NC + i];
      }
    }
    
    ln_bwd(dy, &c->xnf[t*NC], c->invf[t], m->gf, 
           &dh2[t*NC], g->gf, g->bf);
  }
  
  // 2. FFN 역전파
  for (int t = 0; t < NT; t++) {
    for (int i = 0; i < NC; i++) {
      dh1[t*NC + i] += dh2[t*NC + i];
    }
    
    real *df2 = &dh2[t*NC];
    real dfa[NF] = {0};
    
    for (int i = 0; i < NF; i++) {
      real a = 0;
      for (int j = 0; j < NC; j++) {
        g->W2[i*NC + j] += c->fa[t*NF + i] * df2[j];
        a += df2[j] * m->W2[i*NC + j];
      }
      dfa[i] = a;
    }
    for (int j = 0; j < NC; j++) g->bf2[j] += df2[j];
    
    // ReLU backward
    real df1[NF];
    for (int i = 0; i < NF; i++) {
      df1[i] = c->f1[t*NF + i] > 0 ? dfa[i] : 0;
    }
    
    // FFN up backward
    real dln2y[NC] = {0};
    for (int i = 0; i < NC; i++) {
      for (int j = 0; j < NF; j++) {
        g->W1[i*NF + j] += c->xn2[t*NC + i] * df1[j];
        dln2y[i] += df1[j] * m->W1[i*NF + j];
      }
    }
    for (int j = 0; j < NF; j++) g->bf1[j] += df1[j];
    
    real dh1b[NC];
    ln_bwd(dln2y, &c->xn2[t*NC], c->inv2[t], m->g2, 
           dh1b, g->g2, g->b2);
    for (int i = 0; i < NC; i++) {
      dh1[t*NC + i] += dh1b[i];
    }
  }
  
  // 3. 어텐션 역전파
  memset(dao, 0, sizeof(dao));
  for (int t = 0; t < NT; t++) {
    for (int i = 0; i < NC; i++) {
      dh0[t*NC + i] += dh1[t*NC + i];
    }
    
    real *dap = &dh1[t*NC];
    for (int i = 0; i < NC; i++) {
      for (int j = 0; j < NC; j++) {
        g->Wo[i*NC + j] += c->ao[t*NC + i] * dap[j];
        dao[t*NC + i] += dap[j] * m->Wo[i*NC + j];
      }
    }
  }
  
  memset(dA, 0, sizeof(dA));
  for (int t = 0; t < NT; t++) {
    for (int u = 0; u <= t; u++) {
      real s = 0;
      for (int i = 0; i < NC; i++) {
        s += dao[t*NC + i] * c->v[u*NC + i];
      }
      dA[t*NT + u] = s;
    }
    for (int u = 0; u <= t; u++) {
      for (int i = 0; i < NC; i++) {
        dv[u*NC + i] += c->A[t*NT + u] * dao[t*NC + i];
      }
    }
  }
  
  real isc = 1.0 / sqrt((real)NC);
  for (int t = 0; t < NT; t++) {
    real dot = 0;
    for (int u = 0; u <= t; u++) {
      dot += dA[t*NT + u] * c->A[t*NT + u];
    }
    
    real dscore[NT] = {0};
    for (int u = 0; u <= t; u++) {
      dscore[u] = c->A[t*NT + u] * (dA[t*NT + u] - dot) * isc;
    }
    
    for (int u = 0; u <= t; u++) {
      for (int i = 0; i < NC; i++) {
        dq[t*NC + i] += dscore[u] * c->k[u*NC + i];
        dk[u*NC + i] += dscore[u] * c->q[t*NC + i];
      }
    }
  }
  
  // 4. Q/K/V 역전파
  for (int t = 0; t < NT; t++) {
    real yln1[NC];
    for (int i = 0; i < NC; i++) {
      yln1[i] = c->xn1[t*NC + i] * m->g1[i] + m->b1[i];
    }
    
    real dln1[NC] = {0};
    for (int i = 0; i < NC; i++) {
      for (int j = 0; j < NC; j++) {
        g->Wq[i*NC + j] += yln1[i] * dq[t*NC + j];
        g->Wk[i*NC + j] += yln1[i] * dk[t*NC + j];
        g->Wv[i*NC + j] += yln1[i] * dv[t*NC + j];
        
        dln1[i] += dq[t*NC + j] * m->Wq[i*NC + j]
                 + dk[t*NC + j] * m->Wk[i*NC + j]
                 + dv[t*NC + j] * m->Wv[i*NC + j];
      }
    }
    
    real dh0b[NC];
    ln_bwd(dln1, &c->xn1[t*NC], c->inv1[t], m->g1, 
           dh0b, g->g1, g->b1);
    for (int i = 0; i < NC; i++) {
      dh0[t*NC + i] += dh0b[i];
    }
  }
  
  // 5. 임베딩 그래디언트
  for (int t = 0; t < NT; t++) {
    for (int i = 0; i < NC; i++) {
      g->Wte[c->X[t]*NC + i] += dh0[t*NC + i];
      g->Wpe[t*NC + i] += dh0[t*NC + i];
    }
  }
  
  // 주의: PLE 테이블 그래디언트는 계산하지 않음 (고정)
  // g->ple_table 는 NULL 또는 skip
}

#define NPAR (int)(sizeof(Model) / sizeof(real))
