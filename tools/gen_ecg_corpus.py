#!/usr/bin/env python3
"""
gen_ecg_corpus.py — 심박/호흡 시계열 데이터를 문자열 코퍼스로 변환

ECG(심전도) 및 호흡 신호를 심볼릭 시퀀스로 변환하여 Qapla'/PLE-HandGPT 가
학습할 수 있는 텍스트 코퍼스를 생성합니다.

사용법:
    python tools/gen_ecg_corpus.py --input ecg_data.csv --output corpus/ecg_train.txt
    python tools/gen_ecg_corpus.py --demo  # 데모 데이터 생성
"""

import argparse
import math
import random
from pathlib import Path

# 심볼릭 알파벳 정의 (총 16 심볼 → NV=16 으로 설정 가능)
# 각 심볼은 심박수/호흡의 특정 상태를 나타냄
SYMBOLS = {
    'N': '정상',      # Normal rhythm
    'T': '빈맥',      # Tachycardia (fast)
    'B': '서맥',      # Bradycardia (slow)
    'I': '불규칙',    # Irregular
    'A': '무호흡',    # Apnea (no breathing)
    'S': '얕은호흡',  # Shallow breathing
    'D': '깊은호흡',  # Deep breathing
    'R': '급격한변화', # Rapid change
    'n': '정상미세',  # Normal micro-variation
    't': '빈맥미세',
    'b': '서맥미세',
    'i': '불규칙미세',
    '.': '구분자',     # Segment separator
    '\n': '샘플구분',   # Sample separator
}

def generate_demo_ecg(samples=100, seq_len=64):
    """
    데모 ECG/호흡 데이터 생성
    실제 센서 데이터가 없을 때 테스트용으로 사용
    """
    print(f"데모 ECG 데이터 생성: {samples} 샘플, 각 {seq_len} 심볼")
    
    corpus = []
    
    for i in range(samples):
        # 정상 리듬 (70% 확률)
        if random.random() < 0.7:
            # 정상 심박: 주로 'N'과 'n' 조합
            seq = [random.choices(['N', 'n'], weights=[0.7, 0.3])[0] 
                   for _ in range(seq_len)]
        else:
            # 이상 리듬 (30% 확률)
            anomaly_type = random.choice(['T', 'B', 'I', 'A', 'S', 'D', 'R'])
            
            # 이상 패턴 생성
            if anomaly_type in ['T', 'B', 'I']:  # 심박 이상
                base = anomaly_type.upper()
                seq = [random.choices([base, base.lower(), 'N'], 
                                      weights=[0.5, 0.3, 0.2])[0] 
                       for _ in range(seq_len)]
            else:  # 호흡 이상
                seq = [random.choices([anomaly_type, anomaly_type.lower(), 'N'], 
                                      weights=[0.4, 0.3, 0.3])[0] 
                       for _ in range(seq_len)]
        
        # 구분자 추가 (8 심볼마다)
        formatted = []
        for j, s in enumerate(seq):
            formatted.append(s)
            if (j + 1) % 8 == 0 and j < seq_len - 1:
                formatted.append('.')
        
        corpus.append(''.join(formatted))
    
    return '\n'.join(corpus)

def ecg_to_symbols(ecg_values, threshold_fast=100, threshold_slow=50):
    """
    ECG 값 (BPM) 을 심볼로 변환
    
    Args:
        ecg_values: 심박수 리스트 (BPM)
        threshold_fast: 빈맥 기준 (기본 100 BPM)
        threshold_slow: 서맥 기준 (기본 50 BPM)
    
    Returns:
        심볼 문자열
    """
    symbols = []
    
    for i, bpm in enumerate(ecg_values):
        if bpm > threshold_fast:
            symbols.append('T')
        elif bpm < threshold_slow:
            symbols.append('B')
        else:
            # 정상 범위에서 미세 변동
            if abs(bpm - 70) < 5:
                symbols.append('n')
            else:
                symbols.append('N')
        
        # 불규칙성 감지 (이전 값과 큰 차이)
        if i > 0 and abs(ecg_values[i] - ecg_values[i-1]) > 20:
            symbols[-1] = 'I' if symbols[-1] not in ['T', 'B'] else 'i'
    
    return ''.join(symbols)

def main():
    parser = argparse.ArgumentParser(description='ECG/호흡 데이터를 심볼릭 코퍼스로 변환')
    parser.add_argument('--input', type=str, help='입력 CSV 파일 (time,bpm,breath)')
    parser.add_argument('--output', type=str, default='corpus/ecg_train.txt', 
                        help='출력 코퍼스 파일')
    parser.add_argument('--demo', action='store_true', help='데모 데이터 생성')
    parser.add_argument('--samples', type=int, default=1000, help='데모 샘플 수')
    parser.add_argument('--seq-len', type=int, default=64, help='시퀀스 길이')
    parser.add_argument('--holdout', type=float, default=0.1, help='검증 세트 비율')
    
    args = parser.parse_args()
    
    if args.demo:
        print("=== BioGPT-S3 데모 코퍼스 생성 ===\n")
        corpus = generate_demo_ecg(args.samples, args.seq_len)
        
        # 파일로 저장
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(corpus, encoding='utf-8')
        
        # 통계 출력
        lines = corpus.split('\n')
        unique_chars = sorted(set(corpus.replace('\n', '')))
        
        print(f"생성 완료: {args.output}")
        print(f"  샘플 수: {len(lines)}")
        print(f"  총 문자 수: {len(corpus)}")
        print(f"  어휘: {unique_chars}")
        print(f"  어휘 크기: {len(unique_chars)}")
        print(f"\n심볼 의미:")
        for sym, desc in SYMBOLS.items():
            if sym in unique_chars:
                print(f"  '{sym}': {desc}")
        
        # Hold-out 세트 생성
        if args.holdout > 0:
            holdout_size = int(len(lines) * args.holdout)
            random.shuffle(lines)
            train_lines = lines[holdout_size:]
            val_lines = lines[:holdout_size]
            
            train_corpus = '\n'.join(train_lines)
            val_corpus = '\n'.join(val_lines)
            
            train_path = output_path.parent / 'ecg_train.txt'
            val_path = output_path.parent / 'ecg_val.txt'
            
            train_path.write_text(train_corpus, encoding='utf-8')
            val_path.write_text(val_corpus, encoding='utf-8')
            
            print(f"\nHold-out 분리 완료:")
            print(f"  훈련: {len(train_lines)} 샘플 ({train_path})")
            print(f"  검증: {len(val_lines)} 샘플 ({val_path})")
        
        return
    
    if not args.input:
        print("에러: --input 또는 --demo 중 하나를 지정하세요")
        parser.print_help()
        return
    
    # 실제 CSV 처리 (구현 필요)
    print(f"CSV 파일 처리: {args.input}")
    print("이 기능은 실제 ECG 데이터 포맷에 맞게 확장하세요")

if __name__ == '__main__':
    main()
