# Docker 빌드 수정 요약

**날짜:** 2026-02-12 | **목표:** vLLM v0.15.1 + CUDA 12.8 + RTX 5090 (sm_120) 소스 빌드

## 이슈 및 수정 사항

| # | 에러 | 원인 | 수정 방법 | 커밋 |
|---|------|-----|----------|------|
| 1 | `setuptools` 라이선스 필드 파싱 실패 | 베이스 이미지의 setuptools 70.3이 PEP 639 미지원 | `pip install "setuptools>=75.0" "packaging>=24.2"` | `25dfb9d` |
| 2 | PyTorch 다운로드 중 디스크 부족 | `requirements/build.txt`가 torch를 재다운로드 (~5GB) | `--no-build-isolation`으로 베이스 이미지의 torch 재사용 | `25dfb9d` |
| 3 | triton 클론 중 `No space left on device` | cmake FetchContent가 triton 전체 클론 (460MB+) | `--depth 1`로 사전 클론 후 `sed`로 cmake 파일에 경로 주입 | `f2afc90` |
| 4 | `libmpi.so.40` 파일 없음 | 정리 과정에서 PyTorch가 의존하는 `/opt/hpcx/` 삭제 | 정리 목록에서 `/opt/hpcx/` 제거 | `596925a` |
| 5 | `"Float8_e8m0fnu" is not a member` | PyTorch 2.6에 vLLM v0.15.1이 필요로 하는 Blackwell FP8 타입 없음 | vLLM 빌드 전 `pip install torch --index-url .../whl/cu128`으로 업그레이드 | `9c17f82` |

## 잘못된 수정 (되돌림)

| 커밋 | 변경 내용 | 잘못된 이유 |
|------|---------|-----------|
| `20c99f8` | arch를 `9.0+PTX`로 변경 | 콜드 스타트 시 JIT 컴파일로 30-60초 지연 발생 |
| `e21538c` | arch를 `10.0` (sm_100)으로 변경 | sm_100 = 데이터센터 Blackwell, RTX 5090 (sm_120)에서 실행 불가 |

## CUDA 아키텍처 참조표

| TORCH_CUDA_ARCH_LIST | sm_ | GPU | 비고 |
|---------------------|-----|-----|------|
| 8.0 | sm_80 | A100 | Ampere 데이터센터 |
| 8.6 | sm_86 | A40 | Ampere 워크스테이션 |
| 8.9 | sm_89 | L40, L40S | Ada Lovelace |
| 9.0 | sm_90 | H100 | Hopper |
| **12.0** | **sm_120** | **RTX 5090** | **Blackwell 컨슈머** |
| 10.0 | sm_100 | B100/B200 | Blackwell 데이터센터 (RTX 5090 아님!) |

## 빌드 요구 사항

| 구성 요소 | 필요 버전 | 베이스 이미지 버전 | 조치 |
|----------|----------|----------------|------|
| CUDA 툴킷 | 12.8+ (sm_120 컴파일용) | 12.8.1 | 불필요 |
| PyTorch | 2.7+ (Float8_e8m0fnu 타입용) | 2.6 | cu128 wheel로 업그레이드 |
| setuptools | 75.0+ (PEP 639 지원용) | 70.3 | 업그레이드 |
| 디스크 여유 공간 | 빌드에 ~15GB 필요 | ~14GB (GitHub Actions) | 정리 + triton 얕은 클론 |
