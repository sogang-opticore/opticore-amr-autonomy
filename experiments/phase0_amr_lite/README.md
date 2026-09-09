# Phase 0 — AMR-Lite Technical Spike

이 디렉터리는 AMR-Lite Phase 0에서 얻은 결정, 결과, 알려진 한계와 선별 산출물을 보존한다. 실행 코드는 계속 개선할 수 있도록 `packages/amr_lite/`에 분리했다.

## 기준선

- Rule baseline, BC, DAgger-lite, PPO, Stable PPO 비교
- 8개 시나리오 × 2개 seed의 paired evaluation
- Safety Shield OFF/ON 비교
- Stable PPO best: 51,200 timestep
- Stable PPO best Shield OFF 성공률: 87.5%
- 주요 실패: `D1` 정면 접근 충돌, `S4` Shield saturation

이 수치는 기술 가능성을 확인한 제한적 실험 결과이며 일반화 또는 운영 안전성을 의미하지 않는다.

## 문서와 산출물

- [RESULTS.md](RESULTS.md): 정량 결과와 해석
- [DECISIONS.md](DECISIONS.md): 설계 결정과 변경 근거
- [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md): 알려진 제약과 미해결 문제
- [artifacts/](artifacts/): 결과 CSV, 그래프, 대표 데모, Stable PPO best

Phase 0 원본 코드·문서 상태는 구조 변경 직전 커밋 `585969a`에서도 확인할 수 있다.
