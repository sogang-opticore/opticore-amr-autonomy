# Experiment artifacts

이 디렉터리에는 Phase 0 결론을 검토하는 데 필요한 산출물만 선별해 버전 관리한다.

포함 항목:

- `full/`: 초기 BC/PPO 전체 평가 결과와 대표 데모
- `reward_v2/`: reward-v2 비교 결과
- `ppo_best/`: shared-encoder BC warm-start best 평가 결과
- `ppo_stable/`: Stable PPO 학습 곡선, 최종 paired 평가, 대표 데모
- `checkpoints/ppo_stable_full_best.pt`: 51,200 timestep에서 선택된 Stable PPO best
- Stable PPO 학습 metrics와 training log

제외 항목:

- 다시 생성 가능한 teacher/DAgger 데이터셋
- smoke 실행 결과
- Matplotlib 및 pytest cache
- 중간·실패 실험 checkpoint
- Stable PPO final checkpoint

전체 pipeline과 각 학습 명령은 `packages/amr_lite/README.md`를 참고한다.
