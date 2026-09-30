# M1 Reproducibility

M1에서 고정한 평가 protocol, 결과 해석과 검토 가능한 선별 산출물을 보존한다.

- `RESULTS.md`: 공식 결과와 실패 집중 구간
- `artifacts/REPORT.md`: 실행이 생성한 요약 보고서
- `artifacts/manifest.json`: 원본 commit, 환경, dataset/checkpoint hash
- `artifacts/resolved_config.json`: 실제 적용 설정
- `artifacts/test/`: 집계·시나리오·학습 seed·paired comparison CSV
- `artifacts/plots/`: 성공률, 충돌률, episode time 그래프

원본 episode CSV와 학습 checkpoint는 크기와 재생성 가능성을 고려해 Git에 넣지 않는다.
원본 실행의 절대 경로가 manifest에 남아 있을 수 있으며, 이는 provenance를 보존하기
위한 값이다. 현재 패키지에서 새로 실행한 결과는 `packages/amr_lite/artifacts/m1/`
아래에 생성한다.
