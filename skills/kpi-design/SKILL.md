---
name: kpi-design
description: KPI Selection Guidelines에 따라 KPI Plan을 작성하고, 7가지 품질 규칙으로 검증한다.
---

# KPI Design — Validate

`game-directing` 스킬이 생성한 KPI 플랜을 7가지 품질 규칙으로 검증하는 도구.

## KPI Selection Guidelines
KPI Plan(섹션 10~12) 작성 시 `kpi-catalog.md`를 참조하여 다음 규칙을 준수한다.

1. `kpi-catalog.md`를 참조하되, 카탈로그에 없는 게임 고유 KPI도 추가 가능하다.
2. Stage Profile의 제약(max_kpis, excluded_categories)을 준수한다.
3. 반드시 retention 지표 1개 이상, early-session 품질 지표(activation/onboarding) 1개 이상 포함한다.
4. Player Promise·핵심 재미와 직접 연결되는 KPI를 우선 선택한다.
5. 모든 KPI에 Decision Rule을 필수로 작성한다.

## Workflow

1. `game-directing` 스킬이 `kpi-catalog.md`를 참조하여 KPI Plan(섹션 10~12)을 작성한다.
2. 작성된 KPI Plan 마크다운을 이 스킬의 validate 스크립트로 검증한다.
3. FAIL 시 에이전트가 Blocking Issues를 수정하고 재검증한다.

## Included Files

- **Validator script**: `scripts/kpi_design.py`
- **KPI reference catalog**: `kpi-catalog.md` — 에이전트가 KPI 선택 시 참조하는 레퍼런스

## How To Run

프로젝트 루트에서 실행:

```bash
# 기본 검증
python3 skills/kpi-design/scripts/kpi_design.py --input /absolute/path/to/kpi-plan.md --stage prototype

# JSON 출력
python3 skills/kpi-design/scripts/kpi_design.py --input /absolute/path/to/kpi-plan.md --stage soft-launch --format json

# Stage 자동 감지 (문서 내 Stage 필드에서 추출)
python3 skills/kpi-design/scripts/kpi_design.py --input /absolute/path/to/kpi-plan.md
```

## Inputs

| Parameter | Values | Required | Default |
|-----------|--------|----------|---------|
| `--input` | KPI plan 마크다운 파일 경로 | Yes | — |
| `--stage` | `prototype`, `soft-launch`, `live` | No | 문서에서 자동 감지 |
| `--format` | `text`, `json` | No | `text` |

## Validation Checks (7가지)

1. **Decision Rule 누락** (Error): 모든 KPI에 Decision Rule이 있는지 확인
2. **Vanity Metric 감지** (Warning): 절대 수치 기반 지표에 의사결정 연결이 없는 경우
3. **KPI 수 초과** (Error): stage별 상한(prototype 5 / soft-launch 6 / live 7) 초과 시
4. **Retention 지표 부재** (Error): retention 관련 KPI가 최소 1개 필요
5. **Early-session 지표 부재** (Error): activation/onboarding KPI가 최소 1개 필요
6. **Stage 적합성** (Error/Warning):
   - Error: 스테이지 필수 카테고리 KPI 누락 시 (예: live에서 monetization 부재)
   - Warning: 해당 스테이지 제외 카테고리 포함 시 (예: prototype에서 monetization)
7. **Instrumentation Events 부재** (Warning): KPI를 측정하는 이벤트가 섹션 12에 매핑되어야 함

## Expected Input Format

스크립트가 파싱하는 KPI Plan 마크다운 구조:
(`## KPI Plan` 또는 `## 10) KPI Plan` 같은 번호형 헤딩 모두 허용)

```markdown
## KPI Plan
- Stage: prototype
- Primary Outcome Metric: Core Loop Completion Rate
  - Definition: users_completed_first_loop / users_started_first_loop
  - Target: >= 80%
  - Window: per session
  - Rationale: ...
  - Decision Rule: If < 80%: simplify onboarding

## Supporting KPIs
- D1 Retention
  - Formula: users_returned_day1 / users_installed_day0
  - Target: 30-40%
  - Window: daily cohort
  - Rationale: ...
  - Decision Rule: If < 30%: audit tutorial flow
```

## Output

### text (기본)

```
VALIDATION: PASS / FAIL
ERRORS: N
WARNINGS: N

Blocking Issues:
- [issue]

Recommended Improvements:
- [issue]
```

### json

```json
{
  "status": "PASS / FAIL",
  "errors": [ ... ],
  "warnings": [ ... ]
}
```

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Validation PASS |
| 1 | Validation FAIL (blocking errors found) |
| 2 | I/O or argument error |

## Quality Gate
1. [ ] KPI Plan 작성 후 `python3 skills/kpi-design/scripts/kpi_design.py --input <파일경로> --stage <스테이지>` 실행으로 검증했는가
