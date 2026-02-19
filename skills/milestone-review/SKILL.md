---
name: milestone-review
description: 마일스톤마다 플레이테스트 결과를 Direction One-Pager와 대조하여 Milestone Review Report를 작성하고, 검증 스크립트로 구조 품질을 확인한다.
---

# Milestone Review — Validate

`game-directing` 스킬이 작성한 Milestone Review Report를 3가지 규칙으로 검증하는 도구.

## Milestone Review Workflow

Direction One-Pager 작성 이후, 마일스톤마다 플레이테스트 결과를 받아 방향성을 검증하고 조정한다.

### 전제 조건
- Direction One-Pager가 이미 존재해야 한다.
- 팀이 `skills/milestone-review/playtest-input-template.md` 양식으로 플레이테스트 결과를 제출해야 한다.

### 판단 기준

#### Promise Delivery 판정
- **Delivering**: 핵심 재미가 의도한 대로 전달되고 있다. 정량 지표가 목표 범위 안에 있고, 정성 관찰에서 Player Promise에 부합하는 행동/반응이 확인된다.
- **At Risk**: 부분적으로 전달되나 특정 구간에서 약속이 깨진다. 정량 지표 일부가 목표 미달이거나, 정성 관찰에서 혼란/이탈 신호가 있다.
- **Broken**: 핵심 재미/핵심 경험이 전달되지 않고 있다. 정량 지표 다수가 목표 미달이고, 정성 관찰에서 Player Promise와 무관한 행동이 주로 나타난다.

#### 방향 조정 수준
- **Stay**: 현재 방향 유지. 실행 품질만 개선하면 된다. Priority Matrix 변경 없음.
- **Adjust**: 방향은 유지하되 Priority Matrix 일부를 조정한다. 변경 항목과 근거를 반드시 명시한다.
- **Reconsider**: Player Promise 또는 핵심 재미 자체를 재검토해야 한다. Decision Workflow를 다시 수행한다.

### 리뷰 절차 (고정 순서)
1. **참조 기준 확인**: 원본 Direction One-Pager에서 Player Promise와 핵심 재미를 인용한다.
2. **입력 데이터 분석**: 정량 데이터와 정성 관찰을 분리하여 읽는다. 정량 데이터는 KPI 목표와 대조하고, 정성 관찰은 Player Promise 전달 여부 관점에서 해석한다.
3. **Promise Delivery 판정**: 위 판정 기준에 따라 Delivering / At Risk / Broken 중 하나를 선택한다.
4. **핵심 발견 도출**: 입력 데이터에서 방향성에 가장 큰 영향을 미치는 발견을 최대 3개 추출한다. 각 발견은 구체적 데이터를 인용해야 한다.
5. **방향 조정 결정**: Stay / Adjust / Reconsider 중 하나를 선택한다. Adjust 시 Priority Matrix 변경 내용과 근거를 명시한다.
6. **집중 과제 설정**: 다음 마일스톤까지 팀이 집중해야 할 과제를 최대 3개 설정한다.
7. **Decision Log 업데이트**: 이번 리뷰에서 내린 결정, 근거, 버린 대안을 기록한다.

## Milestone Review Report Template

```markdown
# Milestone Review — [마일스톤명]

## 1) 참조 기준
- Player Promise: (원본 Direction에서 인용)
- 핵심 재미: (원본 Direction에서 인용)

## 2) Promise Delivery 판정
- 판정: [Delivering / At Risk / Broken]
- 근거: (입력 데이터에서 구체적 인용)

## 3) 핵심 발견 (최대 3개)
1.
2.
3.

## 4) 방향 조정
- 수준: [Stay / Adjust / Reconsider]
- Priority Matrix 변경: (있으면)
  - 변경 내용:
  - 변경 근거:

## 5) 다음 마일스톤 집중 과제 (최대 3개)
1.
2.
3.

## 6) Decision Log 추가 항목
- 결정:
- 근거:
- 버린 대안:
```

## Gate Policy: Milestone Review
- 목적: 마일스톤별 방향성 검증 및 조정
- 전제: Direction One-Pager가 존재하고, 플레이테스트 입력이 제출됨
- 처리: 아래 Hard Gate 중 1개라도 실패하면 Report 전달 차단

### Hard Gate (차단 조건)
1. Promise Delivery 판정(Delivering / At Risk / Broken)이 명시되어야 한다. ← script #1
2. 판정 근거가 입력 데이터를 구체적으로 인용해야 한다.
3. 원본 Direction의 Player Promise를 참조해야 한다.
4. 방향 조정 수준(Stay / Adjust / Reconsider)이 명시되어야 한다. ← script #2
5. Priority Matrix 변경 시 변경 근거가 있어야 한다.
6. 다음 마일스톤 집중 과제가 최소 1개 명시되어야 한다.

## Included Files

- **Validator script**: `scripts/milestone_review.py`
- **Playtest input template**: `playtest-input-template.md` — 팀이 플레이테스트 결과를 구조화하여 제출하는 양식

## How To Run

프로젝트 루트에서 실행:

```bash
# 기본 검증
python3 skills/milestone-review/scripts/milestone_review.py --input /absolute/path/to/review-report.md

# JSON 출력
python3 skills/milestone-review/scripts/milestone_review.py --input /absolute/path/to/review-report.md --format json
```

## Inputs

| Parameter | Values | Required | Default |
|-----------|--------|----------|---------|
| `--input` | Milestone Review Report 마크다운 파일 경로 | Yes | — |
| `--format` | `text`, `json` | No | `text` |

## Validation Checks (3가지)

1. **Promise Delivery 판정값 검증** (Error): Delivering / At Risk / Broken 외의 값을 사용하면 차단
2. **방향 조정 수준값 검증** (Error): Stay / Adjust / Reconsider 외의 값을 사용하면 차단
3. **집중 과제 수 초과** (Warning): 집중 과제가 3개를 초과하면 focus 분산 경고

## Expected Input Format

스크립트가 파싱하는 Milestone Review Report 마크다운 구조는 위의 **Milestone Review Report Template**을 참조한다.

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
