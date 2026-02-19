---
name: drift-check
description: 마일스톤 사이에 팀의 빌드가 합의된 방향에서 벗어나는지 점검하고 Drift Report를 작성한다.
---

# Drift Check

마일스톤 사이에 팀의 빌드가 합의된 방향에서 벗어나는지 점검한다. → Milestone Review가 플레이어 경험을 검증한다면, Drift Check는 팀 산출물이 범위 안에 있는지 검증한다.

## 전제 조건
- Direction One-Pager(Priority Matrix 포함)가 이미 존재해야 한다.
- 팀이 `skills/drift-check/change-list-template.md` 양식으로 변경 사항 리스트를 제출해야 한다.

## 점검 절차 (고정 순서)
1. **Priority Matrix 매핑**: 각 변경 사항을 원본 Priority Matrix의 Must/Should/Won't에 매핑한다.
2. **경계 위반 식별**: Won't 항목이 빌드에 포함되었거나, Should가 Must 리소스를 밀어내는 경우를 식별한다. → Won't 포함은 합의 외 범위 확장, Should→Must 밀어냄은 우선순위 역전 신호다. 이 두 패턴이 scope creep의 가장 흔한 진입 경로다.
3. **표류 판정**: 아래 기준에 따라 Aligned / Drifting / Off-Track 중 하나를 선택한다.
4. **Decision Log 업데이트**: 판정 결과, 위반 항목, 권고 조치를 기록한다. → 표류는 한 번의 판정이 아니라 누적 추이로 심각성을 판단한다. 기록이 있어야 패턴을 볼 수 있다.

## 판정 기준
→ 단계를 셋으로 나누는 이유: 소수의 이탈은 즉시 교정 가능하지만, 다수의 이탈이나 Must 지연은 방향 자체가 흔들리는 구조적 문제다. 조기에 Drifting을 잡아야 Off-Track으로 넘어가는 것을 막을 수 있다.
- **Aligned**: 모든 변경이 Must/Should 범위 안에 있다.
- **Drifting**: Won't 항목이 1~2개 포함되었거나, Should가 Must를 밀어내는 징후가 있다.
- **Off-Track**: Won't 항목이 다수 포함되었거나, Must 항목이 지연/축소되고 있다.

## Output Format

```markdown
# Drift Report — [점검 기간]

## 1) 참조 기준
- Priority Matrix: (원본 Direction에서 인용)

## 2) 표류 판정
- 판정: [Aligned / Drifting / Off-Track]

## 3) 경계 위반 항목
| 변경 사항 | 원래 우선순위 | 위반 유형 |
|-----------|-------------|----------|
| | | |

## 4) 권고 조치
- [제거 / 재조정 / 수용 후 Priority 업데이트] + 근거

## 5) Decision Log 추가 항목
- 결정:
- 근거:
- 버린 대안:
```

## Gate Policy: Drift Check
- 목적: 마일스톤 사이 범위 이탈 조기 감지
- 전제: Direction One-Pager가 존재하고, 변경 사항 리스트가 제출됨
- 처리: Soft Gate만 적용 (경량 점검) → Drift Check는 조기 경보다. 차단(Hard Gate)은 반복 속도를 불필요하게 늦추고, 팀이 점검 자체를 회피하게 만든다. 인지와 교정이 목적이므로 권고로 충분하다.

### Soft Gate (권고)
1. 모든 변경 사항이 Priority Matrix 카테고리에 매핑되어야 한다.
2. 위반 항목에 권고 조치가 있어야 한다.

## Included Files

- **Change list input template**: `change-list-template.md` — 팀이 변경 사항 리스트를 구조화하여 제출하는 양식
