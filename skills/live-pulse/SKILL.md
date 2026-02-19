---
name: live-pulse
description: 출시 후 실제 플레이어 데이터로 방향성을 검증하고 Live Pulse Report를 작성한다.
---

# Live Pulse Review

출시 후 실제 플레이어 데이터로 방향성을 검증한다. → Milestone Review는 통제된 플레이테스트에서 Player Promise 전달을 확인했지만, 통제 환경의 가정은 실제 시장에서 깨질 수 있다. 플레이어 규모, 플레이 맥락, 경쟁 환경이 모두 다르기 때문에 출시 후에도 디렉터의 방향 판단이 필요하다.

## 전제 조건
- Direction One-Pager(KPI Plan 포함)가 이미 존재해야 한다.
- 팀이 `skills/live-pulse/live-pulse-input-template.md` 양식으로 KPI 대시보드 스냅샷과 플레이어 피드백 요약을 제출해야 한다.

## 리뷰 절차 (고정 순서)
1. **KPI 상태 판정**: 각 KPI를 원본 KPI Plan의 Decision Rule에 따라 Green / Yellow / Red로 판정한다. → Decision Rule은 Direction 설계 시 이미 합의된 조건이다. 실제 수치 대조는 기계적으로 수행하며, 디렉터의 주관이 개입할 지점은 여기가 아니다.

### KPI 상태 판정 기준
- **Green**: Decision Rule 조건에 해당하지 않으며, 목표 범위를 충족한다.
- **Yellow**: 목표에 근접하나 Decision Rule 발동 직전 수준이거나, 최근 추세가 하락 중이다.
- **Red**: Decision Rule 조건에 해당한다. 명시된 조치를 실행한다.

2. **플레이어 시그널 추출**: 정성적 피드백에서 Player Promise와 관련된 신호를 추출한다. → KPI는 행동을 측정하지만 동기와 감정은 놓친다. 정성 피드백이 "왜 그 수치가 나왔는지"를 보완한다.
3. **종합 판단**: KPI 상태와 플레이어 시그널을 교차하여 방향 유효성을 판단한다. → KPI Green이라도 플레이어가 의도한 재미가 아닌 다른 이유로 머물 수 있고, KPI Red라도 핵심 재미 자체는 전달되고 있을 수 있다. 수치와 맥락을 함께 봐야 올바른 판단이 나온다.
4. **대응 수준 결정**: 아래 기준에 따라 Tweak / Pivot / Rethink 중 하나를 선택한다.
5. **Decision Log 업데이트**: 판단 근거, 대응 방향, 고려했으나 선택하지 않은 대안을 기록한다.

## 대응 수준
→ Milestone Review의 Stay/Adjust/Reconsider와 별도 체계를 두는 이유: 출시 후에는 실제 플레이어가 있으므로 변경의 비용과 영향 범위가 근본적으로 다르다. 개발 중 방향 조정은 계획 변경이지만, 라이브 대응은 기존 플레이어 경험에 직접 영향을 미친다.
- **Tweak**: Player Promise는 전달되고 있으나 수치 조정이 필요하다. → 밸런스, 보상량, 난이도 곡선 등 파라미터 수준의 변경. 방향은 유지한다.
- **Pivot**: Player Promise 전달에 구조적 약점이 있어 특정 기능/시스템을 수정해야 한다. → 핵심 루프의 일부 단계나 Priority Matrix의 Should 항목 조정. 방향은 유지하되 실행 방식을 바꾼다.
- **Rethink**: Player Promise 자체가 시장에서 유효하지 않다. → 핵심 재미 또는 타겟 플레이어 가정의 근본 재검토. Decision Workflow를 다시 수행한다.

## Output Format

```markdown
# Live Pulse Report — [리뷰 기간]

## 1) 참조 기준
- Player Promise: (원본 Direction에서 인용)
- KPI Plan: (원본 목표값 인용)

## 2) KPI 상태
| KPI | 목표 | 실측 | 상태 | Decision Rule 조치 |
|-----|------|------|------|-------------------|
| | | | [Green/Yellow/Red] | |

## 3) 플레이어 시그널
- Player Promise 강화 신호:
- Player Promise 약화 신호:
- 예상 외 신호:

## 4) 종합 판단
- 대응 수준: [Tweak / Pivot / Rethink]
- 판단 근거: (KPI 상태와 플레이어 시그널을 교차 인용)

## 5) 조치 권고
- 구체적 방향:
- 우선순위 변경: (있으면)

## 6) Decision Log 추가 항목
- 결정:
- 근거:
- 버린 대안:
```

## Gate Policy: Live Pulse Review
- 목적: 출시 후 실제 시장 데이터 기반 방향 판단
- 전제: Direction One-Pager가 존재하고, KPI 대시보드 스냅샷과 플레이어 피드백 요약이 제출됨
- 처리: Soft Gate + 조건부 Hard Gate → 라이브 환경에서 속도가 중요하므로 기본은 Soft Gate이지만, Rethink 판정은 게임의 근본 방향을 바꾸는 결정이므로 데이터 근거를 강제한다.

### Soft Gate (권고)
1. 모든 KPI에 Green / Yellow / Red 상태가 매핑되어야 한다.
2. 플레이어 시그널이 Player Promise 관점에서 해석되어야 한다.
3. 조치 권고에 구체적 방향이 포함되어야 한다.

### Hard Gate (Rethink 판정 시)
1. Player Promise 자체의 문제점이 데이터로 뒷받침되어야 한다. → Rethink는 기존 플레이어에게도 영향을 미치는 근본 변경이므로, 감이 아닌 근거가 필요하다.

## Included Files

- **Live pulse input template**: `live-pulse-input-template.md` — 팀이 KPI 대시보드 스냅샷과 플레이어 피드백 요약을 구조화하여 제출하는 양식
