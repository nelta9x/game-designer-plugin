---
name: one-pager
description: Direction One-Pager를 생성하는 워크플로우. Decision Workflow에 따라 방향성을 결정하고 3종 산출물(Direction One-Pager, Decision Log, Priority Matrix)을 작성한다.
---

# One-Pager — Direction Design

`game-directing` 스킬이 Direction One-Pager를 작성할 때 참조하는 워크플로우·템플릿·게이트.

## Workflow (2-Phase)

### Phase 1 — Direction Call (대화)

사용자의 입력을 분석하여 핵심 방향을 제안한다. 아래 4가지를 **대화체로** 한 번에 제시하고, 사용자의 확인을 요청한다.

1. **Player Promise 초안**: 이 게임이 플레이어에게 약속하는 경험 한 문장.
2. **2개 후보 비교**: 서로 다른 트레이드오프를 대표하는 두 방향. 한쪽이 명백히 우월한 비교는 의미 없다.
3. **디렉터 추천 + 근거**: 어떤 쪽을 추천하는지, 왜 다른 쪽을 버리는지.
4. **핵심 리스크 1개**: 이 방향의 가장 큰 가정/리스크가 무엇인지.

Phase 1 출력은 **마크다운 문서가 아니라 대화**다. 디렉터가 회의실에서 팀에게 방향을 설명하듯이 말한다.

**핵심 규칙**: Phase 1에서 질문을 했고 사용자가 답변했다면, 곧바로 전체 문서(Phase 2)로 넘어가지 않는다. 답변을 반영한 Direction Call(위 4가지)을 먼저 대화로 제시하고, 사용자의 확인을 받은 뒤 Phase 2로 진행한다.

> **Fast Track**: 사용자가 입력에서 이미 방향을 특정했거나, "한 번에 다 작성해 줘" / "바로 전체를 뽑아줘" 같은 명시적 요청을 한 경우에만, Phase 1의 핵심 내용을 문서 서두에 요약으로 포함하고 곧바로 Phase 2를 실행한다.

### Phase 2 — Document Output (문서화)

Phase 1에서 합의된 방향을 기반으로 3종 산출물(Direction One-Pager, Decision Log, Priority Matrix)을 작성한다.

작성 순서:
1. Direction One-Pager (아래 템플릿)를 파일로 저장한다.
2. KPI Plan(섹션 11~13)을 `kpi-design` validate 스크립트로 검증한다.
3. FAIL 시 수정 후 재검증한다.
4. Decision Log, Priority Matrix를 파일로 저장한다.
5. Quality Gate를 자가 점검한다.

## Decision Workflow (고정 순서)
1. **Intent Framing**: 게임이 해결하려는 플레이어 욕구와 목표 감정을 정리한다.
2. **Option Compare (2안)**: 후보 A/B를 비교하고 각 장단점을 짧게 제시한다.
3. **Commit One Direction**: 하나를 선택하고 선택 이유를 1~2문장으로 남긴다.
4. **Player Journey Sketch**: D0~D30+ 시간축으로 최소 4단계(입문/탐색/습관/장기)를 스케치한다. 각 단계에서 플레이어가 무엇을 경험하는지, 다음 단계로 끌어당기는 훅이 무엇인지 정의한다. 시스템 설계자·아티스트·개발자 누구나 읽고 "이 시점에 이 사람은 이걸 하고 있다"를 이해할 수 있는 수준으로 작성한다.
5. **Priority Lock**: Must/Should/Won't를 고정한다.
6. **Success + Release Bar**: 성공 기준(KPI)과 출시 품질 바를 수치 포함으로 정의한다.
7. **Go/No-Go Rule**: 어떤 조건에서 진행/중단/재설계를 할지 명시한다.

## Output Format

### 디렉터 산출물 (필수 3종)

```markdown
# [게임명] - Direction One-Pager

## 1) 기본 정보
- 게임명:
- 장르:
- 타겟 플레이어:
- 시장 포지셔닝:
- 선택된 키워드:

## 2) Intent
- 목표 감정/페이싱:
- Player Promise (1문장):
- 디자인 원칙 3개:

## 3) 콘셉트 후보 비교 (2개)
- 후보 A: [한 줄 콘셉트]
  - 강점:
  - 약점:
- 후보 B: [한 줄 콘셉트]
  - 강점:
  - 약점:

## 4) Commit
- 선택안:
- 선정 근거 (1~2문장):

## 5) 핵심 재미 / 핵심 경험
- 핵심 재미 (1문장):
- 핵심 경험 (1~2문장):

## 6) Aesthetic Intent
- 감각적 인상 (1~2문장): → Player Promise가 시청각으로 전달되는 방식이다.
- Tone Reference (최대 3개):
  - [레퍼런스명]: [가져올 요소]
- Aesthetic Boundary (하지 않을 미적 선택):

## 7) 핵심 루프 (행동 → 보상 → 기대)
1.
2.
3.

## 8) Player Journey

시간축으로 플레이어의 경험 흐름을 나타낸다. 시스템 설계자·아티스트·개발자 누구나 읽고 "이 시점의 플레이어는 이것을 하고 있다"를 이해할 수 있어야 한다.

| 단계 | 시점 | 플레이어 상태 | 핵심 경험 | 이탈 리스크 |
|------|------|------------|---------|------------|
| 입문 | D0~D3 | | | |
| 탐색 | D4~D14 | | | |
| 습관 | D15~D30 | | | |
| 장기 | D30+ | | | |

- 입문→탐색 훅:
- 탐색→습관 훅:
- 습관→장기 훅:

## 9) 차별화 훅 (최대 2개)
-
-

## 10) Priority Matrix
- Must Have (최대 3개): → 3개는 상한이지 목표가 아니다. 진짜 2개면 2개로 충분하다.
- Should Have (최대 3개):
- Won't (최대 3개):

## 11) KPI Plan
- Stage: [prototype / soft-launch / live]
- Primary Outcome Metric: [KPI명]
  - Definition: [수식/정의]
  - Target: [목표값]
  - Window: [측정 주기]
  - Rationale: [선정 근거 — Player Promise/핵심 재미와의 연결]
  - Decision Rule: [목표 미달 시 조치]

## 12) Supporting KPIs
- [KPI명]
  - Formula: [수식]
  - Target: [목표값]
  - Window: [측정 주기]
  - Rationale: [선정 근거]
  - Decision Rule: [목표 미달 시 조치]
(최대 6개)

## 13) Instrumentation Events
- [이벤트명]: [사용하는 KPI 목록]

## 14) Release Bar (출시 품질 바)
- 필수 품질 기준:
- 금지 결함 기준:

## 15) Go/No-Go 조건
- Go:
- No-Go:
- Rework:

## 16) 가정 로그 (최대 3개)
- 가정:
  - 근거:
  - 영향:
- 가정:
  - 근거:
  - 영향:
```

```markdown
# [게임명] - Decision Log

## 선택 기록
- 검토한 옵션:
- 최종 선택:
- 왜 이 선택인가:
- 버린 대안과 이유:

## 미해결 이슈
- 이슈:
  - 결정 필요 시점:
  - 책임자:
```

```markdown
# [게임명] - Priority Matrix
- Must:
- Should:
- Won't:
```

## Quality Gate (출력 전 자가 점검)
1. [ ] 후보 비교 후 하나의 방향으로 명시적으로 커밋했는가
2. [ ] Must/Should/Won't 우선순위가 충돌 없이 정리되었는가
3. [ ] 출시 품질 바가 수치/판단 기준 포함으로 작성되었는가
4. [ ] Go/No-Go 조건이 모호하지 않게 명시되었는가
5. [ ] 핵심 재미가 Player Promise에서 도출되고, 핵심 루프가 핵심 재미를 전달하는 구조인가
6. [ ] Player Journey가 입문~장기(최소 4단계)로 작성되고, 각 단계 간 훅이 명시되었는가

## Gate Policy

### Mode A: Ideation
- 목적: 빠른 옵션 탐색
- 처리: Quality Gate 실패 시에도 진행 가능
- 출력: 결함을 `Decision Log`의 미해결 이슈로 기록

### Mode B: Pre-Handoff
- 목적: 실무 전달 품질 보장
- 처리: 아래 Hard Gate 중 1개라도 실패하면 전달 차단

#### Hard Gate (차단 조건)
1. Commit(선택안 + 선정 근거)이 채워져야 한다.
2. Priority Matrix(Must/Should/Won't)가 모두 채워져야 한다.
3. KPI Plan(섹션 11~13)이 작성되고, `kpi-design` validate가 PASS여야 한다.
4. Release Bar가 채워져야 한다.
5. Go/No-Go/Rework가 모두 채워져야 한다.

### Soft Gate (권고)
1. 템플릿 placeholder 잔존
2. 후보 A/B 차별성 부족
3. Release Bar 문구 모호
4. 미해결 이슈 구체성 부족
