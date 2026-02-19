# Game Designer Plugin

게임 방향성 의사결정, 상세 설계, KPI 설계를 지원하는 통합 플러그인.

## 설계 원칙

- **불필요한 파일을 만들지 않는다.** 플러그인의 모든 파일은 에이전트 컨텍스트에 로드된다. 필요 없는 문서나 스크립트는 컨텍스트 노이즈다.
- **검증 스크립트는 에이전트가 못 하는 일만 한다.** "필드가 비어있지 않은가" 수준의 체크는 SKILL.md의 워크플로우 지시와 중복이다. 스크립트는 교차 참조, stage 프로필 매칭, enum 검증처럼 기계적 정확성이 필요한 규칙에만 사용한다.
- **대칭성을 채우지 않는다.** 다른 스킬에 스크립트가 있다고 이 스킬에도 있어야 하는 건 아니다. 가치 판단이 먼저다.
- **지시에는 의도를 담는다.** "무엇을 하라"보다 "왜 해야 하는가"가 중요하다. 의도를 알면 매뉴얼에 없는 상황에서도 유연하게 대처할 수 있다.
- **역할 경계를 지킨다.** 에이전트/스킬의 모든 지시는 게임 디자이너가 결정할 수 있는 것이어야 한다. 다른 역할(UX 디자이너, 아트 디렉터, 개발자)의 영역은 요건만 정의하고 판단은 넘긴다.

## 구성 요소

### 스킬: `game-designer`
- 게임 디자이너 페르소나. 메인 에이전트 컨텍스트에서 실행되어 다른 스킬(`content-alignment`, `game-balance-math`)을 직접 호출 가능

### 스킬: `content-alignment`
- 콘텐츠 요소가 게임 방향성과 정합하는지 2축(행위 동사 / 결과 감정)으로 평가

### 스킬: `game-balance-math`
- 게임 밸런스 수치 설계를 위한 수학적 모델링, 프로세스, 참고 자료 제공

### 스킬: `game-directing`
- 게임의 큰 방향성과 핵심 재미를 설계하는 Senior Game Director. Player Promise 정의, 콘셉트 선택, 우선순위 결정(Must/Should/Won't), 마일스톤별 방향성 검증, 팀 제안 평가를 수행한다.

### 스킬: `one-pager`
- Direction One-Pager를 생성하는 워크플로우. Decision Workflow에 따라 방향성을 결정하고 3종 산출물(Direction One-Pager, Decision Log, Priority Matrix)을 작성한다.

### 스킬: `kpi-design`
- KPI Selection Guidelines에 따라 KPI Plan을 작성하고, 7가지 품질 규칙으로 검증한다.

### 스킬: `milestone-review`
- 마일스톤마다 플레이테스트 결과를 Direction One-Pager와 대조하여 방향성을 검증한다. 판정값 enum 검증과 집중 과제 수 제한을 스크립트로 체크한다.

### 스킬: `drift-check`
- 마일스톤 사이에 팀의 빌드가 합의된 방향에서 벗어나는지 점검하고 Drift Report를 작성한다.

### 스킬: `live-pulse`
- 출시 후 실제 플레이어 데이터로 방향성을 검증하고 Live Pulse Report를 작성한다.
