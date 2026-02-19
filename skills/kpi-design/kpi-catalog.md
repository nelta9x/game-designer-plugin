# KPI Reference Catalog

## Stage Profiles

> ※ retention과 activation은 스테이지와 무관하게 항상 필수이다 (KPI Selection Guidelines 참조).

### prototype
- Primary 후보: Core Loop Completion Rate, Unprompted Replay Rate
- 필수 카테고리: activation, engagement
- 제외 카테고리: monetization
- 최대 KPI 수: 5

### soft-launch
- Primary 후보: D1 Retention, D7 Retention
- 필수 카테고리: retention, engagement
- 제외 카테고리: (없음)
- 최대 KPI 수: 6

### live
- Primary 후보: LTV, D30 Retention
- 필수 카테고리: retention, monetization, engagement
- 제외 카테고리: (없음)
- 최대 KPI 수: 7

---

## KPI Entries

### Universal

#### D1 Retention
- Category: retention
- Formula: users_returned_day1 / users_installed_day0
- Target: 30-40%
- Window: daily cohort
- Rationale: Industry benchmark for casual-mid core; below 30% signals weak first session
- Decision Rule: If < 30%: audit tutorial flow and first-session reward pacing
- Events: session_start, install

#### D7 Retention
- Category: retention
- Formula: users_returned_day7 / users_installed_day0
- Target: 10-15%
- Window: daily cohort
- Rationale: Validates mid-term engagement loop; top quartile mobile games hit 15%+
- Decision Rule: If < 10%: review progression curve and content unlock pacing
- Events: session_start, install

#### D30 Retention
- Category: retention
- Formula: users_returned_day30 / users_installed_day0
- Target: 5-8%
- Window: daily cohort
- Rationale: Long-term stickiness; required for LTV modeling
- Decision Rule: If < 5%: investigate endgame content depth and social hooks
- Events: session_start, install

#### Session Length Distribution
- Category: engagement
- Formula: percentile(session_duration, [25, 50, 75, 90])
- Target: median 8-15 min
- Window: per session
- Rationale: Validates session design fits target loop length
- Decision Rule: If median < 5 min: core loop too short or lacks depth; if > 25 min: check fatigue signals
- Events: session_start, session_end

#### Sessions Per Day
- Category: engagement
- Formula: total_sessions / DAU
- Target: >= 2 sessions
- Window: daily
- Rationale: Repeat visits indicate habit formation
- Decision Rule: If < 1.5: add session-start hooks (daily rewards, energy refill, social prompts)
- Events: session_start

### Prototype

#### Core Loop Completion Rate
- Category: activation
- Formula: users_completed_first_loop / users_started_first_loop
- Target: >= 80%
- Window: per session
- Rationale: First loop must be completable to test fun hypothesis
- Decision Rule: If < 80%: simplify onboarding or fix blocking UX issues
- Events: loop_start, loop_complete

#### Tutorial Completion Rate
- Category: activation
- Formula: users_completed_tutorial / users_started_tutorial
- Target: >= 85%
- Window: per user lifetime
- Rationale: Tutorial dropout means players never reach the core fun
- Decision Rule: If < 85%: shorten tutorial or integrate learning into gameplay
- Events: tutorial_start, tutorial_complete

#### First Session Duration
- Category: engagement
- Formula: median(first_session_end - first_session_start)
- Target: >= 5 min
- Window: per user first session
- Rationale: Players spending <5 min likely didn't reach core fun
- Decision Rule: If < 5 min: move the fun moment earlier in the flow
- Events: session_start, session_end

#### Unprompted Replay Rate
- Category: engagement
- Formula: users_started_2nd_session_within_24h / users_completed_1st_session
- Target: >= 40%
- Window: 24h after first session
- Rationale: Organic return without push is the strongest fun signal
- Decision Rule: If < 40%: core loop may lack replayability; test variation/randomness
- Events: session_start, session_end

### Soft-launch

#### Session Frequency
- Category: engagement
- Formula: sessions_per_week / active_users
- Target: >= 4 sessions/week
- Window: weekly cohort
- Rationale: Validates habit loop across first week
- Decision Rule: If < 4/week: test notification timing and daily reward structure
- Events: session_start

#### Early Monetization Signal
- Category: monetization
- Formula: users_viewed_store_day1_to_7 / active_users_day1_to_7
- Target: >= 20%
- Window: first 7 days cohort
- Rationale: Store interest predicts monetization potential without requiring purchase
- Decision Rule: If < 20%: improve store visibility and surface contextual offers
- Events: store_view, session_start

### Live

#### ARPU
- Category: monetization
- Formula: total_revenue / total_active_users
- Target: varies by genre
- Window: monthly
- Rationale: Revenue efficiency per user; primary live-ops health metric
- Decision Rule: If declining: segment by spender tier and diagnose which segment dropped
- Events: purchase_complete, ad_revenue_event

#### ARPPU
- Category: monetization
- Formula: total_revenue / paying_users
- Target: varies by genre
- Window: monthly
- Rationale: Spending depth of payers; helps diagnose whale vs. minnow balance
- Decision Rule: If ARPPU rises but revenue flat: losing payer count, fix conversion funnel
- Events: purchase_complete

#### LTV
- Category: monetization
- Formula: ARPU * (1 / churn_rate) or cohort-based projection
- Target: > CPI * 1.5
- Window: 90-day cohort projection
- Rationale: Must exceed acquisition cost with margin for sustainable growth
- Decision Rule: If LTV < CPI * 1.3: pause UA spend and optimize retention/monetization
- Events: purchase_complete, ad_revenue_event, session_start, install

#### Churn Prediction Score
- Category: retention
- Formula: ML model or heuristic: no_session_3_days AND declining_session_length
- Target: < 15% monthly churn for engaged cohort
- Window: rolling 7-day window
- Rationale: Proactive churn detection enables re-engagement before loss
- Decision Rule: If rising: trigger win-back campaigns for at-risk segment
- Events: session_start, session_end

---

### Genre: Roguelike / Deckbuilder

#### Run Completion Rate
- Category: engagement
- Formula: runs_completed / runs_started
- Target: 15-30%
- Window: per run
- Rationale: Too high means no challenge; too low means frustration
- Decision Rule: If < 10%: difficulty spike too early; if > 40%: add harder modifiers
- Events: run_start, run_complete, run_fail

#### Avg Run Length
- Category: engagement
- Formula: mean(run_end_time - run_start_time)
- Target: 10-20 min
- Window: per run
- Rationale: Run length should match target session budget
- Decision Rule: If > 30 min: runs too long for session target; add mid-run save or shorten stages
- Events: run_start, run_complete, run_fail

#### Card/Item Diversity Per Run
- Category: engagement
- Formula: unique_items_used_per_run / total_available_items
- Target: >= 30% of pool touched per run
- Window: per run
- Rationale: Low diversity = dominant strategy; high diversity = healthy meta
- Decision Rule: If < 20%: buff underused items or nerf dominant picks
- Events: item_pick, card_pick, run_start

### Genre: Puzzle

#### Levels Per Session
- Category: engagement
- Formula: levels_completed / session_count
- Target: 3-8 levels
- Window: per session
- Rationale: Measures session pacing; too few = stuck, too many = no challenge
- Decision Rule: If < 2: check for difficulty wall at specific level ranges
- Events: level_start, level_complete, session_start, session_end

#### Hint Usage Rate
- Category: engagement
- Formula: hints_used / levels_completed
- Target: < 0.3 hints/level
- Window: per session
- Rationale: High hint use means puzzles are frustrating, not challenging
- Decision Rule: If > 0.5: ease difficulty curve in flagged level range
- Events: hint_used, level_complete

#### Difficulty Wall Detection
- Category: engagement
- Formula: level_fail_rate[level_N] / level_fail_rate[level_N-1]
- Target: < 2x spike
- Window: per level cohort
- Rationale: Sudden fail-rate spikes indicate poor difficulty curve
- Decision Rule: If > 2x spike: rebalance the flagged level or add optional hint
- Events: level_start, level_complete, level_fail

### Genre: Action

#### Deaths Per Level
- Category: engagement
- Formula: total_deaths_at_level / attempts_at_level
- Target: 1-3 deaths/level avg
- Window: per level cohort
- Rationale: Some deaths = challenge; too many = frustration
- Decision Rule: If > 5 avg: ease encounter or add checkpoint
- Events: player_death, level_start, level_complete

#### Skill Progression Curve
- Category: engagement
- Formula: avg_deaths_per_attempt[attempt_N] trend over attempts
- Target: decreasing trend by attempt 3
- Window: per level, across attempts
- Rationale: Players should visibly improve; flat curve = poor feedback
- Decision Rule: If flat after 3 attempts: improve feedback or add progressive hints
- Events: player_death, level_start, level_complete

#### Combo Mechanic Usage Rate
- Category: engagement
- Formula: combo_triggered / combat_encounters
- Target: >= 30%
- Window: per session
- Rationale: Core mechanic adoption validates design intent
- Decision Rule: If < 20%: improve discoverability or tutorial for combo system
- Events: combo_triggered, combat_start

### Genre: Strategy

#### Match Duration
- Category: engagement
- Formula: median(match_end - match_start)
- Target: 5-15 min for mobile; 15-30 min for PC
- Window: per match
- Rationale: Match length must fit platform session budget
- Decision Rule: If too long: add comeback mechanics or reduce resource complexity
- Events: match_start, match_end

#### Win Rate Balance
- Category: engagement
- Formula: wins_per_faction_or_class / total_matches_per_faction_or_class
- Target: 45-55% per faction/class
- Window: weekly aggregate
- Rationale: Extreme imbalance kills competitive fairness
- Decision Rule: If any faction < 40% or > 60%: flag for balance patch
- Events: match_end, faction_select

#### Build Diversity
- Category: engagement
- Formula: unique_build_archetypes_used / total_matches (top-N analysis)
- Target: >= 4 viable archetypes in meta
- Window: weekly aggregate
- Rationale: Healthy meta requires multiple competitive strategies
- Decision Rule: If < 3: buff underperforming archetypes or nerf dominant one
- Events: build_select, match_end

---

### Business Model: Premium

#### Player Satisfaction (NPS)
- Category: engagement
- Formula: (promoters - detractors) / total_respondents * 100
- Target: >= 40 NPS
- Window: post-completion or monthly survey
- Rationale: Premium games rely on word-of-mouth; NPS predicts organic growth
- Decision Rule: If < 30: run qualitative survey to identify top pain points
- Events: survey_submit

#### Completion Rate
- Category: engagement
- Formula: users_reached_ending / users_started_game
- Target: >= 30%
- Window: per user lifetime
- Rationale: Premium content should be experienced; low completion wastes content investment
- Decision Rule: If < 20%: identify drop-off chapter and ease difficulty or improve pacing
- Events: chapter_start, chapter_complete, game_complete

### Business Model: Ads

#### Ad View Rate
- Category: monetization
- Formula: ad_impressions / DAU
- Target: 3-6 views/DAU
- Window: daily
- Rationale: Revenue scales with impressions; too many kills retention
- Decision Rule: If < 2: improve ad placement visibility; if > 8: reduce frequency to protect retention
- Events: ad_impression, session_start

#### Ad-to-Session Ratio
- Category: monetization
- Formula: ad_impressions / total_sessions
- Target: 1-2 ads/session
- Window: daily
- Rationale: Controls ad density per play session
- Decision Rule: If > 3: players likely feel spammed; reduce or improve ad reward value
- Events: ad_impression, session_start

#### Opt-in Ad Conversion
- Category: monetization
- Formula: rewarded_ad_watched / rewarded_ad_offered
- Target: >= 50%
- Window: daily
- Rationale: High opt-in means reward value matches player perception
- Decision Rule: If < 40%: increase reward value or improve offer timing
- Events: rewarded_ad_offered, rewarded_ad_watched

### Business Model: IAP

#### IAP Conversion Rate
- Category: monetization
- Formula: first_time_payers / total_active_users
- Target: 2-5%
- Window: monthly cohort
- Rationale: Industry standard for F2P; < 2% means offer or pricing problem
- Decision Rule: If < 2%: test starter pack pricing and offer timing
- Events: purchase_complete, store_view, session_start

#### First Purchase Timing
- Category: monetization
- Formula: median(days_from_install_to_first_purchase)
- Target: day 2-5
- Window: per payer cohort
- Rationale: Too early = aggressive; too late = missed window
- Decision Rule: If > day 7: surface offers earlier; if < day 1: check if pay-wall feels forced
- Events: purchase_complete, install
