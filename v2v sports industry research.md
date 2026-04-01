# Sports-V2V: Multi-Sport Video-to-Video Reasoning Research Roadmap

> **Document type**: Technical roadmap / research whitepaper
> **Status**: Draft v0.1 — internal planning
> **Scope**: Taxonomy design, scorer parameters, data architecture, paper outline
> **Foundation**: Extends [VRM-Soccer](https://github.com/pppop00/VRM-Soccer) + [VBVR](https://github.com/Video-Reason) paradigms

---

## 1. Motivation & Positioning

### 1.1 The Gap

VBVR established that video-to-video reasoning can be rigorously benchmarked at scale — 2.015M synthetic clips, 5 cognitive faculties, rule-based scoring with ρ > 0.9 human alignment. But VBVR's tasks are domain-agnostic (mazes, physics simulations, shape manipulations). No comparable benchmark exists for **grounded, real-world video domains** where:

- Temporal structure is dense and physically constrained
- Ground truth is verifiable without human annotation
- Tactical/strategic intelligence has real-world value

Sports is the ideal entry point: game footage is abundant, tracking data provides exact ground truth, and reasoning tasks map cleanly to established sports analytics questions.

### 1.2 The V2V Paradigm (vs. VBVR)

| Dimension | VBVR | Sports-V2V |
|-----------|------|-----------|
| Input modality | First frame (image) + prompt | Video clip (N seconds) + prompt |
| Output modality | Generated video (model prediction) | Predicted future video (next M seconds) |
| Ground truth source | Programmatically generated | Real game footage |
| Domain | Synthetic / abstract | Real-world sports |
| Evaluation | Rule-based scorers on generated frames | Rule-based scorers comparing prediction to actual footage |
| Scale target | 2M+ samples | 500K+ samples (Phase 1: 50K BEV) |

The core V2V loop:
```
input_clip.mp4  (N seconds of game footage)
      +
prompt.txt      ("Which player receives this pass?")
      ↓
[Video Reasoning Model]
      ↓
predicted_clip.mp4  (M seconds — model's prediction of what happens next)
      ↑ compared against ↑
ground_truth.mp4    (actual M seconds of footage, withheld during inference)
```

### 1.3 Why Sports is Ideal for V2V Benchmarking

1. **Verifiable ground truth**: Tracking data provides exact positions, events, outcomes — no human re-annotation needed
2. **Dense temporal structure**: Every second contains multiple reasoning-relevant events (passes, movements, formations)
3. **Physics-constrained**: Ball/player trajectories obey real physics — invalid predictions are detectable by rule
4. **Multi-scale reasoning**: Tasks span 0.5s (trajectory) to 90min (game management) windows
5. **Existing ecosystem**: SoccerNet, MLB Statcast, Roland Garros Open Data, NBA tracking APIs

---

## 2. Task Family Taxonomy

### 2.1 Design Principles

**Family-first, objectivity-first.** Two constraints drive every design decision:

1. **Family-first over sport-first** — each family defines a universal reasoning operation that manifests across all sports. This makes cross-sport transfer a primary research question rather than an afterthought.

2. **Code-computable ground truth only** — every family's ground truth must be derivable by code from tracking data, with no human annotation and no subjective judgment. Tasks requiring pose skeletons, subjective intent labeling, or proprietary-only data are excluded.

**Theoretical grounding:**

| Framework | How it maps |
|-----------|------------|
| **RPD Model** (Klein 1993) | Simple Match (F4) → Situational Diagnosis (F1, F2, F3, F6) → Mental Simulation (F5, F7, F8) |
| **Bloom's Taxonomy** | Understand (F4) → Apply (F1, F6) → Analyze (F2, F3, F5) → Evaluate (F7, F8) |

**Gap vs. existing benchmarks** (all confirmed absent before this work):

| Family | Closest Existing Work | Why It Falls Short |
|--------|----------------------|-------------------|
| F1: Pre-Contact Anticipation | SoccerNet-BAA (2504.12021) | Anticipates action *after* trigger; we predict *before* contact |
| F2: Tactical Phase Reading | None | No benchmark labels continuous tactical phases |
| F3: Space & Affordance | None | Pitch control never tested as a video model capability |
| F4: Configuration Recognition | SoccerNet Game State | Broadcast-only; BEV tracking as input is novel |
| F5: Causal Event Attribution | Sports-QA causality | At event level only; not tactical/spatial level with BEV grounding |
| F6: Physical Dynamics | Partial (Statcast labels exist) | Never tested as a *visual inference* task from video |
| F7: Counterfactual Option Ranking | None | Counterfactual *option ranking* from video completely absent |
| F8: Rule-Conditioned Outcome | SportR (2511.06499) | Requires human CoT annotation; ours is fully geometric |

### 2.2 The Eight Task Families

---

**F1 · Pre-Contact Anticipation** ✅ FULLY AUTO
> *The clip freezes 0.3s before the kick. Where does the ball go?*

Predict the outcome of an action **before** it executes, from the kinematic information visible in the agent's body and ball motion. This is the direct operationalization of the expert "quiet eye" advantage in cognitive sports science — experts predict from pre-contact cues; novices wait for the ball to move.

| Sport | Instantiation | Ground Truth | Package |
|-------|--------------|-------------|---------|
| Soccer | Pass direction / shot destination from body orientation before foot contact | Ball (x,y) N frames post-contact | `LaurieOnTracking`, `databallpy` |
| Tennis | Serve type + direction from ball toss + racket path before hit | Ball bounce coordinates + box label | Ball trajectory from 2D tracking |
| Basketball | Shot zone from jab step + shoulder turn before release | Ball coordinates at board/net contact | NBA tracking feed |
| Baseball | Pitch type from grip + arm angle before release | `pitch_type` from Statcast | `pybaseball.statcast()` |

*Horizon:* 0.5–2s pre-contact window → predict next 0.5–3s

---

**F2 · Tactical Phase Reading** ⚠️ MODEL-PARAM
> *Is this team pressing, sitting in a mid-block, or building from the back?*

Classify the **macro game phase** the team is executing — not a single action, but a continuous state lasting several seconds. Ground truth is derived by code from ball zone + team centroid + velocity vectors using a pinned rule-set model.

| Sport | Instantiation | Ground Truth | Package |
|-------|--------------|-------------|---------|
| Soccer | Build-up / Midfield progression / Counter-attack / High press / Mid-block | Phase label from `unravelsports.PressingIntensity` + ball zone rules | `unravelsports` |
| Tennis | Baseline rally / Approach phase / Net play / Break point pressure | Phase from rally position + velocity vectors | Position heuristics |
| Basketball | Half-court offense / Fast break / Transition defense / Zone / Press | Phase from ball velocity + team spread | Tracking heuristics |
| Baseball | Count-phase macro: early / 2-strike / 3-ball / full count | Count label from event stream | `pybaseball` |

*Horizon:* 5–10s clip → classify current phase | *Note:* ground truth is model-parameterized; model version pinned at benchmark release

---

**F3 · Space & Affordance Detection** ✅ FULLY AUTO
> *Where on the pitch is there open space right now that an attacker could run into?*

Identify the largest available / under-defended zone using pitch control — a fully automatable computation from player positions and velocities. Pitch control assigns each spatial cell a probability that the attacking or defending team controls it; "open space" is the zone with lowest defender control probability.

| Sport | Instantiation | Ground Truth | Package |
|-------|--------------|-------------|---------|
| Soccer | Open pitch control zone / passing lane / channel behind defensive line | Pitch control surface argmax (low-defender zone centroid) | `LaurieOnTracking/Tutorial3_PitchControl.py` |
| Tennis | Open court direction after lateral movement / recovery gap | Unoccupied court cell from player positions | Voronoi from coordinates |
| Basketball | Open 3-point spot / post entry lane / help rotation gap | Unguarded zone from player positions + shot probabilities | Player position geometry |
| Baseball | Defensive gap / shift-exposed territory | Field zone with lowest fielder coverage probability | Fielder position geometry |

*Horizon:* snapshot from current clip | *Key package:* `Friends-of-Tracking-Data-FoTD/LaurieOnTracking`

---

**F4 · Configuration Recognition** ✅ FULLY AUTO
> *What formation is the blue team defending in?*

Classify the collective formation or positional scheme from agent coordinates. Uses Hungarian assignment (linear sum assignment) to match player positions to formation templates — a published, open-source method with no human annotation required.

| Sport | Instantiation | Ground Truth | Package |
|-------|--------------|-------------|---------|
| Soccer | Formation label (4-4-2, 4-3-3, 4-2-3-1, 3-5-2, 5-3-2) | EFPI Hungarian assignment to template set | `unravelsports` (EFPI, arxiv:2506.23843) + `kloppy` |
| Tennis | Doubles formation (both-back, up-back, I-formation, Australian) | Geometric classification from 2 player positions | Position geometry |
| Basketball | Offensive set (5-out, horns, elbow, post-up) / Defensive scheme (man, 2-3 zone, press) | Player position clustering to template | Position geometry |
| Baseball | Defensive alignment (standard, infield shift, Ted Williams shift, outfield depth) | Fielder positions vs. alignment templates | Position geometry |

*Horizon:* 3–8s clip → classify stable configuration

---

**F5 · Causal Event Attribution** ⚠️ MODEL-PARAM
> *The goal just happened. What action 5 seconds earlier actually caused it?*

Identify the action in a possession chain that most increased goal/scoring probability — the causal step in the sequence. Ground truth is computed via VAEP (Valuing Actions by Estimating Probabilities): each action's score is the delta in goal probability it created, attributing "credit" backward through the chain.

| Sport | Instantiation | Ground Truth | Package |
|-------|--------------|-------------|---------|
| Soccer | What action created the shot? What defensive error led to the goal? | VAEP delta chain — action with highest ΔP(goal) | `socceraction` (ML-KULeuven) |
| Tennis | Which shot in the rally set up the winner? | Point win probability delta per shot | Rally xT analog |
| Basketball | What rotation breakdown gave the open shot? | Expected points delta per possession action | `nba_api` + possession value model |
| Baseball | What pitch in the sequence led to the strikeout/hit? | Run expectancy delta per pitch | `pybaseball` run expectancy tables |

*Note:* VAEP model version pinned; ground truth is fully deterministic given pinned model + SPADL action chain

---

**F6 · Physical Dynamics Inference** ✅ FULLY AUTO
> *Watch the pitcher's arm — fastball, slider, or curveball?*

Infer physical properties of ball/body motion from the visual trajectory — spin type, arc quality, velocity class. For baseball, ground truth is directly available from MLB Statcast (no computation needed). For other sports, inferred from ball trajectory coordinates.

| Sport | Instantiation | Ground Truth | Package |
|-------|--------------|-------------|---------|
| Soccer | Free-kick spin direction / header arc type (flat vs. looping) | Arc classification from ball (x,y,t) trajectory curvature | Ball trajectory geometry |
| Tennis | Serve arc type (flat/kick/slice) inferred from 2D bounce angle + depth | Bounce angle + depth classification from tracking | Ball trajectory geometry |
| Basketball | Shot arc quality (flat/optimal/high) / rebound angle | Arc angle at release from ball (x,y,t) | Ball trajectory geometry |
| Baseball | Pitch type FF/SL/CB/CH/FC + exit velocity bin + launch angle zone | `pitch_type`, `release_spin_rate`, `launch_angle` from Statcast | `pybaseball.statcast()` |

*Horizon:* 0.5–2s of ball motion → infer physics class

---

**F7 · Counterfactual Option Ranking** ⚠️ MODEL-PARAM
> *The player passed left. But was the right-side option actually better?*

Given the actual choice made, enumerate the feasible alternatives at that moment and rank them by expected value. The model must identify both what happened AND evaluate whether it was optimal — the highest-level analytical task in the benchmark.

| Sport | Instantiation | Ground Truth | Package |
|-------|--------------|-------------|---------|
| Soccer | Rank pass options by `pitch_control(target) × xT(target)` — was actual choice optimal? | xT surface + pitch control surface at decision frame | `socceraction.xthreat` + `LaurieOnTracking` |
| Tennis | Was this shot direction optimal? (win probability by shot type × placement) | Win probability surface from rally models | Rally analytics models |
| Basketball | Was this the right shot/pass? (expected points by option) | Expected points by zone from tracking + shot chart | NBA `nba_api` shot quality model |
| Baseball | Was this the right pitch? (expected run value by count × pitch type) | Run expectancy 24-state table from Statcast | `pybaseball` run expectancy |

*Note:* ground truth = argmax over value surface; deterministic given pinned model version

---

**F8 · Rule-Conditioned Outcome** ✅ FULLY AUTO
> *Was the attacker offside when the pass was played?*

Apply sport rules to player/ball coordinates to determine the correct ruling — fully deterministic from geometry. This is the only family where the answer is known with zero model uncertainty: rule definitions are fixed and player positions are measured.

| Sport | Instantiation | Ground Truth | Package |
|-------|--------------|-------------|---------|
| Soccer | Offside (last-defender x at pass moment vs. attacker x) / corner vs. goal kick | Geometric computation from `agent_ids` positions at event frame | Coordinate geometry (no external package) |
| Tennis | Serve in/out + service box classification | Ball bounce (x,y) vs. service box boundary polygon | Coordinate geometry |
| Basketball | Shot-clock violation (time elapsed > 24s) / three-second lane violation | Timing + position from tracking feed | Coordinate geometry + timestamps |
| Baseball | Ball/strike (pitch crossing coordinates vs. strike zone polygon) / fair vs. foul | Pitch (x,z) at plate vs. batter-height strike zone | `pybaseball` Statcast `zone` column |

*Horizon:* 1–3s around rule-triggering event → binary or multi-class ruling

---

### 2.3 Task Distribution

```
Sports-V2V: 192 tasks total (144 open + 48 leaderboard-reserved)
= 8 families × 4 sports × 6 tasks per cell
                          (4 open + 2 leaderboard-reserved per cell)

Family                        | Soccer | Tennis | Basketball | Baseball | Total
──────────────────────────────┼────────┼────────┼────────────┼──────────┼──────
F1 Pre-Contact Anticipation   |   6    |   6    |     6      |    6     |  24
F2 Tactical Phase Reading     |   6    |   6    |     6      |    6     |  24
F3 Space & Affordance         |   6    |   6    |     6      |    6     |  24
F4 Configuration Recognition  |   6    |   6    |     6      |    6     |  24
F5 Causal Event Attribution   |   6    |   6    |     6      |    6     |  24
F6 Physical Dynamics          |   6    |   6    |     6      |    6     |  24
F7 Counterfactual Ranking     |   6    |   6    |     6      |    6     |  24
F8 Rule-Conditioned Outcome   |   6    |   6    |     6      |    6     |  24
──────────────────────────────┼────────┼────────┼────────────┼──────────┼──────
Total per sport               |  48    |  48    |    48      |   48     | 192
```

Two evaluation axes:
- **Family score** (across sports): how universally does the model apply this reasoning operation?
- **Sport score** (across families): how comprehensively does the model understand this sport?

### 2.4 Ground Truth Computation Stack

| Family | Objectivity | Primary Package | Data Requirement |
|--------|-------------|----------------|-----------------|
| F1: Pre-Contact Anticipation | ✅ FULLY AUTO | `LaurieOnTracking`, `databallpy` | Ball (x,y) at 10+ Hz |
| F2: Tactical Phase Reading | ⚠️ MODEL-PARAM | `unravelsports.PressingIntensity` | Tracking + possession label + velocity vectors |
| F3: Space & Affordance | ✅ FULLY AUTO | `LaurieOnTracking/Tutorial3_PitchControl.py` | Player (x,y) + velocity vectors |
| F4: Configuration Recognition | ✅ FULLY AUTO | `unravelsports` EFPI + `kloppy` | Outfield player (x,y) per frame |
| F5: Causal Event Attribution | ⚠️ MODEL-PARAM | `socceraction` VAEP + SPADL | Event stream or event-detected tracking |
| F6: Physical Dynamics | ✅ FULLY AUTO | `pybaseball.statcast()` (baseball); trajectory geometry (other sports) | Statcast feed / ball trajectory |
| F7: Counterfactual Ranking | ⚠️ MODEL-PARAM | `socceraction.xthreat` + `LaurieOnTracking` pitch control | Tracking + value model (pinned version) |
| F8: Rule-Conditioned Outcome | ✅ FULLY AUTO | Coordinate geometry (no external package) | Player/ball (x,y) + timestamps |

**⚠️ MODEL-PARAM** families (F2, F5, F7) are still fully deterministic and reproducible — their ground truth is parameterized by a versioned, open-source model (not human judgment). The model version is frozen at benchmark release and documented in `answer.json:metadata.scorer_version`.

### 2.5 Task ID Convention

```
{FAM_CODE}-{NNN}_{sport_code}_{description}
```

| Code | Family | Code | Family |
|------|--------|------|--------|
| `PCA` | Pre-Contact Anticipation | `PDI` | Physical Dynamics Inference |
| `TPR` | Tactical Phase Reading | `COR` | Counterfactual Option Ranking |
| `SAD` | Space & Affordance Detection | `RCO` | Rule-Conditioned Outcome |
| `CR` | Configuration Recognition | | |
| `CEA` | Causal Event Attribution | | |

| Sport Code | Sport |
|-----------|-------|
| `soc` | Soccer |
| `ten` | Tennis |
| `bbk` | Basketball |
| `bsb` | Baseball |

**Examples:**
- `PCA-001_soc_pass_direction_pre_contact`
- `TPR-002_soc_high_press_vs_midblock`
- `SAD-001_soc_pitch_control_open_zone`
- `CR-003_bbk_defensive_scheme_label`
- `CEA-001_soc_causal_action_before_goal`
- `PDI-001_bsb_pitch_type_statcast`
- `COR-001_soc_optimal_pass_xT_ranking`
- `RCO-001_soc_offside_detection`

### 2.6 Representative Task Table (Phase 1 — Soccer + Tennis)

| Task ID | Family | Sport | Description | Input | Ground Truth |
|---------|--------|-------|-------------|-------|-------------|
| `PCA-001_soc` | Pre-Contact Anticipation | Soccer | Pass direction from body orientation before contact | 3s BEV (ends 0.3s before kick) | Ball (x,y) at reception |
| `PCA-001_ten` | Pre-Contact Anticipation | Tennis | Serve direction from ball toss + racket path | 1s clip before hit | Bounce (x,y) + box label |
| `PCA-001_bsb` | Pre-Contact Anticipation | Baseball | Pitch type from grip + arm angle | 1s pre-release clip | Statcast `pitch_type` |
| `TPR-001_soc` | Tactical Phase Reading | Soccer | High press vs. mid-block classification | 8s BEV | Phase label (pressing intensity threshold) |
| `SAD-001_soc` | Space & Affordance | Soccer | Widest open zone in final third | 3s attacking BEV | Pitch control argmin zone centroid (m) |
| `CR-001_soc` | Configuration Recognition | Soccer | Defensive formation label | 5s defending clip | Formation string (EFPI output) |
| `CR-001_ten` | Configuration Recognition | Tennis | Doubles positioning strategy | 3s rally clip | Position label (both-back/up-back/etc.) |
| `CEA-001_soc` | Causal Event Attribution | Soccer | Which action in the chain caused the shot opportunity? | 10s possession clip | Action index with highest VAEP delta |
| `PDI-001_soc` | Physical Dynamics | Soccer | Free-kick arc type (left-curl/right-curl/straight) | Setup + 1.5s post-kick | Arc class from trajectory curvature |
| `PDI-001_bsb` | Physical Dynamics | Baseball | Pitch type classification | 1s delivery clip | Statcast `pitch_type` + `release_spin_rate` bin |
| `COR-001_soc` | Counterfactual Ranking | Soccer | Was this pass the xT-optimal choice? | 3s BEV at pass moment | xT × pitch_control rank of actual vs. alternatives |
| `RCO-001_soc` | Rule-Conditioned Outcome | Soccer | Offside detection at pass moment | 2s around pass | Binary + offside `agent_id` (geometry) |
| `RCO-001_ten` | Rule-Conditioned Outcome | Tennis | Serve in/out + service box | 1s serve clip | Binary + box label (boundary geometry) |
| `RCO-001_bsb` | Rule-Conditioned Outcome | Baseball | Ball/strike call | Pitch clip | `zone` column from Statcast |

---

## 3. Data Architecture

### 3.1 Video Modality Tiers

```
Tier 1: BEV Synthetic      ← Phase 1 (exists: soccer_bev_pipeline.py)
Tier 2: BEV Real           ← Phase 2 (overhead cameras, homography alignment)
Tier 3: Broadcast Standard ← Phase 3 (SoccerNet, tournament broadcast)
Tier 4: Multi-angle Fusion ← Future work
```

**Phase 1 (BEV Synthetic)** is fully implemented for soccer via `soccer_bev_pipeline.py`. Extension to tennis and basketball requires:
1. New `BEVRenderer` subclass with sport-specific court geometry
2. New `*Adapter` class for tracking data source
3. Sport-specific `analyze_clip_events` equivalent

### 3.2 Canonical Data Format (Multi-Sport Extension)

Extending the existing canonical columns:

**Required (all sports):**
```
frame_id, agent_id, agent_type, x, y
```

**Sport-specific optional columns:**

| Column | Soccer | Tennis | Basketball | Baseball |
|--------|--------|--------|-----------|---------|
| `period` | Match half | Set | Quarter | Inning |
| `timestamp` | Seconds | Game time | Game clock | At-bat time |
| `possession_team` | home/away | server/returner | home/away | batting/fielding |
| `shot_clock` | — | — | Seconds remaining | — |
| `serve_speed_kmh` | — | km/h | — | — |
| `pitch_type` | — | — | — | FF/SL/CB/CH/etc. |
| `ball_spin_rpm` | — | rpm | — | rpm |
| `event_tag` | pass/shot/tackle | serve/return/winner | assist/shot/block | pitch/hit/fielded |

### 3.3 Data Sources & Licensing

| Sport | Open Tracking Data | Open Video | Partnership Path |
|-------|-------------------|-----------|-----------------|
| Soccer | Metrica Sports (CC-BY), SkillCorner Open Data | SoccerNet (research license) | SkillCorner commercial |
| Tennis | Roland Garros Open Data (Hawk-Eye, CC-BY-NC) | ITF/tournament (contact) | ATP Media, Tennis Abstract |
| Basketball | NBA Stats API (aggregate), OpenNBA | — | Second Spectrum, Synergy |
| Baseball | MLB Statcast / Baseball Savant (public) | MLB Film Room (licensed) | MLBAM Advanced Media |

**Phase 1 open-data-only stack:**
- Soccer: Metrica Sample Game 2 + SkillCorner Open Data (6 matches)
- Tennis: Roland Garros 2023 Hawk-Eye tracking (point-level ball/player positions)
- Basketball: Deferred to Phase 3 (no fully open tracking exists)
- Baseball: Statcast pitch-by-pitch (no real-time tracking, frame-level TBD)

### 3.4 Court/Field Geometry

| Sport | Length (m) | Width (m) | Key Zones |
|-------|-----------|----------|----------|
| Soccer | 105.0 | 68.0 | Penalty area, goal area, center circle |
| Tennis (singles) | 23.77 | 8.23 | Service boxes (6.4m deep), baseline, net |
| Tennis (doubles) | 23.77 | 10.97 | Doubles alley (1.37m each side) |
| Basketball | 28.0 | 15.0 | Paint (5.8m × 4.9m), 3-point arc (6.75m), key |
| Baseball | ~120 (diamond) | ~120 | Bases (27.4m apart), pitcher mound (18.4m), outfield |

### 3.5 Adapter Architecture

Extending `soccer_bev_pipeline.py` class hierarchy:

```
BaseTrackingAdapter (abstract)
├── MetricaAdapter          ← existing
├── SkillCornerAdapter      ← existing
├── SkillCornerV2Adapter    ← existing
├── RolandGarrosAdapter     ← Phase 2: Hawk-Eye JSON → canonical
├── NBATrackingAdapter      ← Phase 3: SportVU/Second Spectrum
└── StatcastAdapter         ← Phase 4: Baseball Savant CSV
```

Each adapter outputs the same canonical `CanonicalTrackingSource` with `long_df` containing required columns, enabling all downstream parser/renderer/scorer logic to be sport-agnostic.

---

## 4. V2V Task Instance Format

### 4.1 File Structure

```
sports_v2v/
└── {task_id}/
    └── {task_id}_{8-digit-index}/   ← e.g., SOC-ANT-001_00000042/
        ├── input_clip.mp4           ← N seconds, model sees this
        ├── prompt.txt               ← reasoning question
        ├── ground_truth.mp4         ← M seconds after input (evaluation only)
        ├── first_frame.png          ← first frame of input_clip
        └── answer.json             ← structured ground truth for rule-based scorer
```

### 4.2 answer.json Schema

```json
{
  "task_id": "DP-001_soc",
  "instance_index": 42,
  "sport": "soccer",
  "family": "destination_prediction",
  "input_duration_s": 3.0,
  "prediction_horizon_s": 2.0,
  "ground_truth": {
    "receiver_agent_id": "home_9",
    "pass_completion": true,
    "ball_destination_m": [67.3, 28.1],
    "event_frame": 75,
    "event_time_s": 1.5
  },
  "scorer_weights": {
    "receiver_id_correct": 0.40,
    "completion_correct": 0.30,
    "spatial_accuracy": 0.30
  },
  "metadata": {
    "source_dataset": "metrica",
    "source_match_id": "Sample_Game_2",
    "start_frame": 1250,
    "seed": 42,
    "git_commit": "ec5f2f2"
  }
}
```

### 4.3 Prompt Templates by Task Family

**F1 · Pre-Contact Anticipation (Soccer):**
> "The clip ends just before the red player makes contact with the ball. Based on their body orientation, run-up direction, and the positions of teammates visible in this clip, predict where the ball will go after contact. Report the predicted landing position as (x, y) in meters from the bottom-left corner of the pitch."

**F1 · Pre-Contact Anticipation (Baseball):**
> "Watch the pitcher's delivery — the clip ends just before ball release. Based on the grip, arm angle, and release path visible, classify the pitch type: Four-seam Fastball (FF), Slider (SL), Curveball (CB), Changeup (CH), or Cutter (FC)."

**F2 · Tactical Phase Reading (Soccer):**
> "Based on the player positions and movement patterns visible in this clip, classify the defensive team's current tactical phase. Choose from: high press, mid-block, low block, or counter-press."

**F3 · Space & Affordance Detection (Soccer):**
> "The red team is in possession in the attacking third. Based on the positions and movements of all players visible in this clip, identify the largest open space that an attacking player could run into. Report the center coordinates of this space in meters."

**F4 · Configuration Recognition (Soccer):**
> "Based on the defensive shape of the blue team in this clip, identify their formation. Choose from: 4-4-2, 4-3-3, 4-2-3-1, 3-5-2, or 5-3-2."

**F4 · Configuration Recognition (Basketball):**
> "Based on the offensive positioning of the white team visible in this clip, classify their offensive set: 5-out, horns, elbow, post-up, or transition."

**F5 · Causal Event Attribution (Soccer):**
> "A shot on goal occurs at the end of this clip. Looking back through the possession sequence, identify the single most important action (pass, movement, or positioning change) that created the shooting opportunity. Describe the player's position and the action they took."

**F6 · Physical Dynamics Inference (Soccer):**
> "A free kick is struck in this clip. Based on the ball's trajectory visible after contact, classify the ball flight as: left-curving, right-curving, or straight. Also estimate whether the arc is flat (< 2m peak height), medium (2–4m), or high (> 4m)."

**F7 · Counterfactual Option Ranking (Soccer):**
> "At the moment the red player passes left, observe all available teammates. Rank the three visible passing options (including the actual choice) from best to worst based on the space and defensive positioning visible in the clip. Identify which option you believe had the highest expected value."

**F8 · Rule-Conditioned Outcome (Soccer):**
> "A through ball is played at the moment shown. Based on the positions of all players at the exact moment of the pass, is the attacker who runs onto the ball in an offside position? Answer yes or no. If yes, identify which player was offside by their approximate position on the pitch."

**F8 · Rule-Conditioned Outcome (Baseball):**
> "Watch the pitch and the umpire's call. Based on where the ball crosses the plate relative to the batter's stance and the standard strike zone, was this pitch a ball or a strike? Identify which zone (top/middle/bottom × inside/middle/outside) the pitch crossed."

---

## 5. Rule-Based Scorer Design

### 5.1 Design Principles (Inherited from VBVR)

1. **No LLM judges** — all scoring is deterministic and reproducible
2. **Verifiable answers** — every task has objectively correct ground truth from tracking data
3. **Weighted sub-criteria** — each task decomposes into 2–4 independently scorable dimensions
4. **Human alignment target** — Spearman's ρ > 0.85 with human preference (same threshold as VBVR)
5. **0–1 normalized** — all sub-scores normalized; final score is weighted sum

### 5.2 SportsV2VScorerConfig Parameters

```python
@dataclass
class SportsV2VScorerConfig:
    """
    Rule-based scorer configuration for Sports-V2V future prediction tasks.
    Extends soccer_bev_pipeline.RealismConfig to multi-sport prediction evaluation.
    All distance thresholds are in meters; all time thresholds in seconds.
    """

    # ── Spatial Accuracy ────────────────────────────────────────────────────
    position_error_threshold_m: float = 1.5
    # Max acceptable RMSE for predicted player positions.
    # Sport calibration: Soccer 1.5m | Basketball 0.8m | Tennis 0.3m | Baseball 1.0m

    ball_trajectory_dtw_threshold_m: float = 2.0
    # Dynamic Time Warping distance threshold for ball path comparison.
    # Lower = stricter. DTW handles temporal misalignment better than frame-by-frame MSE.

    ball_destination_zone_radius_m: float = 3.0
    # Radius for "destination zone correct" binary check (pass landing, shot target).

    # ── Temporal Accuracy ───────────────────────────────────────────────────
    event_timing_tolerance_s: float = 0.5
    # Window (±s) in which a predicted event is counted as temporally correct.

    phase_transition_iou_threshold: float = 0.6
    # Min temporal IoU between predicted and actual phase-change windows.

    # ── Event Classification ─────────────────────────────────────────────────
    event_type_top1_accuracy: float = 0.70
    # Min top-1 accuracy for event type classification (pass/shot/dribble/serve/etc.)

    receiver_identification_accuracy: float = 0.60
    # Min ratio of correctly predicted pass/play receivers (agent_id level).

    formation_f1_threshold: float = 0.65
    # Min macro-F1 across formation classes for tactical recognition tasks.

    # ── Physical Plausibility ────────────────────────────────────────────────
    max_speed_violation_ratio: float = 0.02
    # Max fraction of predicted frames where agent speed exceeds sport-legal maximum.
    # Sport maxima: Soccer player 12 m/s | Basketball 10 m/s | Tennis ball 80 m/s | Baseball pitch 47 m/s

    trajectory_smoothness_min: float = 0.85
    # Minimum smoothness score (1 - normalized jerk integral) for predicted trajectories.
    # Penalizes teleportation artifacts and physically implausible acceleration spikes.

    out_of_bounds_violation_ratio: float = 0.02
    # Max fraction of predicted frames where ball/key players exceed field boundaries.

    # ── Tactical Coherence ───────────────────────────────────────────────────
    possession_transition_validity: float = 0.80
    # Min ratio of predicted possession transitions matching actual game events.
    # Inherits and extends soccer_bev_pipeline._stabilize_label_sequence logic.

    support_ratio_min: float = 0.55
    # Min ratio of frames where possession holder has at least one teammate
    # within support_distance_m (inherited from RealismConfig).

    support_distance_m: float = 20.0
    # Distance threshold for "supporting player" classification (sport-adjusted).
    # Soccer: 20m | Basketball: 5m | Tennis: N/A | Baseball: 15m

    # ── Sport-Specific ───────────────────────────────────────────────────────

    # Soccer (inherited from RealismConfig)
    min_ball_in_bounds_ratio: float = 0.95
    min_attack_progress_m: float = 3.0
    min_majority_possession_ratio: float = 0.50

    # Tennis
    in_service_box_accuracy: float = 0.85
    # Fraction of predicted serve outcomes correctly classifying in/out + service box.

    rally_length_tolerance_strokes: int = 2
    # Acceptable difference between predicted and actual rally length (stroke count).

    shot_direction_zone_accuracy: float = 0.75
    # Accuracy of predicted shot direction (cross-court/down-the-line/body).

    # Basketball
    shot_clock_tolerance_s: float = 1.0
    # Acceptable timing error for shot-clock-dependent predictions.

    scoring_zone_accuracy: float = 0.85
    # Accuracy of predicted shot zone (paint/mid-range/corner-3/above-break-3/etc.)

    # Baseball
    pitch_zone_quadrant_accuracy: float = 0.75
    # Accuracy of predicted pitch landing zone (strike zone divided into 9 quadrants).

    exit_velocity_mae_mph: float = 5.0
    # Max mean absolute error for predicted batted ball exit velocity (mph).

    launch_angle_mae_deg: float = 8.0
    # Max mean absolute error for predicted launch angle (degrees).
```

### 5.3 Scoring Dimensions & Weights

| Dimension | Default Weight | Primary Metric | Notes |
|-----------|---------------|---------------|-------|
| Spatial accuracy | 25% | Position RMSE, ball path DTW | Scaled 0–1 via threshold sigmoid |
| Temporal accuracy | 20% | Event timing IoU | Binary within tolerance window |
| Event classification | 25% | Type accuracy, receiver ID | Per-task sub-criteria |
| Physical plausibility | 15% | Speed violations, smoothness | Gate: fail if > max_speed_violation_ratio |
| Tactical coherence | 15% | Possession validity, formation F1 | Inherited from realism filter logic |

**Per-task weight overrides**: `answer.json:scorer_weights` field allows each task to customize weights. Example: `SOC-PHY-001` (free-kick trajectory) up-weights physical plausibility to 40%, down-weights tactical coherence to 0%.

### 5.4 Score Aggregation

```
instance_score  = Σ(dimension_score_i × weight_i)          [0, 1]
                  (weights from answer.json:scorer_weights)

task_score      = mean(instance_scores for task_id)         [0, 1]

family_score    = mean(task_scores for family, all sports)  [0, 1]
sport_score     = mean(task_scores for sport, all families) [0, 1]

family×sport_score = mean(task_scores for family+sport cell) [0, 1]
                     (the 8×4 = 32-cell breakdown matrix)

overall_score   = mean(family_scores)                        [0, 1]
```

The **family × sport breakdown matrix** is the primary analysis artifact — it directly visualizes where cross-sport transfer succeeds or fails:

```
                          Soccer  Tennis  Basketball  Baseball  Family Avg
F1 Pre-Contact Anticip.   0.xx    0.xx      0.xx       0.xx      0.xx
F2 Tactical Phase         0.xx    0.xx      0.xx       0.xx      0.xx
F3 Space & Affordance     0.xx    0.xx      0.xx       0.xx      0.xx
F4 Config Recognition     0.xx    0.xx      0.xx       0.xx      0.xx
F5 Causal Attribution     0.xx    0.xx      0.xx       0.xx      0.xx
F6 Physical Dynamics      0.xx    0.xx      0.xx       0.xx      0.xx
F7 Counterfactual Rank    0.xx    0.xx      0.xx       0.xx      0.xx
F8 Rule-Conditioned       0.xx    0.xx      0.xx       0.xx      0.xx
─────────────────────────────────────────────────────────
Sport Avg    0.xx    0.xx      0.xx       0.xx      overall
```

Human alignment validation: For each scorer, compute Spearman's ρ between automated scores and human preference scores on a 100-instance stratified sample (balanced across families and sports). Target: ρ > 0.85.

---

## 6. BEV Renderer Extension Plan

### 6.1 Soccer (Existing)

`BEVRenderer` in `soccer_bev_pipeline.py` handles:
- 105m × 68m pitch, FIFA-standard markings
- 800 × 520px output, avc1/mp4v codec
- Team colors: red home / blue away / yellow ball

### 6.2 Tennis Court Renderer (Phase 2)

```python
# Proposed TennisBEVRenderer spec
court_length_m = 23.77        # baseline to baseline
court_width_singles_m = 8.23  # singles sideline to sideline
court_width_doubles_m = 10.97 # doubles sideline to sideline
net_height_center_m = 0.914   # rendered as thick line
service_line_depth_m = 6.40   # from net to service line
baseline_clearance_m = 3.0    # render margin behind baseline

# Agent types: AgentType.HOME (server), AgentType.AWAY (returner), AgentType.BALL
# Ball rendered larger when >1m above ground (trajectory arc visualization)
# Surface color coding: clay (terracotta), grass (green), hard (blue/grey)
```

### 6.3 Basketball Court Renderer (Phase 3)

```python
court_length_m = 28.65  # NBA full court
court_width_m  = 15.24
three_point_radius_m = 7.24   # NBA arc (corner starts at 6.70m from basket)
paint_width_m  = 4.88
paint_depth_m  = 5.79
free_throw_line_m = 4.57      # from baseline

# Agent types: 5 HOME players, 5 AWAY players, 1 BALL
# Shot clock overlaid as arc segment in corner
```

---

## 7. Research Paper Outline

**Proposed Title:** *Sports-V2V: A Multi-Sport Video-to-Video Reasoning Benchmark for Tactical Intelligence*

**Venue target:** NeurIPS Datasets & Benchmarks Track / CVPR Workshop on Sports Analytics

---

### Abstract (draft framing)
We introduce Sports-V2V, a large-scale benchmark for video-to-video reasoning grounded in real sports. Unlike prior video reasoning benchmarks that rely on synthetic environments, Sports-V2V tasks models to predict future game footage given a short input clip, evaluated against actual game footage using fully rule-based scorers. We organize 192 tasks across eight sport-agnostic reasoning families — Destination Prediction, Agent Selection, Phase Transition Timing, Configuration Recognition, Open Space Detection, Physical Property Inference, Strategic Intent Recognition, and Rule-Conditioned Outcome — each instantiated across soccer, tennis, basketball, and baseball. This design makes cross-sport transfer a first-class evaluation axis: a model's score on Destination Prediction in soccer directly predicts its generalization to the same family in tennis. Sports-V2V enables rigorous measurement of whether video reasoning capabilities transfer across domains, and where sport-specific knowledge remains necessary.

---

### §1 Introduction
- V2V reasoning as next frontier after image→video
- Existing benchmarks gap: synthetic-only, single-domain, annotation-dependent evaluation
- Contribution: taxonomy, pipeline, scorer, 50K+ BEV + broadcast clips, baseline results

### §2 Related Work
- **Video reasoning benchmarks**: VBVR, EgoSchema, Sports-QA, ActivityNet-QA
- **Sports analytics**: SoccerNet, StatsBomb, SportVU/Second Spectrum, Statcast
- **Video prediction models**: VBVR-Wan, Sora 2, Veo 3, CogVideoX, HunyuanVideo
- **Tactical AI**: DeepMind AlphaStar, MERLIN, Google Research football env

### §3 The Sports-V2V Dataset
- Taxonomy design rationale: 8 sport-agnostic task families (vs. VBVR's 5 cognitive faculties)
- Why family-first over sport-first: enables cross-sport transfer as primary research question
- Task instance construction pipeline (deterministic generation from tracking data)
- Scale table: 192 tasks × N instances × 4 sports × 2 modality tiers
- Prediction horizon distribution: micro (< 1s) / short (1–3s) / medium (5–10s) / long (> 10s)
- Difficulty analysis: per-family, per-sport, per-horizon

### §4 Data Pipeline
- Multi-sport adapter architecture (extending `soccer_bev_pipeline.py`)
- BEV synthetic tier: deterministic generation, shard-safe indexing
- Broadcast video tier: SoccerNet + Roland Garros alignment
- Canonical output format: `input_clip.mp4`, `ground_truth.mp4`, `answer.json`

### §5 Evaluation Framework
- Rule-based scorer design principles (no LLM judges)
- `SportsV2VScorerConfig` parameter derivation and calibration
- Human alignment study methodology and results
- Per-dimension scoring rationale

### §6 Benchmark Results
- Evaluated models: VBVR-Wan 2.2, CogVideoX-5B, HunyuanVideo-I2V, Sora 2, Veo 3
- **Primary result**: 8 × 4 family × sport breakdown matrix per model
- **Radar chart**: per-family scores (8 axes), one polygon per model
- Cross-sport transfer analysis: train on soccer, zero-shot eval on tennis within same family
- Scaling curves: score vs. training data size, per family

### §7 Analysis
- Difficulty distribution by family and prediction horizon
- **Cross-sport correlation matrix**: for each family, Pearson r between soccer score and tennis/basketball/baseball scores across models — high r means the family tests universal reasoning; low r means sport-specific knowledge dominates
- Which families benefit most from cross-sport pre-training (F1/F6 expected high transfer; F4/F7/F8 expected low transfer due to sport-specific rules/patterns)
- Failure mode taxonomy by family: spatial hallucination (F1/F5), temporal drift (F3), physics violation (F6), rule misapplication (F8)
- Human baseline study: domain expert vs. casual sports fan, per family

### §8 Conclusion & Future Work
- Broadcast-video tasks (Phase 3)
- Multi-angle fusion
- Online play-by-play prediction (streaming inference)
- Integration with real-time coaching tools

---

## 8. Implementation Roadmap

### Phase 1 — Soccer V2V Foundation (Months 1–2)
- [ ] Add `answer.json` exporter to `soccer_bev_pipeline.py`
- [ ] Define 48 soccer tasks (6 per family × 8 families, full Task ID list + prompt templates)
- [ ] Build `SoccerV2VScorer` class implementing `SportsV2VScorerConfig`
- [ ] Generate 5K BEV synthetic instances (Metrica + SkillCorner)
- [ ] Human alignment validation on 100-instance sample
- **Output**: Soccer V2V mini-benchmark, scorer module, alignment report

### Phase 2 — Tennis BEV (Months 3–4)
- [ ] `RolandGarrosAdapter` (Hawk-Eye point-tracking JSON → canonical)
- [ ] `TennisBEVRenderer` (court geometry, surface types)
- [ ] Tennis task list (48 tasks = 6 per family × 8 families)
- [ ] `TennisV2VScorer` extension (reuse family scorer logic, sport-specific params)
- [ ] Generate 5K tennis BEV instances
- **Output**: Tennis V2V dataset tier, combined Soccer+Tennis benchmark

### Phase 3 — Broadcast Video Integration (Months 5–6)
- [ ] SoccerNet video + tracking alignment pipeline
- [ ] Homography estimation BEV↔broadcast (OpenCV)
- [ ] Roland Garros broadcast clip alignment
- [ ] Hybrid instance format (BEV + broadcast paired clips)
- **Output**: First broadcast-video task instances, dual-modality evaluation

### Phase 4 — Basketball + Baseline Models (Months 7–8)
- [ ] `NBATrackingAdapter` (open aggregate data or partnership)
- [ ] `BasketballBEVRenderer`
- [ ] Basketball task list
- [ ] Model evaluation runs (VBVR-Wan, CogVideoX, HunyuanVideo)
- [ ] Paper draft
- **Output**: Full Sports-V2V v1.0, CVPR/NeurIPS submission

### Phase 5 — Baseball + Full Benchmark (Months 9–10)
- [ ] `StatcastAdapter`
- [ ] Baseball task list (pitch-prediction focused)
- [ ] Leaderboard infrastructure (50 reserved tasks, blind evaluation)
- [ ] Human baseline study (domain experts)
- **Output**: Sports-V2V v1.0 release, public leaderboard

---

## 9. Open Questions & Research Discussions

### 9.1 Prediction Horizon Design
- What N (input) and M (prediction) durations are appropriate per task?
- Short horizon (0.5–2s): ball trajectory, pass receiver — high precision needed
- Long horizon (5–10s): play development, formation shift — more model degrees of freedom
- Recommendation: define per-task horizon in `answer.json:prediction_horizon_s`

### 9.2 BEV vs. Broadcast as Primary Modality
- BEV is more tractable (no occlusion, exact positions) but less realistic
- Broadcast requires player re-identification, camera calibration, occlusion handling
- Recommendation: BEV for taxonomy development and scorer calibration; broadcast for final benchmark tier

### 9.3 Multi-View Consistency
- Should models be penalized for physically inconsistent predictions (e.g., player teleports)?
- The `trajectory_smoothness_min` and `max_speed_violation_ratio` parameters handle this at scorer level
- Consider: should violations be hard gates (fail instantly) or soft penalties?

### 9.4 Cross-Sport Transfer as a Research Question
- Can a model trained on soccer BEV clips generalize to tennis? (different geometry, agent count, game flow)
- This becomes a primary analysis axis in §7: measures domain generalization of tactical reasoning

### 9.5 Scorer vs. VLM Judge Hybrid
- VBVR uses 100% rule-based scoring. For open-ended tactical reasoning tasks (e.g., formation narrative), rule-based scoring may be insufficient
- Option: Rule-based for objective sub-dimensions (spatial, temporal) + constrained VLM judge for narrative sub-dimensions (with human validation)
- Recommendation: keep rule-based for v1.0; add VLM hybrid as ablation

---

## 10. Connection to Existing Codebase

| Research Concept | Existing Implementation | File |
|-----------------|------------------------|------|
| BEV clip generation | `export_vbvr_clip()` | `soccer_bev_pipeline.py` |
| Rule-based realism gating | `evaluate_clip_realism()`, `RealismConfig` | `soccer_bev_pipeline.py` |
| Possession/event analysis | `analyze_clip_events()`, `_stabilize_label_sequence()` | `soccer_bev_pipeline.py` |
| Prompt generation | `generate_clip_prompt()` | `soccer_bev_pipeline.py` |
| Deterministic sampling | `sample_clip_specs()` | `soccer_bev_pipeline.py` |
| Canonical adapter contract | `BaseTrackingAdapter`, `CANONICAL_REQUIRED_COLUMNS` | `soccer_bev_pipeline.py` |
| Regression test framework | `AdapterTests` | `tests/test_adapters.py` |
| Batch S3 runner | `run_pipeline_to_s3.sh` | `scripts/` |

The `answer.json` exporter and `SportsV2VScorerConfig` are the primary new modules. Everything else extends existing interfaces.

---

*Last updated: 2026-03-10 | VRM-Soccer repo: github.com/pppop00/VRM-Soccer | VBVR: github.com/Video-Reason*
