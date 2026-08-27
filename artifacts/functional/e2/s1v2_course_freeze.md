# S1-v2 course freeze (Part 0, 0 LLM)

protocol: `s1v2_forward_course_v1`  git: `98fc1fd947621ba139385f130252a3300ce9c169`

**COURSE_NOT_CONSTRUCTIBLE**

the arithmetic precheck did not produce a course; no live run is started.  See the precheck block for the first unmet condition.

## Half-protocol margins (recomputed from ps0b sealed counts)

| unit | hampel in oracle | census | quarter margin | **half margin** | half >= 2x | excluded |
|---|---|---|---|---|---|---|
| `GunPointAgeSpan__impulse_v2` | True | LEARNABLE | 3.75 | **7.00** | True | dual-source A (PS-0) |
| `GunPointOldVersusYoung__impulse_v2` | True | LEARNABLE | 4.15 | **5.00** | True | PS-2 / W-1 exam unit |
| `PowerCons__burst_cls2` | True | LEARNABLE | 2.22 | **5.00** | True |  |
| `PowerCons__impulse_v2` | True | LEARNABLE | 2.44 | **5.00** | True | dual-source B (PS-0c) |
| `GunPoint__impulse_v2` | True | LEARNABLE | 1.40 | **3.00** | True |  |
| `GunPointMaleVersusFemale__impulse_v2` | True | LEARNABLE | 1.35 | **2.00** | True | M-1 margin-gate unit |
| `BeetleFly__burst_cls2` | True | HELDOUT_ONLY | - | **-** | False |  |
| `GunPointMaleVersusFemale__burst_cls2` | True | HELDOUT_ONLY | - | **-** | False |  |
| `Herring__impulse_v2` | True | HELDOUT_ONLY | - | **-** | False |  |
| `MoteStrain__impulse_v2` | True | HELDOUT_ONLY | - | **-** | False |  |
| `SonyAIBORobotSurface2__burst_cls2` | True | HELDOUT_ONLY | - | **-** | False |  |
| `ToeSegmentation2__burst_cls2` | True | HELDOUT_ONLY | - | **-** | False |  |

## Course (frozen order)

| # | role | unit | menu oracle | half margin | census | coarsest half n |
|---|---|---|---|---|---|---|

## Transfer graph

- (none: the course is not constructible)

## Treatment-group precheck (arithmetic, pre-LLM)

- producers: PowerCons__burst_cls2, GunPoint__impulse_v2
- distinct task_episode_id: True
- five-axis Scope non-empty: True (19 pattern leaves)
- beneficiary: None -- no held-in LEARNABLE unit machine-matches the producers' Scope once the units spent on other books are excluded
- identity units: BeetleFly__impulse_v2, BirdChicken__burst_cls2
- HELDOUT_ONLY unit: GunPointMaleVersusFemale__burst_cls2
- expected card boundary: after position None
- expected first divergence: position None

- **Delta_material** (regret gate) = max_u(1/n_slice_u) = None

## First unmet condition

- no beneficiary: every hampel-bearing unit left after the four book exclusions is HELDOUT_ONLY, so held-in feedback cannot approve the supplied family on any of them

### Candidate beneficiaries, scored

| unit | Scope match | held-in census | half margin |
|---|---|---|---|
| `BeetleFly__burst_cls2` | False | HELDOUT_ONLY | - |
| `GunPointMaleVersusFemale__burst_cls2` | True | HELDOUT_ONLY | - |
| `Herring__impulse_v2` | False | HELDOUT_ONLY | - |
| `MoteStrain__impulse_v2` | False | HELDOUT_ONLY | - |
| `SonyAIBORobotSurface2__burst_cls2` | False | HELDOUT_ONLY | - |
| `ToeSegmentation2__burst_cls2` | False | HELDOUT_ONLY | - |

### If an exclusion were released

| unit | spent on | Scope match | census | half margin | would qualify |
|---|---|---|---|---|---|
| `GunPointAgeSpan__impulse_v2` | dual-source A (PS-0) | True | LEARNABLE | 7.00 | **True** |
| `GunPointMaleVersusFemale__impulse_v2` | M-1 margin-gate unit | True | LEARNABLE | 2.00 | **True** |
| `GunPointOldVersusYoung__impulse_v2` | PS-2 / W-1 exam unit | True | LEARNABLE | 5.00 | **True** |
| `PowerCons__impulse_v2` | dual-source B (PS-0c) | True | LEARNABLE | 5.00 | **True** |

