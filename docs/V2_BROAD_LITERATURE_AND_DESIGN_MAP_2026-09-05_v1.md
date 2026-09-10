# V2 Broad Literature and Design Map

Status: independent literature survey and design map for Fable.  
Date: 2026-09-05.  
Role: cross-domain literature researcher, not an advocate for Method v2.  
Scope: read-only on the project; no code, contract, threshold, artifact, experiment, or git changes.  
Novelty of any joint claim in this document is **uncertain** until a later method freeze; this file only maps options, counterexamples, and closest work.

This report is not a preregistration, not a contract, and not a claim that any candidate will produce a positive KDD replay. Exposed KDD / HEC-1 numbers are cited only as *already-measured constraints*, never as new capability evidence.

---

## 0. How this survey was bounded

Question under study:

> With no clean ground truth, and only noisy delayed downstream feedback, how can an agent compile data-preparation experience into a persistent, revisable Skill that has restricted execution rights, can be reused across windows and datasets, and does not harm unseen entities?

HEC-1 constraints used as design tests, not as optimization targets:

| ID | Constraint | Source (project, already exposed) |
| --- | --- | --- |
| C1 | Deployment-time visible features do not reliably predict future victims | D1: AUC 0.45–0.65, `NO_OUTCOME_FREE_SEPARATOR` |
| C2 | New Scope entrants carry most severe harm | D1/D5: 8/10 severe harms are new entrants |
| C3 | In pooled Consumer, gain and harm travel mainly through model routing | D5: `ROUTE_DOMINANT`, algebraic share ≈ 73%, interaction ≈ 0 |
| C4 | Online/frozen treatment is sparse; physical call budget often exhausts before a full decision | HEC-1: 62/69 ties; 5-call ≈ 1 proposal |

Current Method v2 (not assumed correct): transferable Program + PatternScope, plus target-local entity evidence / deployment eligibility; unseen entities only probed; training-side preparation split from serving-side model routing.

Search practice: public keywords only; official conference / ACM / IEEE / PMLR / ACL / OpenReview / arXiv / author pages / code repos preferred; MDPI excluded; abstracts plus method/experiment sections checked where HTML/PDF was available; recent arXiv marked **preprint**. Papers that could not be opened beyond the abstract are labeled as such.

---

## 1. Executive Summary

### 1.1 Eight conclusions

1. **“No evidence, no Fast-only deploy” is a mature *safety pattern*, not a mature *identity table*.** The closest established names are conservative / safe contextual bandits, safe policy improvement with baseline bootstrapping (SPIBB), selective classification / reject option, and eligibility-gated dynamic treatment regimes. A hard UID allowlist is the *degenerate special case* of those methods when covariates have no predictive power. HEC-1 C1 makes that degeneration locally rational; it does not make it the best long-run architecture.

2. **Hard entity evidence is at high risk of collapsing into ID memorization.** Hierarchical Bayes / mixed-effects / CATE / empirical-Bayes search systems exist precisely to avoid the complete-pooling vs no-pooling extremes. If v2 stores `{positives, harms, quarantined}` keyed only by entity ID, reviewers can correctly call it a routing table. The literature fix is *partial pooling*: a population prior over Program×Pattern×Consumer, shrunk by entity-level residual, with uncertainty—not a count threshold—controlling execution rights.

3. **Zero-shot transfer and held-in probe are different objects.** Across AutoML, TSFM, domain generalization, and agent skill libraries, what can move without target labels is *procedure + applicability prior + safety protocol*. What cannot move is *who is safe to treat*. WikiSkill, Voyager, TimeClaw, and FFORMA all transfer a compiled artifact and then either re-verify or mix; none transfer a source entity ledger.

4. **Delayed downstream reward is a first-class obstacle, not an implementation detail.** Joulani et al. (ICML 2013) show delay adds regret even in stochastic bandits. Conservative methods that wait for two clean confirmations (v2’s proposed `k=2`) become coverage-starved under delay + entity churn. Promotion / probation / revocation exist, but in mature systems they are *uncertainty- and recency-aware*, not irreversible allowlists.

5. **The four decisions v2 wants to fuse are distinct in other fields.** Program validity ≠ training-row selection ≠ serving-context preparation ≠ serving-model routing. Forecast combination (FFORMA), MoE gating, and causal policy learning treat the last of these as a *soft mixture*, not a hard switch. D5 says routing carries both gain and harm; that is an argument for *decoupling and shrinking*, not for a second hard allowlist.

6. **When the same route carries mean gain and tail harm, mean-only optimization is the wrong objective.** CVaR / chance-constrained policy learning / distributionally robust optimization / SPIBB all exist because maximizing average sMASE will re-select the route that created HEC-1’s severe tails. Constraining `max_single_series_harm` is already in the project contract; literature says the *optimizer* must see that constraint, not only the gate.

7. **“No evidence ⇒ never deploy” can freeze coverage.** Conservative bandits, explore-then-commit, and Soft-SPIBB all spend a *safety budget* to probe. Permanent abstention on unseen entities is safe and also a known way to get a trivial policy (identity forever) when the entity population is non-stationary. C2 is real; the mature response is budgeted, baseline-mixed exploration, not a lifetime ban.

8. **Self-evolution is not a per-entity ledger update.** WikiSkill, Voyager, TimeClaw, and Self-Harness all treat evolution as *change to a reusable procedure or harness surface*, gated by held-out/validation evidence. Updating `EntityEvidence[uid]` is target-local deployment adaptation. It supports safety. It does not, by itself, support A5.

### 1.2 What in v2 is most likely right vs most dangerous

Most likely right (aligned with measured C1–C3 and with SPI / selective prediction):

- Do not use deployment-time features as a harm predictor (C1).
- Do not give cross-domain execution rights to source entity IDs.
- Split training-set intervention from serving-time routing (C3).
- Keep identity / raw as the default baseline; require evidence to leave it.
- Report coverage beside utility; do not hide abstention.

Most dangerous:

- Hard `positives ≥ k` keyed by UID, with no shrinkage, no recency, no mixture.
- Calling that ledger “Skill evolution” or “self-evolution”.
- Freezing Fast-only to a binary program-model switch (FFORMA’s main lesson is the opposite: averaging beats selecting).
- Never probing unseen entities on Fast-only, then claiming cross-domain competence on datasets whose roster churns.
- Using exposed-KDD v2 replay as evidence of a new capability.

### 1.3 Is there something more mature than an entity allowlist?

Yes. The more mature object is:

```text
population prior over (Program, Pattern, Consumer)
  + entity-level residual posterior
  + LCB / posterior P(benefit) execution right
  + soft mix between raw and program models
  + baseline bootstrap on unseen or uncertain entities
  + recency / change-point decay
```

An allowlist is what you get if you set the prior variance to infinity, the mixture weight to `{0,1}`, and the LCB threshold to “at least k raw successes.” That is a legitimate conservative corner. It should be an ablation, not the method identity.

---

## 2. Literature Map

Screened: ~45 candidates. Core set below: 28. Types: **closest** / **mechanism** / **evaluation-only** / **counterexample**.

| paper | year / venue / status | paper type | problem | feedback / supervision | unit of adaptation | knowledge representation | safety mechanism | transfer mechanism | evaluation unit | relevance | stable URL |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Krishnan et al., ActiveClean | 2016 / VLDB | mechanism | prioritize cleaning for convex ML | human oracle + model gradient | record / mini-batch | SGD state, not a skill | convex-loss convergence, not tail harm | none; single table | dataset | downstream-aware cleaning exists, but needs an oracle and a convex Consumer | https://arxiv.org/abs/1601.03797 |
| Krishnan et al., BoostClean | 2017 / arXiv **preprint** | mechanism + counterexample | select detector×repair by boosting | **requires a clean hold-out** | dataset | ensemble of cleaning ops | none beyond val accuracy | none | dataset | closest “cleaning as algorithm selection”; unusable as-is because we have no clean hold-out | https://arxiv.org/abs/1711.01299 |
| Rekatsinas et al., HoloClean | 2017 / PVLDB | counterexample | probabilistic cell repair | integrity constraints + statistical signals | cell | factor graph | constraint satisfaction, not downstream | weak (constraints) | table | repairs toward *clean values*, not downstream utility; GT-like constraints | https://arxiv.org/abs/1702.00820 |
| Berti-Équille, Learn2Clean | 2019 / WWW | closest (cleaning) | RL over preprocessing sequence | downstream ML metric on one dataset | dataset | Q-table over operators | none | none | dataset | executable prep program + downstream reward; no entity safety, no freeze, no cross-domain | https://doi.org/10.1145/3308558.3313602 |
| Mahdavi et al., Raha | 2019 / SIGMOD | mechanism | configuration-free error detection | few labeled tuples | cell / tuple | detector ensemble + classifier | labeling budget | historical-dataset filter | dataset | transfer is detector ranking, not a Skill; needs labels | https://doi.org/10.1145/3299869.3324956 |
| Mahdavi & Abedjan, Baran | 2020 / PVLDB | mechanism | configuration-free correction | few repair examples | cell | context models + transfer | precision via ensemble | transfer learning from past repairs | dataset | still cell-GT oriented | https://m-mahdavi.github.io/assets/pdfs/mahdavi2020baran.pdf |
| Li et al., CleanML | 2021 / PVLDB experiments | **counterexample** + evaluation | does cleaning help ML? | real errors, no single GT | dataset × model × error type | none (benchmark) | statistical testing | none | dataset | cleaning can *hurt*; train vs test cleaning are different decisions | https://chu-data-lab.github.io/downloads/CleanML.pdf |
| Li et al., DiffPrep | 2023 / SIGMOD | mechanism | differentiable preprocessing search | labeled val loss | feature | continuous pipeline weights | none | none | dataset | feature-wise pipelines; assumes differentiable model + labels | https://doi.org/10.1145/3589328 |
| Talagala, Hyndman, Athanasopoulos, FFORMS | 2018 WP / 2023 J. Forecasting | closest (AutoML) + counterexample | per-series model *selection* from features | forecast error on held-out origin | series | RF over tsfeatures | none | meta-model trained on a reference corpus | series / competition set | static features → hard choice; project P4 already found features lose to a fixed choice | https://doi.org/10.1002/for.2963 |
| Montero-Manso et al., FFORMA | 2020 / Int. J. Forecasting | closest (routing) + **counterexample to hard switch** | per-series model *averaging* | forecast error | series | GBDT weights over methods | implicit hedge via mixture | feature meta-learner | M4 series | averaging beat selection (M4 2nd); direct warning against v2 binary route | https://doi.org/10.1016/j.ijforecast.2019.02.011 |
| Shah et al., AutoAI-TS | 2021 / SIGMOD | mechanism | zero-conf forecast pipelines | holdout accuracy | dataset / pipeline | pipeline templates + T-Daub ranking | none | none | dataset | preprocessing is inside a pipeline, not a scoped Skill | https://doi.org/10.1145/3448016.3457557 |
| Salles et al., TSPred | 2022 / Neurocomputing | evaluation + mechanism | nonstationary TS transforms + models | prediction error | series | transform×model process object | none | none | series | closest named “transformation selection” toolkit; not agentic, not safe | https://doi.org/10.1016/j.neucom.2021.09.067 |
| Wu, Shariff, Lattimore, Szepesvári, Conservative Bandits | 2016 / ICML | closest (safety) | never much worse than a baseline arm | bandit reward | time / arm | UCB + baseline slack | **uniform-in-time return constraint vs baseline** | none | synthetic | canonical form of “stay on identity until slack exists” | http://proceedings.mlr.press/v48/wu16.pdf |
| Kazerouni et al., Conservative Contextual Linear Bandits | 2017 / arXiv; AISTATS lineage | closest (safety) | same, with context | linear reward | context | CLUCB | baseline-percentage constraint, additive conservative penalty | none | synthetic / ads-like | context is allowed; unseen context falls back to baseline | https://arxiv.org/abs/1611.06426 |
| Laroche, Trichelair, Tachet, SPIBB | 2019 / ICML | closest (execution right) | improve a known baseline from batch data | offline trajectories | state-action | policy constrained to baseline where counts are low | **bootstrap baseline on under-visited pairs** | assumes baseline known | MDP / DQN nav | almost the mathematical form of entity eligibility, with a *count of (s,a)* not a UID | https://proceedings.mlr.press/v97/laroche19a/laroche19a.pdf |
| Nadjahi et al., Soft-SPIBB | 2019 / ECML | mechanism | soften SPIBB’s binary set | offline | state-action | uncertainty-proportional policy change | constrained deviation, still SPI | none | MDP | the obvious alternative to v2’s hard quarantine | https://www.microsoft.com/en-us/research/wp-content/uploads/2019/07/ECML2019___Soft_SPIBB-8.pdf |
| Joulani, György, Szepesvári | 2013 / ICML | mechanism | delayed feedback | delayed loss | time | black-box reduction | none | none | bandit | delay is additive in stochastic, multiplicative in adversarial; `k=2` confirmations are expensive | https://proceedings.mlr.press/v28/joulani13.html |
| Kakade & Langford, Conservative Policy Iteration | 2002 / ICML | mechanism | never-much-worse policy update | on-policy advantage | policy | mixture with incumbent | mixture until advantage proven | none | MDP | A3-as-incumbent is this pattern | historical; ICML 2002 |
| Murphy, Optimal Dynamic Treatment Regimes | 2003 / JRSS-B | closest (personalization) | sequential individualized rules | delayed outcome Y | person × time | regime = list of rules | eligibility typically encoded in the rule | none | person | “treat only if history satisfies …” is DTR eligibility, not an ID list | https://doi.org/10.1111/1467-9868.00389 |
| Wager & Athey, Causal Forests | 2018 / JASA | mechanism | CATE / HTE | potential outcomes, unconfoundedness | individual | forest of honest trees | honest splitting / CI | none | individual | entity effect without ID lookup, *if* covariates work; C1 says they currently don’t | https://doi.org/10.1080/01621459.2017.1319839 |
| Gelman & Pardoe; Gelman & Hill | 2006 / 2007 | mechanism | partial pooling | hierarchical likelihood | group | random effects | shrinkage, not a hard gate | population prior | group | default statistical answer to “avoid ID memorization” | Gelman & Hill, *Data Analysis Using Regression and Multilevel/Hierarchical Models*, CUP 2007 |
| Gama et al., Concept Drift Survey | 2014 / ACM CSUR | mechanism | when old knowledge expires | streaming labels | time | memory + forgetting | change detection | none | stream | evidence needs a forgetting rate matched to drift rate | https://doi.org/10.1145/2523813 |
| Gama, Sebastião, Rodrigues | 2013 / Machine Learning | evaluation | prequential error | interleaved test-then-train | time | fading factors / windows | change detection via error | none | stream | evaluation unit for nonstationary learning; matches held-in windows better than CV | https://doi.org/10.1007/s10994-012-5320-9 |
| Geifman & El-Yaniv | 2017 / NeurIPS | closest (abstention) | selective classification | i.i.d. labels | example | confidence + reject | high-probability risk at chosen coverage | none | ImageNet/CIFAR | coverage–risk curve is the honest report for v2 | https://arxiv.org/abs/1705.08500 |
| Angelopoulos et al., Conformal Risk Control | 2024 / ICLR | mechanism + warning | control E[loss] of any monotone loss | exchangeable labeled calibration | example | threshold on nested losses | finite-sample risk bound | covariate-shift extension exists, weak for TS drift | image / NLP | **not drop-in**: we lack exchangeable clean labels; delayed sMASE is not their loss | https://arxiv.org/abs/2208.02814 |
| Farinhas et al., Non-exchangeable CRC | 2023 / arXiv **preprint** | mechanism + warning | CRC under drift / change points | weighted nonconformity | example / time | relevance weights | weighted risk bound | recency weighting | synthetic + real | usable only if a weight that tracks temporal relevance can be defined | https://arxiv.org/abs/2310.01262 |
| Wang et al., Voyager | 2023 / arXiv **preprint** | closest (skill library) | open-ended skill compilation | environment success in Minecraft | skill / code fn | executable JS + embedding index | execution error loop, not tail risk | zero-shot to new worlds | game session | executable skills + verification; **verifiable success label** | https://arxiv.org/abs/2305.16291 |
| Shinn et al., Reflexion | 2023 / NeurIPS | mechanism | verbal RL | task success on retries | episode | natural-language reflection | none | none | task instance | needs retries and a success signal; stateless across tasks | https://arxiv.org/abs/2303.11366 |
| Zhao et al., ExpeL | 2024 / AAAI | mechanism | cross-task experiential memory | train-task success | task | insights + retrieved trajectories | none | retrieve by task similarity | HotpotQA / ALFWorld | memory ≠ execution right; raw traces in prompt (project already rejected this for Fast) | https://arxiv.org/abs/2308.10144 |
| Tang et al., WikiSkill | 2026 / arXiv **preprint** | closest (skill evolution) | compile experience → wiki → skills | **ground-truth answers**, val accuracy gate | skill / wiki page | Raw / Wiki / Skill layers | rollback skills if val score drops; wiki never rolls back | cross-model skill transfer | 5 benchmarks × 3 seeds | three-layer split is independently convergent; **depends on GT**; Inference Agent must *not* read the wiki | https://arxiv.org/abs/2608.27454 |
| Li et al., TimeClaw | 2026 / arXiv **preprint** | closest (TS harness) | TS-native agent harness | benchmark scores; evolution uses held-out success rate | task instance | tools + fingerprint memory + evolved routines | admit tool iff held-out success > γ | fingerprint kNN of *successful* traces | CiK / TSRBench / TSAIA | A5/A3 is a kNN switch; GT used to verify evolution; no per-series harm budget | https://arxiv.org/abs/2606.05404 |
| AegisTS | 2026 / arXiv **preprint** | closest (TS cleaning agent) | MTS cleaning pipeline via hierarchical RL | dual reward: issue drop **and** downstream; cleaning metrics use injected GT in code | dataset | RL policy over issue order × operator | none for unseen series tails | freeze RL, zero-shot new dataset | dataset | downstream-aware TS cleaning; **GT-heavy**; single global pipeline, not scoped entities | https://arxiv.org/abs/2605.04902 |
| Chow et al., CVaR MDPs | 2014–2015 / NeurIPS | mechanism | mean–CVaR sequential decisions | return distribution | policy | CVaR of cost/return | tail of the return, not mean | robustness interpretation | small MDP | correct objective family for C3 | https://arxiv.org/abs/1406.3339 |
| Duchi & Namkoong, DRO | 2021 / Ann. Statist. | mechanism | uniform performance over shifts / subpopulations | empirical + ambiguity set | model | f-divergence DRO, CVaR as special case | worst subpopulation | intended for latent groups | several ML tasks | “optimize the harmed subgroup” without naming UIDs | https://arxiv.org/abs/1810.08750 |
| Shi et al., Time-MoE | 2025 / ICLR | mechanism | sparse experts for TSFM | forecast loss | token | MoE gating | auxiliary load-balance, Huber loss | pretrain corpus | zero-shot datasets | soft routing at token level; **routing collapse** is a named failure | https://arxiv.org/abs/2409.16040 |
| Liu et al., Moirai-MoE | 2024 / arXiv **preprint** | mechanism + counterexample | learned vs frequency routing | forecast | token | sparse experts | none | zero-shot 39 datasets | dataset | frequency heuristic is a bad Scope analogue; learned token routing beats it | https://arxiv.org/abs/2410.10469 |
| Das et al., TimesFM | 2024 / ICML | evaluation + mechanism | TSFM zero-shot | forecast | series | patched decoder + instance norm | none | pretrain mixture | dataset | what TSFMs transfer is representation + norm, not a prep program | https://arxiv.org/abs/2310.10688 |
| TSFM normalization study | 2025 / arXiv **preprint** | evaluation | normalization vs zero-shot | MASE | dataset (LODO) | RevIN vs dataset-level stats | none | leave-one-dataset-out | dataset | **dataset-as-unit** already used in TSFM; prep choice dominates architecture | https://arxiv.org/abs/2512.02833 |
| Gulrajani & Lopez-Paz, DomainBed | 2021 / ICLR | evaluation + counterexample | domain generalization | labeled domains | domain | various DG algorithms | none | leave-one-domain-out | domain | ERM often wins when tuned; LODO is hard; do not overclaim DG | https://arxiv.org/abs/2007.01434 |
| Petrik, Ghavamzadeh, Chow, SPI | 2016 / ICML | mechanism | robust baseline regret | batch | state | robust MDP vs baseline | high-probability improvement | none | MDP | ancestor of SPIBB; needs a baseline | ICML 2016 PMLR |

Papers screened and **not** retained as core: MDPI forecast-ensemble survey; generic “baxter-ts” GitHub AutoML; unvetted 2026 preprints that only repeated skill-library ideas without a verification story (SkillX kept as a pointer, not a core row); hardware-oriented safe-RL papers whose constraint is physical cost, not delayed downstream utility.

---

## 3. Closest-Work Matrix

Legend: Y = present; P = partial / different object; N = absent. All 2026 agent papers are **preprints**.

| dimension | ActiveClean / Learn2Clean | FFORMS | FFORMA | SPIBB / Cons. bandits | Causal forest / DTR | WikiSkill | TimeClaw | AegisTS | Voyager | DiffPrep | CRC / selective class. | **this project (target)** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| no clean truth | N (oracle / constraints / labels) | P (uses forecast error, not cell GT) | P | P (rewards observed) | N (potential outcomes assumed) | N (GT answers) | N (benchmark GT for evolution) | N in code (injected GT for cleaning metrics) | P (env success, not cell GT) | N (val labels) | N (calibration labels) | **Y** |
| delayed downstream feedback | N / weak | P (origin holdout, not online delay) | P | P (Joulani needed) | Y (DTR) | N (immediate val score) | N | P (proxy then final) | N (env step) | N | N | **Y** |
| executable data-prep program | Y (ops / Q-learning) | N (forecast method, not prep) | N | N | N | P (scripts in skills, not TS DSL) | P (TS tools, not scoped prep) | Y (cleaning ops) | Y (code skills) | Y (feature pipelines) | N | **Y** |
| Task / Consumer conditioning | P (metric in Learn2Clean) | N | N | N | P (outcome Y) | N | P (task prompt) | Y (forecast / clf / cluster) | N | P (one model) | N | **Y** |
| entity-local evidence | N | P (per-series features, not history of treatment) | P | P (counts on (s,a) or context) | Y (person history) | N | N (fingerprint of *task*, not series UID) | N | N | N (feature-wise, not entity) | P (per-example reject) | **Y (v2 proposes)** |
| cross-domain transferable knowledge | N | P (meta-model) | P | N | N | Y (skills across models) | P (memory across tasks) | P (frozen RL zero-shot) | Y (new worlds) | N | N | **required for A5** |
| execution-right promotion / revocation | N | N | N | **Y** | P (eligibility rules) | **Y** (val gate + skill rollback; wiki never rolls back) | Y (tool admit if success > γ) | N (train then freeze) | P (add on success, weak revoke) | N | Y (reject option) | **required** |
| held-in → held-out evaluation | P | P | P | N (batch) | P | Y (train/val/test) | Y (train/test) | P (cross-dataset) | P | Y | Y (calib/test) | **Y, Fast-only freeze** |
| autonomous Harness modification | N | N | N | N | N | P (skills, not risk/scope compiler) | P (new tools) | N (RL weights) | P (skill library) | N | N | **required for self-evolution** |

**Net:** no row is this project. The dangerous near-duplicates are (i) SPIBB with `(s,a)` replaced by `uid`, (ii) FFORMS with features replaced by an ID lookup, (iii) TimeClaw memory with fingerprints replaced by UIDs, (iv) WikiSkill without GT, which is *not* WikiSkill.

---

## 4. Design Pattern Cards

### Card 1 — Evidence-bounded eligibility (conservative / SPIBB)

- **Mechanism.** Keep a known-safe baseline (identity / raw model). Switch to a new action only where visit counts or lower confidence bounds certify improvement.
- **Form.** Wu et al.: cumulative return of learner ≥ α × baseline, uniformly in t. SPIBB: if `N(s,a) < N_∧`, force `π(a|s) = π_b(a|s)`; else optimize. v2’s `positives ≥ k ∧ ¬quarantined` is SPIBB with `s = uid`, `a = program-route`, `π_b = identity`.
- **Observations.** On-policy or batch counts of (context, action, delayed reward).
- **Assumptions.** Baseline is safe; counts are comparable; stationarity of the (s,a) value.
- **Maps to HEC-1.** Directly attacks C2 (unseen ⇒ baseline) and C1 (do not predict, count). Does **not** fix C3 (hard switch still routes the whole model) or C4 (counts accrue slowly under delay).
- **Main risk.** Count keyed by UID ⇒ ID table; under drift, old counts lie; under roster churn, coverage → 0.
- **Minimal check.** 0-LLM replay: SPIBB-style LCB vs hard `k=2` vs Soft-SPIBB mixture, on the same HEC-1 per-series ledger. If LCB/soft dominate hard k, the allowlist is not necessary.

### Card 2 — Hierarchical partial pooling

- **Mechanism.** Entity parameter `θ_i ~ P(θ | ψ)`; `ψ` estimated from all entities. Little data ⇒ shrink to `ψ`; lots of data ⇒ entity residual.
- **Form.** Gelman: `y_{it} = μ + u_i + ε_{it}`, `u_i ~ N(0, τ²)`. CATE forests estimate `τ(x)` not `τ(id)` when `x` works; mixed-effects ML puts a random intercept *on top of* `x`.
- **Observations.** Repeated outcomes per entity; optional covariates.
- **Assumptions.** Exchangeable residuals given the hierarchy; `τ` estimable. **Fails if entities are not exchangeable and covariates are empty**—which is the HEC-1 local world.
- **Maps to HEC-1.** C1 says `x` is weak, so the random intercept *is* the entity residual. That is still not a hard allowlist: `P(Δ_i > 0 | data)` can stay a probability.
- **Main risk.** If `τ` is huge, partial pooling ≈ no pooling ≈ ID lookup. Must report pooling factor.
- **Minimal check.** Fit a two-level model on Support/delayed gains: series random intercept + Pattern features. If the intercept absorbs almost all variance, v2’s ID table is statistically honest—and the paper must say so. If Pattern or Consumer random effects absorb variance, the ID table is overfit.

### Card 3 — Conservative policy improvement / CPI mixture

- **Mechanism.** Never replace the incumbent; mix until advantage is proven (Kakade & Langford 2002; Petrik et al. 2016; SPIBB).
- **Form.** `π' = (1-α) π_b + α π̂`, choose α so that a high-probability lower bound on `ρ(π') - ρ(π_b)` is ≥ 0 (or ≥ −ε).
- **Observations.** Off-policy or on-policy advantage; need a baseline policy, not just a baseline *score*.
- **Assumptions.** Baseline identifiable; importance weights or counts available. CRM (Swaminathan & Joachims, ICML 2015) also needs propensities—which this project **does not log**. Do not build a propensity platform (already rejected in the 2026-08-19 theory note).
- **Maps to HEC-1.** A3 as incumbent, A5 as candidate, identity as `π_b` for unseen entities. Soft mix attacks C3’s binary route.
- **Main risk.** Without propensities, “conservative” becomes heuristic. That is acceptable if labeled as a policy, not a theorem.
- **Minimal check.** On HEC-1 ledger, compare hard route vs `α_i = clip(n_i / (n_i + n0))` mix of raw and program predictions.

### Card 4 — Safe exploration with abstention

- **Mechanism.** Abstention is an action with zero incremental reward/loss (Cortes et al., NeurIPS 2024 bandits-with-abstention; Geifman & El-Yaniv selective classification). Exploration is allowed only while a safety budget vs baseline remains.
- **Form.** Conservative UCB: explore when empirical slack vs baseline > UCB of exploration cost. Selective: output reject if confidence < t(r*), choose t for risk r* at coverage c.
- **Observations.** Immediate or delayed reward; a well-defined baseline.
- **Assumptions.** Selective classification’s high-probability bounds need i.i.d. labels—**false here**. Use the *shape* (coverage–risk curve), not the theorem.
- **Maps to HEC-1.** C2/C4: unseen entities abstain on Fast-only *unless* held-in slack remains to buy a probe. Distinguishes “probe in held-in” from “deploy in held-out.”
- **Main risk.** Permanent reject ⇒ trivial identity policy. Reviewers will ask for the coverage–utility Pareto, not a single gated mean.
- **Minimal check.** Plot coverage vs harm vs gain for k∈{1,2,3}, for LCB, and for a small held-in exploration quota on new entrants.

### Card 5 — Temporal evidence decay / change-point reset

- **Mechanism.** Match forgetting rate to drift rate (Webb et al. 2018; Gama CSUR 2014). Sliding window, exponential decay, or reset on detected change.
- **Form.** `n_i ← λ n_i + 1{event}`; or ADWIN-style window; or CPTC-style state-conditional conformal (2025 **preprint**).
- **Observations.** Time-stamped outcomes; optional drift detector on residual utility.
- **Assumptions.** Drift is not faster than the window; false resets cost coverage.
- **Maps to HEC-1.** Transfer audit 10/34 Support-safe → +144-safe. Old positives must expire. v2’s `last_window` field is a stub; without decay it will deploy on stale members (D6 already saw flip rates on continuing members; n is small, so this is a *risk*, not a proven law).
- **Main risk.** Too much decay ⇒ reverts to permanent cold start (C4 worse).
- **Minimal check.** Replay with λ ∈ {1.0, 0.7, 0.4} and with a hard “invalid after 1 skipped window.” Report coverage and msh together.

### Card 6 — Two-stage preparation vs routing

- **Mechanism.** Training-set intervention and serving-time action are different causal objects. CleanML explicitly separates “clean train” vs “clean test.” D5’s 2×2 is the TS analogue: `{raw, program} model × {raw, prepared} context`.
- **Form.** Decision T: which training rows enter the program corpus. Decision S: for each served entity, mix or switch models/contexts. In pooled Ridge, T has **interference**: preparing i’s train rows changes predictions for j.
- **Observations.** Per-series gain under the four cells (already collected in D5).
- **Assumptions.** Evaluator can serve the four cells (already true). Extra fits are billed.
- **Maps to HEC-1.** C3. v2’s two-decision table is the right *split*. It is not yet the right *estimators*: T should not copy S’s allowlist blindly (interference).
- **Main risk.** If T = all PatternScope matches, unseen entities still contaminate the program model (spillover harm). If T = allowlist, program model may be too small to beat raw.
- **Minimal check.** Already proposed in v2 §5: T∈{PatternScope, evidence-qualified} on recorded HEC-1 actions. Add a third: T = down-weighted / sample-influence, not hard inclusion.

### Card 7 — Soft raw/program mixture (FFORMA lesson)

- **Mechanism.** Bates–Granger combination; FFORMA learns *weights*, not a winner. Forecast combination puzzle: learned hard selection often loses to simple averages.
- **Form.** `ŷ_i = w_i ŷ^{prog}_i + (1-w_i) ŷ^{raw}_i`, `w_i = σ(a + b·evidence_i)` or GBDT(features). Bound `w_i=0` when no evidence (conservative corner).
- **Observations.** Both predictions (D1 already computes them for S1).
- **Assumptions.** The two predictors are not perfectly correlated; mixture is cheap at serve time.
- **Maps to HEC-1.** C3: same pipe carries gain and harm, so *fractional* routing can cut tails faster than means. C1: weights from evidence, not from weak features—or from features *only as a prior on w*.
- **Main risk.** Two full models at serve (already the architecture). Reviewers: “this is just blending.” Need to show w is *compiled from Skill evidence*, not a free per-series oracle.
- **Minimal check.** 0-LLM: w from evidence counts vs w from FFORMA-style features vs hard switch, same four-line gates.

### Card 8 — Cross-domain skill compiler (WikiSkill / Voyager shape)

- **Mechanism.** Raw traces immutable; a compiled knowledge layer; executable skills gated on validation; skills can roll back, compiled knowledge may persist (WikiSkill). Voyager: code skill admitted after execution success. TimeClaw: tool admitted if held-out success > γ.
- **Form.** WikiSkill: Inference Agent sees *skills only*; Wiki Maintainer compiles; Proposer is atomic; accept iff `R_val > R_best`. Ablation: giving the Inference Agent the wiki **hurt** (−2.8 in the project’s prior reading; paper §5.1: wiki access during training negatively affects skill development).
- **Observations.** Trajectories + a **scalar success**. That scalar is the hole: they have GT accuracy; we have delayed, tail-aware downstream utility.
- **Assumptions.** Validation distribution ≈ test; success is well-defined.
- **Maps to HEC-1.** This is the *only* pattern that can be called self-evolution. Entity ledgers are not this pattern. Fast must not read raw Episode banks (already T233). Slow compiles Program + PatternScope + *rules* for evidence, not UID lists.
- **Main risk.** Copying WikiSkill’s `R_val > R_best` with noisy delayed utility will promote on noise (HEC-1 already had 1 REVISABLE, 0 revisions). Need Support then delayed, already in the governance.
- **Minimal check.** Do not implement a wiki. Implement: atomic Slow patch; deterministic gate; rollback of Program/Scope; persistence of census/statistics. That is already the project’s outer loop. The missing compile target is “evidence *rules* + mixture prior,” not markdown.

### Card 9 — Tail-aware objective (CVaR / DRO / chance constraint)

- **Mechanism.** Optimize mean subject to CVaR or worst-group loss (Chow et al.; Duchi & Namkoong). Bayesian safe policy learning uses posterior P(new policy worse | x) as a chance constraint (JRSS-A 2025 application paper; methods are older).
- **Form.** `max_π E[U] s.t. CVaR_α(-U) ≤ c` or `max_π inf_{Q∈𝒰} E_Q[U]`.
- **Observations.** Distribution of per-entity utilities, not only the mean.
- **Assumptions.** Enough entities to estimate a tail; ambiguity set not vacuous.
- **Maps to HEC-1.** C3: mean-optimal route *is* the tail-risky route. Gates on msh/hf are *filters*; they are not the selection objective. validation-search maximized Support safety then still failed +144 (10/34).
- **Main risk.** CVaR on n≈20 series per window is noisy. Use it as a *ranking regularizer*, not a new threshold (do not retune 0.20/0.30).
- **Minimal check.** Rank Support candidates by mean − λ·worst-k or mean − λ·CVaR, freeze λ before looking at +144. Compare future-safe retention to HEC-1’s 10/34.

### Card 10 — Dataset-as-unit / leave-one-dataset-out

- **Mechanism.** DomainBed; TSFM LODO normalization study; FFORMA’s reference corpus vs M4. Statistical unit is the dataset; domain is the transfer axis.
- **Form.** Train meta-knowledge on D\{d}, adapt on d’s held-in, test on d’s held-out. Average first within dataset, then within domain.
- **Observations.** Multiple independent datasets; exposure ledger.
- **Assumptions.** Datasets are not secretly the same generating process (KDD origins are *not* independent datasets—P4d already corrected this).
- **Maps to HEC-1.** HEC-1 cannot support L4. This card is the *next* experiment, not a KDD replay.
- **Main risk.** DomainBed: fancy DG often loses to tuned ERM. A5 must beat A3 *and* Static after LODO, not only after a lucky order.
- **Minimal check.** Data-asset audit (already planned): ≥2 datasets/domain, exposure, serving adapter, whether Program DSL even runs.

---

## 5. Candidate Method Architectures

All three keep: typed Program DSL, PatternScope as *probe* prior, held-in / freeze / Fast-only, four-line risk gates, A5 = audited K_t + target held-in calibration. None retune 0.005/0.20/0.30.

### 5.A Strengthened evidence-bounded eligibility (v2+decay+slack)

The current v2, with the minimum literature patches that stop it from being a pure ID list.

- **State / memory.**  
  Transferable: Program, PatternScope, k, quarantine rule, decay λ.  
  Target-local: `EntityPosterior[consumer][uid] = {n_pos, n_harm, last_t, quarantined}` with exponential decay. No UID in cross-domain Skill.
- **Held-in update.** PatternScope match → probe → four-line gate. Pass: `n_pos ← λ n_pos + 1`. Severe harm: quarantine. Continuing flip: decay already shrinks n_pos.
- **Fast-only.** Deploy program model iff `n_pos ≥ k` and not quarantined **and** last evidence inside a TTL. Else raw. New roster members: raw.
- **Transfer.** Program + PatternScope + (k, λ, TTL). Entity table wiped on new dataset.
- **Cold start.** All-raw until held-in probes fill the table. A5 helps only if PatternScope increases probe hit-rate (funnel, not magic deployment).
- **Drift.** TTL / λ. Optional: if cohort-level delayed utility drops, freeze new promotions (change-point).
- **Why it could be called self-evolution.** Only if Slow revises Program or PatternScope from census of the ledger (which families of *patterns* keep failing). The ledger itself is not evolution.
- **Why it may fail.** Coverage collapse; D6-style flips among continuing members; still a hard switch (C3); reviewers: ID lookup.
- **Cost.** Cheap. Extra fits only if training subset B is chosen.
- **Harness delta.** Entity memory + Fast routing predicate + decay. Smallest code change.

### 5.B Hierarchical / meta-learning, no hard ID allowlist

- **State / memory.**  
  Transferable: Program library; PatternScope; population prior `ψ` over expected gain given Pattern×Consumer (empirical Bayes from source datasets).  
  Target-local: random intercept `u_i` per entity, posterior `p(u_i | data_i, ψ)`. Features may inform `ψ`, never override a negative posterior.
- **Held-in update.** Each delayed outcome updates `u_i` and, slowly, `ψ`. Unseen i: `u_i = 0`, uncertainty = prior.
- **Fast-only.** Deploy or mix using `LCB(μ + u_i) > 0` (or P(gain>material) > 1-δ). Unseen ⇒ LCB uses prior only—if C1 holds, prior LCB is ≤ 0 ⇒ abstain. This *looks* like A at t=0, but similar entities can share strength via Pattern clustering / random effects at Pattern level, not UID.
- **Transfer.** `ψ` and Pattern clusters move; `u_i` does not. New dataset: all `u_i` reset, `ψ` is a probe prior.
- **Cold start.** Partial pooling on Pattern if any cluster has members; else abstain. This is the only honest zero-shot deployment path that is not ID leakage.
- **Drift.** Random-walk `u_i` (BayesCNS-style forgetting toward prior) or mixture-of-regimes.
- **Why self-evolution.** `ψ` and Pattern clusters are reusable Skills; they change Fast behavior on a *new* dataset before any UID is seen (A5−A3).
- **Why it may fail.** C1: Pattern-level `ψ` may be as weak as D1’s AUC. Then B collapses to A. That collapse is a **falsifier**, not a bug to hide.
- **Cost.** Higher: hierarchical fit per window or per held-in round. Still 0 LLM if Slow is deterministic. Fits must be billed.
- **Harness delta.** Replace boolean eligibility with a posterior + LCB. Do not add a meta-feature AutoML platform; the prior is Pattern×Program×Consumer only.

### 5.C Explicit two-decision + soft routing

- **State / memory.**  
  Transferable: two PatternScopes or two predicates, `T` (train membership) and `S` (serve mix prior); mixture shrinkage hyperparameter n0.  
  Target-local: per-entity weights `w_i` and per-entity train-inclusion `z_i`, both in [0,1].
- **Held-in update.**  
  T: include or down-weight training rows using influence / Support-safe entities only (reduce spillover).  
  S: `w_i = n_pos / (n_pos + n0)` or LCB-based mix of raw vs program *predictions* (not a hard model switch).
- **Fast-only.** Serve `ŷ = w_i ŷ_prog + (1-w_i) ŷ_raw` with frozen `w_i`. Unseen: `w_i=0`. Optionally prepare serving context only if `w_i>w_ctx` (second knob; default off until Stage A says context matters).
- **Transfer.** T/S *predicates* and n0; not `w_i`.
- **Cold start.** All-raw mixture. A5 can transfer a better T predicate (“which Pattern’s training rows helped pooled models before”) as a *probe prior* only.
- **Drift.** Decay `n_pos`; if w_i was high and delayed harm occurs, pull w_i toward 0 (revocation as continuous).
- **Why self-evolution.** Slow may split T from S, revise n0, or revise which Program is the expert in the mixture—those are Skill changes. Updating w_i is not.
- **Why it may fail.** Stage A GAIN_LOST: if routing *is* the only source of gain, shrinking w_i also shrinks utility. Then the honest result is “cannot have both mean gain and tail safety under pooled Consumer,” which is Track B material, not a hyperparameter hunt.
- **Cost.** Two predictions already exist in `scoped_evaluate`. Mixture is free. Training-subset refits are the cost (v2 §5 already budgets this).
- **Harness delta.** Serving mix; optional train-inclusion weights. Larger than A, smaller than a new AutoML.

---

## 6. Recommended Synthesis

Do **not** pick A because it is closest to the current draft, and do **not** pick B because hierarchical Bayes has more papers. Pick the smallest coherent system that (i) respects C1–C4, (ii) is not an ID table, (iii) can still instantiate A5.

### 6.1 Blueprint (working name: Prior-gated mixture, entity residual, two decisions)

```text
Input:  TaskSpec, Consumer, deployment-visible Pattern, frozen K_t
Output: Fast action per entity =
          (z_train inclusion for building program model,
           w_serve mix between raw and program predictions)
        plus abstain/probe bits for held-in
```

**Transferable state K_t (may cross datasets, never contains UIDs):**

- Program library (typed, ≤2 steps)
- PatternScope_probe (who may be probed)
- Optional PatternScope_T (prior on whose *training rows* were useful under this Consumer family)
- Evidence *rules*: decay λ, TTL, LCB level / n0, quarantine on severe harm
- Census summaries: which Program×Pattern×Consumer cells had repeated +/− (Shared Capability bar unchanged)

**Target-local state (wiped on new dataset):**

- Per (consumer, uid): counts with decay, last_t, quarantined
- Frozen at held-out: `w_i`, `z_i`

**Evidence update (held-in only):**

```text
on delayed / Support outcome for entity i:
  g_i = gain vs identity
  n_i ← λ n_i + 1
  s_i ← λ s_i + g_i
  if g_i < -0.30: quarantined_i ← true     # existing severe line, not a new threshold
  # posterior mean
  m_i ← s_i / max(n_i, ε)
```

**Execution-right rule (Fast-only):**

```text
if quarantined_i: w_i = 0
else:             w_i = max(0, n_i / (n_i + n0)) * 1{m_i > 0}
# unseen ⇒ n_i=0 ⇒ w_i=0  (C2 by construction, but soft in n)
# never use UID as a PatternScope predicate
```

**Training-side decision T:**

```text
z_i = 1 if PatternScope_T matches AND (n_i ≥ 1 and not quarantined)
      else 0
# default if PatternScope_T absent: z_i = 1 for all probe-matched (v2 option A)
# choose A vs evidence-qualified by the already-planned 0-LLM replay, freeze once
```

**Serving-side decision S:** mix predictions with frozen `w_i`; do not hard-switch the Consumer unless Stage A says per-channel is required.

**Promotion / revision / revocation:**

- Entity `w_i` up/down every held-in unit: **not** P2, not self-evolution.
- Program ADD / Scope narrowing / n0 or λ change: Slow, atomic, Support then delayed, versioned: **this** is Skill evolution.
- Quarantine of an entity is risk control. Quarantine of a Pattern clause after repeated new-entrant failures is Scope evolution.

**Abstention:** Fast-only never probes. Held-in may probe PatternScope matches even if `n_i=0` (that is how n_i becomes positive). Optional: a small held-in exploration quota for near-miss Patterns, billed, capped—this is Card 4, not a license to explore on held-out.

**Domain transition:**

```text
new dataset:
  wipe all (n_i, s_i, w_i, z_i, quarantine)
  keep K_t
  A5 vs A3 = does PatternScope_probe + Program supply raise hit-rate
             and shorten time-to-first-safe-w_i
             without raising held-out msh
```

### 6.2 Pseudocode

```text
# held-in round r
skills ← K_t ∪ target_local
for unit in held_in:
  candidates ← Fast(skills, PatternScope_probe, memory_supply)  # 0 extra LLM if memory-supplied
  probe ⊆ candidates under semantic budget (≥1 full decision)
  observe Support; update n,s,quarantine
  if authority_gate(Support):
    observe delayed; update n,s,quarantine
    maybe Slow.patch one surface if first-fault says so
freeze (Program, PatternScopes, w, z, rules)
# held-out
for entity in served:
  ŷ ← w_i * program_pred + (1-w_i) * raw_pred
open outcomes once, score, no writeback
```

### 6.3 Necessary assumptions (falsifiable)

1. Identity/raw is an acceptable baseline (already the Static arm).
2. Per-entity delayed gain is a noisy but same-sign estimator of the next-window gain for *continuing* members more often than chance. If D6-scale flips dominate, decay must be aggressive or the whole entity-residual idea dies.
3. Program model and raw model are not identical; mixture can interpolate (D5 interaction ≈ 0 suggests interpolation is meaningful).
4. PatternScope has nonzero probe value on a new dataset, or A5−A3 will be ~0 and accumulation is not instantiated (report funnel, do not claim knowledge failed).
5. Pooled Consumer remains the object of study until Stage A says otherwise. This blueprint does **not** assume per-channel is better.

---

## 7. Self-Evolution Definition

| level | name | what must change | what evidence is allowed | what it is **not** |
| --- | --- | --- | --- | --- |
| L1 | Target-local deployment adaptation | `w_i`, `z_i`, quarantine bits, Fast mix | same-dataset held-in feedback | not a paper claim of self-evolution; routing/memory update |
| L2 | Reusable Skill evolution | Program bytes, PatternScope predicate, evidence *rules* (λ, n0, T vs S split), versioned revision that later re-encounters | Support then independent delayed; re-encounter on a later unit | not ADD then REVOKE only (`RISK_CONTROL_ONLY`, already in AGENTS.md) |
| L3 | Cross-domain Harness evolution | K_t that alters Fast on a **new** dataset *before* that dataset’s UIDs have evidence: better probe supply, better abstain, faster time-to-safe-w | A5−A3 on dataset-as-unit, held-out Fast-only, treatment funnel nonzero | not copying source UIDs; not KDD replay; not a non-empty K0 that never matches |

**Operational tests:**

- If deleting all UIDs and keeping Program+PatternScope+rules leaves Fast unchanged on a new dataset except for slower filling of `w_i`, the UID table was L1 only.
- If A5 and A3 fill `w_i` at the same rate and with the same Programs, accumulation was not instantiated (funnel).
- If Slow never changes Program/Scope/rules, the system is a contextual bandit with a frozen action set, regardless of LLM wrappers.

WikiSkill’s independently useful split: Inference Agent must not read the compiled wiki; Fast must not read entity ledgers or raw episodes. Compiler ≠ executor.

---

## 8. Experiment Implications

No experiments are run here. This is evidence logic for whoever designs the next contract.

### 8.1 Organization

- **Learning unit:** window / origin (as now).
- **Statistical unit:** dataset. Origins of one KDD cache are repeated evaluation points (P4d correction stands).
- **Transfer axis:** domain (pre-declared generating process), never a dataset name in Prompt/Scope/Skill.
- Per domain ≥ 2 datasets when claiming domain transfer. Single long series: call it temporal generalization, not cross-domain.
- LODO variant (K = all other datasets, no order confound) for the transfer *claim*; ordered curriculum for the *curve* claim. Do not mix.

### 8.2 Arms (minimum)

| arm | role |
| --- | --- |
| Static | no adaptation |
| validation-search | equal-budget search, candidate order frozen *before* the run |
| A3 | target-only from public h0 |
| A5 | K_t + same held-in budget |
| A5-frozen / A3-frozen | evolution identifiability; need discordance door (HEC-1 closeout) |
| Optional: FFORMA-style feature mixer | AutoML baseline; already expected weak given P4 |
| Optional: TimeClaw-style fingerprint kNN | memory-without-rights baseline |
| Not as a live arm until spike: AegisTS | GT/RL mismatch; STRUCTURALLY_INCOMPATIBLE remains unless a new adapter is explicitly funded |

Best-Safe-Global stays an **oracle wall**, not a method.

### 8.3 Ablations (each one claim)

1. Hard k vs LCB vs soft n/(n+n0) — “allowlist necessary?”
2. Decay on/off — “evidence TTL necessary?”
3. T = PatternScope vs T = evidence-qualified vs T = all — “train interference?”
4. Hard switch vs prediction mix — “C3 decoupling?”
5. PatternScope prior on vs off (A5 vs A3) — “accumulation instantiated?”
6. Entity residual on vs population-only — “ID vs hierarchy?”

Do not run 2×2×2×2. Serial first-fault: 1 → if hard loses, drop it → 4 → 3 → 5.

### 8.4 Instantiation metrics (mandatory)

K0 → Match → Supply → Selection → Admission → Deployment → Re-encounter → Marginal gain. Zero at any ring = “treatment not instantiated,” not “knowledge failed.” Also: discordance count, coverage, time-to-first-nonzero-w, fraction of served entities with w>0 at freeze.

### 8.5 Report together

Utility (aggregate sMASE vs Static), safety (msh, hf, severe count, new-entrant severe = 0 by construction for Fast-only if w=0), coverage, cost (physical LLM = ledger = backend; fits), adaptation speed (units until w>0 for continuing entities). Never utility without coverage and harm.

### 8.6 Ridge then TSFM

Ridge remains the instrument until a dataset/domain contract exists. TSFM is a Consumer swap on the **same** split, only if Ridge reaches the pre-registered L1 or the paper needs a Consumer-generality column. TSFM preprocessing literature (RevIN etc.) is a *sensitivity* study, not a license to add operators.

### 8.7 Sealed evaluation

At most one unsealing, and only for a configuration that earned SUPPORTED on its own preregistration (HEC-1 closeout §5). Exposed KDD is development. Do not call a v2 replay on exposed windows a capability result.

### 8.8 Claim ↔ experiment

| claim | experiment |
| --- | --- |
| L0 governance | already have HEC-1 |
| L1 identifiable writeback | needs discordance door + semantic budget; not HEC-1 |
| Entity residual beats Pattern router | ablation 1+6 on development, then one live |
| Soft mix cuts tail at acceptable mean cost | ablation 4; Stage A GAIN_* still governs Consumer |
| Cross-domain accumulation | dataset-as-unit A5−A3, funnel nonzero, held-out Fast-only |
| Self-evolution L2 | versioned Scope/Program revision + re-encounter, not UID updates |
| Self-evolution L3 | LODO or multi-dataset order, K_t without UIDs |

---

## 9. Novelty and Reviewer-Risk Map

Novelty of the *joint* problem is **uncertain** in the strong sense: several pieces are old; the conjunction with TS data-readiness + delayed downstream-only feedback + Fast-only freeze + per-entity tail gates is not something any core paper fully occupies. That is a *gap statement*, not a contribution statement.

| reviewer line | why it bites | evidence that would defuse it |
| --- | --- | --- |
| “Just ID lookup / allowlist” | v2 as written *is* this if PatternScope never does work | Ablation: wipe UIDs, keep Pattern+Program; A5 still changes probe supply. Report pooling factor. Soft w_i not `{0,1}`. |
| “Just conservative contextual bandit” | SPIBB + Wu 2016 are the safety core | Bandits do not compile an executable prep program, do not split T vs S, do not have Consumer-conditioned quality semantics, do not freeze to Fast-only across a domain. Show Program revision (L2), not only arm counts. |
| “Just AutoML / FFORMS” | pipeline selection is old | P4: features lost to fixed choice. No clean val. Tail gates. Held-out has zero feedback. FFORMA averaging is a *baseline*, not the system. |
| “Just Agent memory (TimeClaw / ExpeL / Voyager)” | skill libraries are the 2023–2026 fashion | They use GT or env success; Fast in those systems reads traces or skills without a tail budget; A5/A3 is a kNN switch. Show negative + conflict + abstain memory, execution-right ≠ recording, Fast cannot read raw episodes. |
| “Just WikiSkill without GT” | three-layer compile is independently convergent | WikiSkill *requires* GT val accuracy; Inference Agent wiki access hurts. Our compiler is deterministic, reward is delayed utility with harm, entity residual is target-local. Do not claim the wiki. |
| “Just AegisTS” | TS cleaning + downstream reward | AegisTS: injected GT in cleaning reward, global pipeline, RL then freeze, no entity harm budget, no Fast-only freeze vs online. Keep STRUCTURALLY_INCOMPATIBLE unless an honest adapter exists. |

**Most dangerous neighbors, in order:** SPIBB (mechanism), FFORMA (routing), WikiSkill (evolution narrative), TimeClaw (TS harness narrative), Learn2Clean (downstream prep). Cite them in related work as *ancestors*, and put the matrix in the paper.

**What is likely a combination, not a discovery:** conservative baseline + reject option + hierarchical shrinkage + forecast mixing + train/test cleaning split.

**What might still be a real problem (uncertain):** compiling *data-readiness* Skills under *tail-aware delayed downstream-only* feedback, with *Consumer-conditioned* quality, *scoped* programs, and *cross-domain* transfer that is not UID leakage. That sentence has many adjectives; each one must be instantiated in the funnel or it should be deleted.

---

## 10. Falsification and Stop Rules

### 10.1 Entity-evidence scheme is not the right lever

Stop treating UID residual as the method if **any** of these hold on a preregistered development replay *or* the next live course:

1. Continuing members’ sign-flip rate is statistically indistinguishable from new entrants (D6 was n-small; a larger course can kill C2’s “members are stable” half).
2. Soft mix / LCB / partial pooling dominate hard k on harm-constrained utility **and** coverage; the allowlist is then a dominated policy.
3. After decay, coverage stays near 0 on every Fast-only face (churn > TTL).
4. Pattern-level random effects absorb the gain; UIDs add nothing (then B’s prior is enough—or nothing is).
5. Per-series oracle gap cannot be approached without using evaluation-face outcomes (i.e., the only way to pick victims is cheating).

Then: keep conservative baseline + PatternScope, drop the ID table, or go Track B.

### 10.2 Route decoupling is not the right lever

1. Stage A: GAIN_LOST under per-channel or under mix (mean gain was the route; shrinking the route kills the method).
2. D5-style 2×2 on a new Consumer (TSFM) shows CONTEXT_DOMINANT or INTERACTION_DOMINANT.
3. T vs S ablations never change held-out ranking once Program is fixed.

Then: do not split ScopeSpec further. Either change Consumer family (already gated by Stage A 2×2) or accept that data-prep cannot be isolated from model choice in that Consumer—Track B mechanism chapter, not more knobs.

### 10.3 Cross-domain self-evolution has no usable signal

1. A5 treatment funnel is zero on Match or Supply across ≥2 datasets (K0 non-empty but inert).
2. A5−A3 ≤ 0 on dataset-macro average, with discordance door passed (so the comparison was identifiable).
3. The only transferable object that helps is a dataset-name prior (illegal) or a UID that reappears (not a domain).
4. PatternScope match rate on new datasets is ~0 after a honest observable-only predicate (Observation gap, not a memory gap).

Then: A3 vs Static may still be a component result. It does **not** complete A5. Do not relabel Target-local L1 as L3.

### 10.4 When to take Track B instead of more configs

Track B is already licensed by the HEC-1 closeout: governance worked; evolution was treatment-sparse; binding constraint is Support↛future safety plus pooled routing. Switch to Track B as the *paper spine* (not a consolation) if:

- Stage A lands GAIN_LOST or NO_SHAPE, **or**
- the next identifiable live course still has no L2 revision–reencounter, **or**
- dataset-as-unit A5−A3 is null after a nonzero funnel, **or**
- the only working policy is UID allowlist with coverage too low to matter.

Do not spend the sealed unsealing on a config that only won by covering fewer series. Do not retune risk lines to manufacture a positive. Do not replay exposed KDD as a new method result.

---

## 11. Direct answers to RQ1–RQ12

**RQ1.** The mature names are: conservative bandits, SPIBB / safe policy improvement, selective classification / reject option, and DTR *eligibility*. Allowlist of IDs is the degenerate case. Personalization is broader and usually *shrinks*, not *gates*.

**RQ2.** Yes: hierarchical Bayes / mixed effects / empirical Bayes / CATE with honest inference. Use a population prior + entity residual + uncertainty. If posterior variance stays huge, say so—that is evidence that ID lookup is the statistical truth of this data, not a design flex.

**RQ3.** Zero-shot: Program library, Pattern probe prior, safety protocol, mixture hyperparameters, abstain-by-default. Must probe: entity residual, train-inclusion set, serve weights, any execution right beyond probe. Cold start in recsys/DTR: side information + partial pooling + baseline policy. Here side information failed (C1), so cold start = baseline.

**RQ4.** Promotion/probation/revocation exist in SPI/CPI (mixture until proven), WikiSkill (val gate, skill rollback, wiki persist), TimeClaw (γ on held-out success), conservative bandits (slack). Evidence decay is standard in stream learning. Delayed reward: Joulani—budget more time, do not require two noiseless confirmations without a coverage plan.

**RQ5.** Literature uses: two-stage cleaning (CleanML train vs test), MoE / FFORMA soft weights, causal policy learning for who to treat, sample selection for who to train on. Dual Scope is reasonable. Hard double allowlist is not the only, or the best, encoding.

**RQ6.** Constrained PI, CVaR, DRO, SPIBB, chance constraints. Put tail into the *ranking objective* of held-in selection; do not add a new numeric gate.

**RQ7.** Yes, no-evidence-no-deploy can freeze coverage. Safe exploration = baseline mixture + slack-funded probes in held-in + Soft-SPIBB. Held-out stays conservative.

**RQ8.** TTL, exponential decay, sliding window, change-point reset, versioned Skills. Match forgetting to drift (Webb). HEC-1 10/34 already says λ=1 is optimistic.

**RQ9.** See §7. Harness self-evolution = L2/L3. UID ledger = L1.

**RQ10.** Dataset is the statistical unit; domain is the transfer axis; window is the learning step. LODO + ordered curriculum, macro-average dataset then domain. KDD origins are not datasets.

**RQ11.** vs WikiSkill: we lack GT; we have tail risk; Fast must not read wiki/episodes; we compile Programs+Scopes not SKILL.md. vs TimeClaw: no GT-verified tool admit; per-series harm; frozen Fast-only vs kNN memory. vs AegisTS: no injected GT; scoped not global; online held-in vs train-then-freeze RL. vs TSPred/FFORMS: closed-loop delayed feedback, harm, freeze, Consumer conditioning; P4 already beat static features with a fixed choice. Covered novelty: three-layer compile, conservative baseline, forecast mixing. Uncertain novelty: the conjunction in §9.

**RQ12.** Alternatives: (B) hierarchical residual + LCB; (C) two-decision soft mix. Recommended synthesis: §6, which is C’s serving mix + A’s conservative unseen default + B’s prior *if* Pattern pooling empirically moves `ψ`. If `ψ` does not move, do not fake it.

---

## Appendix A. Search log (non-exhaustive)

Public queries included: task-aware data cleaning; FFORMS/FFORMA/AutoAI-TS; conservative/safe contextual bandits; SPIBB; delayed bandits; CATE / DTR / partial pooling; concept drift / prequential / forgetting; conformal risk control / selective classification; Voyager / Reflexion / ExpeL / WikiSkill; TimeClaw / AegisTS; MoE TSFM; TSFM normalization LODO; DomainBed; CVaR MDP; DRO.

Primary corpora: arXiv HTML/PDF, PMLR, ACM DL, author PDFs, NeurIPS/ICLR proceedings. Excluded: MDPI.  

WikiSkill, TimeClaw, AegisTS, TSFM-normalization, Moirai-MoE, several 2024–2026 bandit/CRC papers: **preprint / not journal-of-record** at time of reading. AegisTS full PDF fetch failed in this pass; mechanism details also rest on the project’s 2026-08-17 source-level reading of local `a-evolve/AegisTS` (cleaning metrics use injected GT; proxy vs final; λ_issue > λ_downstream). Do not treat that as a new independent full-paper audit.

BoostClean remains an arXiv 2017 preprint without a located major-venue version in this pass.

---

## Appendix B. What this document is not

- Not a freeze of Method v2.  
- Not a finding that v2 will pass its own §5 replay.  
- Not permission to retune harm lines.  
- Not a claim that per-channel is better.  
- Not self-evolution evidence.  
- Not related-work text for a paper until Fable rewrites it in the paper’s voice and drops development numbers that do not belong there.
