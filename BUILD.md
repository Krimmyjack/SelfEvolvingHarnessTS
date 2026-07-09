# SelfEvolvingHarnessTS 鈥?鏋勫缓涓庡疄楠屾枃妗ｏ紙BUILD锛?

> **鈿?2026-07-05 鐘舵€佹敞璁?*锛氭湰鏂囧疄鐜扮姸鎬佸仠鍦?2026-06-21锛涘叾鍚?Stage 0锛堢閬撲慨澶?鏃ュ織鍩哄缓锛変笌 Stage 1锛堣韩浠藉垽鍐筹細E-1.1 鍥涜疆/S0.7 绠楀瓙璇氬疄鎬?F0 鍓傞噺鎵弿/E-3.2 鍏噦/confirmatory seeds 20鈥?9锛夊凡鍏ㄩ儴鎵ц瀹屾瘯锛?*279 娴嬭瘯杩?*銆傜粓灞€鍒ゅ喅瑙?`results/STAGE1_VERDICT.md`锛圥attern 鏉′欢鍖栬矾鐢?level-1 纭珛锛汣6=妯″瀷鏃犲叧澧炵泭璇佷吉锛夈€傚綋鍓嶅叆鍙?= `../idea/Component_Optimization_and_Integration_Plan.md`锛圫tage 2 缁勪欢浼樺寲涓庣郴缁熷悎娴侊細缂洪櫡娓呭崟 D1鈥揇10銆丳atternSpec/RouterPolicy 鍚堟祦銆佸紶閲忋€佹洿鏂伴棴鐜級銆?
>
> **Stage 2 critical-review sync锛?026-07-07锛?*锛氬綋鍓嶈韩浠戒笌鎵ц椤哄簭浠?`../idea/Current_Vision_Scheme_and_Experiment_Setting_2026-07-07.md`銆乣../idea/Critical_Project_Review_and_Redesign_2026-07-07.md` 鍜?`../idea/Reference_Project_Audit_and_Transfer_2026-07-07.md` 涓烘渶鏂扮患鍚堣瀹氥€傛柊澧炵绾胯瘉鎹細`results/Stage2/BatchScan/report.json` 涓哄彂鐜伴泦鎵弿锛宍P0_kmeans` 鐣ヨ儨 `P1b_kmeans`锛坥racle agreement 0.3333 vs 0.3289锛泈ithin-batch response variance 4.0923 vs 4.4216锛夛紝鍥犳 **P1b 涓嶈浆姝?*锛沗results/Stage2/C1Lite/report.json` 涓?`P1b-memory` 鑳?frozen/random-memory 浣嗚緭 `P1b-static`锛坮egret 0.3424 vs 0.2976锛変笖 first-unseen harm +0.1072 瓒呭畨鍏ㄧ嚎锛屽洜姝?**Memory 涓嶄綔涓虹嫭绔嬫満鍒惰浆姝?*銆俙results/Stage2/ReadinessAdversaries/` 浠庡喕缁?`S2_replication/records_s2.jsonl` 鏋勯€犻潪 API銆侀潪鑷姇鍠傜殑 oracle-actionable 鏍囩锛坮aw=`v_none`锛?72 rows锛沷racle actionable rate 0.929锛夈€傜粨鏋滐細`dp_abstain` 鐩稿 raw 闄?mean regret锛?.2762 vs 0.4005锛変笖 gain_vs_raw=+0.1243锛屼絾 recall 浠?0.604銆乭arm_rate=0.391锛沗P1b-static` regret=0.2965銆乬ain_vs_raw=+0.1040銆乭arm_rate=0.344銆傜粨璁烘槸锛氬凡鏈?policy 鏈夋暟鎹瓫閫変环鍊硷紝浣嗗綋鍓嶈嚜閫傚簲杩樹笉瀹夊叏锛涗笅涓€姝ュ繀椤讳紭鍏堝仛 support/uncertainty gate 鍜?harm calibration锛岃€屼笉鏄棤杈圭晫 24h LLM 闀胯窇銆侭1b proposer 浠嶆槸椤圭洰韬唤闂細鍙湁 LLM proposer 鍦ㄥ悓涓€寮€鏀?`ProgramSpec` 绌洪棿銆佸悓棰勭畻銆両TT no-op 涓?worst-group safety 涓嬭儨杩?deterministic search锛屾墠鍙啓鎴?LLM-driven harness evolution銆傚綋鍓?immediate order锛欱1b 韬唤闂ㄤ紭鍏堬紱Pattern-Batch 浠呭仛 fresh-namespace confirmatory锛沠ull M0-M3 Memory promotion 鏆傚仠鍒?support/escalation gate 鏄庣‘銆?
>
> **Stage 2 safety / Pattern-Batch / EvidencePacket sync锛?026-07-08锛?*锛氬凡鎸?2026-07-08 璁″垝钀藉湴涓変欢浜嬶紝鍧囦笉璋冪敤 LLM/API銆傗憼 `evaluators/safety_gate_lite.py` + `run_safety_gate_lite.py` 鐢熸垚 `results/Stage2/SafetyGateLite/`锛歚dp_abstain` harm=0.391/gain=+0.1243锛沗abstain_to_raw` harm=0.220/gain=+0.0974锛況outer-support q50 harm=0.095/gain=+0.0841/serve_frac=0.329锛宷75 harm=0.141/gain=+0.0963/serve_frac=0.470锛宷95 harm=0.201/gain=+0.0941/serve_frac=0.583銆傜粨璁猴細鏀寔搴﹂棬鎺ф湁鏁堥檷 harm锛屼絾浠嶆槸绂荤嚎鎶樹腑锛屼笉鏋勬垚閮ㄧ讲瀹夊叏澹版槑銆傗憽 `evaluators/pattern_batch_scan.py` + `run_pattern_batch_scan.py` 鐢熸垚 `results/Stage2/PatternBatchConfirmatory/`锛歭egacy_cell oracle_agreement=0.222/response_var=6.2844锛孭0_kmeans=0.333/3.9828锛孭1b_kmeans=0.265/6.2072銆傜粨璁猴細P0 杩炵画 Pattern-Batch 鏄庢樉浼樹簬 legacy锛孭1b 鍦ㄦ湰娆′弗鏍?10d P0 瀵圭収涓嬩笉杞銆傗憿 `policy/evidence_packet.py` 寤虹珛 `skill_memory_evidence_packet_v1`锛屾妸 Pattern 鎽樿銆丼kill cards銆丮emory 鎽樿銆丄ctionMenu meta銆乻afety constraints 鍥哄畾涓?LLM composer 鍓嶇殑鍙祴璇曡緭鍏ラ潰锛屽苟浠ュ崟娴嬬姝?`L_test`/oracle/arms/`X_t`/raw series 娉勬紡銆傜獎鍥炲綊锛歚test_safety_gate_lite` + `test_readiness_adversaries` + `test_pattern_batch_scan` + `test_evidence_packet` 鍏?14 passed銆?
> **Stage 2 slow-path validation / ablation runner sync（2026-07-08）**：在 code review 后补齐 slow-path 最小部署闭环与 fast-path 消融入口，仍不调用真实 LLM/API。`slow_path/evidence_miner.py` 现在从 `EvidenceRecord.conditioning_key.task`/cell 推断 task，不再硬编码 forecast，并输出可被现有 `policy.edits.MemoryWrite` 消费的 utility-bound payload；`slow_path/proposal_schema.py` 会把 `MemoryWrite` proposal 校验到 `MemoryWrite(EditOp)` schema，把 `ProposeRiskRule` proposal 校验到 `RiskRule.validate()`；新增 `slow_path/promotion.py` 的 `PromotionGate`/`ProposalValidationOutcome`，只做 validate + compile to `EditOp`，不自动 apply 到 `PolicyBundle`。同时新增 `fast_path/ablation.py`：`raw` arm 作为正常 `v_none` baseline 执行而非 fallback failure；支持显式 skill-surface override，使 memory-only arm 的 `EvidencePacket.skills=[]` 且 memory 保留。新增测试 `test_slow_path_promotion.py` 与 `test_fast_path_ablation.py`；当前 focused tests 为 slow-path evidence/proposal 5 passed、promotion 3 passed、ablation 2 passed。下一步从 skeleton runner 进入固定数据集上的 raw/deterministic/skill-only/memory-only/skill+memory/composer(+gate) 报表。
> **Stage 2 FastPathAblation no-API run（2026-07-08）**：新增 `run_fast_path_ablation.py` 与 `fast_path/ablation.py` 的 summary/report 输出。命令 `D:\Anaconda_envs\envs\project\python.exe -m SelfEvolvingHarnessTS.run_fast_path_ablation --n-records 4` 写入 `results/Stage2/FastPathAblation/report.json` 与 `records.jsonl`：8 arms × 4 synthetic forecast records = 32 results，`api_calls=0`。arm 顺序为 raw / deterministic_router / skill_only_deterministic / memory_only_selector / skill_memory_deterministic / composer_skill / composer_skill_memory / composer_skill_memory_safety。当前 run 只证明 no-API ablation pipeline、EvidenceRecord 写入和 report contract 可复现；默认 `role_b_proxy` 还不是 utility/harm reporter，因此不能用该小切片宣称 Memory/Composer 收益。下一步应接入固定 reporter 或小型 oracle slice，再把 EvidenceStore 输出送入 slow-path mining/promotion gate。
> **Stage 2 FastPathAblation utility/harm + slow-path mining sync（2026-07-08）**：`run_fast_path_ablation.py` 现在接入固定 `synthetic_oracle_proxy_v1` reporter，向 EvidenceRecord downstream 写入 `raw_loss_proxy`、`selected_loss_proxy`、`utility_delta_vs_raw`、`harm_delta_vs_raw`，并把同一 EvidenceStore 送入 `DeploymentEvidenceMiner -> PromotionGate`，输出 `slow_path_proposals.jsonl`。重新运行 4-record synthetic slice：32 results，`api_calls=0`，slow-path proposals=2，accepted=2。proposal 明细：① `forecast|snrHigh|full` 上 `v_median` mean utility=-0.000312 / harm=0.000312，生成 scoped `ProposeRiskRule` ban to `v_none`；② `forecast|snrLow|miss` 上 `v_median` mean utility=+0.063176 / harm=0，生成 `MemoryWrite`。review 发现 `v_none` 实际是 `impute_linear` baseline 而非 strict raw，因此 `slow_path/evidence_miner.py` 已修正为默认不把 `raw_action` 的正 utility 提升为 MemoryWrite，避免把 baseline 误当成 reusable skill evidence。该 run 仍是 synthetic proxy，不是论文级性能结论；它验证的是 ablation -> utility/harm evidence -> slow-path proposal/promotion gate 的闭环。
> **Stage 2 escalation fast-path sync锛?026-07-08锛?*锛歚policy/escalation.py` 鏂板鍗囩骇寮?fast-path decision layer锛屽苟浠?`policy/__init__.py` 瀵煎嚭銆傚畠鎶?`SkillRetriever -> MemoryEvidence rows -> EvidencePacket -> optional SkillMemoryComposer -> SafetyGate` 涓叉垚涓€涓彲娴嬭瘯鐨勯儴缃插墠鍐崇瓥闈細楂樻敮鎸佷笖 top skill 缃俊瓒冲鏃惰蛋 deterministic candidate锛涗綆鏀寔銆丱OD銆乪vidence conflict 鎴栫粍鍚堢粨鏋勮Е鍙?composer锛沗abstain_to_raw`銆佸急鏀寔銆乭arm policy銆乤ction menu 瓒婄晫銆乻kill/action 涓嶅尮閰嶉兘浼氳 SafetyGate 鏀瑰啓涓?raw fallback锛堥粯璁?`v_none`锛夈€傝灞備粛涓嶆墽琛屽姩浣溿€佷笉璋冪敤鐪熷疄 LLM/API銆佷笉鏇夸唬 `ActionCompiler/Overlay`锛涘畠鐨勮緭鍑烘槸 typed candidate + route + safety decision + packet锛屼緵涓嬩竴姝ユ帴缂栬瘧涓?ablation銆傛柊澧?`tests/test_escalation_fast_path.py`锛岀浉鍏?Skill/Memory/Packet/Composer/SafetyGate/Pattern-Batch 鍥炲綊鍏?28 passed锛宍policy/escalation.py` 涓?`policy/__init__.py` 缂栬瘧閫氳繃銆?> **Stage 2 escalation compile bridge sync锛?026-07-08锛?*锛歚policy/escalation.py` 缁х画琛ヤ笂 `conditioning_key_from_record` 涓?`compile_fast_path_decision`銆傜幇鍦?accepted `EscalationDecision` 鍙互閫氳繃鐜版湁 `ActionCompiler` 缂栬瘧涓?`fast_path.compose.Program`锛堜緥濡?note=`tmpl:{action_id}`锛夛紝鑰?`raw_fallback` 榛樿涓嶇紪璇?rejected action锛岄伩鍏?SafetyGate 涔嬪悗鍙堟妸鍗遍櫓鍊欓€夐€佸叆鎵ц閾俱€傝妗ユ帴浠嶄笉鎵ц artifact銆佷笉璋冪敤鐪熷疄 LLM/API锛涘畠鍙妸 `TypedCandidate -> ActionCompiler/Program` 鐨勫绾﹁ˉ榻愶紝涓嬩竴姝ユ墠鏄?overlay/execute runner 涓?ablation銆傜浉鍏冲洖褰掓洿鏂颁负 30 passed锛岀紪璇戞鏌ラ€氳繃銆?> **Stage 2 escalation execute bridge sync锛?026-07-08锛?*锛歚policy/escalation.py` 缁х画琛ヤ笂 `ExecutedFastPathDecision` 涓?`execute_fast_path_decision`銆傜幇鍦?accepted decision 鍙寜 `TypedCandidate -> ActionCompiler/Program -> execute()` 璺戝嚭 artifact锛汼afetyGate/raw fallback銆佺紪璇戞湭閫氳繃銆佹墽琛屽紓甯搞€乤rtifact 闈炴湁闄愭垨褰㈢姸涓嶇閮戒細 fail-closed 杩斿洖 raw銆傝妗ユ帴浠嶄笉鍐?EvidenceStore銆佷笉璺?Downstream Validator銆佷笉璋冪敤鐪熷疄 LLM/API锛涘畠鍙槸鎶?pipeline 鎺ㄨ繘鍒?Execute/fallback raw 鐨勫彲娴嬭瘯杈圭晫銆傜浉鍏冲洖褰掓洿鏂颁负 32 passed锛宲y_compile 閫氳繃銆?> **Stage 2 downstream validation / EvidenceStore sync锛?026-07-08锛?*锛歚policy/escalation.py` 缁х画琛ヤ笂 `DownstreamValidationResult`銆乣validate_fast_path_output` 涓?`emit_fast_path_evidence`銆傜幇鍦?escalation fast path 宸茶兘鍦?execute/fallback raw 鍚庤繍琛岄粯璁?`role_b_proxy` validator 鎴栨敞鍏ヨ嚜瀹氫箟 downstream validator锛屽苟鎶?`conditioning_key`銆乧ompiled/raw-fallback program銆乪xecution trace銆乿erification_result銆乺outing/candidate/safety/packet provenance 鍐欐垚鐜版湁 `memory.EvidenceRecord`锛屽彲閫夊啓鍏?`EvidenceStore`銆俽aw fallback 涓?SafetyGate reject 浼氳褰曚负 not passed锛屼笖涓嶄細瀛?rejected program銆傝灞備粛涓嶅仛鐪熷疄 LLM/API 璋冪敤锛屼篃涓嶅仛 slow-path promotion锛涗笅涓€姝ユ槸 ablation runner銆傜浉鍏冲洖褰掓洿鏂颁负 35 passed锛宲y_compile 閫氳繃銆?> **Stage 2 Step 1 宸茶惤鍦帮紙2026-07-05锛孋omponent Plan v1.1b锛?*锛?
> 鈶?**D6 淇**锛歚fast_path/compose.is_operator_eligible(op, task, harness, banned)` 缁熶竴璧勬牸鍒ゅ畾
> 锛堟ā鏉?heuristic/LLM/recovery 鍥涜矾 + `run_gates` 鏂板 **Contract gate** 鐗╃悊澶嶆煡锛夆€斺€攁nomaly
> recovery 涓嶅啀娉勫叆 winsorize锛坄tests/test_d6_contract.py`锛夈€?
> 鈶?**policy/ 濂戠害灞傛柊鍖?*锛?.0-鈶?鈶/鈶級锛歚pattern_spec.py`锛圥0=鐜?10 缁村喕缁撳揩鐓э紝鍚?feature
> order/missing 璇箟/period estimator ID/渚濊禆鎸囩汗/code SHA 澶嶇幇瀛楁锛夛紜 `action_spec.py`
> 锛圓ctionSpec/ActionMenu v1[SHA]/ActionCompiler鈥斺€攙_*/f0_* 鍔ㄤ綔 ID 鈫?鍙墽琛?Program 鐨勬墽琛屽绾︼紝
> 鍗曚竴鐪熸簮=_VARIANT_SPECS+F0_DOSAGE_GRID锛夛紜 `router_policy.py`锛圧outerPolicy 缁熶竴鎺ュ彛锛?
> FrozenArmRouterPolicy 鍖呰 frozen_arms.joblib锛孲HA 瀹堝崼锛夛紜 `deploy.py`锛坮outed_process opt-in
> 鍚堟祦鍏ュ彛锛岄粯璁?fast_path 琛屼负闆舵敼鍔級銆傚洓灞傜瓑浠锋€ф祴璇曞叏杩囷細action ID / 缂栬瘧 Program / 鎵ц
> artifact bit 绾?/ provenance锛坄tests/test_policy_contract.py`锛夛紜 dev 鍏?uid picks 閫愪竴涓€鑷?
> 锛坉p_abstain/global/d_lookup锛宍tests/test_router_policy.py`锛夈€?
> 鈶?**寮犻噺鍗忚钀界洏**锛歚stage2_protocol.py` 鈫?`results/Stage2/tensor_protocol.json`
> 锛坣amespace 娉ㄥ唽琛?action menu SHA/3 妯″瀷 pilot+LSTM 闂ㄦ帶/澶辫触绛栫暐/dominance-interaction
> 棰勬敞鍐屽垎鏀鍒欙紱缁撴瀯搴?v2 鍥涙柊鏃?draft 寰呭喕缁擄紱**holdout 鏈В閿併€佹湭璇讳换浣曟柊鏁版嵁**锛?
> 瀹堝崼=`tests/test_stage2_protocol.py`锛夈€?
> 鍏ㄥ簱 **309 娴嬭瘯杩?*锛堝惈 2 live LLM suite锛沗test_slow_path` 鐨?S0.4 闄嶇骇娴嬭瘯鍦烘櫙鍥?D6 鐗╃悊鎷︽埅
> 鑰屾敼鍐欎负濂戠害鍚堟硶鐨?forecast 閲嶅墏閲?median(w25) 鍦烘櫙锛屾満鍒跺畧鍗笉鍙橈級銆?
>
> **Step 1.1 鏀跺彛 + A0 绗竴姝ワ紙2026-07-05锛屽悓鏃ワ紱Component Plan v1.1c锛岃瘎瀹＄浜屽崄涓夎疆鎶介獙锛?*锛?
> 鈶?**task scope 寮哄埗**锛欰ctionCompiler 瀵?task_constraints 杩濆弽 **fail-loud**锛堥槻"Router 閫?
> v_median銆乤nomaly 瀹為檯鎵ц鈮坴_none"鐨勮涔夋紓绉伙級锛汧rozenArmRouterPolicy `task_scope=("forecast",)`
> 闈?forecast 鎷掔粷銆傗憽 **menu 璇箟 SHA**锛歛ction_menu_v1 鏋勫缓鏃跺畬鏁?resolve params锛坉efaults 鈯?
> override锛? meta.operator_defaults_sha 鈫?defaults 鏀瑰姩鍗?SHA 鍙橈紱override 浼樺厛绾ф渶楂?鈫?鏃?menu
> 璺ㄧ増鏈噸鏀剧ǔ瀹氥€傗憿 **寮犻噺鍗忚 v1**锛坄results/Stage2/tensor_protocol_v1.json`锛宑onfig_sha=
> 4cf04acb鈥︼紱v0 鍘熸牱淇濈暀锛夛細holdout 瑙ｉ攣鏀圭嫭绔?append-only `holdout_access_log.jsonl`銆?5 鍔ㄤ綔
> 涓夊垎褰掑睘锛坈ore 10/ablation 3/savgol 鍓傞噺璇婃柇 2锛夈€丏Linear=per-series 鐙珛璁粌鏄庣‘銆乨ominance
> 鍒ゆ嵁鍔?worst-group 瀹夊叏渚э紙F0 season 鏁欒锛夈€傗懀 **runtime pinning**锛氬喕缁撹噦鍔犺浇姣斿 blob
> sklearn/numpy 鐗堟湰锛屼笉鍖归厤 fail-loud锛?*OOD 鍝ㄥ叺**锛歊outingDecision.provenance 鍔?out_of_support
> 鏍囪锛坉ev 鏀寔闆?z-score kNN 璺濈>璁粌鑷窛绂?p95锛屽彧璁板綍涓嶆嫤鎴紱C 閫氶亾 abstain=2.2-鈶ワ級銆?
> 鈶?**A0 绗竴姝?*锛歚conditioning/period.py` 鍏变韩鍛ㄦ湡妯″潡鈥斺€旀劅鐭ョ legacy_fft_v0锛圥0 缂洪櫡鍐荤粨锛?
> 涓庣畻瀛愮 robust_v1锛圫0.7 淇鐗堬級**閫愬瓧杩佸叆**鍗曚竴瀹氫箟鐐癸紙bit 绾т笉鍙橈紝鍐呰仈鏃у疄鐜板鐓ф祴璇曞畧锛夛紝
> key.py/s1_denoise/s1_decompose 涓夊鎺ョ嚎涓哄叡浜湰浣擄紱鏂板 top_k_periods锛圥1 鐗瑰緛涓撶敤锛屼笉杩?
> 绠楀瓙璺緞锛夛紱D1 鍒嗗弶璇佹嵁鍥哄寲涓烘祴璇曪紙trend+season锛歭egacy 琚姭鎸?vs robust 鎵惧洖 24锛夈€?
> 鍏ㄥ簱 **324 娴嬭瘯杩?*锛堟柊 suite锛歵est_period_shared 8锛泃est_policy_contract/router_policy/
> stage2_protocol 鎵╁锛夈€俽outed_process 瀹氫綅=P0 绛栫暐绛変环鎵ц**閫傞厤鍣?*锛堝彔鍔犲紡鍚堟祦=2.0-鈶?2.5锛夈€?
>
> **Step 1.1 琛ヤ竵锛堝悓鏃ワ紝璇勫绗簩鍗佸洓杞獙鏀讹細Step 1.1/A0 鎴愮珛锛孭1 寮€宸ャ€佸紶閲?pilot 鏆傜紦锛?*锛?
> 鈶?**寮犻噺鍗忚 v2**锛坄results/Stage2/tensor_protocol_v2.json`锛宑onfig_sha=f007d6cf鈥︼紱v1/v0 鍘熸牱
> 淇濈暀锛夛細**DLinear 璁粌鍗曚綅鍕樿**鈥斺€攙1 澹扮О per-series 鐙珛涓斾笌 STAGE1 reporter 涓€鑷达紝瀹為檯
> report_target.py 涓?*鍩熷唴姹囨€绘瀯绐椻啋璁粌涓€涓叡浜?DLinear鈫掗€愬簭鍒楄瘎浼?*锛泇2 涓诲彛寰?within-domain
> pooled锛? 椤圭洰鍘熷璁惧畾"鐢ㄥ鐞嗗ソ鐨勬暟鎹缁冧笅娓告ā鍨?锛夛紝per-series 鐙珛闄嶄负璇婃柇鑷?
> dlinear_per_series锛岃ˉ estimand/utility_vs_report/coupling_note銆?*鍐荤粨娴佺▼鐗堟湰鍖?*鈥斺€斿崗璁枃浠?
> 姘镐笉鏀瑰姩锛岀粨鏋勫簱鍐荤粨=鍙﹀瓨 v3锛涘紶閲忚鏂?寮犻噺鐢熸垚**鍙帴鍙楀叏鍐荤粨鐗堟湰**锛坴1/v2 姘镐箙涓嶅悎鏍硷紱淇
> v1 "涓嶅彲鍙樷埀鍐荤粨鏃舵敼鏈枃浠?鑷浉鐭涚浘锛夈€傗憽 **PatternSpec 浠ｇ爜鎸囩汗鈫掓彁鍙栧櫒闂寘**锛坘ey.py+period.py锛?
> A0 鍚庡懆鏈熷疄鐜板湪 period.py锛屽彧 hash key.py 浼氭紡锛夛紱P0 config_sha=e4f10d11 閽夋杩涙祴璇曘€傗憿 **runtime
> 鏍搁獙/鏀捐钀?provenance**锛坮ecorded/runtime/mismatch/allowed_mismatch 鈫?RoutingDecision.
> provenance.runtime_check锛夛紱support 鍝ㄥ叺璁?source+n_train锛堟寕璐︼細S1/閮ㄧ讲鍓嶉』缁戣繘 PolicyArtifact锛夈€?
> 鈶?**Component Plan v1.1d锛歅/D/C/蠁 鐗瑰緛杈圭晫鍐荤粨**鈥斺€斚?P,D,a,m) 鍔ㄤ綔浜や簰鐗瑰緛锛坵indow/period銆?
> 鍊欓€夌獥鍙钩婊戣兘閲忥級涓嶅叆 PatternSpec锛孯outer 鍐崇瓥鏃剁敱 P 鍘熸枡鐜扮畻锛?.1 P1 鑷傜浉搴旀敼鍐欍€?
>
> **P1a 绗竴寮犺〃锛?026-07-05锛岃瘎瀹＄浜屽崄浜旇疆瀹氭鍚庡綋鏃ユ墽琛岋紱results/Stage2/P1a/锛?*锛?
> 鎻愬彇鍣?`conditioning/p1a.py`锛圥atternSpec=P1a锛歮ask-aware 鏃堕棿杞?robust period+top-k/缂哄け
> 鎷撴墤 D/鏈€灏?C 閫氶亾锛沗period.py` 鍔?robust_period_diag 涓?v1 濮旀墭涔嬧€斺€攂it 绛変环瀹堝崼杩囷級銆?
> 閲嶆斁=`run_p1a_replay.py`锛氬喕缁撴姌+鍐荤粨 L_train/L_test 鍙崲鐗瑰緛閲嶆嫙鍚?dp 鑷傦紱**瀹堝崼鈶犺鏂欓噸寤?
> 鍚?P0 鐗瑰緛 480/480 bit 绾у鐜般€佸畧鍗憽p0 閲嶆斁 picks鈮″喕缁?records 鍏ㄨ繃**銆傜粨鏋滐細棰勬敞鍐屼笁鑷?
> 锛坒ixp/pd/pdc锛壩擱egret CI 鍏ㄨ法 0锛沠ollow-up 鏈哄埗闅旂鑷?**fixpc锛圥0-D+P1a-P+C锛塪p_gbdt
> 螖Regret=鈭?.0673 [鈭?.1315, 鈭?.0096]锛圕I 涓嶈法 0锛宖rac_pos=0.9%锛?*锛屽鐩婇泦涓?S_trend
> 锛?.534鈫?.371锛変笌 S_both锛寃orst-group LCB 鈭?.427鈫掆垝0.110锛?*abstain 鍦ㄤ慨濂界壒寰佷笂鏃犲閲?*
> 锛?.1987鈮?.1960鈥斺€攁bstain 涓€鐩村湪琛ョ壒寰佺己闄凤級銆侾1a-D 鐨?gap 鎷撴墤鍦ㄦ湰璇枡锛?% 鍧囧寑闅忔満缂哄け锛?
> 涓哄櫔澹扮淮=**璇枡鏃犳硶璇勪及**锛堢暀 S2 璇枡锛夛紝闈炵壒寰佹窐姹般€俽esponse R虏 0.505鈫?.640 鍗曡皟鍗囥€?
> 鈿?鍙戠幇闆嗕箰瑙傛€?caveat 宸查娉ㄥ唽杩涜〃澶达紱fixpc 涓洪潪棰勬敞鍐?follow-up鈥斺€斿彧瀹?Router 杞壒寰侀泦
> 锛?*P0-D(2)+P1a-P(9)+C(3)**锛夛紝headline 椤?S2 dev 澶嶅埗銆?
>
> **S2 澶嶅埗锛堝悓鏃ワ紝绗簩鍗佷竷杞寚浠わ紱results/Stage2/S2_replication/锛?*锛歚s2_corpus.py` 8 鏃忓叏鍐荤粨
> 锛? v1 鏃忛€愬瓧 + intermittent/hetero/regime/multiseason锛沵iss-topology 绗竴绫昏酱
> random/block/burst脳{0,3,6,12}%锛沨oldout 涓嶇墿鍖栵級鈫?**鍗忚 v3**锛坈3d78ae1鈥︼紝frozen_full锛?
> 寮犻噺 gate 鍚堟牸锛屽弬鏁板揩鐓?鐢熸垚鍣ㄥ崟涓€鐪熸簮锛夆啋 672 uid dev 鈫?Phase B nested 鏍囩锛?2min锛?
> checkpoint锛? Phase C 涓冭噦锛堝畧鍗?p0鈮nchor 杩囷級銆?*鍒ゅ喅鍙岄棬 FAIL**锛坧rereg 搂1 鍏戠幇锛夛細
> fixpc +0.0511 [鈭?.013,+0.124]銆亀g LCB 鈭?.85 vs p0 鈭?.65锛泂q +0.006 鈫?**鍥為€€ P0+abstain锛?
> fixpc/sq 涓嶈浆姝ｏ紝鏃犵涓夎疆 finetuning**銆傚垎瑙ｏ細D1 淇鍦ㄥ鑺傜粨鏋?*澶嶇幇**锛圫_both 0.393鈫?
> 0.269/0.154@sq銆丼_season 鈫擄級锛汼_trend 澧炵泭=**鍙戠幇闆嗕吉褰?*锛圫2 涓?1.049鈫?.443 鍙嶅櫖锛屾崯浼?
> 闆嗕腑鏂版嫇鎵?block/burst鈥斺€擯1a 鐗瑰緛鍙湪鍧囧寑闅忔満缂哄け涓婇獙璇佽繃锛夛紱abstain 涓夊害澶嶈瘉=鐗瑰緛缂洪櫡
> 琛ヤ竵锛堟柊鍒嗗竷涓?p0_abstain 鍥炲埌 鈭?.020锛夈€備袱閫熶弗璋ㄧ郴缁熷湪绗竴娆＄嫭绔嬪鍒跺綋鍦烘嫤鎴箰瑙傛€с€?
> **鍒囩墖 v2 涓夋柟娑堟锛?026-07-06锛岀涓夊崄浜旇疆锛況esults/Stage2/SkillSliceV2/锛?
> prereg_skill_slice_v2.md锛岀涓冨紶鍐崇瓥琛級**锛氬叚鑷?{A/C_llm_v2/D_llm_v2/C_llm_verify/
> **Bplus_v2**/D_bplus_v2}锛孌ataView v2 鍙屼慨锛坧eriod+decomp 杩?core+鍙潬鎬ф爣娉?姹傝瘉瑙勮寖锛夈€?
> verify 寮哄埗涓ゆ銆丅+=featurized DataView(P0+robust diag+decomp+gap 17 缁? per-uid GBDT
> Phase-B 鍐荤粨鎶?OOF銆?*娉ㄥ唽棰勬祴绮剧‘鍏戠幇+涓€涓噸纾?*锛?*Bplus_v2 cum 0.2976 vs frozen
> 0.3677锛圙2 CI [鈭?.109,鈭?.040] 骞插噣锛夛紝S_both 棣栭亣鐮嶅崐锛?.26/0.34 vs 0.61/0.67锛?
> per-family 鈭?.340锛夈€丼_trend 鈭?.145锛坒ixpc 浼奖鏈鐜帮級銆丼_multiseason 鈭?.107**锛?
> LLM 鎷跨潃 core 閲岀殑 robust 璇佹嵁浠?16/16 median-w5锛?no strong seasonality"鐓у啓锛夛紝
> verify 寮哄埗姹傝瘉鏃犳敼鍠勶紱**G-main +0.105 [+0.061,+0.156]=LLM 鐙珛浠峰€间笉鎴愮珛锛堟敞鍐岄娴嬶級**銆?
> **鏈哄埗涓夎繛**锛氣憼robust_v1 鍚堝彇闂ㄤ綆鍣笅浜屽€煎寲澶辩湡锛坧eak_ratio 38 浣?acf 0.10<0.2鈫?
> "detected=0"锛夛紱鈶?*B+ 璧?娑堣垂杩炵画璇佹嵁锛圦5"淇濇寔杩炵画"瑁佸畾瀹炶瘉锛?*锛孡LM 杈?璇讳簩鍊艰緭鍑?
> 蹇借 decomp 鏁板€硷紱鈶_both 鑳滆礋鎵?median_w15脳47鈥斺€?*F0 椋庨櫓瑙勫垯锛坰eason 鐪燂級鏃忕洸鍘嬬缉鍚?
> 鎶?LLM 鎺ㄧ姝ｇ‘鍓傞噺=钂搁鐭ヨ瘑杩囨硾鍖?*銆侭+ 鐨?harm pocket 娓╁拰锛圫_regime +0.032/S_ar
> +0.012锛宮ax 鍧?0.118>未_safe鈫扜1 璐モ€斺€旇浆姝ｉ』甯?abstain/escalation 浣滅敤鍩燂級銆?*鍒嗘敮鎵ц**锛?
> 淇褰掑睘 Pattern鈥斺€擝+ 鐗瑰緛闆?P1b 鍊欓€夎繘鍏ュ鍒跺紡棰勬敞鍐岋紙鏈疆涓嶈浆姝ｏ級锛沝eployment LLM
> composer 閫€鍑轰富绾匡紱**鎱㈣矾寰?proposer 鍗囦负 LLM 涓绘搨鍙?*锛堢嫭绔嬮娉ㄥ唽锛夈€?60 娴嬭繃銆?
>
> **LLM-Skill 鍥涜噦鍒囩墖锛?026-07-06锛岀涓夊崄鍥涜疆锛況esults/Stage2/SkillSlice/锛?
> prereg_skill_slice.md鈥斺€擫LM 棣栨涓婂満锛岀鍏紶鍐崇瓥琛級**锛歋killSpec v1锛? skill 鍗曞皠瑕嗙洊
> 10 鍔ㄤ綔锛岄闄╄鍒?钂搁宸插彂琛ㄧ粨璁猴紝sha=52bc954c锛? DataView锛坔istory-only tool-mediated锛?
> 鏍稿績 4 瑙嗗浘+鍙姹?3 瑙嗗浘锛寁iew log 钀界洏锛? 涓ゆ寮?flash t=0 composer锛?6 鍞竴鍧楀喅绛栧叡
> 17 璋冪敤锛? 瑙ｆ瀽澶辫触鈥斺€旂閬撻浂鏁呴殰锛夈€傚畧鍗叏杩囷紙A鈮rozen 璐︽湰 80 鍧?bit 绾с€丅 wrapper
> 寰€杩旀亽绛夛級銆?*鍒ゅ喅 G1 璐?deployment LLM composer 鍦ㄦ淇℃伅闈㈣鎷?*锛欳 cum 0.4059 /
> D 0.3943 vs frozen 0.3677锛汫2 D鈭扐 +0.023 CI 涓嶅惈 0锛堝姡锛夛紱G3 缁勫悎闈惰触锛圫_both LLM
> 0.76/0.82 vs frozen 0.61/0.67锛夈€?*姝诲洜=aliasing 绗簲灞傦紙淇℃伅闈㈠眰锛夊疄閿?*锛歋_both 鐨?
> P0 璇绘暟 seasonal_strength med=0.000/period med=434锛圖1 娲诲寲鐭筹級鈫?core structure 瑙嗗浘
> 濡傚疄杞堪鐩茶鏁?鈫?LLM 淇′箣锛坮ationale 鏄庡啓"no seasonality"锛夆啋 15/16 鍧?1-call 鍐崇瓥銆?
> 涓嶈姹?period/decomp锛坮obust 璇佹嵁鍦ㄦ娊灞夐噷娌¤鎷匡級鈫?鍏ㄧ嚎 median-w5 鏉′欢鏀跨瓥锛屽湪缁撴瀯鏃?
> 涓婅緭 per-uid GBDT锛圫_both +0.149/S_regime +0.090锛夈€?*缁撹锛歀LM 鏃犳硶瓒呰秺鍠傜粰瀹冪殑
> Pattern 灞傦紱鑷俊鐨勯敊璇憳瑕佹姂鍒舵眰璇佽涓恒€傜摱棰?DataView 璇佹嵁璐ㄩ噺闈?LLM 鎺ㄧ悊** 鈫?涓嬩竴杞?
> 锛堥』鏂伴娉ㄥ唽锛?DataView v2锛歳obust 鍛ㄦ湡/鍒嗚В璇佹嵁杩?core 瑙嗗浘锛堝崟鍙橀噺閲嶈窇 C/D锛屾伆濂?
> 涔熸槸 view-log鈫扨1b 鍙岄噸韬唤鐨勫厬鐜帮級銆侴4 鎸?prereg 鍘熸枃 FAIL锛?4/16 鍞竴瑙﹀彂>60%锛?
> 瀹炰緥绾?40/80=50%鈥斺€斿喅绛栧叡浜娇"鍞竴鍐崇瓥鎴愭湰"搴﹂噺閫€鍖栵紝濡傚疄骞舵姤锛夈€?56 娴嬭繃銆?
>
> **LLM 閲嶅叆瀹¤ + 鍧楃骇鏂伴鎬ч棬锛?026-07-06锛岀涓夊崄浜岃疆锛況esults/Stage2/BlockGate/锛?
> Component Plan 搂12锛?*锛氱敤鎴风洿瑙?澶栭儴 reviewer 瑙﹀彂璺嚎瀹¤锛屾牳楠屼笁鐪熶竴鍋囷紙鐪燂細pipeline
> 浼樺厛绾?LLM 闆跺嚭鍦?updater 绾緥绂?LLM/鐪熸簮浜斾慨鏀归潰 vs 璁″垝缁堟€佷竴闈紱**鍋囷細reviewer 澹扮О
> "Memory 宸茶褰?**锛? 鐙珛鏂板彂鐜?compose_llm"涓嶈鍘熷簭鍒?銆傝瀹氾細瀹為獙搴忓垪娌″亸锛堥樁姊邯寰?
> 姝ｇ‘+纭畾鎬у姏绔瘉鏄庡埌鎵嬶級锛?*鐩爣鏋舵瀯鏂囨。鏀剁缉涓虹湡**鈫捖?2 淇涓哄崌绾у紡涓夊眰锛堢‘瀹氭€ч粯璁?
> 闂ㄨЕ鍙?LLM composer+鎱㈣矾寰?proposer+frozen fallback锛夈€?*鍧楃骇闂ㄩ獙璇侊紙17s锛? 瀹堝崼锛?*锛?
> P0 绌洪棿缃崲涓ゆ牱鏈楠岋紙伪=0.01 鏃犳棆閽棤鏍囩锛夛紝棣栭亣 recall 74.3%/FPR 22.5%/**harm 瑕嗙洊
> 70.2%/闈欐€侀噸鏀?c2 0.198鈫?.060**锛汼_trend 鏈€澶?harm 0.688 鍏ㄦ帴浣忋€?*娈嬩綑鐩茬偣=S_both 缁勫悎
> aliasing 绗洓灞?*锛堟紡缃?0.302 涓?0.245锛夛細缁勫悎鏃忚惤宸茶鏃忕壒寰佸嚫鍖呭唴鈫掔壒寰佺┖闂存柊棰栨€ф娴?
> 浠讳綍鑱氬悎绾у埆缁撴瀯鎬уけ鏄?鈬?LLM 闈跺瓙=缁勫悎**璇嗗埆**锛堚墵check13/14 鍚︽帀鐨勭粍鍚堢敓鎴愶級銆備笅涓€姝?
> 鍥涜噦 LLM-Skill 鍒囩墖锛圓/B/C/D锛岄』鏂伴娉ㄥ唽锛汱LM 淇℃伅闈㈤』鏂板 skill-conditioned 瑙佸師搴忓垪
> 鍏ュ彛锛涢閬囧鎵?frozen锛夈€倁pdater v4 鏆傚仠锛?47 鍏ㄩ噺缁?+ 4 鏂板畧鍗崟璺戠豢锛堟湰杞函鏂板妯″潡锛?
> 鏈Е鍙婃棦鏈変唬鐮侊級銆?
>
> **updater v3 = response-aware support锛?026-07-06锛岀涓夊崄涓€杞紱results/Stage2/Updater3/锛?
> prereg_updater3.md锛?*锛氳瘎瀹¤竟鐣屽叏閲囩撼锛堢鍚?鍙插唴 rolling-origin 杞绘帰閽堬紝API 鎷?>CUT 杈撳叆
> 鈫掓爣绛剧墿鐞嗕笉鍙揪锛? 鎺㈤拡脳1 鍒囩偣脳3 缁寸‖甯斤紱寮犻噺琛屼笉浣滅鍚嶏紱v2/frozen 閫愬瓧澶嶇敤 ckpt锛夈€?
> 瀹堝崼 G-A锛坴3 娴伱桺0 绌洪棿 bit 绾у鐜?v2 璐︽湰锛? G-B/C/D 娴嬭瘯鍏ㄨ繃銆?*鍒ゅ喅 FAIL锛? 鎵块噸闂ㄨ繃锛?*锛?
> cum 0.4033銆侀閬?harm 0.284銆佸鐜板鐩?鈭?.019銆乺ollback 14銆乧overage 0.91锛堥閬?澶嶇幇鍏ㄦ斁琛?
> =鏀寔鍩熸棤鍒ゅ埆鍔涳級銆?*鍙屽眰姝诲洜锛坉iag_updater3_signal.json锛?*锛氣憼浼拌灞傗€斺€斿悎娉曞崟鍒囩偣鎺㈤拡
> vs oracle 鍝嶅簲 Spearman 浠?0.10/0.28/鈭?.02锛涒憽琛ㄧず灞傦紙鏍癸級鈥斺€?*oracle 鍝嶅簲鏈韩涓嶅垎鏃?*
> 锛? 缁?1-NN 0.466銆佸叏 9 缁翠笂鐣?0.584銆佹暎甯冩瘮 0.93+锛岃€?P0 鐗瑰緛 0.896锛夆啋 鍝嶅簲宸€肩敱閫€鍖?
> 姘村钩涓诲銆佷笉鎼哄甫鏈哄埗韬唤锛宺esponse-aware 鏂瑰悜鍦ㄦ湰 gym 琚瘉浼簬鏍归儴銆?*姝ｉ潰鍓骇鐗?*锛氭満鍒?
> 韬唤淇℃伅涓€鐩村湪 P0 鐗瑰緛閲岋紙0.896锛夆€斺€攙2 鐨?aliasing 澶辫触鏄敮鎸佸煙**璇箟**闂锛坧er-uid NN
> 鍥㈠潡锛夐潪淇℃伅闂 鈫?鍏ヨ处鏂伴娉ㄥ唽鍋囪锛?*鍧楃骇锛坧opulation锛夋柊棰栨€ч棬**锛垀42 uid/鍧楃殑涓ゆ牱鏈?
> 妫€楠岋紝鈭歯 澧炵泭锛夈€侺LM proposer 缁х画涓嶈В閿併€?
>
> **updater v2 涓夎噦锛?026-07-06锛岀涓夊崄杞紱results/Stage2/Updater2/锛宲rereg 搂3锛?*锛?6 鍗婂潡
> 脳 5 棰勯攣鎺掑垪锛堝惈澶嶇幇杞达級锛寋frozen, naive v1, OOD-aware v2=canary 褰卞瓙鍧?per-uid 鏀寔鍩熸贩鍚坿銆?
> **鍒ゅ喅 PARTIAL(4/6)**锛歷2 鍏ㄩ潰鑳?v1锛坈um 0.378 vs 0.401銆侀閬?harm 0.198 vs 0.337銆佸け璐?7 vs
> 18銆佸鐜板鐩?+0.003 vs 鈭?.014銆乧overage 0.68 闈炰綔寮婏級鈥斺€旈樁姊涓夌骇鎴愮珛锛涗絾 c1锛坈um 浠嶇暐杈?
> frozen +0.010锛宮in 鎺掑垪宸茶耽锛変笌 c2锛堥閬?harm>未_safe锛夋湭杩団€斺€?*canary+P0 kNN 鏀寔鍩熸尅涓嶄綇
> "鐗瑰緛鐩镐技鏈哄埗涓嶅悓"鐨?aliasing 棣栭亣**銆傞娉ㄥ唽鍒嗘敮鎵ц锛歀LM proposer(v3) 涓嶈В閿侊紱涓嬩竴姝?
> response-aware support / episodic memory锛堜笉璋冮槇鍊硷紱寮犻噺琛?response 绛惧悕鍘熸枡锛夈€?
> propose_update 鎻愰€?3脳锛埼?鍙奖鍝?picks 涓嶅奖鍝?fit鈥斺€旀嫙鍚堜竴娆℃壂 魏锛屼綅绾у悓缁撴灉锛夈€?
>
> **L5 鍏噦锛堝悓鏃ワ紝绗簩鍗佷節杞紱results/Stage2/L5/锛宲rereg_l5_updater2.md 搂1锛孡ODO-family=
> 鍏ㄥ憳棣栭亣鍙ｅ緞锛?*锛?*鍒ゅ喅=KEEP-ACTION-ONLY**锛堥娉ㄥ唽瑙勫垯鑷姩钀斤細joint vs action_only +0.025
> 璺?0 涓?worst-family 鏇村樊锛泂equential 鏄捐憲鏇村樊锛夈€傜粨鏋勪笁鍙戠幇锛氣憼涓夋ā鍨嬩富鑿滃崟 global_pair
> 鍏ㄥ満鏈€浣筹紙瀛︿範鑷傚叏杈撳父鏁板鈥斺€旈閬囦笅瀛︿範涓嶈縼绉伙紝涓?S2 澶嶅埗鍚屾瀯锛夛紱鈶″弻妯″瀷鍓彍鍗?
> **model_only 鍐冲畾鎬ф渶浣?*锛?.463锛寁s action_only 鈭?.137 CI 娓呮櫚锛沜hronos 400/672锛夆€斺€擯0 鐗瑰緛
> 涓嬪敮涓€鍙缁村害="閫夊摢涓ā鍨?锛屼粎鍦ㄥ幓鎺?seasonal_naive 骞叉壈鍚庢樉鐜帮紱鈶oint>sequential 浣嗛兘
> 杩戒笉涓婄畝鍗曡噦銆?鍙屾ā鍨?model-only+action-only overlay"鍏ヨ处涓?*鏂伴娉ㄥ唽鍋囪**锛堜笉鍗虫椂閲囩撼锛夈€?
> updater v2 鑽夋宸查娉ㄥ唽锛圤OD-aware+canary+5 棰勯攣鎺掑垪+recurrence 杞达級銆?
>
> **Harness action-only 鍨傜洿鍒囩墖锛堝悓鏃ワ紝绗簩鍗佸叓杞畾妗堬紱results/Stage2/HarnessSlice/锛?*锛?
> overlay 姝ｅ紡钀藉湴锛坄pipeline.process(forced_program=鈥?` 绛栫暐绋嬪簭鍘?compose锛屽綋鍓?harness 鐨?
> L1/L4/L3/绠楀瓙鐘舵€佸叏鐣欙紝gates 鎷掔粷鈫抮ecovery 涓嶆梺璺級+ EvidenceRecord.routing锛?*2.0-鈶?鏈熼檺
> 鍏戠幇**锛? 纭畾鎬?updater锛堝垎灞傜暀鍑洪獙璇侊細鍧囧€兼敼鍠勨埀worst-group 涓嶆伓鍖栨墠 accept锛? rollback
> 锛堟柊鍧?harm>未_safe 澶嶅師锛夈€?*绗竴寮?Harness 琛?*锛氭満鍒堕摼鍏ㄩ€氾紙672/672 overlay 闆跺彂鏁ｃ€?
> 3 accept/5 reject/2 rollback 鍏ㄧ幆鑺傜湡瀹炶Е鍙戯級锛?*鎬ц兘锛氭湸绱犳寔缁洿鏂拌緭鍐荤粨 0.4149 vs 0.3695**
> 鈥斺€斾激瀹冲叏閮ㄥ湪"鎺ュ彈鍚庨閬囨柊鏃?绐楀彛锛坴1鈫扴_trend harm 0.203銆乿2鈫扴_both 0.083锛?*涓ゆ false
> accept 鍧囪 rollback 鍗曞潡鍐呮崟鑾?*锛夛紱v3 鍚?5 杩?reject=楠岃瘉闂ㄥ垎甯冨唴姝ｅ父銆傝瘖鏂細宸茶鍒嗗竷
> 楠岃瘉闂ㄧ粨鏋勬€ф棤娉曢槻鏈缁撴瀯浼ゅ 鈫?**updater v2 椤?OOD-aware 鎺ュ彈鏈哄埗锛堟柊棰勬敞鍐岋級**銆?
> abstain/L5 鎺緸鎸夌浜屽崄鍏疆鏀剁揣锛坅bstain=鍥為€€鍩虹嚎闈炲畨鍏ㄨ儨鑰咃紱L5_joint=绔嬮」闈炴垚鍔燂紝涓诲疄楠?
> 涓夋ā鍨嬭彍鍗?鍓姤 DLinear+Chronos 瀛愯彍鍗昜share=0.143]锛夈€倀ests/test_overlay.py 鍥涘畧鍗€?
>
> **寮犻噺 pilot 瀹屾垚锛?40/240 妲介浂缂哄け锛屸増25min锛泃ensor_pilot_verdict.json锛?*锛?
> **interaction_share=0.2931 鈮?.15 鈫?branch=L5_joint**锛坅ction脳model 浜や簰涓板瘜锛夛紱dominance
> 鍚﹀喅锛堟渶寮哄 v_none脳chronos 瑕嗙洊 31.9%锛?0%銆亀g LCB 鈭?.60锛夆啋"寮烘ā鍨?杞诲鐞嗘敮閰?涓嶆垚绔嬶紝
> L5 鑱斿悎 (a,m) 閫夋嫨绔嬮」銆係_multiseason 鍛ㄦ湡瀵?(16,128) 涓夐噸绾︽潫锛堥潪鍏害瀵硅 robust_v1
> 鍗?lag ACF 璇嫆=P1b 鏍囩殑锛夈€俠acklog锛氱粨鏋勯棬鎺?P1 鑷傦紙椤绘柊棰勬敞鍐岋級銆?
>
> **Router 绗竴杞紙鍚屾棩锛岀浜屽崄鍏疆鍙岃瘎瀹?Go 鍚庯紱棰勬敞鍐?results/Stage2/prereg_s2_replication.md锛?
> results/Stage2/Router1/锛?*锛歚run_router1.py` 鍏噦浜?fixpc 鐗瑰緛+鍐荤粨鎶橀噸鏀撅紙瀹堝崼鈶犺鏂?bit 绾?
> +瀹堝崼鈶㈠鐓ц噦閫愪綅澶嶇幇 P1a fixpc 鍏ㄨ繃锛夈€?*鑳滆€?= shared-Q(P,D,C,a)+蠁 鐜扮畻**锛歴q 螖Regret
> 鈭?.0321 [鈭?.0693, 鈭?.0021]锛圕I 涓嶈法 0锛夈€亀orst-group LCB 鈭?.1104鈫掆垝0.0708銆丼_trend
> 0.371鈫?.299锛泂q_abstain_kcv 鏈€浼?0.1621锛埼? 浜旀姌鍏ㄩ€?0.5<legacy 1.0锛変絾鐩稿 sq 浠?+0.0017
> 鈥斺€攁bstain 鍦ㄥソ鐗瑰緛+濂藉舰鎬佷笂杈归檯鍐嶆鈮?锛圥1a 鍙戠幇 1 澶嶈瘉锛夈€?*C-gate 鍦ㄥ急鍩哄骇涓婃湁鏁?*锛?
> pa_abstain_cgate 鎶?legacy 瑙﹀彂鐮?95%锛?.20鈫?.01锛変笖鍧囧€?worst-group 鍙屾敼鍠勨€斺€攍egacy 瑙﹀彂
> 澶у钀藉湪楂?C 鍖猴紙璇ュ瀹冩槸閿欑殑锛夛紱sq 鍩哄骇涓婅Е鍙戔埀浣嶤 浜ら泦鈮?銆?*unseen dosage 鑳藉姏瀛樺湪浣嗗急**锛?
> sq_low15 鍦?w15-oracle 瀛愰泦閫変腑鐜?0.14锛坧a 缁撴瀯鎬?0.00锛夈€佸瓙闆?regret 0.42 vs 0.53锛屾€讳綋 CI
> 璺?0锛坣=37锛夆€斺€斿厓鏁版嵁鎻掑€兼垚绔嬶紝骞呭害寰?S2锛堝惈鍓傞噺杞达級澶嶈瘉銆俿q_rank 鍧囧€间笉鏄捐憲浣?worst-group
> LCB 鏈€浼橈紙鈭?.0249锛夈€傚垽瀹氾細sq 杩?S2 澶嶅埗闆嗙 4 琛岋紱Router-2 incumbent=sq锛坅bstain 鍙€?
> 魏_cv锛岃竟闄咃級锛沠amily+鍓傞噺鑷傛寜棰勬敞鍐屾湭瀹炵幇锛坰q 瑕嗙洊鍏惰兘鍔涳紝鏄惧紡澹版槑锛夈€?
>
> **鏂囨。瀹氫綅锛?026-07-05 鏇存柊锛屽綋鍓嶅洓鏂囨。浣撶郴锛?*锛歚../idea/System_Requirements_and_EndToEnd_Workflow.md` = 绯荤粺鐪熸簮锛堢ǔ瀹氳竟鐣屼笌涓嶅彲杩濆弽绾︽潫锛夛紴 `../idea/Component_Optimization_and_Integration_Plan.md` = Stage 2 鍞竴鎵ц璁″垝锛堝 SysReq 澶存敞鍒楀嚭鐨勫緟鍚屾绔犺妭鏈変复鏃跺琛ユ晥鍔涳級锛?`results/STAGE1_VERDICT.md` + 鍚?freeze.json = 鍘嗗彶鍒ゅ喅涓庡喕缁撳崗璁紙鍙锛夛紴 `../idea/TS_Knowledge_Foundations_*.md` = 鐭ヨ瘑鍙傝€冦€傛棫 `../plan.md` 涓庡凡褰掓。鐨?`../idea/previous/.../SelfEvolvingHarness_*.md` 鍙綔鍘嗗彶杩芥函锛屼笉鍐嶆槸鏉冨▉鍏ュ彛銆?*鏈枃妗?* = 瀹炵幇鐘舵€?+ 瀹為獙璁板綍 + 澶嶇幇鎸囧崡銆?
>
> **canonical 浠ｇ爜搴?= `SelfEvolvingHarnessTS/`**锛堝榻?plan.md锛夈€傛棫 `SelfHarnessTS/` 鏄簾寮冪殑鏃╂湡璁捐锛屽叾浠ｇ爜涓嶅鐢紙浠?`exploration/` 鐨?E0鈥揈4 瀹為獙缁撹涓?POC 浣滃弬鐓э級銆?
>
> 鎴嚦 2026-06-21锛歅hase 0鈥? + 涓ら」鎵╁睍锛坈ell-scoped 妯℃澘銆丱PD 淇＄敤鍒嗛厤锛夊叏閮ㄥ疄鐜帮紝**12 涓?test suite / 75 鐢ㄤ緥鍏ㄨ繃**銆傚凡浠庡悎鎴愰暱璺戞帹杩涘埌**鐪熷疄 Monash 璺ㄥ煙闀胯窇**锛埪?.3锛夛紝骞舵帴鍏?*涓夊眰鍙€?grounded 鍒ゅ畼**锛坒rozen / frozen+ensemble / Chronos foundation锛夛紱澶村彿鍙戠幇锛?*鍒ゅ畼鑳藉姏闂ㄦ帶鍙澶撮儴绌洪棿**銆?*鏂板 classify 绔埌绔?*锛埪?.4锛夛細ECG5000 鐪熷疄閿?+ ROCKET 纭畾鎬у垽瀹?+ InceptionLite 鐙珛鎶ュ憡鍣?鈫?**绗簩鏉¤法浠诲姟 C1**锛坉enoise_stl 鍔?forecast 浼?classify + 鍚?cell 鏈€浼樺弽杞級锛?*鑷繘鍖栵紙闃舵B锛夌嫭绔嬮噸鏂板彂鐜拌 C1**锛堝杞绘竻娲楁ā鏉裤€佺鏀瑰舰骞虫粦锛孫PD 褰掑洜鏀舵暃锛岀嫭绔嬫姤鍛婂櫒 mean 螖Perf +0.007鈫?0.043锛夈€?

---

## 1. 瀹炵幇鎬昏

涓€鍙ヨ瘽鐩爣锛氬浐瀹?LLM銆佷笉鏀规潈閲嶃€佷笉鍋?per-dataset 璁粌锛涢潰瀵?(pattern, task) 寮傝川鏃跺簭锛岄€氳繃**鑷繘鍖?Harness**锛圠1 鎸囦护 / L2 绠楀瓙缂栨帓 / L3 璁板繂 / L4 楠岃瘉锛夋妸鍔ｅ寲搴忓垪鍔犲伐鎴愬涓嬫父浠诲姟**灏辩华**鐨勬暟鎹€傛牳蹇冨懡棰?**H\* = f(pattern, task)**銆?

```
SelfEvolvingHarnessTS/
鈹溾攢鈹€ harness/               # R1/R4锛氳杩涘寲鐨勫敮涓€瀵硅薄
鈹?  鈹溾攢鈹€ edit_patch.py        EditPatch + Manifest 濂戠害锛坧roposer鈫攎erger锛?
鈹?  鈹溾攢鈹€ editable_surfaces.py Surface + EDITABLE_SURFACES(18 闈? + validate 鏍￠獙閾?
鈹?  鈹溾攢鈹€ layers.py            L1鈥揕4 dataclass + minimal_l*() + PipelineTemplate/StageDef.from_dict
鈹?  鈹斺攢鈹€ state.py             HarnessState锛歛pply_edit / snapshot+restore / replay / 鐗堟湰
鈹溾攢鈹€ conditioning/          # R3锛氬叏绯荤粺绱㈠紩璇█
鈹?  鈹溾攢鈹€ key.py               struct_feats(10 缁? 绋冲仴 SNR) + quality_profile + build_conditioning_key
鈹?  鈹溾攢鈹€ binning.py           鍐荤粨缃戞牸 鈫?cell_id锛圫NR脳missing脳task锛?
鈹?  鈹溾攢鈹€ distance.py          d = 伪路d_struct + (1-伪)路d_quality + similarity
鈹?  鈹斺攢鈹€ router.py            (pattern_bin, task) 鏍囩
鈹溾攢鈹€ operators/             # R1锛氬彧璇诲熀纭€绠楀瓙搴擄紙鍗曚竴鐪熸簮锛?
鈹?  鈹溾攢鈹€ registry.py          TOOL_REGISTRY(callable) + OPERATOR_METADATA(缁?L2)
鈹?  鈹溾攢鈹€ s1_{impute,denoise,outlier,decompose}.py  s2_align.py  s3_shape.py  _common.py
鈹溾攢鈹€ sandbox/executor.py    # in-process 娴佹按绾挎墽琛?+ 寮傚父鍥村牭
鈹溾攢鈹€ fast_path/             # R2 蹇矾寰勶紙per-input锛屽彧浜ц瘉鎹級
鈹?  鈹溾攢鈹€ perceive.py          conditioning_key(+cell_id)
鈹?  鈹溾攢鈹€ retrieve.py          Retriever锛堟殩鍚姩锛屽喎鍚姩瀹夊叏锛?
鈹?  鈹溾攢鈹€ compose.py           heuristic + compose_llm + cell-scoped 妯℃澘椹卞姩 + cell_banned_ops
鈹?  鈹溾攢鈹€ execute.py / verify.py(Gate閾? cell-scoped Skill) / pipeline.py(process, memory/llm 鍙€?
鈹溾攢鈹€ evaluators/            # R5/R9/R10锛歱roxy + grounded 涓ゅ眰锛屼笁浠诲姟涓夊崗璁紙鐪?torch锛?
鈹?  鈹溾攢鈹€ base.py(娲惧彂) _torch_models.py(DLinear/LSTM/InceptionLite)
鈹?  鈹溾攢鈹€ frozen_probe.py(鍐荤粨 LSTM 缂栫爜鍣?Ridge, 蟽_A鈮?; set/pretrain/load_*_encoder swap鐐? grounded_{forecast,anomaly,classify}.py
鈹?  鈹溾攢鈹€ chronos_probe.py(鈽呯湡 TS foundation 鍒ゅ畼 Chronos-Bolt, 纭畾鎬? grounded_forecast{set_forecast_target/substrate}
鈹?  鈹溾攢鈹€ rocket_probe.py(鈽卌lassify 纭畾鎬у垽瀹?ROCKET-lite+LogReg, 蟽=0) grounded_classify{set_classify_substrate, classify_inception}
鈹?  鈹溾攢鈹€ report_target.py(鈽呯嫭绔嬫姤鍛婂櫒: lstm/dlinear/chronos[fc]路inception/rocket[clf] 鉄?鍒ゅ畼, 螖Perf 涓昏〃鐢? disjoint_targets 寮哄埗鍒嗙)
鈹?  鈹溾攢鈹€ role_a_proxy.py role_b_metrics.py calibration.py  _artifacts/frozen_lstm_{h64,real_h64}.pt(缂撳瓨)
鈹溾攢鈹€ memory/                # R3/L3锛氳瘉鎹笌妫€绱?
鈹?  鈹溾攢鈹€ evidence_store.py(EvidenceRecord+dict-of-lists) retrieval.py(MemoryIndex kNN) signatures.py
鈹溾攢鈹€ slow_path/             # R2 鎱㈣矾寰勶紙per-cell锛? 杩涘寲寮曟搸
鈹?  鈹溾攢鈹€ schedule.py          edit_budget(cosine) + CellSchedule(鍐荤粨鐘舵€佹満)
鈹?  鈹溾攢鈹€ batch_builder.py     鍒嗙粍 + 涓?split(held_in/out_a/out_b)
鈹?  鈹溾攢鈹€ mining.py            weakness/strength 鎶ュ憡
鈹?  鈹溾攢鈹€ validator.py         grounded 涓?split + 鎺ュ彈寰嬶紙鍞竴瑁佸垽锛屼笉姹℃煋 harness锛?
鈹?  鈹溾攢鈹€ proposer.py          鍚屼竴 fixed LLM 鍑?K 鍊欓€夛紙鍙敞鍏ワ級+ cell-scoped 妯℃澘鎻愯
鈹?  鈹溾攢鈹€ merger.py            鍚堝叆 + 鐗堟湰+1 + 璺?cell 鏁村浐锛圫trength鈫掑彈淇濇姢鍖猴級
鈹?  鈹溾攢鈹€ attribution.py       OPD 鍏紡11 outcome-calibrated 淇＄敤鍒嗛厤
鈹?  鈹斺攢鈹€ evolve.py            涓茶 round-robin 涓绘帶锛堝喕缁?瑙ｅ喕 + 閫愬€欓€夐噸楠屽悎骞?+ 褰掑洜锛?
鈹溾攢鈹€ llm/client.py          # OpenAI 鍏煎 DeepSeek 瀹㈡埛绔紙缂撳瓨+閲嶈瘯锛? 浠ｇ爜/JSON 鎶藉彇
鈹溾攢鈹€ config/thresholds.py   # 蔚/蟿/K/S/N_FREEZE/edit_budget/bin/SNR... 闆嗕腑绠＄悊
鈹溾攢鈹€ data/synthetic_gen.py  # (clean,degraded,label) 涓夊厓缁勶細P1鈥? + 4 缃戞牸棰勮
鈹溾攢鈹€ data/load_real.py      # 鈽呯湡瀹?Monash 璺ㄥ煙閿氾細z-score 鐪熷簭鍒?澶嶇敤閫€鍖栫綉鏍尖啋鍚屽舰 RawSeries(闆舵敼鍔? + classify 閿?RealClassSignal/build_real_classify_corpus)
鈹溾攢鈹€ data/load_ecg5000.py   # 鈽卌lassify 鐪熷疄閿?ECG5000(UCR 5-class 浼樺厛, 瀹曟満鍥為€€ TF 浜屽垎绫? 鈫?_artifacts/ecg5000.npz
鈹溾攢鈹€ data/_artifacts/monash_clean.npz  # 83 淇″彿/7 鍩熷ぇ璇枡锛坆uild_real_npz 涓嬭浇锛孒F 鐩磋繛锛?
鈹溾攢鈹€ run_evolve_longrun.py  # 鐪?LLM 澶?cell脳澶?epoch 鍚堟垚闀胯窇杩愯鍣?
鈹溾攢鈹€ run_real_longrun.py    # 鈽呯湡瀹炴暟鎹暱璺戯細--mode diag/evolve --start minimal/degraded --substrate frozen/chronos
鈹溾攢鈹€ run_classify_longrun.py # 鈽卌lassify 鑷繘鍖栭暱璺戯紙闃舵B锛夛細ECG5000 + ROCKET 纭畾鎬у垽瀹?+ 鐪?LLM 鈫?cell-scoped 杞绘竻娲楁ā鏉?+ Ours-evolved 螖Perf(InceptionLite鉄傚垽瀹?
鈹溾攢鈹€ run_main_table.py      # 鈽呬富琛?Data-Readiness 螖Perf 姹囩紪锛堢嫭绔嬫姤鍛婂櫒 脳 final_test split锛屽垽瀹樷焸鎶ュ憡鍣級+ 7 鍥哄畾鍙樹綋/per-cell oracle锛?-task classification 鎺?ECG5000+ROCKET 鍒ゅ畼
鈹溾攢鈹€ run_calibrate_eps.py   # 鈽呂?閰嶅鏍囧畾锛堝悓-batch cur vs cand 閰嶅 蟽_螖 鈫?frozen鈮?.03/chronos鈮?.08锛?
鈹斺攢鈹€ tests/                 # 12 suite / 75 鐢ㄤ緥
```

---

## 2. 杩愯鐜

| 椤?| 鍊?|
|---|---|
| Python | conda `project` 鐜锛歚D:/Anaconda_envs/envs/project/python.exe`锛坱orch 2.12+cu126锛?*CUDA 鍙敤**锛? GPU锛泂klearn/scipy/statsmodels/pandas 榻愶級 |
| base anaconda | **鏃?torch** 鈥斺€?璺?evaluators/slow_path 蹇呴』鐢?project python |
| 杩愯娴嬭瘯 | `PYTHONPATH=<Agent> D:/Anaconda_envs/envs/project/python.exe -m SelfEvolvingHarnessTS.tests.<suite>` |
| 杩愯闀胯窇 | 鍚堟垚 `... -m SelfEvolvingHarnessTS.run_evolve_longrun`锛涚湡瀹?`... -m SelfEvolvingHarnessTS.run_real_longrun [--substrate chronos]` |
| 娉ㄦ剰 | `conda run -n project` 瀵瑰琛?`-c` 浼氬穿 鈫?鐢ㄧ洿鎺?python 璺緞 + `PYTHONPATH` |
| LLM key | `env DEEPSEEK_API_KEY` 浼樺厛锛屽洖閫€ `previous/check7-10` 鐨?key锛涚紦瀛樺湪 `llm/_cache/` |
| Chronos 鍒ゅ畼 | `chronos-forecasting 2.3.0` + `transformers 5.12.1`锛堝凡瑁咃級锛涢娆＄敤涓嬭浇 `amazon/chronos-bolt-small`锛?*HF 鐩磋繛鍙敤锛岄暅鍍?hf-mirror.com 涓嶅彲鐢?*锛屽嬁璁?`HF_ENDPOINT`锛墊
| 鐪熷疄璇枡 | 榛樿 `AdaCTS/data/monash_real.npz`锛?2 淇″彿锛夛紱澶ц鏂?`data/_artifacts/monash_clean.npz`锛?3 淇″彿锛宍load_real.build_real_npz` 涓嬭浇锛墊

---

## 3. 鍏抽敭璁捐鍐冲畾锛堝疄鐜板€掗€?/ 宸插畾鐗堬級

| 鍐冲畾 | 鍐呭 | 鍑哄 |
|---|---|---|
| **EditPatch 濂戠害** | `edited_layer/op/path/value/manifest`锛涙棤 scope 瀛楁锛堢敱 surface 娲剧敓锛夛紱source_type=failure/strength锛涘鍧€涓夋ā寮?leaf/list_scalar/named_object锛岀浣嶇疆绱㈠紩娣辫矾寰?| plan.md 搂3.2 |
| **Surface.type=value 鏈熸湜绫诲瀷** | 闈炲瓧娈靛鍣ㄧ被鍨嬶紱leaf-into-dataclass鈫扤one 鎺ㄦ柇锛沶amed_object鈫掔被鍚?| 搂3.2(c) |
| **L4 鎷嗗瓧娈?* | `evaluator_registry` 鈫?`proxy_evaluators`(step) + `grounded_evaluators`(consolidator-only, protected)锛涘畧 evaluator/optimizer 鍒嗙 | code-review |
| **task_templates 鎵佸钩** | `Dict[name, PipelineTemplate]`锛宯amed_object `::name` 瀵诲潃 | 鈥?|
| **绋冲仴 SNR struct_feat** | 鍘昏秼鍔?+ top-3 棰?+ MAD 鎶楃缇わ紙鏃?MA-11 琚?5蟽 绂荤兢涓诲鎴愯礋鍊硷級锛沗BIN_SNR_SPLIT_DB=4.0` | 闀胯窇2 璇婃柇 |
| **frozen+probe锛堟棤鐪?foundation锛?* | 鍐荤粨 LSTM 缂栫爜鍣紙鐣欏嚭闆嗛璁?缂撳瓨锛? Ridge 澶?鈫?纭畾鎬?蟽_A鈮?锛圗2c 蹇呰鏉′欢澶嶅埢锛?| 鈥?|
| **validator 涓嶆薄鏌撳綋鍓?harness** | snapshot鈫抋pply鈫抏val鈫抮estore锛涢€愬€欓€夊**褰撳墠** harness 閲嶉獙 鈫?娑堥櫎杩囨湡鍊欓€?| 搂6.1 #5 |
| **LLM compose 鍙湪閮ㄧ讲璺緞** | 楠岃瘉鐜矾鐢?heuristic compose锛堢‘瀹氭€?鍏?LLM锛屼笖璇?active/defaults/banned 鈫?缁撴瀯闈㈠彲楠岋級锛沗pipeline.process` 榛樿 memory/llm=None 閫€鍖?Phase0/1 | Phase 2 |
| **cell-scoped 妯℃澘** | pattern-conditioned 妯℃澘鍙敼鏈?cell compose锛坈ompose + verify 鍚岀敤 `cell_banned_ops`锛夆啋 澶╃劧 Pareto 瀹夊叏 鈫?unlock C1 | Phase 2b |
| **OPD 鍏紡11 淇＄敤鍒嗛厤** | 閫?(cell,op) 绱Н validator 瀹炴祴 delta锛屛?1鈭?/鈭?1+N) 鍔犳潈锛涘杺 proposer prefer/avoid | Phase 2b |
| **鎱㈣矾寰勬帶鍒讹紙B.2 #4/#5/#6锛?* | N_FREEZE=3 鍐荤粨 / scope-based 瑙ｅ喕 / FREEZE_RECHECK_EPOCHS=5锛涗覆琛?round-robin 鏃犻攣锛沞dit_budget cosine max=K=3/min=1/total=12 | plan.md 搂6.1 |
| **S_SEEDS substrate-aware** | frozen forecast + anomaly 纭畾鎬р啋S=1锛沜lassify(InceptionLite) 闅忔満鈫扴鈮?锛堝凡鎺ュ叆 validator锛?| code-review #3 |
| **鐪熷疄閿?鐪熶俊鍙?鍙楁帶缃戞牸閫€鍖?* | load_real 閫愬簭鍒?z-score 鐪熷疄 Monash 褰撳共鍑€淇″彿婧愶紝澶嶇敤鍚堟垚閫€鍖栫綉鏍尖啋浜у悓褰?RawSeries锛涗笌鍚堟垚闀胯窇閫?cell 鍙瘮锛屽敮涓€鍙橀噺=鐪熷疄淇″彿缁撴瀯 | 搂4.3 |
| **涓夊眰鍙€?grounded 鍒ゅ畼** | `set_forecast_substrate(frozen|chronos)` + `set_forecast_target(raw|ensemble|seasonal_resid)`锛氬叏灞€ setter銆佸 evaluator/validator 闆朵镜鍏ワ紱榛樿 frozen+raw 淇濇寔鏃㈡湁琛屼负/娴嬭瘯涓嶅彉 | 搂4.3 |
| **Chronos foundation 鍒ゅ畼** | Amazon Chronos-Bolt 闆舵牱鏈洿鎺ラ娴嬪櫒锛坧redict_quantiles 鍙?mean锛?*纭畾鎬?蟽_A=0**锛夛紱plan A.2/A.4 "鎺ラ璁潈閲?钀藉湴锛沚olt-small 鐢滅偣锛坆ase 鍙嶆洿宸笖 6脳 鎱級 | 搂4.3 |
| **鍒ゅ畼鈫旀姤鍛婂櫒鍒嗙锛堜富琛級** | `report_target.py` 鐙珛 target锛坙stm/dlinear/chronos锛夆焸 in-loop 鍒ゅ畼 + `batch_builder.final_test`锛堣繘鍖栨湡涓嶇鐨勭4 split锛? `run_main_table.py` 姹囩紪 螖Perf锛沠rom-scratch鈫掑 seed銆傚畧 readiness 闈炲惊鐜嚜璇?| Exp_Design 搂鈽?2 |
| **classify 纭畾鎬у垽瀹橈紙ROCKET-probe锛?* | classify 鍒ゅ畼浠庨殢鏈?InceptionLite(蟽>0) 鎹负 **ROCKET-lite+LogReg**锛堥殢鏈哄嵎绉牳 seed 鍥哄畾 鈬?蟽=0锛夆啋 淇粨璁?#3 鐨?classify 璁粌鍣０鍊猴紝鏄?frozen-LSTM-probe 鐨?classify 绫绘瘮锛沗set_classify_substrate` 榛樿 inception 淇濆洖褰?| 搂4.4 |
| **classify 鍒ゅ畼鈫旀姤鍛婂櫒鍒嗙** | in-loop 鍒ゅ畼=ROCKET銆佺嫭绔嬫姤鍛婂櫒=InceptionLite from-scratch(`classify_inception` 鏄惧紡锛屽惈 raw NaN fillna)锛沗disjoint_targets` 鍔?rocket/inception锛汦CG5000 閿?UCR 瀹曗啋TF 浜屽垎绫诲洖閫€) | 搂4.4 |
| **蔚 閰嶅閲嶆爣瀹?* | 蔚 鐢?*鍚?batch cur vs cand 閰嶅 螖** 瀹氾紙闈炴棤閰嶅 batch 鏂瑰樊鈫掑悗鑰呭惈妯埅闈㈠紓璐ㄣ€乤ccept 寰嬮噷鎶垫秷銆佽繃浼皛3脳锛夈€俙run_calibrate_eps.py` 閰嶅鐗堬細璺?cell 涓綅 蟽_螖 frozen鈮?*0.028**(鈮堝師0.03,E2c 鏈)/chronos鈮?*0.08**锛涗粎瓒嬪娍 cell snrHigh\|miss 鐪熼珮鍣?0.27~0.79)銆俙EPS_NARROW_REAL_CHRONOS=0.08`+`--eps`銆偽?涓嶆敼鍒ゅ畼鍋忓ソ | run_calibrate_eps |

---

## 4. 瀹為獙璁板綍

### 4.1 code-review锛坸high锛?026-06-20锛?
鍏ㄤ唬鐮佸簱瀹℃煡 鈫?8 finding锛屼慨澶?#1/#2/#3/#5/#6锛?
- **#1** EditPatch.from_dict 鐧藉悕鍗曞凡鐭ュ瓧娈碉紙LLM 濉為澶栭敭涓嶅啀 TypeError 涓㈠€欓€夛級銆?
- **#2** validator proxy 棰勭瓫榛樿 OFF锛堛€屾棤璇佷笉鐢ㄣ€嶏紝calibration 鎺ュ叆鍚庡啀寮€锛涢『甯﹁В #4 forecast 涓?DLinear proxy 姣?frozen-Ridge grounded 鏇磋吹鐨勫€掓寕锛夈€?
- **#3** grounded_val_loss 瀵归殢鏈?substrate 鎸?S_SEEDS 澶?seed 骞冲潎銆?
- **#5** evolve Strength 姣?epoch 绠椾竴娆★紙O(cells) 闈?O(cells虏)锛夈€?
- **#6** mining 涓€娆?fast_path 澶嶇敤 grounded+floor+淇＄敤鍒嗛厤銆?
- **鏈慨锛堝緟鍔烇紝浠呭ぇ瑙勬ā鎵嶇棝锛?*锛?7 retrieval 姣忚緭鍏ュ叏琛ㄦ壂鎻忥紱#8 consolidate 鐗堟湰 churn銆?

### 4.2 鐪?DeepSeek 闀胯窇锛坢inimal/degraded 璧凤紝6 cell脳3 epoch锛?

| 闀胯窇 | 璁剧疆 | 鍏抽敭缁撴灉 | 缁撹 / 涓嬩竴姝?|
|---|---|---|---|
| **L1** 2 cell | 闄嶇骇 harness锛堝叧绂荤兢绠楀瓙锛夛紝P1/P2/P3 鏁版嵁 | winsorize 琚?grounded 鎺ュ彈鎭㈠锛涘啑浣欏€欓€夎嚜鍔ㄦ嫆锛沘nomaly 姝ｇ‘鍐荤粨 | 鏈哄埗瀵癸紝浣?**cell collapse**锛堟暟鎹叏钀?snrLow\|miss锛夆啋 鏀圭ǔ鍋?SNR + 缃戞牸鏁版嵁 |
| **L2** 6 cell | 绋冲仴 SNR + 4 缃戞牸棰勮 | **C1 瀹炴祴涓虹湡**锛坥utlier 绉婚櫎 snrHigh 鏀瑰杽/snrLow 鏈夊锛夛紱grounded 姝ｇ‘閴村埆 winsorize(鍓婂嘲)vs outlier_iqr锛?*璺?cell Pareto 鐪熷疄鐢熸晥**锛? 杩濆弽锛?| 鐡堕=**global toggle 鏃犳硶琛ㄨ揪 cell-conditional 鏈€浼?* 鈫?cell-scoped 妯℃澘 |
| **L3** 6 cell | + cell-scoped 妯℃澘鑳藉姏 | LLM **100% 鎻?cell-scoped 妯℃澘**锛? cell 鍚勮嚜涓嶅悓 specialization 琚帴鍙楋紙snrHigh\|full鈫抎enoise_stl锛泂nrLow\|miss鈫抜mpute_kalman锛夛紱0 Pareto 杩濆弽锛?*held_out(a) 娉涘寲闂ㄩ娆¤Е鍙?*锛堟嫆杩囨嫙鍚堟ā鏉匡級 | unlock C1 楠岃瘉鎴愬姛锛沘ccept 鐜?11% |
| **L4** 6 cell | + OPD 淇＄敤鍒嗛厤 | 褰掑洜琛ㄨ繛璐紙snrHigh\|full: prefer stl/avoid savgol-wavelet-median锛夛紱**LLM 鏄庢樉鍦ㄨ**锛堟ā鏉垮悕鍙?`..._avoid_denoise_savgol`/`prefer_wavelet`锛夛紱snrLow\|full 鍊熷綊鍥犲彂鐜版纭?denoise_stl | accept 鐜囦粛 11%鈥斺€旂摱棰?*浠?瀹氫綅"杞Щ鍒?headroom vs 蔚"** |

> 涓夋闀胯窇鍚勬毚闇蹭竴涓湡闂骞惰涓嬩竴姝ヨВ鍐筹細鈶燾ell collapse鈫掔ǔ鍋?SNR锛涒憽global toggle鈫抍ell-scoped 妯℃澘锛涒憿proposer 涓嶇煡浣曟湁鐩娾啋OPD 褰掑洜锛堚啋鎻ず headroom-vs-蔚 涓烘柊闄愬埗锛屄?.3 杩涗竴姝ヤ慨姝ｄ负**鍒ゅ畼鍙楅檺**锛夈€?

### 4.3 鐪熷疄 Monash 璺ㄥ煙闀胯窇锛圥0锛氶獙 C3 + 鑷繘鍖栵紝2026-06-20锛?

**鏁版嵁 / 杩愯鍣?*锛歚data/load_real.py` 鎶?Monash 鐪熷疄鍗曞彉閲忓簭鍒?*閫愬簭鍒?z-score** 褰?骞插噣淇″彿婧?锛屽鐢ㄥ悎鎴愰€€鍖栫綉鏍硷紙`_degrade`/`_inject_anomalies` 脳 `G_hi/G_lo脳full/miss`锛変骇鍚屽舰 `RawSeries` 鈫?BatchBuilder / evaluators / slow_path 鍏ㄩ摼璺?*闆舵敼鍔?*鍚冪湡鏁版嵁銆俙run_real_longrun.py` 鏃楁爣锛歚--mode diag/evolve`銆乣--start minimal/degraded`銆乣--substrate frozen/chronos`銆乣--encoder synthetic/real`銆乣--forecast-target raw/ensemble/seasonal_resid`銆乣--npz`銆傞粯璁ゆ湰鍦?`AdaCTS/data/monash_real.npz`锛?2 淇″彿 / 3 鍩燂細nn5_daily p7銆乼ourism_monthly p12銆乫red_md p12 寮鸿秼鍔挎棤瀛ｈ妭锛夈€?

**Step 1a 璇婃柇锛堝厤 LLM锛?*锛?2 淇″彿 脳 4 閫€鍖栭璁?鈫?**6 cell 鍏ㄨЕ鍙?*锛坒orecast 4 + anomaly 2锛夛紱binning 椴佹锛堝彧鐢?SNR脳missing锛夈€傜湡瀹炲己瓒嬪娍/缂哄け涓?`period` struct_feat锛團FT 涓婚锛夎浣庨瓒嬪娍涓诲鈮堝簭鍒楅暱锛堜笉褰卞搷 binning锛屾槸鐪熷疄璇婃柇鐜拌薄锛夈€傜湡瀹?std 璺ㄥ煙宸?5 涓暟閲忕骇锛坣n5鈮? vs fred鈮?.5e5锛夆啋 **per-series z-score 蹇呴渶**銆?*缂栫爜鍣ㄨ縼绉昏瘖鏂?*锛氬悎鎴愰璁喕缁?LSTM 鍦ㄧ湡瀹?forecast **3/4 cell 鍔ｄ簬 seasonal_naive**锛涚湡瀹炵暀鍑洪璁紙leave-signal-out锛?*浠嶄笉鐮村簳** 鈫?grounded forecast substrate锛圚=48 鈮?4~7 鐪熷疄鍛ㄦ湡 + fred 瓒嬪娍澶栨帹锛夊ぉ鑺辨澘銆?

**涓ゆ鑷繘鍖栵紙6 cell 脳 3 epoch锛岀湡 DeepSeek锛壝?涓ょ被鍒ゅ畼**锛?

| 瀹為獙 | 鍒ゅ畼 | accept | 鍏抽敭缁撴灉 |
|---|---|---|---|
| **Step2 鎭㈠**锛坉egraded 璧凤紝鍏?winsorize/outlier_iqr/outlier_mad锛?| frozen | 22%(4/18) | **鑷富鎭㈠涓旇秴瓒婂仴搴峰熀绾?*锛氱粡 cell-scoped `denoise_stl`锛?*闈為噸寮€绂荤兢绠楀瓙**锛夆啋 nRMSE 闄?0.30~0.65锛? Pareto 杩濆弽锛涙棤澶撮儴绌洪棿 cell 姝ｇ‘涓嶇 |
| **Step1b C1**锛坢inimal 璧凤級 | frozen | 6%(1/18) | 5 cell 鍐荤粨锛堝仴搴峰熀绾垮己锛夛紱**C1 閾佽瘉 = winsorize 璺ㄤ换鍔″彉鍙?*锛坒orecast +0.08 / anomaly 鈭?.147锛屽悓 pattern 鐩稿弽鏈€浼橈級|
| **Step2 鎭㈠** | **chronos** | 17%(3/18) | 鎭㈠鍚?**鐮?naive 搴?3/4 cell**锛坰nrLow\|miss 1.115鈫?.553鈮猣loor锛夛紱**澶撮儴绌洪棿闅忓垽瀹橀噸瀹氫綅**锛堣秼鍔?cell snrHigh\|miss 宸叉渶浼樷啋姝ｇ‘鍐荤粨锛夛紱鍒ゅ畼濉戦€?*涓嶅悓**鏈€浼樻祦姘寸嚎锛坕mpute_kalman/znorm/...锛墊
| **Step1b C1** | **chronos** | **22%(4/18)** | **鍏?4 forecast cell 鏀硅繘銆乪volved 鍚?4/4 鐮?naive 搴?*锛埼?+0.31~+1.32锛泂nrHigh\|full 0.921鈫?.448<floor锛夛紱鍙?2 anomaly cell 鍐荤粨锛涙纭嫆澶у洖褰掞紙held_out_a 螖鈭?.467锛墊

**寮哄寲 grounded judge锛堣繃绋嬶級**锛欻F 鐩磋繛 huggingface.co 鍙敤锛?*闀滃儚 hf-mirror.com SSL EOF 涓嶅彲鐢?*锛夈€備笁鏉犳潌瀹炴祴鈥斺€斺憼鏇村ぇ璇枡锛?3 淇″彿 / 7 鍩?`monash_clean.npz`锛夊鐪熷疄缂栫爜鍣ㄦ棤鏁堬紝涓旂湡瀹為璁紪鐮佸櫒**姣斿悎鎴愰浂鏍锋湰鏇村樊**锛堝紓璐ㄧ湡瀹為璁?muddy 鐗瑰緛锛夛紱鈶seasonal_resid` 鍙嶆晥鏋滐紙娈嬪樊涓嶅彲棰勬祴鈫掑湪寮?naive 涓婂姞鍣級锛涒憿OOF 鏀剁缉 `ensemble`锛坒rozen鈯晄easonal-naive锛?-fold 鏍锋湰澶栦及 w*锛変弗鏍?鈮?raw銆佽秼鍔?cell 鐮村簳浣嗗己瀛ｈ妭 cell 浠?鈭?.04~鈭?.12锛坵 鏃犳硶鍦?test 鏈潵涓婅皟锛夈€傗啋 鎺ュ叆鐪?foundation锛歚evaluators/chronos_probe.py`锛圕hronos-Bolt 纭畾鎬ч浂鏍锋湰锛夛紝`set_forecast_substrate('chronos')` swap 鐐广€?*bolt-small 鐮?floor 3/4锛坆alanced eval split锛夛紝bolt-base 鍙嶆洿宸笖 6脳 鎱?* 鈫?small 鐢滅偣銆?

> **馃攽 澶村彿鍙戠幇锛氬垽瀹樿兘鍔涢棬鎺?鍙澶撮儴绌洪棿"銆?* Step1b 鍦?frozen 鍒ゅ畼涓?6% accept銆? cell 鍐荤粨锛屾棫缁撹璁颁负"headroom vs 蔚 / heuristic 鍩虹嚎宸插濂?锛?*鍚屽疄楠屼粎鎹?Chronos 鍒ゅ畼 鈫?22% accept銆?/4 forecast cell 澶у箙鏀硅繘涓?evolved 鍚庣牬 naive 搴?*銆傚嵆锛?*寮卞垽瀹樻劅鐭ヤ笉鍒版暟鎹竻娲楃殑浠峰€硷紙鍏惰嚜韬ぉ鑺辨澘鎺╃洊锛夛紝寮哄垽瀹樻彮绀虹湡瀹炲ご閮ㄧ┖闂?*鈥斺€旀棫"headroom vs 蔚"鏄?*鍒ゅ畼鍙楅檺锛岄潪鏁版嵁鍙楅檺**銆傝嚜杩涘寲 harness 璁虹偣鍦ㄥ己鍒ゅ畼涓嬭鏄捐憲鏇村己鍦板疄璇侊紱涓斿垽瀹樺閫犳渶浼樻祦姘寸嚎 鈫?**H\* = f(pattern, task, judge)**銆?

> 瀹為獙鏃ュ織锛歚_real_step2.log` / `_real_step1b.log`锛坒rozen锛夈€乣_real_step2_chronos.log` / `_real_step1b_chronos.log`锛坈hronos锛夈€?

### 4.4 classify 绔埌绔細绗簩鏉¤法浠诲姟 C1锛?026-06-21锛屾棤 LLM 涓昏〃锛?

> **鍔ㄦ満**锛氭鍓嶈瘉鎹泦涓湪 forecast(+灏戦噺 anomaly)锛沝ata readiness 鎸夊畾涔夎法浠诲姟锛?鏃犲叏灞€鏈€浼?浠诲姟鍐茬獊"椤诲浠诲姟鎵嶅畬鏁淬€傛湰鑺傝ˉ classify 绔埌绔紝缁欏嚭**鐙珛浜?winsorize(forecast鈫攁nomaly) 鐨勭浜屾潯 C1 璇佹嵁**銆傛湰杞?浠?C1 涓昏〃锛堟棤杩涘寲/鏃?LLM锛夈€?

**鏂版帴鍏?*锛堣瑙?搂3 鍐冲畾琛ㄦ柊澧炶锛夛細
- **鐪熷疄鍒嗙被閿?ECG5000**锛坄data/load_ecg5000.py`锛夛細UCR 5-class 婧愶紙tsc.com锛夊綋鍓?**502 瀹曟満 鈫?鑷姩鍥為€€ TF 浜屽垎绫?csv**锛堟甯?2079/寮傚父 2079锛屽钩琛★紝morphology 鍒ゅ埆锛夈€?*鍗冲綋鍓?classify 鏁版嵁=浜屽垎绫?ECG**锛堥潪 5-class锛沀CR 鎭㈠鍚庡悓浠ｇ爜閲嶈窇鍗冲緱 5 绫伙級銆傞€愬簭鍒?z-score + 澶嶇敤閫€鍖栫綉鏍?G_hi/G_lo脳full/miss锛坄data/load_real.py: build_real_classify_corpus`锛夆啋 涓?forecast 鍚?(SNR脳missing) cell 鍧愭爣锛岄€?cell 鍙瘮銆?
- **classify 纭畾鎬у垽瀹?= ROCKET-lite + LogReg**锛坄evaluators/rocket_probe.py`锛夛細闅忔満鍗风Н鏍?seed 鍥哄畾 鈬?**蟽=0**锛堜慨缁撹 #3 鐨?classify 璁粌鍣０鍊猴紝鏄?forecast frozen-LSTM-probe 鐨?classify 绫绘瘮锛夛紱`set_classify_substrate('rocket')` 鍒囨崲銆?
- **鍒ゅ畼鈫旀姤鍛婂櫒鍒嗙**锛歩n-loop 鍒ゅ畼=ROCKET銆佺嫭绔嬫姤鍛婂櫒=**InceptionLite from-scratch**锛坄classify_inception`锛屼笌鍒ゅ畼涓嶇浉浜わ紝鍚?raw NaN fillna锛夛紱`run_main_table --task classification`銆?

**涓昏〃**锛?00 ECG 淇″彿 / final_test=30 per cell / 2 seed锛?87s锛宍_clf_maintable.log`锛沠orecast 瀵圭収=Monash锛宍_fc_maintable.log`锛夈€傚浐瀹?7 鍙樹綋 螖Perf(vs raw)锛岀嫭绔嬫姤鍛婂櫒锛堚焸鍒ゅ畼锛夛細

| variant锛堝钩婊?绂荤兢寮哄害锛?| forecast 鍧囧€?(lstm/dlinear, 鉄俢hronos) | classify 鍧囧€?(inception, 鉄俽ocket) | rocket 浜ゅ弶鍙傜収 |
|---|---|---|---|
| v_none锛堜粎鎻掕ˉ锛?| +0.000 / +0.000 | +0.008 | +0.000 |
| v_median锛堣交鍘诲皷宄帮級 | 鈭?.003 / **+0.189** | **+0.249** | +0.134 |
| v_savgol锛堜腑骞虫粦锛?| 鈭?.024 / +0.020 | +0.046 | 鈭?.054 |
| **v_stl锛圫TL 閲嶅幓鍣紝鏀瑰舰锛?* | **+0.057 / +0.052** | **鈭?.037** | **鈭?.110** |
| v_wavelet | 鈭?.024 / +0.020 | +0.044 | 鈭?.054 |
| v_winsor锛堢缇ら挸鍒讹級 | +0.029 / +0.093 | +0.141 | +0.073 |
| v_winsor_savgol | 鈭?.018 / +0.065 | +0.186 | +0.081 |

**馃攽 绗簩鏉?C1 = shape-altering 骞虫粦绠楀瓙璺ㄤ换鍔＄鍙风炕杞?+ per-cell 鏈€浼樺弽杞?*锛?
- **`denoise_stl` 鍔?forecast(+0.05) 浣嗕激 classify(鈭?.04~鈭?.11)**鈥斺€擲TL 鎶规帀 classify 鎵€闇€鐨勯珮棰戝垽鍒舰鎬侊紝鍗村埄 forecast銆傝繖鏄嫭绔嬩簬 winsorize 鐨勭浜屼釜璺ㄤ换鍔＄浉鍙嶇畻瀛愩€?
- **鍚?cell 鏈€浼樼洿鎺ュ弽杞?*锛堟渶骞插噣闄堣堪锛夛細`snrLow|miss` 涓?forecast 鐨?oracle 璧㈠ = **v_stl(+0.055)**锛岃€岃绠楀瓙鏄?classify 鐨?*鏈€宸?*(鈭?.188)锛沜lassify 璧㈠ = **v_median(+0.204)**锛岃绠楀瓙鏄?forecast 姝?cell 鐨勮礋璐＄尞(鈭?.063)銆俙snrLow|full` 鍚屽悜鍙嶈浆锛坒orecast v_stl +0.006 / classify v_stl 鈭?.161锛夈€?
- **杞荤缇ょЩ闄?median/winsor)涓や换鍔￠€氬悆**锛堢Щ闄ゆ敞鍏ョ殑 5蟽 缃戞牸绂荤兢=绾櫔澹帮紝闈炰俊鍙凤級鈥斺€旀晠 C1 鐨勫垽鍒畻瀛?*鐗规寚鏀瑰舰骞虫粦(stl)**锛岄潪鎵€鏈夋竻娲椼€?
- per-cell oracle 璧㈠**璺ㄤ换鍔′笉鍚?*锛歠orecast={winsor_savgol, median, winsor, stl}锛宑lassify={median, winsor, median, median}銆?
- **闈炲惊鐜ǔ鍋?*锛氱嫭绔嬫姤鍛婂櫒 InceptionLite 涓庡垽瀹?ROCKET **鏂瑰悜涓€鑷?*浜庡叏閮ㄥ己淇″彿锛坴_median ++銆乿_stl 鈭掆垝銆乿_winsor +锛夆啋 readiness 澧炵泭闈炲垽瀹樼壒寮傘€?

**鑷繘鍖栵紙闃舵B锛岀湡 DeepSeek锛宮inimal 璧凤紝3 epoch锛宍run_classify_longrun.py`锛宍_clf_evolve_minimal.log`锛?*锛氭妸鍥哄畾 7 鍙樹綋鍗囩骇涓?LLM 鑷繘鍖栵紙鍒ゅ畼=纭畾鎬?ROCKET锛?6 calls/847s锛夈€?*杩涘寲鍣ㄧ嫭绔嬮噸鏂板彂鐜?搂4.4 鐨?C1 淇″彿**锛?
- 瀛﹀埌 cell-scoped **杞绘竻娲?*妯℃澘骞?*鏄惧紡绂佹敼褰㈠钩婊?*锛歚snrHigh|full` prefer `denoise_median` / **ban `denoise_savgol`**锛沗snrLow|miss` prefer `denoise_median` 鈫?ROCKET CE **0.397鈫?.202**(螖+0.195)銆?
- **OPD 褰掑洜璺?cell 鏀舵暃鍒拌法浠诲姟妯″紡**锛氭櫘閬?prefer `winsorize`/`denoise_median`(杞诲幓灏栧嘲)銆乤void `denoise_stl`/`savgol`/`wavelet`(鏀瑰舰)鈥斺€旀鏄?鍔?forecast 浼?classify"閭ｆ壒绠楀瓙锛堝敮 snrHigh|miss cell 褰掑洜杈冨櫔锛屼笌 搂4.4 璇?cell 淇″彿鏈氨寮变竴鑷达級銆?
- **闈炲惊鐜鐩?*锛氱嫭绔?InceptionLite 鎶ュ憡鍣?鉄俁OCKET 鍒ゅ畼) 脳 杩涘寲鏈熸湭纰扮殑 final_test 鈫?mean 螖Perf **+0.007鈫?0.043**锛?*0 Pareto 杩濆弽**锛涙纭嫆涓€涓繃鎷熷悎 `winsorize_median` 缁勫悎(held_out_a 鍥炲綊)銆俛ccept 2/12(17%锛屼笌 forecast Step1b 6鈥?2% 鍚屾。锛岃瘹瀹?銆?
- 娉細classify 鏃?seasonal floor 鈫?mining `improvable=False`/`mine_strength=None`锛堜粎鎻愮ず锛屼笉闂ㄦ帶鎻愯锛汸areto 浠嶇敱 validator held_out_b 瀹堬級锛沝egraded 璧风偣瀵?classify 鈮?minimal锛坢inimal 鏈笉骞虫粦锛夋晠鏈崟璺戙€?

> **娉ㄦ剰锛堣瘹瀹炶竟鐣岋級**锛歠orecast 閿?Monash銆乧lassify 閿?ECG5000 鈫?璺ㄤ换鍔″鐓ф槸"**鍚屼竴绠楀瓙闆?+ 鍚屼竴 (SNR脳missing) cell 缃戞牸涓娿€佺畻瀛愭晥搴旈殢浠诲姟缈昏浆**"锛岄潪鍚屼竴搴忓垪鎹㈡爣绛撅紙ECG 闀?140<forecast MIN_LEN锛屾棤娉曞悓搴忓垪鍋?forecast锛夈€傚悎鎴?generator 鍙悓 pattern 璺戜笁浠诲姟锛屼絾鏈疆閫夌湡瀹為敋浠ユ眰 classify 淇″彿寮哄害銆傛棩蹇楋細`_clf_maintable.log` / `_fc_maintable.log` / `_clf_evolve_minimal.log`銆?

---

### 4.5 鈽卾4 S1锛氭祦寮?domain 鎸佺画閫傚簲 + 涓?bootstrap锛?026-06-23锛?

> 鎵?`idea/Refactor_Continual_TaskReadiness_v4.md` + `idea/S1_Implementation_Plan.md`锛堝惈 reviewer 8 鐐圭殑鏈€缁堢粨璁猴級銆係1 = 鎶?per-corpus 杩涘寲鎻愮骇涓?*娴佸紡 domain 鎸佺画閫傚簲**锛坮eset-free锛夛紝鏄?v4 鎸佺画閫傚簲 headline 鐨勬渶灏忓湴鍩恒€?*鏈Е绾㈢嚎**锛歀5 妯″瀷閫夋嫨銆丮_deploy 姹犮€丵5 璁板繂鍘嬬缉銆丵3 闈炲绉板蹇嶆帴鍙楀緥鍏ㄩ儴鎺ㄥ悗锛圫4+锛夈€?

**钀藉湴妯″潡锛? 鏂板缓 / 4 鏀癸級**锛?
| 鏂囦欢 | 鍔ㄤ綔 | 瑕佺偣 |
|---|---|---|
| `models/registry.py` + `models/__init__.py` | 鏂板缓 | 涓夎鑹?`MODEL_METADATA`锛坄allowed_roles` + 瑙掕壊鍚庣紑瀹炰緥 chronos_judge/report锛? `get_models_for_role`鈥斺€?*J鈭㎝_deploy=鈭?闈?API 寮哄埗闈?assert**锛堝畧闈炲惊鐜級銆係1 鍙敤鐜版垚 J/R锛汳_deploy(TSFM/GBDT) 鐧昏 `status=todo` 寰?S4 鎸?|
| `slow_path/deploy_stream.py` | 鏂板缓 | `DomainSpec` + `deploy_stream(mode鈭坰cratch/frozen/updating)` + 鍓嶅悜杩佺Щ JSONL锛沠rozen 鐢ㄤ复鏃?store 涓嶆薄鏌?carried memory锛況eadiness 娴嬩簬 held_out_a |
| `evaluators/readiness.py` | 鏂板缓 | `readiness_score=(J_raw鈭扟_cur)/(J_raw鈭扟_min_ref)`锛堝垎姣嶉€€鍖栤啋nan锛? `aggregate_time_to_readiness`锛?*median+max**锛宧eadline median锛?|
| `run_stream_s1.py` | 鏂板缓 | A/B/C 缂栨帓锛堝厛 updating 浜?checkpoint 鈫?frozen 澶嶇敤 鈫?scratch锛夛紱鍚堟垚榛樿鍏?LLM锛宍--llm flash`/`--npz` 鎺ョ湡瀹?|
| `tests/test_deploy_stream.py` | 鏂板缓 | 7 鐢ㄤ緥锛堜笁 mode/frozen 涓嶆敼 H/lazy 閲嶉獙闄嶇骇/reset-free/enter_new_domain/registry 鍒嗙/readiness锛?|
| `slow_path/evolve.py` | 鏀?| `Evolver.run(domain_idx, on_epoch_end)` + `revalidate_strength()`锛坙azy warm-start 閲嶉獙锛宖loor-margin 浠ｇ悊鎹曡礋杩佺Щ鈫掗檷绾?advisory锛?|
| `slow_path/schedule.py` | 鏀?| `CellSchedule.enter_new_domain(k)`锛歳ound_idx 褰掗浂閲嶇儹 + **瑙ｅ喕**锛坒reeze domain-scoped锛? `domain_idx` meta-閫€鐏ぉ鑺辨澘 |
| `config/thresholds.py` | 鏀?| `DOMAIN_BUDGET_CEILING_DECAY`锛堥粯璁?0锛岄暱娴佽皟 1锛? `READINESS_THRESHOLD=0.8` |
| `evaluators/__init__.py` | 鏀?| 瀵煎嚭 readiness helpers |

**reviewer 8 鐐圭殑鏈€缁堣惤鍦?*锛堝凡鍐欏叆 `S1_Implementation_Plan`锛屼唬鐮侀伒涔嬶級锛氣憼+鈶?registry `allowed_roles` API + 瑙掕壊鍚庣紑瀹炰緥锛涒憿 canonical+2 shuffle(`--order-seed`)锛涒懀 `RawSeries` docstring 閽夋竻锛堢‘璁ゅ瓨鍦ㄤ簬 synthetic_gen.py锛夛紱鈶?`enter_new_domain` 閲嶇儹+**瑙ｅ喕**锛坮eviewer "floor 闈?0"/瀛楁钀界偣宸茬籂姝ｏ級锛涒懃 frozen 璺?fast_path+J+R 涓?evidence 鍏ヤ复鏃?store锛涒懄 閲嶉獙 baseline=鍘荤墖娈靛弽浜嬪疄锛圫1 鐢?floor-margin 浠ｇ悊锛夛紱鈶?ttr 鍗曚綅=round锛堥潪 epoch锛? 鑱氬悎 median+max銆?

**楠岃瘉**锛?
- **鍏?75 娴嬭瘯杩?*锛?8 鏃?+ 7 鏂帮紝`project` env锛?鍒嗛挓锛泂chedule/evolve 鏀瑰姩闆跺洖褰掞級銆?
- **涓?mode 绔埌绔?demo**锛堝悎鎴?forecast K=3 patterns 褰?domain锛宻tub proposer锛屽厤 LLM锛夛細**updating(C) version 璺ㄥ煙鍗曡皟鍓嶈繘锛坮eset-free 鎴愮珛锛?*锛?*frozen(B) k=0 閫€ minimal(ver0)銆乲鈮? 杞藉叆 C 鐨?checkpoint 涓?version 涓嶅墠杩涳紙涓嶆敼 H 鎴愮珛锛?*锛?*scratch(A) 姣忓煙 fresh**銆俽eadiness 鍦ㄦ湁 headroom 鐨?cell=1.0銆佹棤 headroom cell=nan锛堣瘹瀹炩€斺€攎inimal 宸蹭笉鐮?raw 鍒欎笉鍙娴嬶級銆?
- **璇氬疄杈圭晫**锛氬悎鎴?minimal 璧风偣涓?C/A 鐨?time-to-readiness 閮解増1锛坢inimal 鍗冲氨缁級锛?*鍓嶅悜杩佺Щ淇″彿闇€ degraded 璧风偣鎴栫湡瀹炶嚜鐒堕€€鍖栨祦鎵嶆樉鐜?*锛堜笌 D1 涓€鑷达細headline 鍦ㄧ湡瀹炴祦涓婅窇锛夈€備笅涓€姝?= `--npz` 鐪熷疄娴?+ degraded 璧风偣璺戝嚭 B鈭扐/C鈭払 鏇茬嚎锛圫2锛夈€?

**妯″瀷鏉冮噸 provisioning**锛歋1 鐨?J 鍒ゅ畼榛樿 `frozen+probe`锛堟湰鍦?torch锛屾棤闇€涓嬭浇锛夈€?*Chronos-Bolt 鏉冮噸宸蹭笅杞藉氨缁?*锛坄amazon/chronos-bolt-small` ~48M + `chronos-bolt-base` 鍧囧湪 `~/.cache/huggingface/hub`锛孒F 鐩磋繛锛岀‘瀹氭€?蟽_A=0锛夆啋 鍙綔鏇村己 forecast 鍒ゅ畼锛坄set_forecast_substrate("chronos")`锛? R 鎶ュ憡鍣ㄣ€係4 寰呬笅锛歍imesFM/Moirai锛圡_deploy TSFM 姹狅級銆?

### 4.6 鈽卾4 S2锛氬墠鍚戣縼绉绘洸绾垮垎鏋愶紙2026-06-23锛?

> 鎵?搂4.5 S1 涓?`idea/S1_Implementation_Plan.md 搂B.5` / `Refactor_v4 S2`锛圥0锛夈€係2 = 璇?S1 钀界殑 `forward_transfer_*.jsonl` 鈫?鑱氬悎鎴?per-(mode, k) 鏇茬嚎 + headline 鍒ゆ嵁銆傜函鍒嗘瀽灞傦紝涓嶅啀璺戣繘鍖栥€?

| 鏂囦欢 | 鍔ㄤ綔 | 瑕佺偣 |
|---|---|---|
| `slow_path/forward_transfer.py` | 鏂板缓 | `load_transfer_log`锛堝惈 NaN锛宩son 榛樿 allow_nan锛夆啋 `per_domain_points`锛坈ell 鑱氬悎锛歵tr median/max銆乺eadiness@budget median銆乺eady_frac銆乺eval_demote锛夆啋 `forward_transfer_verdict`锛欳(updating/memory-on) vs A(scratch/memory-off) 鐨?mean readiness(C鈭扐)/ttr_gain(A鈭扖)銆佷笁 bootstrap **B鈭扐(璁板繂浠峰€?/C鈭払(缁х画鏇存柊浠峰€?**銆佽礋杩佺Щ鎶ゆ爮銆?*discriminative 瀹堟姢**锛堝樊鍒嗗叏鍦?tol 鍐?鈫?`supported=None` 涓嶄笅缁撹锛岄槻楗卞拰骞冲眬璇垽"鎴愮珛"锛?|
| `run_s2_transfer.py` | 鏂板缓 | 缁堢琛?+ 鍒ゆ嵁 + matplotlib 鍙岄潰鏉匡紙time_to_readiness(k) / readiness@budget(k)锛孋/B/A 涓夋洸绾匡級+ `s2_transfer.json`/`.png`锛泂tdout 寮哄埗 utf-8锛堥槻 Win 鎺у埗鍙?GBK 缂栦笉鍑?鈭?鈫?鈿?宕╋級 |
| `tests/test_forward_transfer.py` | 鏂板缓 | 12 娴嬶細3 bug 鍥炲綊 + S2 鑱氬悎/鍒ゆ嵁/鎶ゆ爮/NaN-file 寰€杩?|

**淇鐨?3 涓?bug**锛坮eview 鎶ュ憡锛夛細
- **#1 [鐪?浣嗛潪宕** `layers.py` `operator_defaults["stl_decompose"]` 榛樿 `"auto"`鈫抈0`銆俙"auto">=2` 纭疄鎶?TypeError锛屼絾琚?`denoise_stl` 鐨?`except Exception` **闈欓粯鍚炴垚 savgol 鍥為€€** 鈫?STL **浠庝笉鐪熻窇**锛堟瘮宕╂洿闅愯斀锛夈€傛敼 `0`=鑷姩鐚滄祴鐪熻窇 STL锛涚旱娣卞姞 `s1_denoise._coerce_period` 璁?`'auto'/None/闈炴硶`鈫掕嚜鍔ㄧ寽娴嬶紙闃?LLM 鍚?`"auto"`锛夈€?
- **#2 [鐪焆** `deploy_stream` 鍏ュ彛鏍￠獙锛歚scratch/updating` 缂?`make_proposer` 鈫?鎶?`ValueError`锛堝師浼氬湪 `evolve_cell` 璋?`proposer.propose` 宕?`AttributeError`锛夛紱`frozen` 浠嶅厑璁?None銆?
- **#3 [鍋囬槼鎬** `denoise_savgol(window, order)` 鍖呰鍣?*鑷韩鎺ュ彈 `order` 骞跺唴閮ㄧ炕璇戞垚 scipy `polyorder`**锛坄s1_denoise.py:18`锛夛紱榛樿 `{"window":11,"order":3}` 姝ｇ‘銆?*鏈敼**鈥斺€旇嫢鎸夋姤鍛?`s/order/polyorder/` 鍙嶈€岄敊锛坄polyorder` 浼氳鍚炶繘 `**_` 闈欓粯蹇界暐锛宍order` 閫€榛樿锛夈€傚凡鍔犲崟娴嬪浐鍖栨濂戠害銆?

**楠岃瘉**锛?*鍏?87 娴嬭繃**锛?5 + 12 鏂帮紝project env锛夛紱S2 鍦?`runs/s1_demo` demo 涓婄鍒扮璺戦€氾紙鍑?json+png锛夈€?*璇氬疄杈圭晫**锛氬悎鎴?demo C/A 鏇茬嚎閲嶅彔锛坢inimal 鍗冲氨缁€佹棤 headroom锛夆啋 `discriminative=False`銆乣supported=None`锛堝伐鍏锋纭姤鍛?鏃犲垎绂讳俊鍙蜂笉鍙粨璁?锛夈€?

**鐪熷疄娴?S1鈫扴2 璺戦€?+ flash vs pro 瀵规瘮锛?026-06-23锛宍runs/s1_flash` / `runs/s1_pro`锛?*锛?
- **淇 4 bug**锛歚run_stream_s1.real_domains()` 鍘熸寜涓嶅瓨鍦ㄧ殑 `s.source`/`s.name` 鍒嗙粍 鈫?鐪熷疄娴佸鎴?K=1锛堟棤鍓嶅悜杩佺Щ锛夈€傛敼鎸?`RealSignal.config` 鈫?**K=4 鐪?Monash 鍩?*锛坈ovid_deaths/fred_md/nn5_daily/tourism_monthly锛夈€傚姞 `--k`/`--max-per-domain`/`--max-domains`/`--min-signals` CLI + per-model proposer cache锛坄stream_s1_{flash,pro}`锛夈€?
- **鍚岄厤缃?*锛坘=2, 2 epoch, frozen-LSTM 鍒ゅ畼, max-per-domain=8, n-per-signal=3锛夛細

| | flash (deepseek-chat) | pro (deepseek-v4-pro) |
|---|---|---|
| 澧欓挓 | 11 min | 29 min |
| updating 绱 ver锛堚増鎺ュ彈鏁帮級 | **2** | **0** |
| 鎻愯椋庢牸锛?0 proposals锛?| 婵€杩涳細鍔?cell-scoped denoise 妯℃澘锛坢edian/stl/wavelet/savgol锛?winsorize | 淇濆畧锛歜an_winsorize / no_fft / toggle winsorize off + 鍙傛暟寰皟锛?1 L1锛?|
| tourism k=3 readiness C/A | **0.443 / 1.000** | **1.000 / 1.000** |
| S2 mean readiness(C鈭扐) | **鈭?.278** | **0.000** |
| discriminative / supported | True / **False**锛堣蹇嗘湁瀹筹級 | False / **None**锛堝钩灞€涓嶅彲缁撹锛?|
| 璐熻縼绉绘姢鏍?fired | False | False |

- **鏍稿績鍙戠幇**锛?*proposer 寮哄害鏀瑰彉澶辫触妯″紡**鈥斺€斿急(flash)杩囩紪杈戔啋鎶?covid/nn5 鐨?denoise 妯℃澘鎼哄甫杩?tourism 鑷?*璐熻縼绉?*锛圕1 璺ㄥ煙绗﹀彿缈昏浆锛夛紱寮?pro)鎭板綋淇濆畧鈫掑琚?grounded 闂ㄦ嫆鈫?*閬垮紑璐熻縼绉?*浣嗗洜鍒ゅ畼寮变害鏃犳澧炵泭銆?
- **澶╄姳鏉?鍒ゅ畼闈炴暟鎹?*锛氬鏁?cell `j_min_ref > j_raw`锛坒rozen-LSTM 涓嶇牬 seasonal-naive 鈫?readiness=nan锛夛紝鍗拌瘉鏃х粨璁?鐡堕鍦ㄥ垽瀹?銆?*鐪熸鍚戣縼绉婚渶鎹?chronos 寮哄垽瀹樻彮 headroom**锛坄set_forecast_substrate("chronos")`锛夈€?
- **鎶ゆ爮缂哄彛**锛歚reval_demote=0`鈥斺€攆lash 璐熻縼绉绘簮鑷惡甯︾殑 **L2 config**锛坥perator_defaults/templates锛夛紝鑰?`revalidate_strength` 鍙畧 consolidated `strength_signatures`銆傝璁″€猴細warm-start 閲嶉獙搴旇鐩?cell-scoped 妯℃澘鎼哄甫銆?
- **娆¤**锛歱ro 鎻愪簡闈炴硶鍙傛暟 `denoise_savgol.window_length`锛堢畻瀛愬疄鍙備负 `window`锛夆啋 proposer prompt 缂虹畻瀛愬弬鏁?schema锛堝緟琛?`_SURFACE_HINT`锛夈€?

**澶嶇幇**锛歚run_stream_s1 --npz ...monash_clean.npz --llm {flash,pro} --k 2 --epochs 2 --max-per-domain 8 --out-dir runs/s1_{flash,pro}` 鈫?`run_s2_transfer --in-dir runs/s1_{flash,pro}`銆?

---

## 5. 绱Н缁撹

1. **妗嗘灦鎸夎鐐瑰伐浣?*锛歚H* = f(pattern, task)` 宸插疄璇佲€斺€斾笉鍚?(pattern脳task) cell 鏀跺埌**涓嶅悓**鐨勬渶浼樻祦姘寸嚎锛圠3 闀胯窇 + 搂4.3 鐪熷疄鏁版嵁锛夛紝涓旀瘡涓?cell-scoped 缂栬緫澶╃劧 Pareto 瀹夊叏銆?*杩涗竴姝ワ細鍒ゅ畼涔熷閫犳渶浼樻祦姘寸嚎 鈫?H\* = f(pattern, task, judge)**锛埪?.3 frozen vs chronos 閫夊嚭涓嶅悓绠楀瓙锛夈€?*璺ㄤ换鍔?C1 鐜版湁涓ゆ潯鐙珛璇佹嵁**锛歸insorize(forecast鈫攁nomaly 绗﹀彿缈昏浆锛屄?.2/4.3) + **denoise_stl(forecast鈫攃lassify 绗﹀彿缈昏浆 + 鍚?cell 鏈€浼樺弽杞紝搂4.4)**鈥斺€斿浠诲姟 data-readiness 涓诲紶鎴愮珛銆?
2. **evaluator / optimizer 鍒嗙鏈夋晥**锛歱roposer锛堝悓涓€ fixed LLM锛夊嚭澶氭牱 cell 鐗瑰紓鍊欓€夛紝**grounded 鏄敮涓€瑁佸垽**锛屽彧鎺ュ彈鐪熸敼鍠勨埀娉涘寲鈭areto 瀹夊叏鑰呫€倃insorize(鍓婂嘲)琚嫆銆乨enoise_stl 琚帴鍙椻€斺€斿垽鍒纭紱鐪熷疄鏁版嵁涓婃纭嫆 held_out_a 鍥炲綊锛埼斺垝2.467锛夈€?
3. **frozen+probe 鏄繀瑕佹潯浠?*锛歡rounded 榛樿搴曞骇纭畾鎬э紙蟽_A=0锛屽疄娴?seed 鏃犲叧锛夛紝鍚﹀垯鎱㈣矾寰勮璁粌鍣０娣规病銆侰hronos-Bolt 鍚屾牱纭畾鎬?鈫?鍙綔鏇村己鍒ゅ畼鑰屼笉寮曞叆璁粌鍣０銆?
4. **~~褰撳墠闄愬埗 = headroom vs 蔚~~ 鈫?淇锛氶檺鍒舵槸鍒ゅ畼鍙楅檺锛岄潪鏁版嵁鍙楅檺**锛埪?.3 澶村彿鍙戠幇锛夈€俧rozen 鍒ゅ畼浠?minimal 浠?11%/6% accept锛屾浘璁颁负"heuristic 鍩虹嚎宸插濂?锛涙崲 Chronos 寮哄垽瀹樺悗鍚屽疄楠?22% accept銆?/4 cell 澶у箙鏀硅繘涓旂牬 naive 搴曗€斺€?*寮卞垽瀹樻劅鐭ヤ笉鍒版竻娲椾环鍊硷紙鑷韩澶╄姳鏉挎帺鐩栵級锛屽己鍒ゅ畼鎻ず鐪熷疄澶撮儴绌洪棿**銆?
5. **OPD 褰掑洜 = 璐ㄩ噺/鏀舵暃鏉犳潌**锛堥潪 accept 鏁伴摱寮癸級锛氬畾浣嶆洿鍑嗐€丩LM 鍙銆?1 cell 鏀舵暃锛涘己鍒ゅ畼涓嬪綊鍥犳洿閿愶紙chronos snrHigh\|miss prefer denoise_stl +0.665锛夈€?
6. **鐪熷疄璺ㄥ煙鎴愮珛锛圕3锛?*锛?2 淇″彿 / 3 寮傝川鍩燂紙寮卞懆 nn5銆佹湀搴?tourism銆佸己瓒嬪娍 fred锛夌粡 load_real 鍏ㄩ摼璺浂鏀瑰姩璺戦€氾紱6 cell 鍏ㄨЕ鍙戯紝鑷繘鍖栨仮澶?C1 鍦ㄧ湡瀹炴暟鎹笂澶嶅埢骞讹紙寮哄垽瀹樹笅锛夎秴瓒婂悎鎴愮粨璁恒€?

---

## 6. 鍚庣画璁″垝

| 浼樺厛绾?| 椤?| 鍔ㄦ満 | 钀界偣 |
|---|---|---|---|
| 鉁?**P0** | ~~鎺㈡洿澶?headroom 鐨?cell锛堢湡瀹炶法鍩熼敋锛墌~ **宸插畬鎴?*锛埪?.3锛夛細鐪熷洜涓嶆槸鏁版嵁 headroom 灏忥紝鑰屾槸**鍒ゅ畼鍙楅檺**锛涙帴 Chronos 寮哄垽瀹樺悗浠?minimal 涔熻澶уご閮ㄧ┖闂?| 鈥?| `data/load_real.py` + `run_real_longrun.py` + `chronos_probe.py`锛堝凡钀藉湴锛?|
| 鉁?**P0** | ~~蔚 鎸?Chronos 灏哄害閲嶆爣瀹殈~ **宸插仛锛堥厤瀵圭増锛?*锛毼?鍚?batch cur vs cand 閰嶅 蟽_螖锛堥潪鏃犻厤瀵?batch 鏂瑰樊锛岄偅杩囦及~3脳锛夆啋 frozen鈮?.03(鍘熷€兼湰瀵?/chronos鈮?.08锛涗粎瓒嬪娍 cell 鐪熼珮鍣緟 per-cell 蔚銆俙--eps` | `run_calibrate_eps.py` / `config/thresholds` |
| **P0** | **Chronos 鍒ゅ畼澶?epoch 闀胯窇** | 寮哄垽瀹樺ご閮ㄧ┖闂村ぇ 鈫?鍊煎緱璺戞洿澶?epoch 鐪?per-cell 鏀舵暃 + OPD 鍔犻€?| `run_real_longrun --substrate chronos --epochs 鈮?` |
| **P1** | **LLM-in-loop 楠岃瘉** | L1 鑷敱鏂囨湰 prompt/constraint 缂栬緫鍦?heuristic 楠岃瘉涓?螖=0 涓嶅彲 accept锛涜楠岃瘉椤?LLM compose 杩涢獙璇佺幆璺紙鎴愭湰鏉冭　锛?| `slow_path/validator` 鍙€?LLM compose 璺緞 |
| **P1** | **鏇撮暱澶?epoch 闀胯窇** | 瑙傚療 OPD 褰掑洜鏄惁鍔犻€?per-cell 鏀舵暃锛堝叾鐪熷疄浠峰€艰酱锛?| `run_evolve_longrun`锛堝姞 epoch锛?|
| **P2** | calibration 鎺ュ叆 validator 鈫?鏈夋潯浠堕噸鍚?proxy 棰勭瓫 | code-review #2 鐣欑殑鍙ｅ瓙锛沺roxy 浠呭湪 cell calibration 蟿 璇佸彲淇℃椂鍚敤 | `validator` + `evaluators/calibration` |
| **P2** | retrieval 鎺ョ湡瀹炲 domain 閿?| 楠?C3 / 鏆栧惎鍔ㄨ法鍩?| `data/load_real` + `memory/retrieval` |
| **P3** | 鏁堢巼锛?7 retrieval 鎸佷箙 kNN 绱㈠紩 / #8 consolidate 浠呭彉鍖栨椂鍐?| 浠呰瘉鎹銆侀暱璺戜箙鎵嶇棝锛涗笉鎻愬墠浼樺寲 | `memory/retrieval` / `slow_path/merger` |
| **P3** | winsorize 榛樿 `limits=0.05` 鍋忔縺杩涳紙鍓婁俊鍙峰嘲锛夊彲璋?| L2 瑙傚療鍒?winsorize 鍓?amp-1 瀛ｈ妭宄?| `config` / `operators/s1_outlier` |

### 宸茬煡鎶€鏈€猴紙闈為樆濉烇級
- `EditPatch.value` 涓?dataclass 鏃?`to_dict/from_dict` 涓嶈兘 round-trip锛坮eplay 璧板唴瀛樺璞★紝瀹¤ JSON 澶熺敤锛夈€?
- SNR struct_feat 浠嶅亸寮憋紙鑼冨洿绐勶級锛涚ǔ鍋ョ増澶熷垎绠变絾闈炵悊鎯充俊鍙?鍣０浼拌銆傜湡瀹炲己瓒嬪娍鏁版嵁 `period` struct_feat 琚秼鍔夸富瀵硷紙涓嶅奖鍝?binning锛夈€?
- classify grounded锛圛nceptionLite锛夐殢鏈猴紝S_SEEDS=2 浠呴儴鍒嗗帇鍣紱闀胯窇灏戣窇 classify銆?
- **搂4.3 鏂板浠ｇ爜锛坄load_real.py` / `chronos_probe.py` / forecast `ensemble`+`seasonal_resid` 鐩爣 + `set_forecast_substrate`锛夌粡鐪熷疄闀胯窇绔埌绔獙璇侊紝浣嗗皻鏃犱笓鐢ㄥ崟娴?*锛?0 suite/62 鐢ㄤ緥鏁版湭鍙橈紱鐜版湁 evaluators/slow_path 娴嬭瘯浠嶅叏杩囷紝鍥犻粯璁?substrate=frozen/target=raw 鏈彉锛夆啋 寰呰ˉ `test_real_data` / `test_chronos`銆?

---

## 7. 娴嬭瘯娓呭崟锛?3 suite / 87 鐢ㄤ緥锛宲roject env 鍏ㄨ繃锛?

| suite | 鐢ㄤ緥 | 瑕嗙洊 |
|---|---|---|
| test_harness_core | 12 | EditPatch 鏍￠獙閾?/ apply / snapshot+restore / replay / from_dict 瀹归敊 / (f)绀轰緥 / conditioning_key |
| test_fast_path | 7 | 閲岀▼纰?ready+EvidenceRecord / C1 浠诲姟鍒嗘祦 / Skill Gate 寮哄埗 / 涓夌骇 fallback / identity / 搴忓垪鍖?|
| test_evaluators | 8 | frozen 蟽_A=0 / grounded 鍒ゅ埆 / anomaly 骞虫粦浼ゅ / calibration / classify / 娲惧彂 / Role B / fast_path鈫攅val 闆嗘垚 |
| test_llm_client | 5 | 浠ｇ爜/JSON 鎶藉彇 / 缂撳瓨鍛戒腑 / 妯″瀷鍒悕 / **live API smoke** |
| test_schedule | 6 | edit_budget cosine / 鍐荤粨 / 鎺ュ彈閲嶇疆 / 瑙ｅ喕+寮烘 / round_idx 鎸佺画 |
| test_slow_path | 5 | validator 鎺ュ彈/鎷掔粷 / Evolver 鎺ュ彈+鐗堟湰鍓嶈繘锛堝嚭鍙ｅ垽鎹級 / 鍐荤粨 / mining |
| test_proposer_live | 2 | **鐪?DeepSeek** 鍑哄悎娉?EditPatch / 鍠?validator |
| test_phase2 | 7 | distance 鍗曡皟 / kNN 妫€绱?鍐峰惎鍔?/ failure 鑱氬悎 / compose_llm(stub) / active_operators 绾︽潫 / 鏆栧惎鍔ㄩ泦鎴?/ **live compose** |
| test_templates | 6 | template from_dict / compose 鐢ㄥ尮閰嶆ā鏉?/ cell-scoped ban / Skill cell-scoped / proposer 瑙ｆ瀽妯℃澘 / validator Pareto 瀹夊叏 |
| test_attribution | 4 | 纬 / ops_credit 绗﹀彿 / store value+summary / evolve 闆嗘垚 |
| test_real_classify | 6 | ROCKET 蟽=0 + 鍒嗘槗鍒嗛泦 / substrate 娲惧彂 / classify disjoint / 鎶ュ憡鍣?NaN 瀹归敊 / classify 璇枡宸ュ巶 / ECG 缂撳瓨 smoke(鍏嶄笅杞? |
| test_deploy_stream | 7 | 鈽卾4 S1锛氫笁 mode 璺戦€?JSONL / frozen 涓嶆敼 H / lazy 閲嶉獙闄嶇骇 / reset-free 璺ㄥ煙鎼哄甫 / enter_new_domain 瑙ｅ喕+閲嶇儹 / registry 涓夎鑹插垎绂?/ readiness 搴﹂噺 |
| test_forward_transfer | 12 | 鈽卾4 S2 + bug 鍥炲綊锛歴tl 榛樿 period=0 鐪熻窇 / denoise_stl 瀹?'auto' / savgol 鎺ュ彈 order(#3 鍋囬槼鎬у绾? / deploy_stream 缂?proposer 鎶?ValueError(#2) / per_domain 鑱氬悎 / verdict 鍔╃泭+閫€鍖?discriminative+鎶ゆ爮 / NaN-file 寰€杩?|

---

## 8. 澶嶇幇鎸囧崡锛圧eproduction锛?

**缁熶竴鍓嶇紑**锛坈onda `project` 鐜锛宼orch+CUDA锛沜hronos 鐢ㄧ洿杩?HF 鍕胯闀滃儚锛夛細
```bash
# git-bash / bash锛?
AGENT="C:/Users/杈?Desktop/Agent"; PY="D:/Anaconda_envs/envs/project/python.exe"
export PYTHONPATH="$AGENT" HF_HUB_DISABLE_SYMLINKS_WARNING=1 PYTHONIOENCODING=utf-8
# PowerShell锛?env:PYTHONPATH="C:/Users/杈?Desktop/Agent"; $env:HF_HUB_DISABLE_SYMLINKS_WARNING="1"
```
涔嬪悗鎵€鏈夊懡浠?= `$PY -m SelfEvolvingHarnessTS.<module> <args>`銆?

| # | 瀹為獙 | 鍛戒护 | 浜у嚭 / 鐢ㄩ€?| LLM | 鏃堕棿 |
|---|---|---|---|---|---|
| T | 鍏ㄦ祴璇曪紙12 suite/75锛?| `... -m SelfEvolvingHarnessTS.tests.test_evaluators`锛堝強 test_slow_path/test_templates/test_real_classify/...锛?| 鍥炲綊 | 鍚?| s |
| **R0** | **鍏?LLM 璇婃柇**锛坈ell 鍒嗗竷 + 缂栫爜鍣ㄨ〃鐜?nRMSE vs floor锛?| `...run_real_longrun --mode diag` | Step 1a锛歝ell 鍒嗗竷銆佺紪鐮佸櫒杩佺Щ璇婃柇 | 鍚?| ~30s |
| R0' | 璇婃柇路鐪熷疄缂栫爜鍣紙leave-signal-out锛?| `...run_real_longrun --mode diag --encoder real --npz SelfEvolvingHarnessTS/data/_artifacts/monash_clean.npz --encoder-cache SelfEvolvingHarnessTS/evaluators/_artifacts/frozen_lstm_real_h64.pt` | E2 缂栫爜鍣ㄥ鐓?| 鍚?| ~2s* |
| R0'' | 璇婃柇路chronos 鍒ゅ畼 | `...run_real_longrun --mode diag --substrate chronos` | chronos 鍩虹嚎 nRMSE 鐮?floor | 鍚?| ~1min |
| **R1** | **Step 2 鎭㈠**锛坉egraded 璧凤紝frozen 鍒ゅ畼锛?| `...run_real_longrun --mode evolve --start degraded` | 鑷富鎭㈠璇佹嵁 | 鏄?| ~12min |
| **R2** | **Step 1b C1**锛坢inimal 璧凤紝frozen 鍒ゅ畼锛?| `...run_real_longrun --mode evolve --start minimal` | C1 鍒嗗寲 + 璇氬疄鍐荤粨 | 鏄?| ~12min |
| R1c/R2c | 涓婁袱鑰吢穋hronos 鍒ゅ畼 | `... --start {degraded,minimal} --substrate chronos --eps 0.08` | 寮哄垽瀹樼増锛堢牬 naive 搴?+ Ours-evolved 螖Perf 琛岃嚜鍔ㄩ殢 `--report-readiness`锛?| 鏄?| ~5鈥?5min |
| **R3** | **涓昏〃**锛坒orecast锛夛細7 鍥哄畾鍙樹綋 + per-cell oracle + single-best | `...run_main_table --judge chronos --seeds 2` | C1 涓婄晫 + raw/minimal/degraded 螖Perf | 鍚?| ~1鈥?min |
| **R5** | **classify 涓昏〃**锛埪?.4 绗簩鏉?C1锛夛細ECG5000 + ROCKET 鍒ゅ畼 + InceptionLite 鎶ュ憡鍣?| `...run_main_table --task classification --max-signals 300 --final-size 30 --seeds 2` | classify per-cell oracle + stl 璺ㄤ换鍔＄鍙风炕杞紙瀵圭収 R3锛?| 鍚?| ~6鈥?min |
| R5' | ECG5000 缂撳瓨棰勫缓锛堥娆★紝鏃犲垯 R5 鑷姩瑙﹀彂锛?| `...data.load_ecg5000` | `data/_artifacts/ecg5000.npz`锛圲CR 瀹曗啋TF 浜屽垎绫诲洖閫€锛?| 鍚?| ~10s |
| **R6** | **classify 鑷繘鍖栭暱璺?*锛堥樁娈礏锛屄?.4 鑷繘鍖栧潡锛夛細minimal 璧?+ ROCKET 鍒ゅ畼 + 鐪?LLM | `...run_classify_longrun --start minimal --epochs 3 --max-signals 300` | cell-scoped 杞绘竻娲楁ā鏉?+ OPD 褰掑洜 + Ours-evolved 螖Perf | 鏄?| ~14min |
| R4 | 蔚 閰嶅鏍囧畾 | `...run_calibrate_eps` | 鍚?batch cur/cand 蟽_螖 鈫?蔚锛坒rozen鈮?.03/chronos鈮?.08锛?| 鍚?| ~1min |
| D | 涓嬫洿澶у共鍑€璇枡锛圚F 鐩磋繛锛?| `$PY -c "from SelfEvolvingHarnessTS.data.load_real import build_real_npz; build_real_npz('SelfEvolvingHarnessTS/data/_artifacts/monash_clean.npz', configs=['nn5_daily','fred_md','tourism_monthly','covid_deaths','us_births','saugeenday','sunspot'], per_config=20)"` | 83 淇″彿/7 鍩?| 鍚?| ~30s |
| **S1** | **鈽卾4 娴佸紡鎸佺画閫傚簲 + 涓?bootstrap**锛堝悎鎴?demo锛屽厤 LLM锛?| `...run_stream_s1 --epochs 2 --out-dir runs/s1` | A/B/C 涓?mode 鍓嶅悜杩佺Щ JSONL + summary.json | 鍚?| ~2min |
| S1' | S1路鐪?DeepSeek proposer | `...run_stream_s1 --llm flash --epochs 3` | 鐪?LLM 杩涘寲鐨勫墠鍚戣縼绉?| 鏄?| ~10min |
| S1'' | S1路鐪熷疄 domain 娴侊紙鎸?source 鍒?domain锛?| `...run_stream_s1 --npz SelfEvolvingHarnessTS/data/_artifacts/monash_clean.npz --llm flash` | 鐪熷疄娴?B鈭扐/C鈭払 鏇茬嚎锛圫2 杈撳叆锛?| 鏄?| ~15min |
| **S2** | **鈽卾4 鍓嶅悜杩佺Щ鏇茬嚎**锛堣 S1 JSONL 鈫?鏇茬嚎+鍒ゆ嵁+鍥撅級 | `...run_s2_transfer --in-dir runs/s1`锛坉emo 鐢?`--in-dir runs/s1_demo`锛?| per-(mode,k) 琛?+ headline 鍒ゆ嵁锛圕 vs A銆丅鈭扐/C鈭払銆佹姢鏍忋€乨iscriminative锛? `s2_transfer.json`/`.png` | 鍚?| ~5s |
| W | **涓?Chronos 鍒ゅ畼鏉冮噸**锛堥娆★紝HF 鐩磋繛锛涘凡灏辩华鍒欑鍥烇級 | `$PY -c "from SelfEvolvingHarnessTS.evaluators.chronos_probe import get_chronos; get_chronos()"` | `~/.cache/huggingface/hub/models--amazon--chronos-bolt-small`锛垀48M锛岀‘瀹氭€у垽瀹?鎶ュ憡鍣級 | 鍚?| ~15s |

> `*` R0' 鐪熷疄缂栫爜鍣ㄩ璺戜細棰勮+缂撳瓨锛堥娆＄◢涔咃級銆侺LM 闀胯窇榛樿缂撳瓨鍦?`llm/_cache/real_{start}.json`锛堝悓 cache_name 閲嶈窇鍛戒腑鈫掔绾э級銆?
> 鏃ュ織褰掓。锛歚SelfEvolvingHarnessTS/_real_step{1b,2}{,_chronos,_chronos_table}.log`銆乣_dl_corpus.log`銆?
> **鏈€灏忓鐜颁富瀹為獙**锛堝崐澶╋級锛歊3锛堜富琛?C1 涓婄晫锛屽厤 LLM锛夆啋 R1c+R2c锛坈hronos 鑷繘鍖栵紝鍑?Ours-evolved 琛岋級鈫?鎷兼垚 搂4.3 / Experiment_Design 搂鈽?鐨勪富琛ㄣ€?




> **Stage 2 FastPathAblation reference/support correction (2026-07-09)**: `fast_path/ablation.py` now reports both reporter-native `mean_utility_delta_vs_raw` and derived `mean_lift_vs_raw_arm` using the same-uid `raw` arm as the ablation reference. This fixes a readout ambiguity where `raw` means executable `v_none`/impute-linear baseline, while the reporter's `raw` reference is the degraded input series. The refreshed 4-record no-API synthetic slice still has 32 results and `api_calls=0`, but all non-raw arms have slightly negative `mean_lift_vs_raw_arm`; therefore this slice validates the pipeline/report contract, not Memory/Composer superiority. `slow_path/evidence_miner.py` and `slow_path/promotion.py` now track `n_unique_cases`/`utility_positive_case_count`/`harm_case_count` and use independent source-uid support for proposal thresholds. This prevents repeated ablation arms over the same uid from inflating support. With 2 records, slow-path proposals are correctly blocked; with 4 records, proposals remain 2 and are backed by `n_unique_cases >= 2`.
> Follow-up in the same stage: `DeploymentEvidenceMiner` also reports `mean_case_utility_delta_vs_raw` and `mean_case_harm_delta_vs_raw`; `MemoryWrite` proposals now use these case-averaged values rather than arm-row-weighted means. Row-level means remain diagnostic only.

> **Stage 2 OracleLedgerAblation no-API replay (2026-07-09)**: `run_fast_path_ablation.py` now supports `--slice oracle-ledger`, which reads the fixed `results/Stage2/S2_replication/records_s2.jsonl` L_test ledger only inside the downstream validator (`ledger_l_test_oracle_v1`). Fast-path packet construction still consumes only pattern/skill/memory/action-menu surfaces; memory rows are built causally from prior same-cell ledger cases, excluding the current uid. Full replay command: `D:\Anaconda_envs\envs\project\python.exe -m SelfEvolvingHarnessTS.run_fast_path_ablation --slice oracle-ledger --n-records 672 --out-dir SelfEvolvingHarnessTS\results\Stage2\FastPathOracleLedgerAblationFull`. Result: 672 records x 8 arms = 5376 results, `api_calls=0`, slow-path proposals=12, all 12 are `ProposeRiskRule` after conflict gating. Arm readout by `mean_lift_vs_raw_arm`: `composer_skill` +0.001821 with harm 0.117378; `deterministic_router`/`skill_only`/`skill_memory_deterministic` -0.017089 with harm 0.067200; `memory_only_selector` -0.080781 with harm 0.306238; `composer_skill_memory` and `composer_skill_memory_safety` -0.080545 with harm 0.306352; raw=0. Safety rejects in deterministic/skill-only arms come from `candidate_abstain_to_raw` and `unknown_skill`; composer+memory no longer has `unknown_skill` after the stub composer was fixed to attach only registry skills that support the selected action. Conclusion: the fixed ledger replay validates the ablation/ledger interface and strongly argues against promoting current memory/composer policy as a deployment improvement; the immediate next gate is risk-rule/scope refinement and memory conflict handling, not real LLM/API.
