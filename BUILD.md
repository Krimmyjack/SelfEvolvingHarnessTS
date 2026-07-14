# SelfEvolvingHarnessTS — 构建与实验文档（BUILD）

> 本文档 = 现在怎么构建/运行/测试这个项目。完整执行历史（P0–P6 逐阶段实现细节、每次 sync 的详细结论）已搬到 [`EXECUTION_LOG.md`](./EXECUTION_LOG.md)（只读追加式历史记录，不要在这里找"现在应该怎么做"）。
>
> **当前阶段状态**：P0–P6 已按 `../idea/Final_Plan_CodeAgentFirst_2026-07-09.md` 全部执行完毕（P5-A.3 终审 + P6 discovery engine 关账，claim = B-null）。当前研究状态的权威摘要见 `../idea/README.md`（文档阅读顺序）与项目 memory 中的 AdaCTS 当前状态整合记录。本文档只覆盖"如何构建/运行/复现"，不重复叙述研究结论。
>
> **canonical 代码库 = `SelfEvolvingHarnessTS/`**（对齐 `plan.md`）。旧 `SelfHarnessTS/` 是废弃的早期设计，其代码不再使用（仅 `exploration/` 的 E0–E4 实验结论与 POC 作参照）。

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


## 4. 实验记录（索引）

完整叙述（每次 run 的详细设置、教训、日志路径）已搬到 `EXECUTION_LOG.md`，按时间顺序追加，不在这里重复。这里只列关键里程碑方便定位：

| 里程碑 | 日期 | 一句话结论 |
|---|---|---|
| code-review（8 finding，修 5） | 2026-06-20 | 详见 EXECUTION_LOG「code-review」条目 |
| 真 DeepSeek 长跑 L1–L4（6 cell×3 epoch） | 2026-06-20 | cell collapse → 稳健 SNR → cell-scoped 模板解锁 C1 |
| 真实 Monash 跨域长跑（P0，验 C3） | 2026-06-20 | 判官能力门控头部空间；接 Chronos 强判官后 accept 率 6%→22%，破 naive 底 |
| classify 端到端（第二条独立 C1） | 2026-06-21 | denoise_stl 助 forecast 伤 classify——跨任务符号翻转 + 逐 cell 最优反转 |
| ★P4 S1：流式 domain 持续适应 | 2026-06-23 | reset-free 成立；updating 模式版本单调前进 |
| ★P4 S2：前向迁移曲线分析 | 2026-06-23 | demo 曲线重叠 → discriminative=False（诚实边界，非负结果掩饰） |
| Stage 0/1（E-1.1 四轮/S0.7/F0/E-3.2/confirmatory seeds 20–39） | 2026-07-02~05 | Pattern 条件化路由 level-1 确立；C6=模型无关增益证伪 |
| Stage 2 P0–P6（code-agent-first 全序列） | 2026-07-09~12 | P5-A.3 终审 = B 分支（败因=generation非选择）；P6 discovery engine 关账 = B-null |

---

## 5. 累积结论（构建相关不变量）

> 完整研究结论清单（含最新状态）以项目 memory 的 AdaCTS 当前状态整合记录 + `../idea/README.md` 为准。这里只保留对"怎么构建/复现"仍然成立、且贯穿全程的不变量。

1. **`H* = f(pattern, task, judge)`**：不同 (pattern×task) cell 收敛到不同最优流水线；判官强弱本身也改变最优流水线（frozen vs Chronos 选出不同算子）——复现任何实验时必须固定判官身份，不能跨判官比较结果。
2. **evaluator / optimizer 分离是硬约束**：proposer（含 LLM）只产生候选，grounded judge 是唯一裁判；任何"复现失败"排查应先确认 judge 没被意外换掉或降级为 proxy。
3. **frozen+probe（σ_A≈0 的确定性判官）是复现可比性的必要条件**；换判官（含升级到 Chronos/DLinear-closed-form）即视为新实验身份，不与旧数据横向比较。
4. **"headroom vs ε 受限" 的根因是判官能力，不是数据本身**——遇到 accept 率异常低时，先怀疑判官是否太弱，不要默认是数据/算子池不够。

---

## 6. 后续计划

> 旧的 P0–P3 优先级列表（Phase 0/1 时代）已在 Stage 2/code-agent-first 序列中全部执行完毕或被取代，不再是当前待办。当前活跃 backlog 见 `../idea/README.md` 阅读顺序表 + Exp-P（pattern-shift evolution）与 Benchmark v2/v3（`../idea/Benchmark_v0_Forecast_Design.md` + v3 addendum）两条主线的"下一步"小节；不在本文件重复维护，避免与 EXECUTION_LOG.md/idea/ 三处各自过期。

### 已知技术债（非阻塞）
- `EditPatch.value` 为 dataclass 时 `to_dict/from_dict` 不能 round-trip（replay 走内存对象，审计 JSON 够用）。
- SNR struct_feat 仍偏弱（范围窄）；稳健版够分但非理想信噪比估计。真实强趋势数据 `period` struct_feat 被趋势主导（不影响 binning，A0 已修正为 robust_v1）。
- classify grounded（InceptionLite）随机，S_SEEDS=2 仅部分压噪。

---

## 7. 测试清单

当前测试套件/用例数一直在增长（P0–P6 序列持续 TDD 新增），**具体数字以 `EXECUTION_LOG.md` 最新一条 sync 记录里报的"全库 N passed"为准**，不在本文件写死静态数字（这正是旧版 BUILD.md 长期过期的原因之一）。

套件命名与覆盖范围的规律：`tests/test_<模块或阶段名>.py`，一个新阶段（Stage/P-序号）落地时通常同时新增一个对应 suite；查某个功能测在哪个 suite，直接按模块名或阶段名搜 `tests/` 目录即可，不需要在本文件维护映射表。

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
