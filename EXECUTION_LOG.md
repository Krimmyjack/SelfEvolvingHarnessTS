# SelfEvolvingHarnessTS — 执行历史日志（EXECUTION_LOG）

> 只读、追加式的研究/实现日志，从原 `BUILD.md` 头部不断增长的同步记录块拆分而来（2026-07-13）。**这里只记录"发生过什么、当时的结论是什么"，不代表当前状态**——当前状态以 `BUILD.md`（怎么构建/运行）+ `../idea/README.md`（当前研究结论）+ 项目 memory 的 AdaCTS 当前状态整合记录为准。新条目继续往本文件末尾追加，不要回填修改历史条目（如需勘误，新增一条注明勘误对象）。
>
> 内容顺序：先是 Phase 0/1 时代（2026-06-20~23）的早期实验记录，然后是 Stage 0/1/2（2026-07-05 起）的同步日志——后半部分保留原文件的原始顺序（该部分本身是新条目陆续插在顶部形成的，不是严格时间正序，未重新排序，避免引入转录错误）。
>
> **已知问题**：本文件（包括原 BUILD.md 迁移过来的这部分内容）存在历史遗留的字符编码损坏——部分中文显示为乱码（如"鈥?锛?等），是文件本身字节层面的问题，不是渲染问题。已确认是 GBK/UTF-8 误转导致，理论上可通过编码转换大部分还原，但自动还原会引入新的错误替换（已实测约 10% 残留错误），风险大于收益，故本次未做自动修复，原样保留。如需要，可另开一个专门任务做人工校对式修复。

---


## Phase 0/1（2026-06-20~23）早期实验

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


---

## Stage 0/1/2 同步日志（2026-07-05 起）

> **鈿?2026-07-05 鐘舵€佹敞璁?*锛氭湰鏂囧疄鐜扮姸鎬佸仠鍦?2026-06-21锛涘叾鍚?Stage 0锛堢閬撲慨澶?鏃ュ織鍩哄缓锛変笌 Stage 1锛堣韩浠藉垽鍐筹細E-1.1 鍥涜疆/S0.7 绠楀瓙璇氬疄鎬?F0 鍓傞噺鎵弿/E-3.2 鍏噦/confirmatory seeds 20鈥?9锛夊凡鍏ㄩ儴鎵ц瀹屾瘯锛?*279 娴嬭瘯杩?*銆傜粓灞€鍒ゅ喅瑙?`results/STAGE1_VERDICT.md`锛圥attern 鏉′欢鍖栬矾鐢?level-1 纭珛锛汣6=妯″瀷鏃犲叧澧炵泭璇佷吉锛夈€傚綋鍓嶅叆鍙?= `../idea/Component_Optimization_and_Integration_Plan.md`锛圫tage 2 缁勪欢浼樺寲涓庣郴缁熷悎娴侊細缂洪櫡娓呭崟 D1鈥揇10銆丳atternSpec/RouterPolicy 鍚堟祦銆佸紶閲忋€佹洿鏂伴棴鐜級銆?
>
> **Stage 2 critical-review sync锛?026-07-07锛?*锛氬綋鍓嶈韩浠戒笌鎵ц椤哄簭浠?`../idea/Current_Vision_Scheme_and_Experiment_Setting_2026-07-07.md`銆乣../idea/Critical_Project_Review_and_Redesign_2026-07-07.md` 鍜?`../idea/Reference_Project_Audit_and_Transfer_2026-07-07.md` 涓烘渶鏂扮患鍚堣瀹氥€傛柊澧炵绾胯瘉鎹細`results/Stage2/BatchScan/report.json` 涓哄彂鐜伴泦鎵弿锛宍P0_kmeans` 鐣ヨ儨 `P1b_kmeans`锛坥racle agreement 0.3333 vs 0.3289锛泈ithin-batch response variance 4.0923 vs 4.4216锛夛紝鍥犳 **P1b 涓嶈浆姝?*锛沗results/Stage2/C1Lite/report.json` 涓?`P1b-memory` 鑳?frozen/random-memory 浣嗚緭 `P1b-static`锛坮egret 0.3424 vs 0.2976锛変笖 first-unseen harm +0.1072 瓒呭畨鍏ㄧ嚎锛屽洜姝?**Memory 涓嶄綔涓虹嫭绔嬫満鍒惰浆姝?*銆俙results/Stage2/ReadinessAdversaries/` 浠庡喕缁?`S2_replication/records_s2.jsonl` 鏋勯€犻潪 API銆侀潪鑷姇鍠傜殑 oracle-actionable 鏍囩锛坮aw=`v_none`锛?72 rows锛沷racle actionable rate 0.929锛夈€傜粨鏋滐細`dp_abstain` 鐩稿 raw 闄?mean regret锛?.2762 vs 0.4005锛変笖 gain_vs_raw=+0.1243锛屼絾 recall 浠?0.604銆乭arm_rate=0.391锛沗P1b-static` regret=0.2965銆乬ain_vs_raw=+0.1040銆乭arm_rate=0.344銆傜粨璁烘槸锛氬凡鏈?policy 鏈夋暟鎹瓫閫変环鍊硷紝浣嗗綋鍓嶈嚜閫傚簲杩樹笉瀹夊叏锛涗笅涓€姝ュ繀椤讳紭鍏堝仛 support/uncertainty gate 鍜?harm calibration锛岃€屼笉鏄棤杈圭晫 24h LLM 闀胯窇銆侭1b proposer 浠嶆槸椤圭洰韬唤闂細鍙湁 LLM proposer 鍦ㄥ悓涓€寮€鏀?`ProgramSpec` 绌洪棿銆佸悓棰勭畻銆両TT no-op 涓?worst-group safety 涓嬭儨杩?deterministic search锛屾墠鍙啓鎴?LLM-driven harness evolution銆傚綋鍓?immediate order锛欱1b 韬唤闂ㄤ紭鍏堬紱Pattern-Batch 浠呭仛 fresh-namespace confirmatory锛沠ull M0-M3 Memory promotion 鏆傚仠鍒?support/escalation gate 鏄庣‘銆?
>
> **Stage 2 safety / Pattern-Batch / EvidencePacket sync锛?026-07-08锛?*锛氬凡鎸?2026-07-08 璁″垝钀藉湴涓変欢浜嬶紝鍧囦笉璋冪敤 LLM/API銆傗憼 `evaluators/safety_gate_lite.py` + `run_safety_gate_lite.py` 鐢熸垚 `results/Stage2/SafetyGateLite/`锛歚dp_abstain` harm=0.391/gain=+0.1243锛沗abstain_to_raw` harm=0.220/gain=+0.0974锛況outer-support q50 harm=0.095/gain=+0.0841/serve_frac=0.329锛宷75 harm=0.141/gain=+0.0963/serve_frac=0.470锛宷95 harm=0.201/gain=+0.0941/serve_frac=0.583銆傜粨璁猴細鏀寔搴﹂棬鎺ф湁鏁堥檷 harm锛屼絾浠嶆槸绂荤嚎鎶樹腑锛屼笉鏋勬垚閮ㄧ讲瀹夊叏澹版槑銆傗憽 `evaluators/pattern_batch_scan.py` + `run_pattern_batch_scan.py` 鐢熸垚 `results/Stage2/PatternBatchConfirmatory/`锛歭egacy_cell oracle_agreement=0.222/response_var=6.2844锛孭0_kmeans=0.333/3.9828锛孭1b_kmeans=0.265/6.2072銆傜粨璁猴細P0 杩炵画 Pattern-Batch 鏄庢樉浼樹簬 legacy锛孭1b 鍦ㄦ湰娆′弗鏍?10d P0 瀵圭収涓嬩笉杞銆傗憿 `policy/evidence_packet.py` 寤虹珛 `skill_memory_evidence_packet_v1`锛屾妸 Pattern 鎽樿銆丼kill cards銆丮emory 鎽樿銆丄ctionMenu meta銆乻afety constraints 鍥哄畾涓?LLM composer 鍓嶇殑鍙祴璇曡緭鍏ラ潰锛屽苟浠ュ崟娴嬬姝?`L_test`/oracle/arms/`X_t`/raw series 娉勬紡銆傜獎鍥炲綊锛歚test_safety_gate_lite` + `test_readiness_adversaries` + `test_pattern_batch_scan` + `test_evidence_packet` 鍏?14 passed銆?
> **Stage 2 slow-path validation / ablation runner sync（2026-07-08）**：在 code review 后补齐 slow-path 最小部署闭环与 fast-path 消融入口，仍不调用真实 LLM/API。`slow_path/evidence_miner.py` 现在从 `EvidenceRecord.conditioning_key.task`/cell 推断 task，不再硬编码 forecast，并输出可被现有 `policy.edits.MemoryWrite` 消费的 utility-bound payload；`slow_path/proposal_schema.py` 会把 `MemoryWrite` proposal 校验到 `MemoryWrite(EditOp)` schema，把 `ProposeRiskRule` proposal 校验到 `RiskRule.validate()`；新增 `slow_path/promotion.py` 的 `PromotionGate`/`ProposalValidationOutcome`，只做 validate + compile to `EditOp`，不自动 apply 到 `PolicyBundle`。同时新增 `fast_path/ablation.py`：`raw` arm 作为正常 `v_none` baseline 执行而非 fallback failure；支持显式 skill-surface override，使 memory-only arm 的 `EvidencePacket.skills=[]` 且 memory 保留。新增测试 `test_slow_path_promotion.py` 与 `test_fast_path_ablation.py`；当前 focused tests 为 slow-path evidence/proposal 5 passed、promotion 3 passed、ablation 2 passed。下一步从 skeleton runner 进入固定数据集上的 raw/deterministic/skill-only/memory-only/skill+memory/composer(+gate) 报表。
> **Stage 2 FastPathAblation no-API run（2026-07-08）**：新增 `run_fast_path_ablation.py` 与 `fast_path/ablation.py` 的 summary/report 输出。命令 `D:\Anaconda_envs\envs\project\python.exe -m SelfEvolvingHarnessTS.run_fast_path_ablation --n-records 4` 写入 `results/Stage2/FastPathAblation/report.json` 与 `records.jsonl`：8 arms × 4 synthetic forecast records = 32 results，`api_calls=0`。arm 顺序为 raw / deterministic_router / skill_only_deterministic / memory_only_selector / skill_memory_deterministic / composer_skill / composer_skill_memory / composer_skill_memory_safety。当前 run 只证明 no-API ablation pipeline、EvidenceRecord 写入和 report contract 可复现；默认 `role_b_proxy` 还不是 utility/harm reporter，因此不能用该小切片宣称 Memory/Composer 收益。下一步应接入固定 reporter 或小型 oracle slice，再把 EvidenceStore 输出送入 slow-path mining/promotion gate。
> **Stage 2 FastPathAblation utility/harm + slow-path mining sync（2026-07-08）**：`run_fast_path_ablation.py` 现在接入固定 `synthetic_oracle_proxy_v1` reporter，向 EvidenceRecord downstream 写入 `raw_loss_proxy`、`selected_loss_proxy`、`utility_delta_vs_raw`、`harm_delta_vs_raw`，并把同一 EvidenceStore 送入 `DeploymentEvidenceMiner -> PromotionGate`，输出 `slow_path_proposals.jsonl`。重新运行 4-record synthetic slice：32 results，`api_calls=0`，slow-path proposals=2，accepted=2。proposal 明细：① `forecast|snrHigh|full` 上 `v_median` mean utility=-0.000312 / harm=0.000312，生成 scoped `ProposeRiskRule` ban to `v_none`；② `forecast|snrLow|miss` 上 `v_median` mean utility=+0.063176 / harm=0，生成 `MemoryWrite`。review 发现 `v_none` 实际是 `impute_linear` baseline 而非 strict raw，因此 `slow_path/evidence_miner.py` 已修正为默认不把 `raw_action` 的正 utility 提升为 MemoryWrite，避免把 baseline 误当成 reusable skill evidence。该 run 仍是 synthetic proxy，不是论文级性能结论；它验证的是 ablation -> utility/harm evidence -> slow-path proposal/promotion gate 的闭环。
> **P0 contracts & hygiene sync（2026-07-09，Final_Plan_CodeAgentFirst §P0）**：按 `../idea/Final_Plan_CodeAgentFirst_2026-07-09.md`（code-agent-first 拍板 + P0–P6 计划）落地 P0 全部四件，全程 TDD、不调用 LLM/API、现任行为零扰动（全库 **494 passed** 无一失败）。① `policy/task_spec.py` 新建：TaskSpec/MetricSpec 一等公民（task_type/target_semantics/label_availability/metric/horizon/downstream_model_class/forbidden_modifications；sha 身份；forbidden 按 registry canonical 名判定），默认口径 forecast=nRMSE+dlinear_shared、classification=accuracy+rocket_ridge、anomaly=F1+residual_zscore_detector（P2 rig 预留）。② `policy/evidence_packet.py`：拆除 `"task": {"type": "forecast"}` 硬编码——新增 `task_spec` 参数（缺省=forecast_task_spec_v1()，与历史隐式口径一致），packet["task"] 保留 legacy `type` 键并携带完整 TaskSpec 字段，provenance 记 `task_spec_sha`。③ `policy/action_semantics.py` 新建：raw 语义三拆——`v_raw_identity`（严格恒等，空程序）/ `v_impute_linear`（`v_none` 的 canonical 语义名）/ `v_ledger_baseline`（历史台账口径别名）；`RAW_SEMANTICS_NOTE` 为 migration note（历史 gain_vs_raw/regret-vs-raw 参照物全部是 impute-linear baseline，此后不得混称 strict raw）；`raw_identity_action_spec()` 空 steps + `ActionCompiler.to_program` 空步显式分支 → 空 Program（不走 compose heuristic，防"空模板静默合成插补链"的语义漂移）；冻结 menu v1 不注入新动作（SHA 不变）。④ `policy/program_edit.py` 追加 **ProgramSpec grammar v1**（v0=B1b 冻结面 bit 级不动）：`ProgramSpecV1`（task_type 显式 + pattern_guard 白名单特征 P_FEATS∪{snr,missing_rate} + risk_budget_beta∈[0,1] + max_modified_fraction≤β + fallback∈menu∪语义名 + prog1_ SHA 身份/chain_sha 去重键）、`validate_v1`（v0 机械规则全保留 + registry allowed_tasks 按任务过滤，anomaly 物理禁平滑/删改）、`guard_matches`（缺特征 fail-loud）、`check_execution_invariants`（保长 + 观测点修改率预算；NaN 填补不计入 distortion）、`to_action_spec_v1`（defaults⊕override 完整 resolve）。新增测试 40 条（test_task_spec 10 / test_action_semantics 7 / test_program_spec_v1 18 / test_evidence_packet +2）。预注册骨架落 `results/Stage2/prereg_codeagent_first_P1_P5.md`（ε 占位待 P3 种子供给校准后注册；P5 confirmatory seeds=40–59 一次性，20–39 已被 STAGE1 消耗不复用）。下一步=P1：EvidencePacket v2（连续证据+trace）、CodeAgentComposer 默认接线（stub+cached-DeepSeek 双后端、ITT）、SafetyGate 后接与按面 harm 台账。
> **P0 code review + P1 code-agent-first wiring sync（2026-07-09）**：①对 P0 diff 做 8 角度 code review（git 仓库实际不存在，环境头误报；按会话内 diff 审查），5 findings 全部处置：validate_v1 窗参数改**精确 int 判定**（int(w) 隶属判定会放行 9.0/"9" 后在真实执行烧预算）、ProgramSpecV1.sha() 归一到 resolved_budget（None 与显式=β 同身份，防 action_id 级重复执行）、_known_fallbacks 加 lru_cache（menu 冻结可缓存，validate_v1 是 gym 热点）、去掉 check_execution_invariants 冗余 isfinite 项、ActionCompiler.to_harness 对空步 spec 不再注册永不匹配的 stages=[] 死模板。②**P1 全落地（TDD，25 条新测，全库 519 passed）**：`policy/evidence_packet.py` 增 **packet v2**（`skill_memory_evidence_packet_v2`=v1+continuous_evidence[数值 fail-loud，bool/非数值/NaN 拒收=R1 禁二值读数]+trace_summaries[泄漏 lint]+allowed_grammar[单一真源=program_edit 常量]）；`policy/code_agent_composer.py` 新建 **CodeAgentComposer**（stub=确定性规则合成 no-API/CI 安全/bit 级复现，llm=缓存优先 DeepSeek 经 with_deepseek()，ITT：invalid/empty/不可解析→candidate=None 且 api_calls 照记，输出只能是携带 ProgramSpec v1 的 TypedCandidate）；`policy/escalation.py` 增 `EscalationConfig.composer_first`（默认 False=旧行为 bit 级不变；True=code agent 默认上场+packet v2 输入面）、`EscalationDecision.program_action`、SafetyGate program 候选路径（spec_v1_from_dict→validate_v1→guard_matches，拒绝原因 invalid_program_spec/program_grammar_rejected/pattern_guard_unsatisfied/pattern_guard_feature_missing，novel program 跳过 menu 成员检查）、compile 直编译分支（reason=`compiled_program_spec_v1`）；`program_edit.py` 增 spec_v1_to_dict/from_dict（严格反序列化 fail-loud）。③`run_p1_codeagent_first.py` 验收 runner：三臂（raw / incumbent_control / code_agent_first_stub）× 8 records no-API 切片落 `results/Stage2/P1CodeAgentFirst/`（manifest+report+records.jsonl），**按面 harm 台账**（baseline_raw/router/program/gate_fallback）上线，stub 双跑 bit 级一致（复现出口判据），code-agent 臂 8/8 服务 prog1_ 程序端到端执行；raw 臂正 utility(+0.039) 再次实证 v_none=插补基线非 strict raw。**本 run 仅接线/复现验收，非性能主张**（synthetic proxy 非论文判官）。P1 出口判据①②③④⑤全过。下一步=P2：最小 anomaly rig（一周硬封顶）+ 动机表（复用 classify C1 符号翻转与 F0 season harm，新增 anomaly 臂）。
> **P2 anomaly rig + 动机表 sync（2026-07-09）**：TDD 全程（10 新测，全库 **529 passed**），no-API。① `evaluators/anomaly_rig.py` 新建（老 grounded_anomaly.py 保留不动）：注入协议（point amp=6.0 + contextual level-shift 2.5×len5，位置互斥、margin/min_sep 纪律）+ 冻结检测器（**leave-one-out** rolling-median(w=11) 残差鲁棒 z-score，NaN 安全，threshold=4.0）+ F1/AUROC 判官（命中容差 ±ANOM_TOL=2；AUROC=严格正类 vs 严格负类、容差环排除）。**两个 rig 设计教训入 DETECTOR_SPEC 注释**：(a) 窗含中心点会自吸收残差 → MAD 低估 ~27%、z 虚高、假告警成簇（首轮实测 21 个假 flag），故基线只用邻居中值；(b) 告警尺度若随 artifact 自标定，平滑把残差与 MAD 一起压扁 → z 反而爆炸、recall 假性保持（首轮实测 96 flag），故 **scale 在 raw 输入上冻结标定**（scale_calibration=frozen_on_raw_input，运营告警语义）；另修 `_pick_positions` count=0 逻辑洞（曾注入 16 段=81 标签）。判官冻结（frozen=True，改参数=新判官身份）。② escalation gate 新增 **task_forbidden_op**：packet 携带的 TaskSpec.forbidden_modifications 成为活语义（canonical 名比较，registry allowed_tasks 之外的实例级收紧），`decide_fast_path` 增 task_spec 透传。③ `run_p2_motivation.py` 动机表（论文实验 1）：同一批 40 序列（anomaly|snrHigh|full ×20 + anomaly|snrLow|miss ×20，历史含注入异常）双任务判官——forecast=seasonal-naive h24 nRMSE（对干净未来）、anomaly=冻结 z-score F1（raw 上标定）；6 程序行 × Δ vs **v_raw_identity**（P0 语义纪律）× 90% bootstrap CI（B=1000）×部署契约列。**正式结果（results/Stage2/P2Motivation/，seed=20260709）**：median_w9 fc **+0.972**[+0.766,+1.160] / an **−0.642**[−0.689,−0.596]；winsor +0.667/−0.586；universal_cleaner +0.973/**−0.848**；task_conditioned +0.716/±0.000（无害侧保全）；v_impute_linear 本切片 ≡ raw（miss 块未落 seasonal-naive 尾窗，两列 Δ=0——记录为切片特性非 bug）。**翻转判据（mean>ε=0.01 ∧ CI 同侧）3 程序全过 → fresh 任务对（forecast×anomaly）=1 + frozen 引用对（forecast×classification，`_clf_maintable.log`：v_median +0.249/+0.134 助 classify、v_stl −0.037/−0.110 伤 classify 而 stl/savgol 助 forecast）=1 → 出口判据（≥2 任务对）PASS**；契约列显示翻转已被 D6 物理编码（median/winsor 在 anomaly 下 deploy=N）。**motivation-grade 合成切片，非 confirmatory**；anomaly rig 就此冻结（一周封顶纪律，rig=判官非研究对象）。下一步=P3：TS-Readiness Replay Gym（保真度硬门）+ 种子供给/ε 校准。
> **P2 code review + P3 gym/保真度/headroom/ε sync（2026-07-09）**：①P2 review 3 findings 全处置（注入容量不足 fail-loud ×2、缺失块无合法起点可读报错、delta 回填 O(n²) 重扫改直接引用）。②**P3 全落地（TDD，19 新测，全库 548 passed，no-API）**：`policy/seed_programs.py` **skill bank v1**（8 条冻结种子：forecast 6=period/fft/ema 系插补×{stl,median,ma,savgol}+winsor 复合两剂量、anomaly 2=period_complete/impute_fft 单步；全部 grammar 合法+menu v1 不可表达[`is_novel_v1` tests 守]；prereg 草案中不存在算子按红线剔除）；`readiness_gym.py` **TS-Readiness Replay Gym v1**（0-API：proxy_eval/finalize/abstain、预算、结构化 trace[R1]、true 只落 result 永不进 observation[泄漏 tests 守]、invariants 报 trace 不拒绝[拒绝语义归 P4 SafetyGate，β 计数/幅度空间待重校准——设计注记]）；`run_p3_gym.py` 判决 runner。**③三个方法教训（全部实测抓获并修复/记录）**：(a) 单折回测考卷噪声 → **rolling-origin K=6** 折均值（within ρ 0.35→0.57）；(b) **pooled 保真度是 Simpson 陷阱**（pooled −0.32 而候选均值层完全单调对齐、within +0.35）→ 主判据改 **within-series 排序保真**；(c) 近简并平局海制造 ρ=−1.0 假象 → distinct<5 不计、覆盖率<0.5 判 insufficient_variance。**④正式判决（results/Stage2/P3Gym/，n=60，B=1000，ρ_min=0.70 预注册）**：forecast within ρ=**0.5699 FAIL**（<0.70，不做门槛购物）；anomaly deployable 面 **insufficient_variance**（合法面=插补类属预期）；anomaly 违约诊断组 pooled ρ=**0.8504 PASS**（proxy 有牙齿）→ **两任务 escalate-only：gym-proxy 只可作搜索中间信号，P4 validator/P5 identity gate 验收一律 true 判官**（R4 硬门拦住了 C5 式 proxy 盲点的重演——门起作用了）。**headroom=0.0000（0/60）机制查明**：seasonal-naive 判官只读尾部两周期 → 插补轴不可见、种子塌缩 menu 等价物（seed_period_median9≡f0_median_w9 逐值相同）——**substrate 局限性记录，不否证真实 corpus 供给价值**（S1 witness ΔL1+0.046 在全序列判官下）。**ε=0.02 正式注册**（max(0.02, ci90_lo=0) 触发下限）；prereg §3 冻结、§4/§5 挂 true-判官 binding 后果。P3 出口四件全过（gym 0-API✓/保真度报告✓/skill bank v1✓/ε 注册✓）。下一步=P4：慢路径闭环（typed EditOps→PromotionGate→**true 判官** held-out validator→versioned PolicyBundle+rollback）。
> **P4 慢路径完整晋升周期 sync（2026-07-09）**：TDD 全程（10 新测，全库 **558 passed**，no-API）。①`slow_path/bundle_store.py` 新建：PolicyBundle 持久化（bundle↔dict 往返带 sha 一致性核验[篡改 fail-loud]、版本 artifact **不可变禁覆盖**、chain.json=HEAD+append-only 事件流、rollback 只移 HEAD 不删版本[审计可回放]）。②`slow_path/true_judge_validator.py` 新建：**true 判官双段 validator**（P3 binding 兑现：forecast=seasonal-naive vs 干净未来、anomaly=冻结 z-score F1，proxy 不进验收）；serving 全程走部署消费面 `apply_edits→compile_bundle→RiskAwareRouterPolicy`（§13.4 焊点，无平行实现——Critical Review §4 知识断路教训）；四判据=held-in 方向∧held-out≥ε=0.02（P3 冻结值）∧per-cell worst-group≥−0.05∧非目标行 bit 级不变；`SubstrateRouterPolicy`=v0 现任（F0 时代剂量启发式 snrLow→w25，在季节 substrate 已知错误=被改进对象，真实证据驱动非摆拍）。③`run_p4_promotion.py` 完整周期：held-in 挖掘→枚举 proposer 3 条 SlowProposal（§13.2 arm①好坏都提裁决在判官）→PromotionGate→true 判官验证→apply_edits 版本升级→BundleStore 落盘→回归重放→rollback 演示→rejected buffer。**正式结果（results/Stage2/P4Promotion/，n=60，held-in/out=30/30 按 cell 分层）**：晋升 `bundle_v0.e1`=`mined_ban_f0_median_w25_snr_low`（ban w25→w9 @ cell_snr=low）——held-in **+0.237**/held-out **+0.264**（≥ε=0.02，n=30，fired 15）、per-cell snrLow **+0.528**/snrHigh **0.0**（作用域干净）、回归重放目标行 +0.502×30 且**非目标行与 anomaly 面 bit 级零扰动**、rollback head→v0 后 serving 与原 v0 bit 级一致再恢复晋升头（chain.json 2×save+2×rollback 全留痕）；reverse 规则以零效应（v0 在 low 本就服务 w25，ban w9 永不触发→held-in +0.0000）、harmful 规则以 held-in **−0.374** 被拒进 rejected_edits.jsonl。**七环闭合=cycle_complete**（typed EditOp/gate/双段验证/版本落盘/回归/回滚/拒绝缓冲）；**机制验收非性能主张，"self-evolving" 一词继续锁定至 P6**；Memory M0–M3 按 prereg §4 条件线显式挂起（risk-veto 自 P1 起已在 gate 活跃）。P4 出口判据全过。下一步=P5：identity gate 六臂（confirmatory seeds 40–59 一次性）+ pattern-vs-domain 四象限 + safety 收口。
> **P5 正式判决包 sync（2026-07-10）**：prereg §5.0 开跑前冻结（δ_safe=0.05/K_novel=3/主臂=ca_skills 先验声明/B=3/四象限轴/safety 一次性协议），**confirmatory seeds 40–59 一次性消耗**；7 新测，全库 **565 passed**。三 runner：`run_p5_identity_gate.py`（checkpoint/resume[A-36]、ITT、grouped bootstrap by seed、CountingClient 成本台账）、`run_p5_quadrant.py`（真 Monash 12 基底×4 preset、series_uid 泄漏守卫）、`run_p5_safety.py`（P4 production bundle 重载+sha 核验）。**判决三连（results/Stage2/P5Verdict/VERDICT.md）**：①**P5-A claim 分支 = self-updating deterministic with LLM-optional**——ca_skills−det = **−0.799** CI90[−0.940,−0.651]（504 真实 DeepSeek 调用/1173s，四判据全败）；**机制归因：接口可靠性之败非生成质量之败**——ca_plain 180/180 malformed（无示例=格式塌缩）、ca_skills 179/180 死于 pattern_guard_feature_missing（**harness 契约不一致缺陷实测抓获**：allowed_grammar 宣传 guard 特征而 gym 观测面不提供；LLM 模仿 skill 卡提交 guarded 程序=合理行为被 100% 拒）、ca_skills_memory served-conditional **+0.852 ≥ det +0.799**（n=18 观察性）→ serving-LLM L1 合规塌缩在 proposer 席重现，check13/14 精化为"瓶颈在合规/接口可靠性非生成质量"；retrial=P5-A.2 须新预注册+新种子（guard 契约一致化+格式合规面），本轮不重跑。过程中实测抓获并修复 guard None-值 TypeError（LLM-in-the-loop 才触发的路径，2 回归测试守）。②**P5-B 冻结轴上假设被推翻（诚实负结果）**：dd/sp regret **0.885** vs sd/dp **0.170**，配对差 +0.715 CI90[+0.369,+1.090] 反向不跨零——退化结构 pattern 轴上 **domain 完胜 pattern**（domain 携带 period/季节/尺度等内在结构；与 E-1.1/E-3.2 自洽：degradation-only 不够、P0 结构特征才承重）；论文禁写"pattern 胜 domain"，主张降级为内在结构轴+更大语料的新预注册实验（不做轴购物）。③**P5-C PASS**：bundle_v0.e1 confirmatory 一次性——coverage 33.3%/gain +0.131/harm 3.3%/worst-cell LCB +0.255≫−δ_safe/anomaly 面 bit 级零扰动；S2 sealed holdout 显式延期为独立注册访问。**P0–P5 主张阶梯更新**：任务条件化✓/safety-gated serving+晋升✓（substrate 级）/慢路径机制✓（self-evolving 仍锁定）/pattern 胜 domain✗（退化轴负结果）/LLM-driven evolution 未确立（分支 B+可修复瓶颈已定位）。下一步=P6（条件性：多周期累积才解锁 self-evolving）或 P5-A.2 retrial 预注册。
> **P5-A.2 retrial + 外评采纳首批 sync（2026-07-10）**：`prereg_p5a2_retrial.md` 开跑前冻结（唯一两处变更=guard 契约一致化[gym_fingerprint_v2：P0 指纹从观测序列现算进 observation+guard 同源评估]+格式合规面[prompt_v2_exemplar+repair_retries=1]；**seeds 60–79 一次性消耗**，缓存隔离 p5a2_identity）。**形式判决：分支 B 维持**——ca_skills−det=**−0.150** CI90[−0.242,−0.077]（549 composer 调用/1363s），但判据③首次达标 **novel_effective_edits=21**。**接口归因完全验证**：ca_plain **180/180 malformed → 0/180**（60/60 served +0.705，示例修复格式塌缩）、guard 拒绝从 feature_missing 全部变为 guard_unsatisfied（契约生效；残余 62 候选=LLM 写 guard 过紧的真实判断差）、主臂 gap **−0.799→−0.150（81% 败因是接口）**、served-conditional +0.687 vs det +0.780。**诚实注记：仍非终审**——外评核查证实 runner `packet=dict(obs)` 绕过 packet v2，连续证据（R1）从未喂给 LLM（slice v2 证明的翻盘杠杆）→ **终审=P5-A.3**（外评路线图⑤：真实 Monash+packet v2 契约真源+ReadinessPlan→deterministic compiler 臂+外部基线+四指标分解，**seeds 80–99 新预注册**）。**协议碰撞透明留痕**：并行会话外评路线图原拟 seeds 60–79 做更大 retrial，本会话按先冻结的窄版 prereg 消耗之——两审各自协议有效并列入档，路线图⑤顺延 80–99。**勘误（外评②）**："真实调用"字段=composer 调用次数含缓存命中，归档 run 未分离；runner 已接 `client_stats`（n_api/n_hit）自 P5-A.3 起落盘。**安全（外评①代码侧执行）**：`llm/client.py` 硬编码 fallback key 删除→环境变量+网络前 fail-loud（缓存重放不受影响；budget 测试改显式假 key；live smoke 无 key 自动 skip）；**🔴 平台侧旧 key 轮换作废=用户动作，未完成**。过程中修 LLM-only 路径 bug：guard None 值 TypeError→KeyError fail-safe（2 回归测试）。全库 **570 passed+1 skipped**。下一步=外评路线图 ③packet v2 进 runner ④ReadinessPlan→compiler 臂 ⑤P5-A.3 终审预注册（seeds 80–99）；P6 命名锁定至 ≥2 轮真实 promotion 累积。
> **外评二审采纳 + P5-A.3 机器建成 sync（2026-07-10）**：①二审全部数字**逐一复核精确成立**（novel 去重 21次→11 distinct SHA[7×/5× 重复]、跨臂 random 41/31 / ca_plain 23/23 / ca_skills 21/11 / memory 9/7 / det 13/1、尾部 +1.032 vs +0.892、per-cell 0.581/0.727 vs 0.715/0.909）→ VERDICT 措辞修正落盘：**"deficit 缩小 81%"**（非"81% 严格由接口造成"；跨 seed 批+双变更→强受控归因非 factorial）、判据③改 **SHA-distinct 去重口径**（11≥3 仍稳过）、正确结论="**LLM candidate supply 存在**"非"优于随机"、**臂序诊断**（ca_plain +0.705 > ca_skills +0.630 = skill cards 负向锚定[过度 guard/模板模仿/多样性塌缩 23→11 distinct]→强证 packet v2 必要性：问题是"知识以何种证据形式提供"）、命名勘误（anomaly|*=series family 非 task）。②**路线图③④⑤机器全建成（TDD，9 新测）**：`policy/readiness_plan.py`=**ReadinessPlan→deterministic compiler**（LLM 只出语义意图，合规按构造保证——穷举 property 测试：4 impute×2 clip×6 family×3 strength 全部编译过 validate_v1；guard 只留有背书谓词、丢弃/任务过滤全留痕=semantic 与 compliance 可分离计量）+ PlanComposer（stub/llm+repair，ITT）；`run_p5a3_final.py`=**architecture-complete confirmatory** runner：真实 Monash（12 信号×presets，seeds 80–99 每 seed 3 episodes）、**packet v2 契约真源**（R1 兑现：continuous_evidence 从 P5Quadrant 真实 records 聚合，同 preset 排除同 series_uid=LODO 防泄漏）、7 臂（frozen/random/det[固定基线]/pv2_direct/**pv2_plan_compiler=PRIMARY**/no_ce[R1 增量]/no_skills[ablation]；外部 code-agent 基线显式延期、closed-menu 臂因 w5 不在 grammar 网格记录为设计事实）、**四指标分解**（semantic=guard/step 丢弃台账、compliance=有效率+repair 计数、**selection regret=决策后离线评估全部候选 true delta 只入报告不回流**、harness benefit=主对比+worst-group per preset）、series_family/pattern_preset 命名字段、checkpoint/resume、client_stats。`prereg_p5a3_final.md` 冻结（判据③=SHA-distinct ≥3；败则 headline="self-updating deterministic harness with **LLM novelty supplier**"）。③**正式 run 被 preflight 硬门挡住（by design）**：`--backend llm` 强制先过 preflight（新 key 注入+n_api/n_hit 分离核验）；🔴 旧 key 平台撤销+新 key 注入=用户动作，完成后一条命令开跑。
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



> **Stage 2 FastPathAblation reference/support correction (2026-07-09)**: `fast_path/ablation.py` now reports both reporter-native `mean_utility_delta_vs_raw` and derived `mean_lift_vs_raw_arm` using the same-uid `raw` arm as the ablation reference. This fixes a readout ambiguity where `raw` means executable `v_none`/impute-linear baseline, while the reporter's `raw` reference is the degraded input series. The refreshed 4-record no-API synthetic slice still has 32 results and `api_calls=0`, but all non-raw arms have slightly negative `mean_lift_vs_raw_arm`; therefore this slice validates the pipeline/report contract, not Memory/Composer superiority. `slow_path/evidence_miner.py` and `slow_path/promotion.py` now track `n_unique_cases`/`utility_positive_case_count`/`harm_case_count` and use independent source-uid support for proposal thresholds. This prevents repeated ablation arms over the same uid from inflating support. With 2 records, slow-path proposals are correctly blocked; with 4 records, proposals remain 2 and are backed by `n_unique_cases >= 2`.
> Follow-up in the same stage: `DeploymentEvidenceMiner` also reports `mean_case_utility_delta_vs_raw` and `mean_case_harm_delta_vs_raw`; `MemoryWrite` proposals now use these case-averaged values rather than arm-row-weighted means. Row-level means remain diagnostic only.

> **Stage 2 OracleLedgerAblation no-API replay (2026-07-09)**: `run_fast_path_ablation.py` now supports `--slice oracle-ledger`, which reads the fixed `results/Stage2/S2_replication/records_s2.jsonl` L_test ledger only inside the downstream validator (`ledger_l_test_oracle_v1`). Fast-path packet construction still consumes only pattern/skill/memory/action-menu surfaces; memory rows are built causally from prior same-cell ledger cases, excluding the current uid. Full replay command: `D:\Anaconda_envs\envs\project\python.exe -m SelfEvolvingHarnessTS.run_fast_path_ablation --slice oracle-ledger --n-records 672 --out-dir SelfEvolvingHarnessTS\results\Stage2\FastPathOracleLedgerAblationFull`. Result: 672 records x 8 arms = 5376 results, `api_calls=0`, slow-path proposals=12, all 12 are `ProposeRiskRule` after conflict gating. Arm readout by `mean_lift_vs_raw_arm`: `composer_skill` +0.001821 with harm 0.117378; `deterministic_router`/`skill_only`/`skill_memory_deterministic` -0.017089 with harm 0.067200; `memory_only_selector` -0.080781 with harm 0.306238; `composer_skill_memory` and `composer_skill_memory_safety` -0.080545 with harm 0.306352; raw=0. Safety rejects in deterministic/skill-only arms come from `candidate_abstain_to_raw` and `unknown_skill`; composer+memory no longer has `unknown_skill` after the stub composer was fixed to attach only registry skills that support the selected action. Conclusion: the fixed ledger replay validates the ablation/ledger interface and strongly argues against promoting current memory/composer policy as a deployment improvement; the immediate next gate is risk-rule/scope refinement and memory conflict handling, not real LLM/API.

> **Stage 2 Phase4 MemoryV2 FastPathAblation（2026-07-09）**: stage4 upgrades the no-API ablation matrix from 8 arms to 10 arms by separating `positive_memory_only`, `risk_memory_only`, and `positive_risk_memory`, while preserving `raw`, deterministic/skill-only, composer+skill, composer+skill+memory, and composer+skill+memory+SafetyGate controls. `fast_path/ablation.py` now supports per-arm `memory_mode`, reports `serve_fraction`, `fallback_fraction`, `abstain_fraction`, `safety_reason_counts`, `safety_evidence_ref_counts`, `worst_cell_mean_utility_delta_vs_raw`, and emits strict JSON (`NaN`/`inf` -> `null`). `run_fast_path_ablation.py` now builds `MemoryEvidenceV2` utility/risk rows, uses causal prior same-cell ledger memory, makes the stub composer prefer `utility_memory`, and abstains to raw on risk-only packets. Smoke runs: synthetic 4 records -> 40 results, `api_calls=0`, slow-path proposals=2; oracle-ledger 8 records -> 80 results, `api_calls=0`, slow-path proposals=2. Full replay command: `D:\Anaconda_envs\envs\project\python.exe -m SelfEvolvingHarnessTS.run_fast_path_ablation --slice oracle-ledger --n-records 672 --out-dir SelfEvolvingHarnessTS\results\Stage2\FastPathOracleLedgerAblation_Phase4_full`; result: 672 records x 10 arms = 6720 results, `api_calls=0`, slow-path proposals=12. Full replay readout (`mean_lift_vs_raw_arm`, `mean_harm_delta_vs_raw`): `composer_skill` is slightly positive (+0.001821, harm 0.117378); deterministic/skill-only are negative (-0.017089, harm 0.067200); `positive_memory_only` and `positive_risk_memory` are strongly negative (-0.080781, harm 0.306238); `composer_skill_memory` and `composer_skill_memory_safety` remain strongly negative (-0.080545, harm 0.306352); `risk_memory_only` abstains/falls back on 663/672 cases with zero lift/harm. Conclusion: stage4 validates the MemoryV2 ablation/reporting interface and gives stronger evidence that current memory selection/conflict handling is the main failure mode. It does not validate real LLM value. Next gate is risk-memory conflict handling and scoped RiskRule promotion, then retest composer identity under the same action space; real LLM/API remains gated.

> **P5-A.3 final trial executed (2026-07-10, user-run after platform key rotation)**: `python -m SelfEvolvingHarnessTS.run_p5a3_final --preflight` PASS (n_api=1/n_hit=1) then `--backend llm` consumed **seeds 80-99 one-shot** under frozen `results/Stage2/prereg_p5a3_final.md`. Integrity: 420 records (7 arms x 60 episodes), resumed=0, 757 real API calls / 0 cache hits / 787 composer calls / 4615.7s wall, budget 1500 not hit. **Headline: claim_branch = self_updating_deterministic_with_llm_novelty_supplier** - pv2_plan_compiler - det_search = **-0.177 CI90 [-0.261, -0.097]**; criteria (1)(2) FAIL, (3) PASS (4 distinct novel-effective SHA; `fe670ade089e` effective 7x across families/presets up to +5.92), (4) PASS. Four-metric decomposition pins the residual on **generation**: supply ceiling (oracle over own pool vs det chosen) -0.114 CI90 [-0.206,-0.027]; compliance solved (compiler semantic drop rate 0; 20/180 invalids all upstream, ITT); pv2_direct regret 0.003 is a pool-collapse artifact (47/60 episodes single-distinct candidate). Exploratory: negative-anchoring double ablation (no_skills/no_ce both above primary, pool ceilings ordered no_ce > no_skills > primary); **random_valid beats the fully-informed LLM +0.242 CI90 [+0.116,+0.368]** and its pool ceiling exceeds det (+0.158, CI>0) while realized +0.065 n.s. -> random's bottleneck is selection (proxy regret 0.093), the LLM's is generation. Verdict: `results/Stage2/P5A3Final/VERDICT.md`; cross-ref appended to `results/Stage2/P5Verdict/VERDICT.md`; plan section 6 updated. Logging gaps queued for P6: persist candidate-level spec dicts, split llm_error parse-vs-network, effect-equivalence novelty classes. Post-run diagnostics script: scratchpad `analyze_p5a3.py` (grouped-by-seed bootstrap, exploratory label).
