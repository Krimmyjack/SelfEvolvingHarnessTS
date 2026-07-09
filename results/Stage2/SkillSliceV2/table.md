# 切片 v2：信息面/推理/粒度三方消歧（六臂；prereg_skill_slice_v2.md）

| arm | cum (5 排列均值) | 升级块数 | 升级块 Δ vs frozen (mean/max) | G1 |
|---|---|---|---|---|
| A_frozen | 0.3677 | 0 | +0.0000 / +0.0000 | ✅ |
| C_llm_v2 | 0.3964 | 80 | +0.0287 / +0.3426 | ❌ |
| D_llm_v2 | 0.3897 | 40 | +0.0440 / +0.3426 | ❌ |
| C_llm_verify | 0.3979 | 80 | +0.0302 / +0.3426 | ❌ |
| Bplus_v2 | 0.2976 | 80 | -0.0702 / +0.1182 | ❌ |
| D_bplus_v2 | 0.3362 | 40 | -0.0630 / +0.1182 | ❌ |

G2 vs A: D_llm_v2: +0.0188 [-0.0072,+0.0432]→False | Bplus_v2: -0.0730 [-0.1088,-0.0402]→True | D_bplus_v2: -0.0338 [-0.0625,-0.0082]→True
**G-main（触发块 LLM−B+）**: +0.1047 [+0.0613,+0.1555] → LLM 独立价值 不成立（=注册预测）
G3 S_both 首遇: [{"perm": 0, "frozen": 0.6708, "C_llm_v2": 0.7587, "C_llm_verify": 0.7728, "Bplus_v2": 0.3405}, {"perm": 1, "frozen": 0.612, "C_llm_v2": 0.8225, "C_llm_verify": 0.8225, "Bplus_v2": 0.2613}, {"perm": 2, "frozen": 0.6708, "C_llm_v2": 0.7587, "C_llm_verify": 0.7728, "Bplus_v2": 0.3405}, {"perm": 3, "frozen": 0.612, "C_llm_v2": 0.8225, "C_llm_verify": 0.8225, "Bplus_v2": 0.2613}, {"perm": 4, "frozen": 0.6708, "C_llm_v2": 0.7587, "C_llm_verify": 0.7728, "Bplus_v2": 0.3405}]
诊断: {"llm_failures": {"v2": 0, "verify": 0}, "verify_violations": 0, "v2_window_requests": 4, "trigger_instance_share": "40/80=0.50（G4 教训：实例级）"}
