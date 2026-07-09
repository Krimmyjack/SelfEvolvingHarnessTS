# LLM-Skill 四臂切片（16 半块 × 5 预锁排列；prereg_skill_slice.md）

| arm | cum (5 排列均值 [min,max]) | LLM 块数 | LLM 块 Δ vs frozen (mean/max) |
|---|---|---|---|
| A_frozen | 0.3677 [0.368,0.368] | 0 | +0.0000 / +0.0000 |
| B_wrapper | 0.3677 [0.368,0.368] | 0 | +0.0000 / +0.0000 |
| C_llm_all | 0.4059 [0.406,0.406] | 80 | +0.0382 / +0.3426 |
| D_escalation | 0.3943 [0.379,0.406] | 40 | +0.0532 / +0.3426 |

G1(C) mean≤0:False max<δ:False | G1(D) mean≤0:False max<δ:False
G2 D−A: +0.0232 [+0.0001,+0.0463] → False
G3 组合靶 S_both 首遇: [{"perm": 0, "block": 9, "llm": 0.7587, "frozen": 0.6708, "d_triggered": false}, {"perm": 1, "block": 3, "llm": 0.8225, "frozen": 0.612, "d_triggered": true}, {"perm": 2, "block": 11, "llm": 0.7587, "frozen": 0.6708, "d_triggered": true}, {"perm": 3, "block": 5, "llm": 0.8225, "frozen": 0.612, "d_triggered": false}, {"perm": 4, "block": 7, "llm": 0.7587, "frozen": 0.6708, "d_triggered": true}]
G4 D 触发 14/16 ≤60%: False | llm_failures=0
**判决**：G1 败：deployment LLM composer 在此信息面被拒；view log 仍收割
