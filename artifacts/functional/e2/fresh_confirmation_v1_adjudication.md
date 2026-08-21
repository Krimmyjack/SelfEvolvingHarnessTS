# fresh_confirmation_v1 主线裁定附录(0 评估)

日期: 2026-08-21。本附录不修改 `fresh_confirmation_v1.json` / `.md` 任何字节;
所有裁定均为对已打开读数的零评估重判读,经外审(sol)复核一致。

## 判定

```text
raw_instrument_verdict(pooled)      = A5_WINS(维持)
raw_instrument_verdict(per_channel) = A5_WINS
adjudicated_verdict(per_channel)    = A5_TIE_TRANSFER_BOUNDARY
reason = 预注册 WINS 与 TIE 条款同时成立且未规定优先级;首正成本差 3 次重训
         系噪声量级,按不利于己方向裁定;总判定不敏感。
overall_verdict = FRESH_A5_DELIVERS(维持)
```

## 措辞修正(对外引用以本节为准)

1. 曝光口径:本 cohort 不是"方法开发从未接触的域"。NOAA family 有旧线报告
   (AGGREGATE_SEEN),20 站曾被退役 screening 扫描(无存留读数)。准确表述:
   在未被本版本 Source Skill 编译与方法开发使用 downstream treatment outcome 的
   NOAA 20 站 cohort 上,held-out 2025 outcome 被一次性打开验证。
   它是 fresh Outcome,不是全新 Domain。
2. 安全口径:零次复用越过 aggregate −0.005 harm 门;但 pooled 确认中两臂
   均出现 1 条受害序列(99999904140,−0.1256,被聚合 +0.0297 掩盖)。
   不得表述为"没有任何伤害"。
3. 两钥匙结构原始计数:3 次低信噪比晋级(g/se 1.09/0.29/0.56)→ 5 次复用尝试
   被拦(3 次当窗确认、2 次确认通过但 v2 门拦)→ 1 次合法放行且保持正向。
   "无门必受害"为已测读数上的反事实推断,非独立消融。
4. 总成本口径:99 vs 195 为真实测量,被"只在 task_A 形成 Draft"协议规则放大;
   协议补全后的乐观估计 ≈ 99 vs 144(57+66+6+15,含探针成功前提),非实测。
   承重读数为不受该规则影响的首正成本 69 vs 123(−43.9%)。

## 对外 Claim 上限(canonical)

在一次冻结的 NOAA held-out 2025 确认中,Source-derived Guidance 使 pooled
Harness 以少 43.9% 的 Consumer 重训达到首个 delayed-positive Workflow;最终
held-out utility 与 harm 与冷启动相同。per-channel Consumer 上无可测迁移优势,
构成迁移边界。真实 Skill store、检索、Target-local 持久化、晋级、召回与当窗
确认路径均被执行。

不得声称: A5 提升最终质量;所有 Consumer 都有迁移收益;完整 Harness 自进化
已闭合;没有任何逐序列伤害;已证明一般跨域泛化。
