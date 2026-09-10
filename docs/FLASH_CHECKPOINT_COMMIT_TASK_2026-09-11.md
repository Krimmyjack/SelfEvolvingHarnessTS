# 给Flash：只保存当前进度并做一次选择性Git提交

用户授权原意：“commit交给flash，后续的内容我还要和你探讨”。
**唯一交付是当前代码/文档/结果的本地Git检查点；不启动、不设计、不实现下一轮实验。**

## 工作目录与现状

`C:\Users\辉\Desktop\Agent\SelfEvolvingHarnessTS-deepseek-guidance-evolution`
（WSL同目录：`/mnt/c/Users/辉/Desktop/Agent/SelfEvolvingHarnessTS-deepseek-guidance-evolution`）。
先读工作区及项目AGENTS。Astra最后检查HEAD为`c845b4b`，暂存区空；你上手时再确认，
其他执行线的未提交改动属于用户，不能reset、覆盖、删除或擅自一并纳入。

TRAIN-2已完成，无需恢复：新旧最终60次决定有效；新Skill全部identity，收益0，
旧Skill−0.164690，形成段固定winsorize→FFT +0.849895。不是稳定自进化正结果。
形成段两条失败、原始Slow响应缺失和序列化恢复已在结果报告披露，不抹除。

## 要做的事

1. 读`docs/DEV_TRAIN2_RESULT_2026-09-11.md`和状态页置顶，核对状态明确写为完成。
   后续设计全部标为待讨论；尤其TRAIN-3文件不得被当作执行授权。
2. 检查下面的提交范围，确认没有凭据、数据集、巨型缓存或其他线未完成工作。
   如需数字核对，仅运行`python _scratch/dev_train1/audit_train2_result.py`：0 LLM/0 fit，
   不带`--write`，不重跑研究实验。没有必要做全仓测试或重新生成历史工件。
3. 逐路径暂存，不使用`git add .`或`git add -A`。检查暂存diff/stat和文件名后做**一次本地commit**。
   建议消息：`Checkpoint TRAIN development and DEV-TRAIN-2 results`。
4. 返回commit短ID、文件数、主要保存内容、仍未提交的目录/文件类别。确认没有push。

## 提交范围

本轮TRAIN实现及已完成的直接支持代码：

- `evaluation/main_protocol_p4/dev_train1_evaluator.py`
- `evaluation/main_protocol_p4/dev_train1_runner.py`
- `evaluation/main_protocol_p4/dev_train1b_observations.py`
- `evaluation/main_protocol_p4/dev_train2_workflow.py`
- `evaluation/main_protocol_p4/run_dev_train1.py`
- `evaluation/main_protocol_p4/smoke_dev_train1.py`
- `evaluation/main_protocol_p4/run_dev_deploy1.py`
- `evaluation/main_protocol_p4/run_dev_deploy2.py`
- `tests/main_protocol/test_dev_deploy_feedback_integrity.py`

设计/现状/报告（检查后保存当前内容，不把提议改成批准）：

- `AGENTS.md`已有的Skill主线、共享训练及取消35%批准记录；不新增方法规则。
- `docs/STATE_ONE_PAGE_2026-09-03.md`
- `docs/SLOW_FEEDBACK_UPDATE_DESIGN_NOTES_2026-09-08.md`
- `docs/MAINLINE_ROADMAP_AND_DEV_TRAIN1_TASK_DRAFT_2026-09-09.md`
- `docs/DEV_TRAIN1_RESULT_2026-09-09.md`
- `docs/DEV_TRAIN1B_RESULT_2026-09-10.md`
- `docs/DEV_TRAIN1B_ROLE_OBSERVATIONS_TASK_2026-09-10.md`
- `docs/DEV_TRAIN2_UNCAPPED_WORKFLOW_SKILL_TASK_2026-09-10.md`
- `docs/DEV_TRAIN2_RESULT_2026-09-11.md`
- `docs/GROK_CODE_REPAIR_RESULT_2026-09-09.md`
- `docs/GROK_REPAIR_AND_ARCHITECTURE_REVIEW_TASK_2026-09-09.md`
- `docs/FRAMEWORK_WALKTHROUGH_2026-09-09.md`
- `docs/SKILL_SURFACES_AND_SELF_HARNESS_AUDIT_2026-09-09.md`
- 本任务书。
- `docs/DEV_TRAIN3_GENERAL_GUIDANCE_TASK_2026-09-11.md`可以作为**未批准讨论草案**保存，
  必须保留顶部授权更正，不执行其内容。
- `artifacts/main_protocol/dev_train2_closeout.json`：精简真实读数、逐决定及卡正文。

以下是被_scratch忽略、但应保留的少量源文件；只对列出的具体文件使用`git add -f -- <path>`，
**不要force-add整个_scratch目录**：

- `_scratch/dev_train1/launch_live.py`
- `_scratch/dev_train1/launch_train1b.py`
- `_scratch/dev_train1/launch_train2.py`
- `_scratch/dev_train1/prepare_train2_recovery.py`
- `_scratch/dev_train1/audit_train2_result.py`
- `_scratch/dev_train1/audit_train1b_roles.py`
- `_scratch/dev_train1/test_runner_contract.py`
- `_scratch/dev_train1/test_train1b_observations.py`
- `_scratch/dev_train1/test_uncapped_contract.py`
- `_scratch/dev_train1/test_train2_parallel.py`

路径不存在就如实记录，不自行补造。若同一路径上有正在进行的外部改动或明显无关内容，
暂不纳入该文件并说明；不要为凑齐名单把未经核对的工作一锅提交。

## 明确不做

- 不启动TRAIN-3、不调用Fast/Slow、不拟合模型、不阅读密封或新数据。
- 不改核心方法、数值、阈值、API配置、模型或代码逻辑；只有修正明显状态措辞需在回执披露。
- 不修改/提交Fable独占维护的`docs/DECISIONS.md`，本次其他Fable审查文档改动也先留工作区。
- 不提交`.dev_*_runs/`、`.aris/`运行目录、完整`_scratch/`、原始数据、模型缓存、Grok会话、
  `idea-stage/`、`refine-logs/`或其他未列出的历史恢复工件。
- 不删除任何本地材料，不清理脏工作区，不amend/reset/rebase，不改全局Git身份、不绕过hook、不push。
- 不新增SHA/哈希清单，不派子agent，不把这项归档工作扩成一次代码重构或大型审计。

原始运行数据继续留本地，commit只保存可复核的精简证据及代码/文档；如实说明它不是全量
本地缓存的异地备份。正常完成后停止，后续研究计划由用户和Astra继续讨论。
