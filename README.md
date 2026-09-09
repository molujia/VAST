# VAST — 最终方案 B

VAST 是微服务故障根因定位研究代码。本版本固定采用 **完整历史特征 + 连续候选 latent + 内部 OSER 修正**，版本号 `1.0.0`。项目所有者已于 2026-09-09 审阅正式结果并选择方案 B。本仓库取代此前的 interim 离散增强快照。

这里发布源码、固定参数、运行入口和结果摘要。数据、具体查询计划、训练模型、逐例排名与原始实验输出需要外部提供。默认入口仅执行 B；历史实验中的 A、历史对照和 Strict LOFO 不是本版本的运行选项。

## 已认可的正式结果

实验标识：`vae-snapshot-rebase-seed42-20260908-01`。2026-09-09 02:01:28（北京时间）完成。下表均为 **B 的 OSER 后最终分数**，比例制、六位小数。

| 数据集 | 监督设置 | 测试案例 | Hit@1 | Hit@3 | Hit@5 | TOP135 | MRR |
|---|---|---:|---:|---:|---:|---:|---:|
| RCABench | Query-only | 427 | 0.742389 | 0.868852 | 0.889930 | 0.833724 | 0.812920 |
| RCABench | Oracle-full | 427 | 0.718970 | 0.899297 | 0.927400 | 0.848556 | 0.816168 |
| AIOps22-pre | Query-only | 72 | 0.680556 | 0.791667 | 0.930556 | 0.800926 | 0.766160 |
| AIOps22-pre | Oracle-full | 72 | 0.708333 | 0.875000 | 0.944444 | 0.842593 | 0.807078 |

对每例完整候选排名，取真实根集合中的最佳名次 `r`。`Hit@k = mean(r <= k)`，`TOP135 = (Hit@1 + Hit@3 + Hit@5) / 3`，`MRR = mean(1/r)`。评分以 `score` 降序、`raw_score` 降序、`entity_id` 升序排序；不能任意改变并列处理。

对应的历史完整 pairwise 对照及 B 的变化如下。Δ 是绝对比例差，不是相对百分比。

| 数据集 / 设置 | 历史 TOP135 | B ΔTOP135 | 历史 MRR | B ΔMRR |
|---|---:|---:|---:|---:|
| RCABench Query | 0.800937 | +0.032787 | 0.775278 | +0.037642 |
| RCABench Oracle | 0.836066 | +0.012490 | 0.804639 | +0.011529 |
| AIOps22-pre Query | 0.824074 | -0.023148 | 0.782568 | -0.016408 |
| AIOps22-pre Oracle | 0.847222 | -0.004630 | 0.823539 | -0.016461 |

**B 在 RCABench 上提升，在 AIOps22-pre 上的 TOP135 和 MRR 均下降。** 四组满足本次实验预设的 `ΔTOP135 >= -0.05` 容忍要求；容忍不等于提升，也不约束单独的 Hit 指标。AIOps Query Hit@3 相比对照下降 `0.069444`。选择 B 是所有者的最终取舍，不能据此声称它在所有数据集上优于 A 或历史方法。

这些结果来自固定普通划分、单个 seed 42，不提供多 seed 统计显著性、Strict LOFO、未见故障或新增服务泛化结论。OSER 是组合方法的内部组件，组合结果不能全部归因于 VAE。

## 方法与数据边界

1. **固定主动查询。** 在完整、无根因标签的外层训练故障池上构建多模态 `global_pca_dim32` 表示，使用 HDBSCAN leaf 聚类与 center 选择。RCABench 的 `min_cluster_size/min_samples = 5/5`，AIOps22-pre 为 `6/2`。两者采用 Euclidean 距离、medoid 中心、`allow_single_cluster=false`，查询预算固定 30，seed 42。没有 DBSCAN 回退。
2. **保留完整历史训练基础。** 使用打包的历史特征准备、正常基线、候选上下文特征和 pairwise-linear 排序器。保留原特征名称与顺序，使历史困难负例选择规则继续有效。根因传播关闭，训练采用 `fault_only`；正常窗口可参与历史特征基线，不作为新增故障监督。Query-only 仅使用选中的 30 例根因及故障类型标签，Oracle-full 使用全部外层训练故障标签。测试标签只用于计算指标。
3. **连续候选表示。** 因子化 CVAE 的机制、传播、上下文三个后验各有 16 维，拼接成每个候选独有的 48 维 latent。输入状态宽度分别为 11、6、10，带缺失模态掩码。CVAE 在完整外层训练故障池的无标签状态上训练，机制 triplet 项只读取已获准监督的根候选及故障类型。
4. **真实均值样本及后验变体。** 每个监督案例保留全部历史特征，附加后验均值；再生成 `K=8` 个 latent 变体。采样为 `mu + epsilon * exp(0.5 * clip(logvar, -12, 8))`，NumPy seed 42，保留候选、真实根和历史行。B 不通过 decoder 生成特征案例，也不把根因移到新服务。RCABench 为 `488 + 48 = 536` 维，AIOps22-pre 为 `168 + 48 = 216` 维。
5. **权重和归一化。** 真实案例的方向性正负 pair 权重保持不变；一个真实父案例的全部后验子案例合计 pair 质量为其真实 pair 质量的 `0.25`，不会随 K 增长。历史特征与 latent 分别标准化，两个 scaler 都只拟合真实监督案例的均值 pair 差，不在变体上重新拟合。底层分类器为历史 `liblinear` LogisticRegression。
6. **内部 OSER。** 使用可观测状态和基础排序分数进行有界残差修正。故障类型 episode 内隔离被留出的真实父案例及其所有子案例；外层 query 仅为真实案例，变体仅进入允许的 support。这里的 episode 隔离是训练正则化，不构成完整流程的 LOFO 评估。推断不接收根因或故障类型标签。

部分状态是特征代理：例如持续时间使用活跃时间戳计数，最早异常使用窗口内排名映射，拓扑深度使用入度。它们不是实测物理时间、传播距离或真实因果路径。精确映射见 `materialize_historical_states`。

| 数据集 | 外层训练故障池 | Query 监督 | Oracle 监督 | 外层测试故障 | Query / Oracle latent 子案例 |
|---|---:|---:|---:|---:|---:|
| RCABench | 995 | 30 | 995 | 427 | 240 / 7,960 |
| AIOps22-pre | 169 | 30 | 169 | 72 | 240 / 1,352 |

采用历史按时间的外层 70/30 划分，正常窗口与故障窗口分别划分；保留内层验证比例 0.2 的划分元数据。本最终配方不会为内层验证再扣除已声明的监督案例。查询只发生在外层训练故障池，Oracle 不经过预算查询器。

## 固定参数

科学配置位于 [`src/vast/default_config.json`](src/vast/default_config.json)。公开入口冻结 seed、预算和科学参数；运行 JSON 只配置路径、设备、数据集与监督设置。

| 部件 | 参数 |
|---|---|
| CVAE 网络 | 3 × 16 latent；hidden 128；SiLU；LayerNorm |
| CVAE 优化 | AdamW；学习率 0.001；weight decay 0.0001；batch 上限 64；150 次更新；梯度裁剪 5.0 |
| CVAE 目标 | masked reconstruction 1.0；target context 0.75；cycle consistency 0.75；KL 在前 50 步升至 0.0125 |
| 监督机制项 | triplet 权重 0.1；margin 0.5；只用获准根候选 |
| B 变体 | K=8；父案例后代 pair 总质量 0.25；推断只用后验均值 |
| OSER | `oser-p02`；冻结的 32 维 backbone；训练 residual/gate 参数；Adam lr 0.01；30 步 |
| OSER episode | meta 权重 0.5；内层更新 1 次；内层 lr 0.05；真实 outer query |
| OSER 修正 | 残差 cap 0.05；gate 阈值 0.5；缺失证据时不施加修正 |

配置中的 `mechanism_consistency_weight=1.5` 是历史记录字段，当前训练目标未读取它；`sampling_radius=0.5` 和 `mechanism_sampling_policy` 也不限制 B 上述高斯后验采样。不得把这些字段误写成 B 的额外损失或采样截断规则。`top135_absolute_decline_tolerance` 是原比较实验的报告标准，单独的 B 运行入口不会自动加载历史对照或根据该阈值选择方法。OSER 的 0.05 是分数单位上限，不是指标变化上限。

正式四组 CVAE 各训练 150 步，四组 B OSER 各训练 30 步，均无 fallback。小样本 smoke 可能因故障类型分组不足进入有记录的 OSER fallback，必须查看 `internal_OSER` 诊断。

## 代码布局与运行环境

```text
src/vast/                 最终方法配置、路径与运行入口
src/vast/_historical/     历史完整特征/训练基础的依赖闭包
src/rcl_study/            已验证的连续 CVAE、B、OSER 及其依赖
src/fixed_active_learning/固定 HDBSCAN 获取实现
scripts/run_vast.py       prepare、smoke/formal、resume、status、report
scripts/select_queries.py 从外部数据重建 seed-42 查询计划
configs/                 运行 JSON 示例
tests/                   历史兼容、权重、复用、接口与合成 smoke 测试
tools/                   来源清单、发布检查、既有模型回放
docs/provenance/          源码哈希与变换说明
```

内部 `rcl_study`、`vae_opt_*`、`final_rcl_*`、`conservative_lofo_*` 等名字保留以减少相对已验证代码的偏移。依赖文件可能保留其他历史辅助函数；这些命名或函数不表示公开入口启用了旧离散实验、outer router 或 LOFO。历史包以私有名字加载，拒绝同一进程混入另一份 `nexusrcl_rebuild`。

**训练/执行目标为 Linux 完整源码 checkout**，使用 `fcntl` 文件锁。控制器和 PyTorch worker 可以来自两个解释器。正式 worker 为 Python 3.8.18、PyTorch 2.4.0+cu121、CUDA 12.1、RTX A6000；控制器使用 Python 3.12 与 scikit-learn 1.5.2。worker 不需要安装控制器的 HDBSCAN 依赖。Windows 可以阅读、整理与运行纯源码检查，完整训练入口不支持 Windows 文件锁。

在自己创建的控制器环境中安装依赖：

```bash
python -m pip install -e '.[test]'
```

`pyproject.toml` 配置控制器依赖，不自动安装或修改外部 PyTorch worker。worker 需具备 PyTorch、NumPy 及其数值依赖。运行时会记录解释器、torch/CUDA、NumPy/sklearn 和线程配置。源码入口会向 worker 传递本 checkout 的 `PYTHONPATH`。请从完整 checkout 执行；仅复制 `vast` 包或安装独立 wheel 不包含本运行流程需要的全部脚本与来源清单。

为复现并列排序，应保留经过验证的数值库与 BLAS 线程配置。正式实验未设置 `OMP_NUM_THREADS`、`OPENBLAS_NUM_THREADS`、`MKL_NUM_THREADS`，worker 的 torch threads 为 32。不同硬件或库版本不保证逐位一致。不要为了运行本仓库直接改动已有实验环境。

## 外部输入契约

复制 [`configs/runtime.example.json`](configs/runtime.example.json) 到仓库外，填入自己的路径。相对路径以该运行 JSON 所在目录为基准。`output_root` 必须与源码、外部输入文件和 feature 目录均不重叠，smoke/formal 使用不同的输出目录。

每个数据集需要：

- `feature_dir`：包含 `windows.csv`、`entity_features.csv`、`metadata.json` 的既有特征 bundle；AIOps 的历史 alias 为 `hd1`。这是特征层入口，不是直接读取原始日志/指标文件的适配器。
- `windows.csv`：唯一 `window_id`，以及 `dataset`、`window_kind`（fault/normal）、`start_ts`、`end_ts` 等窗口元数据；`positive_ids` 用分号分隔多个真实根，`metadata_json` 提供获准监督的故障类型（`fault_type`，或历史兼容字段）。正常窗口可为空根。逐例标签从输入剥离后才进入历史特征准备。
- `entity_features.csv`：每个 `(window_id, entity_id)` 唯一一行，保留完整候选目录、历史实体元数据与原始特征列。`metadata.json` 的 `all_feature_columns` 指定特征名称和顺序。缺省的少数状态代理值不等于可以删减正式历史特征。
- `split_manifest`：独立冻结的外层故障成员，格式为 `{"outer_train_case_ids": [...], "outer_test_case_ids": [...]}`。入口重新进行历史时间划分并核验集合、计数与不重叠性。
- `query_plan`：仅 Query-only 必需，采用 `fixed_active_learning.plan_contract` JSON。必须包含 30 个有序唯一案例、dataset、HDBSCAN/center、seed 42、几何/分区/表示等哈希、records、`selected_case_ids`、`plan_sha256` 与 `plan_id`。入口校验语义哈希及外层训练成员，不能用裸 ID 列表代替。
- `supervision_plans`：可选的原历史 `QueryPlan` JSON，按 `query_only` / `oracle_full` 指定。正式逐位回放应提供原计划，特别是 Oracle 的案例顺序会影响后验随机数分配。没有该字段时，Query 使用固定查询顺序，Oracle 使用特征表中的外层训练顺序；这可能不等同于正式参考顺序。

历史 `QueryPlan` 包含 `dataset`、`normal_cluster_id`、`window_clusters`、`queried_window_ids`、`queried_roles`、`queried_labels`、`pseudo_labels`、`pseudo_confidence`、`metadata`；后两个伪标签映射应为空。运行器重新从获准窗口取根标签，不依赖 JSON 中预填的根标签。Query 计划顺序必须一致，Oracle 必须完整覆盖外层训练故障。

若需要重建主动查询计划：

```bash
python scripts/select_queries.py \
  --config /srv/vast-inputs/fixed-al/configs/rcabench.json \
  --feature-root /srv/vast-inputs/features \
  --output /srv/vast-inputs/rcabench-center-seed42.json \
  --verify-reference
```

该工具需要外部冻结的 `rcl-fixed-active-learning-config/v1` 配置、无标签 candidate pool、eligible feature inventory、表示矩阵参考哈希，以及在开启 `--verify-reference` 时的原查询计划。路径解析与字段约束见 `src/fixed_active_learning/config.py`。历史配置字段 `active_learning_seeds` 固定为 `[41,42,43]`，本版本包装器只运行 42。入口不自行创造数据、特征清单或参考哈希。

## 准备、运行与值守

先运行准备和小样本 smoke，并值守到完成：

```bash
python scripts/run_vast.py run --runtime /srv/vast-inputs/runtime-smoke.json --mode smoke --prepare-only
python scripts/run_vast.py run --runtime /srv/vast-inputs/runtime-smoke.json --mode smoke --resume
python scripts/run_vast.py status --output /srv/vast-runs/B-seed42-smoke
python scripts/run_vast.py report --output /srv/vast-runs/B-seed42-smoke
```

`--prepare-only` 写入输入阶段与 manifest，下一次启动必须加 `--resume`。smoke 每设置只取前 3 个监督案例、前 3 个测试案例，CVAE 5 步；Oracle 的 fit 人口随监督缩小，Query 的无标签 fit 人口仍为完整外层训练池。OSER 仍按其固定配方执行。smoke 报告标记为 `bounded_debug_only`，不能用来替代正式分数。

首次正式运行使用新的外部输出目录，可放入 tmux 值守：

```bash
tmux new-session -s vast-B
python scripts/run_vast.py run --runtime /srv/vast-inputs/runtime-formal.json --mode formal
# 另一个终端检查状态
python scripts/run_vast.py status --output /srv/vast-runs/B-seed42-formal
python scripts/run_vast.py report --output /srv/vast-runs/B-seed42-formal
```

已有正确、兼容的阶段必须复用；不要为了重建报告再启动新实验。中断或下游失败后的恢复：

```bash
python scripts/run_vast.py run --runtime /srv/vast-inputs/runtime-formal.json --mode formal --resume
```

同一个输出目录最多一个 runner，同时最多两个设置的控制器、一个 CVAE/encode 重计算 worker。CVAE、latent、基础拟合、OSER、预测和评估各自写入带哈希的阶段提交；匹配阶段复用，输出已写而提交中断时可以恢复。不同设置不共享拟合后的 ranker 或 OSER。

`manifest.json` 冻结运行配置、源码、外部输入哈希与环境。`--resume` 会拒绝输入、源码或运行环境身份变化，已提交输入损坏也会停止并要求调查。不要通过删除已有模型来绕过检查。原始正式实验使用的是归档工作区的阶段身份；本发布入口不会未经核验把旧实验目录当成新缓存。已有正式结果的校验应使用下一节的回放工具。

每个 unit 的 `controller.log`、`status.json` 和 worker 阶段目录下的 `progress.json` / `worker.log` 提供状态、计数、错误。根目录 `heartbeat.json` 记录活跃与排队单元；它是最后一次写入的状态，进程被强制终止后可能滞后，应结合 PID 和日志时间判断。`run.done.json` 只在所有请求的 B 单元完成并重新核验完整排名、真实根和指标后生成。`report` 本身不启动训练。

## 验证与来源

```bash
python -m pytest -q
python -m tools.snapshot_sources --destination-root . --verify-only
python tools/repository_firewall.py .

# 可选：两解释器的合成小样本完整 CLI + 复用 + 输入变更拒绝
VAST_WORKER_PYTHON=/opt/conda/envs/vast-worker/bin/python \
  python -m pytest tests/test_cli_smoke.py -q --basetemp=/srv/vast-checks/pytest-smoke

# 只用受信任、已完成的参考 B 模型做前向回放，不重新训练
python tools/validate_reference.py \
  --reference-run /srv/vast-reference/vae-snapshot-rebase-seed42-20260908-01 \
  --output-root /srv/vast-checks/reference-replay
```

回放工具要求参考 run 所引用的外部输入、历史监督计划、共享阶段和模型仍可访问。只应加载自己信任的 pickle/checkpoint。输出摘要包括特征一致性、latent/基础/最终分数最大差值、完整排名与指标一致性，不下载或发布原始数据。

源码来自已完成的 `vae-post-al-training-optimization` 修订及 `current_label_free_reference_20260821_final` 历史快照；固定查询来源为 `rcl-active-learning-fixed-v1-20260906`。原正式实验验证了 239 个冻结源码文件、68 个增强阶段 seal；共享 CVAE 为 4 个，A/B 共 8 个独立 OSER。后者是原 A/B 对比矩阵的数字，不是本入口另行运行 A。

正式报告 SHA-256：

- Markdown：`433035f720741866d586a162960bad64f092a15dc4934fa8f39a191d550a50c0`
- JSON：`5cd15b073c1033e953a11b28efa93a45650a26ae59a2164e7a1c76d3178d28e5`

发布验证已完成：45 项控制器回归测试、1 项合成完整 CLI 测试通过；四组正式模型回放的 latent、基础和最终分数最大误差均为 0.0，完整排名及指标完全一致。恢复测试确认 checkpoint 内容与修改时间不变，输入改变后拒绝复用。验证没有重新训练正式模型。

发布源码的逐文件来源及变换记录见 [`docs/provenance/source-inventory.json`](docs/provenance/source-inventory.json)，打包验证记录见 [`docs/provenance/README.md`](docs/provenance/README.md)。`docs/superpowers` 中 2026-09-06 的 interim 设计只作历史记录，当前方法与运行契约以本 README 为准。
