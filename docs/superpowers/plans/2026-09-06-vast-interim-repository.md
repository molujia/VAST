# VAST 暂存仓库实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 将已验证的 HDBSCAN + proxy-mode compatible CVAE + OSER-Meta 方法整理成不含数据和实验结果、可审计且可继续迭代的私有 VAST 暂存仓库。

**架构：** 仓库保留实验代码的原始内部模块名，由 `vast` 提供薄公共门面。一个受测的同步器从权威工作区按显式入口递归解析项目内 import，并额外纳入运行时模块入口；一个独立防火墙在提交前阻断数据、结果、缓存、checkpoint、绝对机器路径和凭据。来源清单为每个复制文件记录源/目标 SHA-256，使本次暂存版本可追溯但不携带数据绑定产物。

**技术栈：** Python 3.12、pytest、NumPy、pandas、SciPy、scikit-learn 1.5.2、threadpoolctl、PyTorch、Git。

---

## 文件结构

- 创建 `tools/snapshot_sources.py`：解析允许的 Python 内部依赖、复制最终方法源码并生成来源清单。
- 创建 `tools/repository_firewall.py`：扫描仓库路径和文本内容中的数据、输出、缓存、checkpoint、凭据及机器绝对路径。
- 创建 `tests/test_snapshot_sources.py`：覆盖递归闭包、运行时入口、排除目录与 SHA-256 清单。
- 创建 `tests/test_repository_firewall.py`：覆盖路径泄漏、秘密泄漏、绝对路径及合法科学配置。
- 创建 `tests/test_vast_facade.py`：覆盖 VAST 版本、冻结方法元数据及延迟加载入口。
- 创建 `src/vast/__init__.py`、`src/vast/method.py`：薄公共门面，不复制训练实现。
- 创建 `src/fixed_active_learning/**`：从固定 HDBSCAN authority 机械同步的代码闭包。
- 创建 `src/rcl_study/**`：从最终 runner 显式入口解析得到的内部代码闭包。
- 创建 `src/nexusrcl_rebuild/training/pairwise_backend.py`：最小 pairwise-linear 后端。
- 创建 `scripts/run_final_rcl_formal.py`、`scripts/run_final_rcl_real_smoke.py`、`scripts/status_final_rcl.py` 及所需准备脚本。
- 创建 `configs/final_rcl/hdbscan_proxy_cvae_oser_seed42.json`：冻结科学配置，不含数据或输出路径。
- 创建 `docs/provenance/source-inventory.json`：逐文件来源与哈希。
- 创建 `pyproject.toml`、`.gitignore`、`README.md`：最小可安装项目、忽略规则与暂存说明。

### 任务 1：实现可重复的源码快照同步器

**文件：**
- 创建：`tests/test_snapshot_sources.py`
- 创建：`tools/snapshot_sources.py`

- [ ] **步骤 1：编写闭包和排除策略的失败测试**

```python
def test_collects_relative_imports_and_explicit_runtime_roots(tmp_path):
    source = make_source_tree(tmp_path)
    selected = collect_snapshot_files(source)
    assert "rcl_study/final_rcl_contract.py" in selected
    assert "rcl_study/helper.py" in selected
    assert "rcl_study/service_continuous_neural_runner.py" in selected

def test_rejects_forbidden_source_path(tmp_path):
    with pytest.raises(SnapshotError, match="forbidden"):
        copy_snapshot_file(tmp_path / "scores/result.json", tmp_path / "out")

def test_inventory_hashes_source_and_destination(tmp_path):
    inventory = materialize_snapshot(source_root, destination_root)
    row = inventory["files"][0]
    assert row["source_sha256"] == row["destination_sha256"]
```

- [ ] **步骤 2：运行测试并确认因模块尚不存在而失败**

运行：`python -m pytest tests/test_snapshot_sources.py -q`

预期：FAIL，错误包含 `ModuleNotFoundError: No module named 'tools.snapshot_sources'`。

- [ ] **步骤 3：实现最小同步器**

```python
PYTHON_ROOTS = (...最终 final_rcl 与运行时入口...)
FORBIDDEN_PARTS = {"reference", "verification", "reproduced", "scores", "candidate_pools", "query_plans", "feature_inventories", "__pycache__"}

def collect_snapshot_files(source_root: Path) -> tuple[str, ...]:
    # 解析 rcl_study/scripts 内部 import，并合并显式 runtime roots、
    # fixed_active_learning 的 .py 文件和最小 pairwise backend。
    ...

def materialize_snapshot(source_root: Path, destination_root: Path) -> dict:
    # 原子地复制选定文本源码并返回含逐文件 SHA-256 的稳定清单。
    ...
```

- [ ] **步骤 4：运行测试确认通过**

运行：`python -m pytest tests/test_snapshot_sources.py -q`

预期：全部通过。

- [ ] **步骤 5：提交任务 1**

```text
git add tools/snapshot_sources.py tests/test_snapshot_sources.py docs/superpowers/plans/2026-09-06-vast-interim-repository.md
git commit -m "build: add reproducible VAST snapshot tooling"
```

### 任务 2：实现数据与秘密防火墙

**文件：**
- 创建：`tests/test_repository_firewall.py`
- 创建：`tools/repository_firewall.py`

- [ ] **步骤 1：编写失败测试**

```python
@pytest.mark.parametrize("path", [
    "outputs/run/score.json", "data/rcabench/case.json",
    "model/checkpoint.pt", "src/pkg/__pycache__/x.pyc",
])
def test_blocks_forbidden_paths(tmp_path, path):
    write(tmp_path / path, "x")
    assert scan_repository(tmp_path)

def test_blocks_token_and_machine_absolute_path(tmp_path):
    write(tmp_path / "bad.txt", "github_pat_abc C:\\\\Users\\\\name\\\\data")
    assert {finding.kind for finding in scan_repository(tmp_path)} == {"secret", "absolute_path"}

def test_allows_scientific_hashes_and_relative_runtime_inputs(tmp_path):
    write(tmp_path / "config.json", '{"profile_sha256":"' + "a" * 64 + '","data":"${DATA_ROOT}"}')
    assert scan_repository(tmp_path) == []
```

- [ ] **步骤 2：运行测试确认正确失败**

运行：`python -m pytest tests/test_repository_firewall.py -q`

预期：FAIL，错误包含 `ModuleNotFoundError: No module named 'tools.repository_firewall'`。

- [ ] **步骤 3：实现防火墙 CLI 与库接口**

```python
def scan_repository(root: Path) -> list[Finding]:
    # 跳过 .git；按路径段和扩展名拦截数据/结果/checkpoint/cache，
    # 按文本规则拦截 token、私钥、Windows/Linux 机器专属绝对路径。
    ...

def main() -> int:
    findings = scan_repository(Path(args.root))
    return 1 if findings else 0
```

- [ ] **步骤 4：运行测试确认通过**

运行：`python -m pytest tests/test_repository_firewall.py -q`

预期：全部通过。

- [ ] **步骤 5：提交任务 2**

```text
git add tools/repository_firewall.py tests/test_repository_firewall.py
git commit -m "test: add VAST data and secret firewall"
```

### 任务 3：建立薄 VAST 门面和项目元数据

**文件：**
- 创建：`tests/test_vast_facade.py`
- 创建：`src/vast/__init__.py`
- 创建：`src/vast/method.py`
- 创建：`pyproject.toml`
- 创建：`.gitignore`

- [ ] **步骤 1：编写失败测试**

```python
def test_method_descriptor_freezes_selected_components():
    descriptor = describe_method()
    assert descriptor["active_learning"] == "hdbscan"
    assert descriptor["augmentation"] == "proxy_mode_cvae_compatible"
    assert descriptor["ranker"] == "weighted_pairwise_linear+oser_meta"
    assert descriptor["outer_router"] is None

def test_load_config_reads_packaged_default():
    assert load_default_config()["method_id"] == "hdbscan-proxy-cvae-compatible-oser-p02"
```

- [ ] **步骤 2：运行测试确认因 `vast` 尚不存在而失败**

运行：`python -m pytest tests/test_vast_facade.py -q`

预期：FAIL，错误包含 `ModuleNotFoundError: No module named 'vast'`。

- [ ] **步骤 3：实现门面和项目配置**

```python
__version__ = "0.1.0.dev0"

def describe_method() -> dict[str, object]:
    return {
        "active_learning": "hdbscan",
        "augmentation": "proxy_mode_cvae_compatible",
        "ranker": "weighted_pairwise_linear+oser_meta",
        "outer_router": None,
    }
```

`pyproject.toml` 使用 `src` 布局，声明 Python 3.12 和 NumPy/pandas/SciPy/scikit-learn/threadpoolctl/PyTorch 依赖，并为 pytest 添加 `src` 与仓库根路径。

- [ ] **步骤 4：运行门面测试确认通过**

运行：`python -m pytest tests/test_vast_facade.py -q`

预期：全部通过。

- [ ] **步骤 5：提交任务 3**

```text
git add src/vast tests/test_vast_facade.py pyproject.toml .gitignore
git commit -m "feat: add interim VAST facade"
```

### 任务 4：同步权威实现与聚焦测试

**文件：**
- 创建：`src/fixed_active_learning/**`
- 创建：`src/rcl_study/**`
- 创建：`src/nexusrcl_rebuild/**`
- 创建：`scripts/**`
- 创建：`configs/final_rcl/hdbscan_proxy_cvae_oser_seed42.json`
- 创建：`tests/test_final_rcl_*.py`
- 创建：`tests/test_run_final_rcl_*.py`
- 创建：`docs/provenance/source-inventory.json`

- [ ] **步骤 1：执行同步器**

运行：`python tools/snapshot_sources.py --source-root .. --destination-root .`

预期：输出复制文件数量，生成的每个路径均不命中禁止项，来源/目标哈希一致。

- [ ] **步骤 2：运行来源清单自校验**

运行：`python tools/snapshot_sources.py --destination-root . --verify-only`

预期：退出码 0，输出 `source inventory valid`。

- [ ] **步骤 3：运行不需要真实数据的聚焦测试**

运行：`python -m pytest tests/test_final_rcl_contract.py tests/test_final_rcl_hdbscan_proxy.py tests/test_final_rcl_training.py tests/test_final_rcl_pairwise.py tests/test_final_rcl_oser.py tests/test_final_rcl_evaluation.py tests/test_final_rcl_execution.py tests/test_final_rcl_neural_request.py tests/test_final_rcl_oser_runner.py tests/test_final_rcl_real_execution.py tests/test_run_final_rcl_formal_cli.py tests/test_run_final_rcl_real_smoke_cli.py -q`

预期：全部通过；如本机缺少 PyTorch，PyTorch 专属测试只允许由 pytest 显式 skip，不允许 import error。

- [ ] **步骤 4：运行内部 import 闭包检查**

运行：`python -m compileall -q src scripts`

预期：退出码 0；随后删除由检查生成的缓存并由 `.gitignore` 阻断其提交。

- [ ] **步骤 5：提交任务 4**

```text
git add src/fixed_active_learning src/rcl_study src/nexusrcl_rebuild scripts configs tests/test_final_rcl_*.py tests/test_run_final_rcl_*.py docs/provenance/source-inventory.json
git commit -m "feat: snapshot the selected VAST implementation"
```

### 任务 5：补充最小暂存文档

**文件：**
- 创建：`README.md`
- 创建：`docs/provenance/README.md`

- [ ] **步骤 1：编写 README**

README 明确：该仓库是非最终暂存版；方法组成、三个实验臂、环境依赖、数据不随仓库分发、运行时输入、smoke/formal/status 命令入口，以及当前不承诺复现实验分数。

- [ ] **步骤 2：编写 provenance 说明**

说明 `source-inventory.json` 字段、authority 名称、同步与验证命令；不写入本机或服务器绝对路径。

- [ ] **步骤 3：运行文档和防火墙检查**

运行：`python tools/repository_firewall.py .`

预期：退出码 0，输出 `repository firewall passed`。

- [ ] **步骤 4：提交任务 5**

```text
git add README.md docs/provenance/README.md
git commit -m "docs: add interim VAST repository guide"
```

### 任务 6：最终验证并推送私有仓库

**文件：**
- 验证：仓库全部已跟踪内容

- [ ] **步骤 1：运行完整本地门禁**

运行：`python -m pytest -q`

预期：0 failures；依赖缺失只能表现为明确 skip。

运行：`python tools/snapshot_sources.py --destination-root . --verify-only`

预期：来源清单验证通过。

运行：`python tools/repository_firewall.py .`

预期：防火墙通过。

- [ ] **步骤 2：检查 Git 内容**

运行：`git status --short`、`git diff --check`、`git ls-files`

预期：无未提交变更、无空白错误、无被禁止路径。

- [ ] **步骤 3：使用临时 credential helper 推送**

```text
git -c 'credential.helper=!f() { echo username=x-access-token; echo password=$git_personal_access_token; }; f' push origin main
```

预期：推送成功；remote URL 与本地 Git 配置均不包含 token。

- [ ] **步骤 4：验证远端提交**

运行：`git ls-remote origin refs/heads/main`

预期：远端 SHA 与 `git rev-parse HEAD` 完全一致。

