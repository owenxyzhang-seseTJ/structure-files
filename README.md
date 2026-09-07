# structure-files

面向 DeepSeek Harness 的化学感知结构文件处理服务。它只读写和审计
CIF、POSCAR/CONTCAR、OUTCAR、XYZ/extXYZ 等结构文件，并提供文件级的
TS-like 候选搜索；不运行 VASP、RASPA、NEB、鞍点优化或任何 DFT/理论计算。

## 设计边界

- 原始文件永不覆盖；所有生成文件经过 `proposal` → `proposal_apply` 两阶段。
- proposal 记录输入 SHA-256，应用前再次核对，源文件变化就拒绝应用。
- CIF 保留占位率、无序和客体信息；不静默把 occupancy 改成 1，不静默选择无序构型。
- “补化学键”先由 `bond_graph` 给出跨周期共价键/金属配位候选和置信度；不会把坐标启发式
  冒充实验键级或无提示写入 `_geom_bond`。
- OUTCAR 只读解析；截断文件只提取最后一个完整离子步，不回写或“修复”原文件。
- 补氢输出 top-k 候选和不确定性。ML 势只能在同一组成、同一模型内优化/排序，不能把
  重原子 CIF 解释成唯一的实验氢位置。

## 本地运行

```bash
uv sync --python 3.12
# The preset uses the equivalent module form with PYTHONPATH=src so it also
# works when macOS marks uv's editable .pth as hidden under Python 3.12.
PYTHONPATH=src uv run --no-sync python -m structure_files.mcp_server
```

`structure-files-mcp` 使用 stdio MCP 协议，Harness 通过
`@deepseek-ai/dsh-mcp-client` 连接。若需要 VASP 文本的可选增强解析：

```bash
uv sync --extra vasp --python 3.12
```

ML 后端不在核心安装中；只有完成 masked-H/TS 路径基准后，才建议在独立环境安装
`mace-torch` 或配置一个 ASE Calculator 工厂。补氢时只固定重原子做 H-only 约束优化并在
同一组成内排序；TS 路径只做同一模型的单点能量/力描述。

## 库选择与可复现边界

完整的版本、替代 ML 后端、许可证/API 风险和验收矩阵见
[`docs/library-research.md`](docs/library-research.md)。

| 层 | 库 | 用途与限制 |
| --- | --- | --- |
| CIF/对称性 | Gemmi | 读取多 block/复杂 Hermann–Mauguin CIF、显式对称操作和占位信息；原始文本不覆盖 |
| 独立交叉解析 | Pymatgen + ASE | 分别解析 CIF/POSCAR/XYZ/extXYZ；解析结果不一致时进入 audit，而不是静默择一 |
| 对称性复核 | Spglib | 仅作为空间群/操作数交叉检查，不把恢复的空间群写回用户要求的 P1 |
| 周期化学图 | NetworkX + 本地共价半径表 | 生成跨周期键和金属配位候选；bond order/价态是筛选启发式，不是键级证明 |
| 可选 H/TS 模型 | ASE Calculator 接口，内置 MACE 适配器 | 只在同一模型、同一组成内做 H-only 或路径帧单点排序；不执行 NEB、鞍点优化或 DFT |
| 可选格式增强 | `dpdata`、`parsevasp`、`rdkit` | 通过 extra 安装；不改变核心 proposal/apply 和化学不确定性边界 |

核心环境可用 `uv.lock` 复现。需要显式启用 MACE 时使用独立环境安装可选依赖：

```bash
uv sync --extra ml --python 3.12
```

然后在 `hydrogen_propose` 或 `ts_search` 的 `ml_json` 中指定
`backend: "mace"`、本地 `model_path` 和 `allow_failure`；服务不会隐式下载 MACE 权重；没有经过同模型基准测试时保持
`enabled: false`。

## Harness 验证

`~/.dsh/.agent-presets/structure-files/` 是实际用户 preset，`preset.yml` 只负责选择器
展示，`agent.cordis.yml` 负责插件组装。部署级 `dsh --profile web --dump-config` 仍可能显示
`agent-presets.default: standard`，因为那是静态组装默认值；运行时的用户设置层由
`agentPresets/list` 决定。验收应看到 `structure-files` 行 `trust: user`、`isDefault: true`
且无 `broken` 字段，并用 `agentPresets/read` 确认内容包含 `mcp-structure-files`。
预设的 MCP 启动项使用 `python -m structure_files.mcp_server` 加显式 `PYTHONPATH=src`，
避免 macOS/Python 3.12 跳过 uv 可编辑安装的隐藏 `.pth` 文件。

## 工具

| 工具 | 作用 |
| --- | --- |
| `scan` | 识别文件类型、大小、哈希和可解析性 |
| `inspect` | 读取晶胞、原子、组分、占位/无序、对称性和 OUTCAR 摘要 |
| `validate` | Pymatgen + ASE 双解析、周期接触、重复位点、键/配位和 Spglib 交叉检查 |
| `convert` | 在 CIF/POSCAR/extXYZ 间提出转换 proposal |
| `transform` | P1、超胞、基矢、原点/原子平移、wrap/unwrap、片段提取 proposal |
| `bond_graph` | 周期化学键图和金属配位候选（只读） |
| `bond_propose` | 把审查后的周期键/配位候选写入带 `_geom_bond` 的 CIF proposal；标准键级保持未知，并附 DSH 推断审计字段 |
| `hydrogen_propose` | 价态驱动的无氢/多取向 top-k 候选，支持可选 H-only ML 排序 |
| `proposal_apply` | SHA-256 保护下应用指定候选到新文件并生成审计 sidecar |
| `outcar_extract` | 只读提取最后完整离子步、能量、力和轨迹 |
| `compare` | 结构、晶胞、组成和原子映射比较 |
| `ts_search` | 从反应物/产物文件生成周期安全插值帧，识别端点键变化并排序 TS-like 候选；可选同一 ASE/MACE 势单点评分 |

## TS search 的文件级边界

`ts_search` 需要反应物和产物结构文件（原子数与元素组成相同）。默认按相同标签映射，
否则按元素出现顺序映射；复杂重排可显式传 `mapping`。它生成内部插值帧、端点启发式
成键/断键变化、周期最小镜像坐标、几何中心性和碰撞惩罚，并把 top-k 帧、完整
`ts-path.extxyz` 和 `ts-search-report.json` 放入 proposal。

可选 `ml_json` 通过 ASE Calculator（内置 MACE 适配器或用户提供的
`module:function` factory）对同一模型的所有帧做单点能量/力描述。能量只在本次路径、
同一模型内归一化；ML 失败时保留几何排序。输出始终标记为 `TS-like candidate`，不能
解释为已找到的过渡态或鞍点，也不会调用 NEB、Sella、VASP、DFT 或任务提交。

最小调用示例：

```json
{
  "reactant_path": "reactant.cif",
  "product_path": "product.cif",
  "n_frames": 17,
  "top_k": 3,
  "output_format": "cif",
  "include_trajectory": true
}
```

补化学键的最小调用：

```json
{
  "input_path": "framework.cif",
  "expand_symmetry": true,
  "include_coordination": true,
  "min_confidence": 0.55
}
```

该调用只创建 proposal；查看 `proposal.manifest_path` 后，再用
`proposal_apply` 选择 `candidate_index: 0` 写入新的 CIF。

## 补化学键的输出边界

`bond_graph` 只读返回周期最小镜像边、距离、启发式 bond order、金属配位类型和置信度。
需要把候选边放进 CIF 时使用 `bond_propose`：它先（默认）将 CIF 展开成完整的 P1 原子表，
再生成新的 CIF proposal。输出包含标准 `_geom_bond` 字段以及
`_dsh_geom_bond_kind/order/confidence/image_*` 审计列；`_geom_bond_type` 保持 `?`，因此不把
半径/距离启发式冒充实验测得的键级。必须通过 `proposal_apply` 写入新文件，并在应用后的
`.audit.json` 中审查边、价态、短接触和金属配位。

启用 ML 的示例（需用户环境自行安装模型）：

```json
{
  "reactant_path": "reactant.cif",
  "product_path": "product.cif",
  "ml_json": "{\"enabled\":true,\"backend\":\"mace\",\"model_path\":\"/path/to/model.model\",\"allow_failure\":true}"
}
```

## 安装与运行

仓库公开地址：[owenxyzhang-seseTJ/structure-files](https://github.com/owenxyzhang-seseTJ/structure-files)。项目要求 Python 3.12（`pyproject.toml` 限制为 `>=3.12,<3.13`），推荐使用 [uv](https://docs.astral.sh/uv/) 安装锁定依赖。

```bash
git clone https://github.com/owenxyzhang-seseTJ/structure-files.git
cd structure-files
uv sync --python 3.12
chmod +x scripts/run_mcp.sh scripts/run_cli.sh
```

核心依赖包括 Gemmi、Pymatgen、ASE、Spglib、NetworkX、NumPy、SciPy、Pydantic、orjson 和 MCP Python SDK。`uv.lock` 是可复现安装的依据。可选依赖按需安装：

```bash
# 可选 VASP 文本增强；核心 OUTCAR 只读解析不依赖它们
uv sync --extra vasp --python 3.12

# 可选孤立分子/官能团辅助检查；不替代周期晶体键图
uv sync --extra chemistry --python 3.12

# 可选 MACE；需要用户自行准备 PyTorch/CUDA 和本地模型
uv sync --extra ml --python 3.12
```

服务不会隐式下载 ML 权重。没有经过 masked-H 和同组成路径基准验证的模型时，应保持 ML 关闭。

### stdio MCP

MCP 服务使用标准输入/输出传输，不监听 TCP 端口：

```bash
PYTHONPATH=src uv run --no-sync python -m structure_files.mcp_server
```

跨工作目录启动请使用项目自带脚本：

```bash
/absolute/path/to/structure-files/scripts/run_mcp.sh
```

`run_mcp.sh` 会依次尝试 `DSH_STRUCTURE_PYTHON`、项目 `.venv/bin/python`、uv 和系统 `python3`，并自动设置 `PYTHONPATH=src`。如果宿主 Harness/文件技能使用根目录策略，可同时传递 `DSH_STRUCTURE_ROOTS`（它是策略元数据，不替代宿主的路径授权）：

```bash
DSH_STRUCTURE_ROOTS=/absolute/path/to/your/structure-work \
  /absolute/path/to/structure-files/scripts/run_mcp.sh
```

### JSON CLI（MCP 不可用时）

CLI 与 MCP 共用同一个 `StructureService`，每次输出一份 JSON；失败时输出结构化错误并返回非零退出码：

```bash
scripts/run_cli.sh scan --path examples
scripts/run_cli.sh inspect --path examples/ethanol_no_h.cif
scripts/run_cli.sh validate --path examples/ethanol_no_h.cif
scripts/run_cli.sh inspect --json '{"path":"examples/ethanol_no_h.cif","include_atoms":true}'
```

也可以直接使用 Python API：

```python
from structure_files.service import StructureService

service = StructureService()
report = service.validate("examples/ethanol_no_h.cif")
print(report["ok"], report["warnings"])
```

## 从 DSH Harness 迁移

### DSH preset

当前用户 preset 位于：

```text
~/.dsh/.agent-presets/structure-files/
├── preset.yml
└── agent.cordis.yml
```

`preset.yml` 负责选择器显示，`agent.cordis.yml` 负责 persona、文件技能、计划模式和 `mcp-structure-files` 组装。把仓库迁移到另一台机器后，需把其中的项目路径和 uv 路径改为目标机器的绝对路径，或使用：

```bash
export DSH_STRUCTURE_PROJECT=/absolute/path/to/structure-files
export DSH_STRUCTURE_UV=/absolute/path/to/uv
```

DSH Web 启动示例：

```bash
dsh web --no-open --port 3080
```

`3080` 是 DSH Web UI 端口，不是 MCP 端口。MCP 由 Harness 以 stdio 子进程启动；直接访问未认证的 Web HTTP 接口得到 `401` 是认证层行为，不代表 MCP 失效。

验收时应在 `agentPresets/list` 看到 `structure-files` 的 `trust: user`、`isDefault: true` 且没有 `broken` 字段，并在 `agentPresets/read` 中确认内容包含 `mcp-structure-files`。部署级 `dsh --profile web --dump-config` 仍可能显示 `agent-presets.default: standard`，因为那是静态组装默认值。

### Codex CLI / Codex Desktop

在仓库目录外配置时使用启动脚本的绝对路径：

```bash
codex mcp add structure-files -- \
  bash /absolute/path/to/structure-files/scripts/run_mcp.sh

codex mcp list
codex mcp get structure-files
```

等价的 Codex TOML 配置为：

```toml
[mcp_servers.structure_files]
command = "bash"
args = ["/absolute/path/to/structure-files/scripts/run_mcp.sh"]
startup_timeout_sec = 120

[mcp_servers.structure_files.env]
# If your host enforces a structure-file root policy, pass it here.
DSH_STRUCTURE_ROOTS = "/absolute/path/to/your/structure-work"
```

Codex Desktop 的 MCP 设置填入同一条 stdio command 即可。不要把 `http://127.0.0.1:3080` 当作 MCP server URL；那是 DSH Web 地址。连接成功后，工具会出现在 `structure-files` server 下。

### Claude Code

仓库已经提供项目级 `.mcp.json`：

```json
{
  "mcpServers": {
    "structure-files": {
      "type": "stdio",
      "command": "bash",
      "args": ["scripts/run_mcp.sh"]
    }
  }
}
```

在仓库根目录运行：

```bash
claude --mcp-config .mcp.json --strict-mcp-config
```

或添加到 Claude Code 用户配置：

```bash
claude mcp add structure-files -- \
  bash /absolute/path/to/structure-files/scripts/run_mcp.sh

claude mcp list
claude mcp get structure-files
```

项目级配置适合团队仓库，用户级配置适合多个项目共享。使用绝对脚本路径可以避免 Claude Code 从不同工作目录启动时找不到项目。

### MCP 被策略禁用时的 plugin/CLI 方案

当前不需要另一个 plugin 才能运行：MCP、CLI 和 Python API 共用同一实现。若组织策略禁止 MCP，直接使用：

```bash
scripts/run_cli.sh inspect --path /absolute/path/to/input.cif
scripts/run_cli.sh transform --json \
  '{"input_path":"/absolute/path/to/input.cif","p1":true,"supercell":[2,2,1]}'
```

若环境只允许插件，可让插件调用 `scripts/run_cli.sh`，或者把同一 `run_mcp.sh` 包装为插件的 stdio server；不要复制一套绕过 proposal/apply 的写入逻辑。当前公开仓库交付的是可直接使用的 MCP + CLI，不包含一个重复实现功能的空壳 plugin。

## 工具参数和文件工作流

| 工具 | 读/写 | 作用 | 关键参数 |
| --- | --- | --- | --- |
| `scan` | 只读 | 扫描文件/目录，识别格式、大小、SHA-256 和可解析性 | `path`, `recursive`, `max_files` |
| `inspect` | 只读 | 晶胞、原子、组分、occupancy/disorder、对称性或 OUTCAR 摘要 | `path`, `expand_symmetry`, `include_atoms` |
| `validate` | 只读 | Pymatgen/ASE 双解析、周期接触、重复位点、候选键/配位和 Spglib 复核 | `path`, `expand_symmetry`, `contact_limit`, `symprec` |
| `convert` | proposal | CIF/POSCAR/extXYZ/XYZ 互转 | `input_path`, `output_format`, `expand_p1`, `occupancy_policy` |
| `transform` | proposal | P1、超胞、基矢、原点/原子平移、片段平移、wrap/unwrap、组分提取 | `supercell`, `basis`, `origin_shift`, `atom_selection`, `extract` |
| `bond_graph` | 只读 | 周期最小镜像共价键/金属配位候选图 | `tolerance`, `metal_tolerance`, `include_hydrogen` |
| `bond_propose` | proposal | 带审计字段的 CIF 候选键注释 | `expand_symmetry`, `include_coordination`, `min_confidence` |
| `hydrogen_propose` | proposal | 价态驱动的补氢、多质子化和多取向 top-k 候选 | `top_k`, `orientation_count`, `hetero_policy`, `ml_json` |
| `proposal_apply` | 唯一写入 | 校验 SHA-256 后选中候选，写目标和 `.audit.json` | `manifest_path`, `candidate_index`, `destination` |
| `outcar_extract` | 只读或 proposal | summary、最后完整离子步或完整 extXYZ 轨迹 | `mode`, `output_format` |
| `compare` | 只读 | 组成、原子数、晶胞和 StructureMatcher 比较 | `input_a`, `input_b`, `ignore_hydrogen`, `primitive` |
| `ts_search` | proposal | 周期安全插值、端点键变化、TS-like 帧排序和可选同模型单点描述 | `reactant_path`, `product_path`, `mapping`, `n_frames`, `ml_json` |

推荐顺序是：

```text
scan/inspect
  → validate
  → bond_graph/compare（只读审查）
  → convert/transform/bond_propose/hydrogen_propose/outcar_extract/ts_search
  → 阅读 proposal.manifest_path、候选和 warnings
  → proposal_apply
  → 读取输出和 .audit.json，再次 validate
```

proposal 默认位于源文件同目录的 `.dsh-structure-proposals/<proposal_id>/`。manifest 至少记录 `source_files`、源 SHA-256、`artifacts`、候选哈希、`requested_output` 和 `metadata`。应用时可以指定新目标：

```json
{
  "manifest_path": "/absolute/path/to/.dsh-structure-proposals/<id>/manifest.json",
  "candidate_index": 0,
  "destination": "/absolute/path/to/result.cif",
  "overwrite": false
}
```

`overwrite=true` 也不能覆盖源文件；它只允许在用户明确选择时覆盖非源目标。

## 端到端调用示例

以下 JSON 可以直接作为 MCP 工具参数，也可以放入 CLI 的 `--json`。相对路径示例要求当前工作目录是仓库根目录。

### 扫描、检查和验证

```bash
scripts/run_cli.sh scan --path examples
scripts/run_cli.sh inspect --json \
  '{"path":"examples/ethanol_no_h.cif","expand_symmetry":true,"include_atoms":true}'
scripts/run_cli.sh validate --json \
  '{"path":"examples/ethanol_no_h.cif","expand_symmetry":true}'
```

`inspect` 的 CIF 报告会分类 asymmetric unit、已预展开/P1、display-oriented 和 partial-occupancy/disorder 表示。`validate` 的 `ok=false` 不应被改写成“结构已清洗”；先阅读 `warnings`、`independent_parsers`、`geometry` 和 `policy`。

### CIF 转换和超胞

非平凡空间群 CIF 转 POSCAR/XYZ 时必须明确 `expand_p1=true`：

```bash
scripts/run_cli.sh convert --json \
  '{"input_path":"examples/ethanol_no_h.cif","output_format":"poscar","expand_p1":true,"output_dir":"work"}'

scripts/run_cli.sh transform --json \
  '{"input_path":"examples/ethanol_no_h.cif","output_format":"cif","p1":true,"supercell":[2,2,1],"output_dir":"work"}'
```

`supercell` 接受三个正对角整数或 9 个整数构成的可逆 3×3 整数矩阵。拓扑改变后旧 `_geom_bond` 会失效并标记为需要重新计算。

### 原点、原子和片段平移

```json
{
  "input_path": "framework.cif",
  "output_format": "cif",
  "p1": true,
  "origin_shift": [0.125, 0.0, 0.0],
  "atom_selection": [0, 1, 2],
  "atom_translation": [0.0, 0.5, 0.0],
  "translation_coordinate_type": "fractional",
  "connected_fragment": true,
  "wrap": true
}
```

非 P1 asymmetric unit 上的选定原子平移会被拒绝，必须先 `p1=true`。整体原点平移会共轭对称操作的仿射平移；无法安全处理时服务会拒绝。

### 键图和键注释

```bash
scripts/run_cli.sh bond_graph --json \
  '{"path":"framework.cif","expand_symmetry":true,"include_hydrogen":true,"tolerance":0.45}'

scripts/run_cli.sh bond_propose --json \
  '{"input_path":"framework.cif","expand_symmetry":true,"include_coordination":true,"min_confidence":0.55,"output_dir":"work"}'
```

`bond_propose` 输出的 `_geom_bond_type` 保持 `?`，并增加 `_dsh_geom_bond_kind/order/confidence/image_*` 审计列。应用前必须审查 `edges`、`validation` 和 `warnings`。

### 补氢和 ML

```bash
scripts/run_cli.sh hydrogen_propose --json \
  '{"input_path":"framework_no_h.cif","top_k":8,"orientation_count":12,"hetero_policy":"enumerate","include_unprotonated_hetero":true,"output_dir":"work"}'

scripts/run_cli.sh hydrogen_propose --json \
  '{"input_path":"framework_no_h.cif","top_k":5,"ml_json":"{\"enabled\":true,\"backend\":\"mace\",\"model_path\":\"/models/mace.model\",\"allow_failure\":true}","output_dir":"work"}'
```

ML 只允许固定重原子和晶胞、移动 H，并在同一 calculator、同一组成内比较。硬碰撞或 H-host 脱离会使 refinement 被拒绝并回退几何候选；不同组成不比较绝对能量。

### OUTCAR 提取

```bash
scripts/run_cli.sh outcar_extract --json \
  '{"input_path":"examples/OUTCAR.sample","mode":"summary"}'

scripts/run_cli.sh outcar_extract --json \
  '{"input_path":"examples/OUTCAR.sample","mode":"last_complete","output_format":"cif","output_dir":"work"}'

scripts/run_cli.sh outcar_extract --json \
  '{"input_path":"examples/OUTCAR.sample","mode":"trajectory","output_format":"extxyz","output_dir":"work"}'
```

截断 position block 会出现在 `summary.truncated_position_block` 和 `warnings` 中，不会被拼接或覆写。

### 文件级 TS search

```bash
scripts/run_cli.sh ts_search --json \
  '{"reactant_path":"reactant.cif","product_path":"product.cif","n_frames":17,"top_k":3,"output_format":"cif","include_trajectory":true,"cell_mode":"require_equal","output_dir":"work"}'
```

反应物和产物须原子数、元素组成相同。默认优先按相同唯一标签，其次按元素出现顺序；复杂重排用 `mapping`。支持 `require_equal`、`reactant`、`product` 和 `linear` 晶胞模式。输出包括候选 CIF、`ts-path.extxyz` 和 `ts-search-report.json`，始终标记 `TS-like candidate`，不调用 NEB、Sella、VASP 或 DFT。

### 比较和应用

```bash
scripts/run_cli.sh compare --json \
  '{"input_a":"reactant.cif","input_b":"product.cif","ignore_hydrogen":true,"primitive":false}'

scripts/run_cli.sh proposal_apply --json \
  '{"manifest_path":"/absolute/path/to/.dsh-structure-proposals/<id>/manifest.json","candidate_index":0,"destination":"work/selected.cif"}'
```

应用后检查 `output_sha256` 和 `audit_path`，再对选中输出运行 `validate`。

## 化学和 ML 的解释边界

- `bond_graph` 的距离、共价半径、启发式 bond order 和金属配位类型只用于候选筛选；`_geom_bond_type` 仍为 `?`。
- 普通 C-H 在价态、周期几何和占位率都清楚时可能接近确定；羧酸/磺酸 O-H、N/S 质子化、配位水、μ-OH/缺陷端基和部分占位位点必须保留多候选。
- ML 只有在用户显式提供本地模型或 `module:function` ASE calculator factory 时才运行；不下载权重，不移动重原子，不改晶胞，不做优化或 NEB。
- H 候选的 ML 排序只在同一模型、同一组成内有意义；`model_preferred` 是模型条件下的偏好，不是唯一实验位置。
- `ts_search` 是反应物/产物映射、周期安全插值、端点键变化和候选帧排序；最高分帧不是确认的过渡态或鞍点。

## 库和方法选型

| 层 | 库/接口 | 在本项目中的职责 | 重要限制 |
| --- | --- | --- | --- |
| CIF 原文/对称性 | Gemmi | 多 block、显式 symmetry loop、occupancy/disorder 和 block 选择 | 解析成功不等于化学正确 |
| 独立解析 | Pymatgen + ASE | CIF、POSCAR、XYZ/extXYZ 的双解析和结构适配 | 解析器不一致进入 audit，不静默择一 |
| 对称性复核 | Spglib | 空间群/操作数的独立信息性检查 | 不把恢复的空间群覆盖用户要求的 P1 |
| 周期几何 | NumPy/SciPy | 分数/笛卡尔坐标、最小镜像、插值和碰撞指标 | 数值指标需要化学角色审查 |
| 周期图 | NetworkX + 共价半径表 | 连通片段、候选共价键和金属配位 | 不证明 bond order、价态或实验拓扑 |
| ML 接口 | ASE Calculator；内置 MACE 适配 | H-only refinement、同模型路径单点描述 | 不运行优化、NEB、DFT；不跨模型比较 |
| 可选增强 | dpdata、parsevasp、RDKit | VASP 文本或孤立分子辅助 | 不改变文件-only 和 proposal/apply 边界 |

CHGNet、ORB、Fairchem/UMA 等可以通过 `calculator_factory: "module:function"` 接入，但必须由用户显式提供本地工厂并先完成同组成基准；服务不会隐式下载权重。更详细的版本、许可证/API 风险和验收矩阵见 [`docs/library-research.md`](docs/library-research.md)。

## 测试和发布前验收

```bash
uv run --no-sync pytest -q
uv run --no-sync ruff check .
uv run --no-sync python -m compileall -q src
uv lock --check
python -m json.tool .mcp.json
scripts/run_cli.sh inspect --path examples/ethanol_no_h.cif
```

发布前还应确认：源文件 mtime/字节数/SHA-256 未变化；输出可由 Pymatgen 和 ASE 重新读取；晶胞、坐标、标签、occupancy 和周期 image 与 proposal metadata 一致；短接触、重复位点、价态和金属配位已审查；`.audit.json` 的输出哈希与实际文件一致；没有生成或提交 VASP/DFT/NEB/job 文件。

## 故障排查

**`ModuleNotFoundError: structure_files`**：从仓库根目录执行 `uv sync --python 3.12`，或直接使用 `scripts/run_mcp.sh` / `scripts/run_cli.sh`；两个脚本都会设置 `PYTHONPATH=src`。

**MCP 启动但没有工具**：检查 `codex mcp get structure-files` 或 `claude mcp get structure-files` 的 command 是否指向绝对路径的 `scripts/run_mcp.sh`，确认脚本可执行；不要把 3080 Web URL 当作 MCP 地址。

**POSCAR 转换被拒绝**：CIF 可能仍是非平凡空间群 asymmetric unit。先 `inspect`/`validate`，再明确 `expand_p1=true`；partial occupancy 必须显式指定策略，不能静默有序化。

**`proposal_apply` 报源文件变化**：这是 SHA-256 保护机制。对当前源文件重新运行操作，不能手动修改 manifest 哈希绕过检查。

**目标文件已存在**：使用新的 `destination`，或在确认目标不是源文件后明确设置 `overwrite=true`。

**ML 加载失败**：检查本地模型路径、Python/CUDA/PyTorch 和 MACE 版本；设置 `allow_failure=true` 可以回退几何排序，但失败不代表结构已确认。

**解析器不一致或 `validate.ok=false`**：阅读 `independent_parsers`、`normalized`、`geometry` 和 `warnings`。常见原因是 leading publication-text block、缺少显式 symmetry loop、预展开副本、special-position 重复或 partial occupancy；先分类表示，再决定是否 P1 展开。

## 目录结构

```text
structure-files/
├── src/structure_files/
│   ├── mcp_server.py       # stdio MCP tools
│   ├── cli.py              # JSON CLI
│   ├── service.py          # shared high-level service
│   ├── parsers.py          # Gemmi/Pymatgen/ASE adapters
│   ├── geometry.py         # periodic contacts and candidate graph
│   ├── operations.py       # P1/supercell/basis/translation
│   ├── hydrogen.py         # deterministic H candidate enumeration
│   ├── ml.py               # explicit ASE/MACE adapter and H-only guard
│   ├── ts_search.py        # interpolation and TS-like ranking
│   ├── outcar.py           # read-only OUTCAR parser
│   └── proposals.py        # SHA-256 proposal/apply gate
├── scripts/run_mcp.sh      # DSH/Codex/Claude stdio launcher
├── scripts/run_cli.sh      # MCP-free JSON launcher
├── .mcp.json               # Claude Code project MCP config
├── AGENTS.md               # Codex agent rules
├── CLAUDE.md               # Claude Code agent rules
├── docs/library-research.md
├── examples/
├── tests/
├── pyproject.toml
└── uv.lock
```

运行时 `.dsh-structure-proposals/`、`.audit.json`、缓存和虚拟环境均被 `.gitignore` 排除，不应提交到 Git。许可证为 MIT。
