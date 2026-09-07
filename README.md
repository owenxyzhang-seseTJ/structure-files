# dsh-structure-files

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
