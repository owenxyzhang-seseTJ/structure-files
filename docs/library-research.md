# 结构文件处理库调研与选型

本文件记录 `structure-files` preset 在 2026-09-03 的依赖选择、替代方案和
可复现边界。版本以项目 `uv.lock` 为准；PyPI 可用性是发布时的参考，实际运行仍应
用锁文件和本地基准测试确认。

## 核心运行时

| 任务 | 当前库 | 为什么选它 | 约束 |
| --- | --- | --- | --- |
| CIF 原始 block、对称操作、occupancy/disorder | [Gemmi](https://gemmi.readthedocs.io/) 0.7.5 | 能读取多 data block、显式对称操作和原始 loop；适合先做 representation 分类 | 不把 Gemmi 解析成功等同于化学正确；block 选择记录在审计元数据 |
| 独立结构解析和结构变换 | [Pymatgen](https://pymatgen.org/) 2026.5.4 | CIF/POSCAR、晶格、StructureMatcher、周期距离和 VASP 生态成熟 | 只作为一个独立 parser/adapter；解析警告不得被隐藏 |
| 独立 I/O、extXYZ、ASE Calculator 接口 | [ASE](https://wiki.fysik.dtu.dk/ase/) 3.29.0 | 能读写 POSCAR/CIF/XYZ/extXYZ，并提供约束、单点力/能量接口 | 不在本 preset 中调用 NEB、优化器或任务提交；ML 仅显式配置时使用 |
| 对称性复核 | [Spglib](https://spglib.github.io/spglib/) 2.7.0 | 作为空间群和操作数的独立 cross-check | 只报告恢复结果，不覆盖用户要求的 P1 输出 |
| 周期键图 | [NetworkX](https://networkx.org/documentation/stable/) 3.6.1 + NumPy | 处理连通片段、配体/金属配位图，便于 AI 工具消费 | 半径/距离只是候选边；不证明实验 bond order |
| 数值和序列化 | NumPy 2.5.2、SciPy 1.18.1、orjson 3.12.0 | 周期坐标、矩阵、排序和稳定 JSON 输出 | 所有坐标变化都写入 proposal/audit，不覆盖源文件 |

## 可选模型与格式增强

### 补氢和 TS-like 排序

- [MACE](https://github.com/ACEsuit/mace) / `mace-torch` 0.3.16：当前内置适配器。
  通过 ASE Calculator 做同一候选束的 H-only 约束优化，或对同一插值路径做单点能量/力描述。
  建议显式提供已验证的 `model_path`；没有模型或加载失败时回退几何排序。
- [CHGNet](https://github.com/CederGroupHub/chgnet) / `chgnet` 0.4.2、
  [ORB](https://github.com/orbital-materials/orb-models) / `orb-models` 0.7.0、
  [Fairchem/UMA](https://github.com/facebookresearch/fairchem) / `fairchem-core` 2.22.0：
  可通过 `calculator_factory: "module:function"` 接入，不放入核心安装，原因是模型权重、
  CUDA、API 和许可条件随版本/环境变化。接入时必须用 masked-H 金标准集和同一组成回归测试。
- 任何模型都不能从重原子唯一推断实验氢位置；候选必须先满足价态、周期碰撞和预期组成，
  再在同一模型、同一组成内比较。

### VASP 文本和分子化学增强

- `dpdata` 1.1.0、`parsevasp` 3.4.0 作为可选 OUTCAR/POSCAR 文本增强；核心 parser 已能
  只读提取完整离子步，因此它们不是必需依赖。
- `rdkit` 2026.3.5 仅适合孤立分子/SMILES 级价态和官能团辅助检查；不能替代周期晶体
  对称性、跨边界键或金属配位图。

## 明确不纳入 preset 的库/流程

本 preset 不调用 VASP、DFT、RASPA、NEB、Sella、鞍点优化器、作业提交或监控服务。ASE 的
Calculator 接口只在用户明确传入 `ml_json` 时用于单点或固定重原子的 H-only 候选排序，
且报告中始终标记为 `TS-like candidate` 或 `model_preferred`，不是过渡态或实验结构结论。

## 与 Harness 工具的对应关系

1. `scan` / `inspect`：先确认格式、block、晶胞、占位率、无序和 SHA-256。
2. `validate`：Pymatgen + ASE 双解析、周期接触、启发式键图、价态/金属配位和 Spglib。
3. `transform`：P1、超胞、unimodular 基矢、原点/原子或连通片段平移、wrap/unwrap、组分提取。
4. `bond_graph`：只读候选边；`bond_propose`：显式写入 `_geom_bond`，但标准 bond type 保持 `?`，
   同时保存 DSH kind/order/confidence/image 审计字段。
5. `hydrogen_propose`：价态驱动的确定/歧义候选束；可选 MACE/ASE H-only 排序。
6. `ts_search`：相同元素组成的反应物/产物映射、周期安全插值、端点键变化和候选帧排序；
   可选同一模型单点评分，不做 NEB/鞍点优化。
7. `proposal_apply`：唯一写入入口，核对所有源 SHA-256，并生成 `.audit.json`。

## 最低验收矩阵

- 正常 P1、非平凡空间群 asymmetric unit、已预展开/display-image、special-position 和
  disorder/partial occupancy CIF；特别测试 leading `publication_text` block。
- 正交和斜晶胞的对角/非对角超胞、unimodular 基矢、跨边界键和周期片段平移。
- C-H 确定候选、羧酸/磺酸/胺/配位水的多质子化和多取向候选；ML 失败时可审计回退。
- 相同/不同晶胞的 TS-like 路径、元素重排显式 mapping、无晶胞 XYZ、端点无键变化和
  ML 单点评分；输出不得被描述成真实过渡态。
- 每次生成后检查：源文件未改写、proposal SHA 门控、Pymatgen/ASE 可重新读取、坐标/标签/
  occupancy 一致、短接触和金属配位可解释、输出路径没有覆盖源文件。

已有 CIF `_geom_bond` loop 会在不改变原子拓扑的直接 round-trip 中保留；P1 展开、超胞、基矢、
平移、组分提取或补氢后会主动使旧 bond loop 失效，避免把旧晶胞的标签/周期 image 误带到新模型。
