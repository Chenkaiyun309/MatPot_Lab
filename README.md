# MatPot_lab

面向材料结构生成、势函数计算与结果分析的本地工作台。项目提供命令行工具和单页网页工作台，覆盖从晶体结构建模到弛豫、声子、相稳定性、弹性性质、应力–应变与分子动力学的完整流程。

![MatPot_lab 顶栏](logo\MatPot_lab_logo.png)

## 功能

- **结构建模**：自由晶胞（a/b/c + α/β/γ + 分数坐标站位）单晶建模；内置 Atomsk 的 Voronoi 多晶建模；外部结构多文件批量导入（VESTA / Materials Studio / VASP 等导出的 CIF / POSCAR / LAMMPS / XYZ 等）。
- **一个项目多个结构**：项目内可保存任意多份结构文件，每个计算任务单独选择输入结构。
- **静态计算**：结构弛豫、声子、相稳定性（固定成分 / 二元扫描）、弹性常数、应力–应变曲线。
- **分子动力学**：NVE / NVT / NPT 运行、MD 弛豫、MD 应力–应变，输出轨迹与热力学日志。
- **势函数**：统一适配层支持 CHGNet、DeePMD-kit、MACE、SevenNet 与 MatGL（仅机器学习势）；网页端实时显示各后端安装状态。
- **任务中心与结果分析**：任务提交、状态轮询、日志查看、终止；结果页按阶段目录预览结构（可交互旋转）、CSV 表格、曲线图与日志。
- **多晶建模内置 Atomsk**：`modules/atomsk` 随项目分发，无需用户单独安装。

## 目录

```text
calculator_projects/  各项目的结构、计算结果与元数据（按 项目ID/阶段 子目录组织）
docs/              技术方案
logo/              品牌资源
models/            势函数模型与注册信息
modules/           命令行计算模块（含势函数适配层与内置 atomsk）
web_code/          单页网页工作台（templates + static SPA）
```

## 快速开始

建议使用独立 Conda 环境（需要 `pymatgen`、`ase`、`numpy`、`matplotlib`；CHGNet 后端另需 `chgnet`），并在项目根目录运行命令。

```bash
conda activate conda_py13
python web_code/web_app.py --check
python web_code/web_app.py
```

网页默认在 `http://127.0.0.1:5055` 启动。若只需启动服务、不自动打开浏览器：

```bash
NO_OPEN_BROWSER=1 python web_code/web_app.py
```

打开网页后：新建/选择项目 → 结构建模（单晶 / 多晶 / 导入）→ 静态计算或分子动力学中选择结构与势函数 → 任务中心跟踪 → 结果分析预览。

## 命令行示例

生成自由晶胞结构（a / b / c 与三个晶轴夹角全部手工输入）：

```bash
python modules/generate_structures.py \
  --builder custom \
  --cell-a 3.0 --cell-b 3.0 --cell-c 3.0 \
  --cell-alpha 90 --cell-beta 90 --cell-gamma 90 \
  --ordered-sites "Fe:0,0,0;Fe:0.5,0.5,0.5" \
  --supercell 2 2 2 \
  --output-dir calculator_projects/<项目>/model/base
```

使用 CHGNet 进行结构弛豫：

```bash
python modules/relax_structures.py \
  --input-dir calculator_projects/<项目>/model/base \
  --output-dir calculator_projects/<项目>/calculations/relax \
  --potential chgnet \
  --model 0.3.0
```

使用 DeePMD-kit 进行结构弛豫：

```bash
python modules/relax_structures.py \
  --input-dir calculator_projects/<项目>/model/base \
  --output-dir calculator_projects/<项目>/calculations/relax \
  --potential deepmd \
  --model models/frozen_model.pb \
  --type-dict "Hf=0,Nb=1,Ta=2,Ti=3,Zr=4" \
  --nsw 200
```

运行 NVT 分子动力学：

```bash
python modules/run_molecular_dynamics.py \
  --input-dir calculator_projects/<项目>/calculations/relax \
  --output-dir calculator_projects/<项目>/calculations/md_relax \
  --potential chgnet \
  --model 0.3.0 \
  --ensemble nvt \
  --temperature 1000 \
  --steps 1000
```

## 势函数后端与扩展

项目通过 `modules/potential_loader.py` 统一加载势函数。新增后端时，可在此适配层注册新的势函数类型，并沿用现有任务、网页表单和结果元数据流程。

当前支持的后端：

- **CHGNet**：默认材料势函数后端。
- **DeePMD-kit**：支持 ASE Calculator、`type_dict`、多任务模型 head 与导入后端选择。
- **MACE**：支持官方基础模型和 `models/mace/` 中的自定义检查点。
- **SevenNet**：支持官方预训练模型和 `models/sevennet/` 中的自定义检查点。
- **MatGL**：支持 TensorNet、M3GNet 等通过 MatGL 模型注册表发布的势函数。

所有后端都通过同一个 `LoadedPotential.make_calculator()` 接口向弛豫、相稳定性、弹性、应力–应变、声子和分子动力学模块提供 ASE Calculator。

### 可选后端安装

不同框架依赖较重，建议按需安装，不要一次性装入生产环境：

```bash
# 一次安装本次新增的三个后端
pip install -r requirements_mlips.txt

# MACE
pip install -r requirements_mace.txt

# SevenNet
pip install -r requirements_sevennet.txt

# MatGL / TensorNet / M3GNet
pip install -r requirements_matgl.txt

# DeePMD-kit
pip install -r requirements_deepmd.txt
```

CUDA 环境应先根据驱动安装匹配的 PyTorch 或 DeePMD-kit 构建。Apple Silicon 可在网页的 Device 字段选择 `Apple MPS`；若某个框架或算子不支持 MPS，请改用 CPU。

内置模型选择包括：


| 后端     | 默认/预置模型                        |
| ---------- | -------------------------------------- |
| MACE     | `medium-mpa-0`                       |
| SevenNet | `7net-0`                             |
| MatGL    | `TensorNet-PES-MatPES-PBE-2025.2`    |
| MatGL    | `TensorNet-PES-MatPES-r2SCAN-2025.2` |
| MatGL    | `M3GNet-PES-MatPES-PBE-v2025.1`      |

基础模型可能在第一次使用时从模型发布平台下载，离线环境应提前缓存模型或在 `models/registry.json` 中注册本地检查点。

使用 MACE 进行结构弛豫：

```bash
python modules/relax_structures.py \
  --input-dir calculator_projects/<项目>/model/base \
  --output-dir calculator_projects/<项目>/calculations/relax \
  --potential mace \
  --model medium-mpa-0 \
  --use-device cpu
```

使用 SevenNet 进行结构弛豫：

```bash
python modules/relax_structures.py \
  --input-dir calculator_projects/<项目>/model/base \
  --output-dir calculator_projects/<项目>/calculations/relax \
  --potential sevennet \
  --model 7net-0 \
  --use-device auto
```

使用 MatGL TensorNet：

```bash
python modules/relax_structures.py \
  --input-dir calculator_projects/<项目>/model/base \
  --output-dir calculator_projects/<项目>/calculations/relax \
  --potential matgl \
  --model TensorNet-PES-MatPES-PBE-2025.2 \
  --use-device cpu
```

### DeePMD-kit 配置

DeePMD-kit 的 CPU/GPU、TensorFlow/PyTorch 后端需要根据计算机环境单独选择。安装匹配版本后，放置模型到 `models/`，或在网页/命令行中传入模型路径：

```bash
pip install -r requirements_deepmd.txt
```

常用参数：


| 参数                            | 说明                         |
| --------------------------------- | ------------------------------ |
| `--potential deepmd`            | 选择 DeePMD-kit 势函数       |
| `--model <path>`                | DeePMD 模型文件或目录        |
| `--type-dict "Ti=0,Nb=1"`       | 元素到模型 type index 的映射 |
| `--dp-head energy`              | 多任务模型的可选 head        |
| `--dp-backend auto|standard|tf` | Calculator 导入策略          |

力学、应力–应变与 NPT 计算要求所选模型能够输出 virial/stress。不同势函数的能量零点不同，请勿直接混合比较它们的绝对能量。

## 输出与追溯

每个任务会输出 CSV、结构文件、轨迹和日志。计算摘要包含以下元数据，便于追溯与筛选：

```text
potential, model_id, model_path, type_dict, dp_head, dp_backend, use_device
```

网页任务的所有输入与输出都保存在 `calculator_projects/<项目ID>/` 对应阶段子目录中；网页会自动读取这些结果用于预览与下载。命令行单独运行模块时请通过 `--input-dir` / `--output-dir` 指定项目内的目录。

## 技术方案

当前 CHGNet → DeePMD-kit 兼容改造方案见 [docs/deepmd_upgrade_plan.md](docs/deepmd_upgrade_plan.md)。
