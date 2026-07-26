# CHGNet Calculator 兼容 DeePMD-kit 势函数技术改造方案

## 1. 背景与目标

当前项目是面向 CHGNet 的材料计算工作流，覆盖结构生成、结构弛豫、相稳定性、力学性质、应力-应变和分子动力学等模块。现有计算模块中，除结构生成外，大部分物性计算最终依赖 ASE `Atoms.calc` 完成能量、力和应力评估，但模型加载、命令行参数和网页界面仍然强绑定 CHGNet。

本次升级目标是将项目从单一 CHGNet 工作流扩展为通用机器学习势函数工作流，优先兼容 DeePMD-kit，并保留现有 CHGNet 默认行为。

目标使用方式：

```bash
python3 modules/relax_structures.py \
  --potential chgnet \
  --model 0.3.0

python3 modules/relax_structures.py \
  --potential deepmd \
  --model models/frozen_model.pb \
  --type-dict "Hf=0,Nb=1,Ta=2,Ti=3,Zr=4"
```

## 2. 官方接口依据

DeePMD-kit 提供 ASE Calculator 接口，常用入口包括：

- `deepmd.calculator.DP`
- 旧版或 TensorFlow 后端环境中的 `deepmd.tf.calculator.DP`

该 Calculator 面向 ASE `Atoms.calc` 使用，核心参数为模型路径 `model`，并支持可选的 `type_dict`、`head` 等参数。它提供 `energy`、`forces`、`stress` 等属性，满足本项目结构弛豫、单点能、力学、应力-应变和基于 ASE 的 MD 所需的主要接口。

参考：

- [DeePMD-kit calculator API](https://docs.deepmodeling.com/projects/deepmd/en/latest/autoapi/deepmd/calculator/index.html)
- [DeePMD-kit TensorFlow calculator API](https://docs.deepmodeling.com/projects/deepmd/en/latest/autoapi/deepmd/tf/calculator/index.html)
- [ASE calculator interface](https://wiki.fysik.dtu.dk/ase/ase/calculators/calculators.html)

## 3. 当前项目兼容性评估

### 3.1 可直接复用的部分

以下模块已经大量使用 ASE `atoms.calc` 模式，适合先接 DeePMD-kit：

- `modules/relax_structures.py`
- `modules/phase_stability.py`
- `modules/mechanical_properties.py`
- `modules/stress_strain_curve.py`

这些模块主要调用：

- `atoms.get_potential_energy()`
- `atoms.get_forces()`
- `atoms.get_stress(voigt=True)`
- ASE `FIRE` 优化器
- ASE trajectory 输出

DeePMD-kit ASE Calculator 支持这些核心接口，因此改造成本较低。

### 3.2 需要重点改造的部分

`modules/run_molecular_dynamics.py` 当前使用 CHGNet 专属：

```python
from chgnet.model.dynamics import MolecularDynamics
```

该封装不能直接替换为 DeePMD-kit，需要将 MD 执行层改为基于 ASE MD 积分器：

- NVE：`ase.md.verlet.VelocityVerlet`
- NVT：`ase.md.nvtberendsen.NVTBerendsen` 或 Langevin
- NPT：`ase.md.npt.NPT` 或 `ase.md.nptberendsen.NPTBerendsen`

### 3.3 当前限制

当前项目限制如下：

- `modules/model_loader.py` 只支持 CHGNet 内置模型和 `.pth.tar` 文件。
- CLI 参数只有 `--model`，没有势函数类型选择。
- Web UI 的模型选择框文案和模型发现逻辑都是 CHGNet 语义。
- `models/registry.json` 只描述 CHGNet 模型。
- 本机当前 Python 环境尚未安装 `deepmd`，实际运行前需要配置 DeePMD-kit 环境。

## 4. 目标架构

升级后采用通用势函数适配层：

```text
Web UI / CLI
    |
    v
potential_loader.py
    |
    +-- CHGNetPotential
    |       +-- CHGNet.load()
    |       +-- CHGNet.from_file()
    |       +-- CHGNetCalculator
    |
    +-- DeePMDPotential
            +-- deepmd.calculator.DP
            +-- deepmd.tf.calculator.DP fallback
```

统一接口建议：

```python
potential = load_potential(
    potential_name=args.potential,
    model_arg=args.model,
    use_device=args.use_device,
    type_dict=args.type_dict,
    dp_head=args.dp_head,
    dp_backend=args.dp_backend,
)

atoms.calc = potential.make_calculator()
```

## 5. CLI 参数设计

所有依赖势函数的脚本统一新增参数：

```text
--potential chgnet|deepmd
--model <builtin-name-or-model-path>
--type-dict <optional>
--dp-head <optional>
--dp-backend auto|standard|tf
```

参数含义：

| 参数 | 说明 | 示例 |
|---|---|---|
| `--potential` | 势函数类型 | `chgnet` / `deepmd` |
| `--model` | 模型名或模型文件路径 | `0.3.0` / `models/frozen_model.pb` |
| `--type-dict` | DeePMD 元素类型映射 | `Hf=0,Nb=1,Ta=2,Ti=3,Zr=4` |
| `--dp-head` | DeePMD 多头模型 head | `energy` |
| `--dp-backend` | DeePMD import 路径策略 | `auto` |

兼容原则：

- 不传 `--potential` 时默认 `chgnet`。
- 不传 `--model` 时默认 `0.3.0`。
- 现有命令行脚本保持可运行。

## 6. 模型注册表设计

建议升级 `models/registry.json`：

```json
[
  {
    "id": "chgnet-0.3.0",
    "label": "CHGNet 0.3.0 - MPtrj",
    "potential": "chgnet",
    "model": "0.3.0",
    "builtin": true,
    "default": true
  },
  {
    "id": "deepmd-frozen",
    "label": "DeePMD frozen_model.pb",
    "potential": "deepmd",
    "model": "frozen_model.pb",
    "type_dict": "Hf=0,Nb=1,Ta=2,Ti=3,Zr=4",
    "default": false
  }
]
```

自动发现规则建议：

- CHGNet：`*.pth.tar`
- DeePMD TensorFlow frozen model：`*.pb`
- DeePMD PyTorch model：`*.pth`、`*.pt`
- DeePMD saved model 或目录模型：目录内包含模型配置或权重文件时作为候选

## 7. 核心适配层设计

新增文件：

```text
modules/potential_loader.py
```

建议结构：

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class LoadedPotential:
    name: str
    model_id: str
    model_path: Path | None
    calculator_cls: Any | None = None
    model_obj: Any | None = None
    calculator_kwargs: dict[str, Any] | None = None

    def make_calculator(self):
        ...


def load_potential(...):
    if potential_name == "chgnet":
        return load_chgnet_potential(...)
    if potential_name == "deepmd":
        return load_deepmd_potential(...)
    raise SystemExit(...)
```

CHGNet 分支：

```python
model = CHGNet.load(model_name=model_arg, use_device=use_device)
return LoadedPotential(
    name="chgnet",
    model_id=model_arg,
    model_obj=model,
    calculator_cls=CHGNetCalculator,
)
```

DeePMD 分支：

```python
try:
    from deepmd.calculator import DP
except Exception:
    from deepmd.tf.calculator import DP

kwargs = {"model": str(model_path)}
if type_dict:
    kwargs["type_dict"] = parse_type_dict(type_dict)
if dp_head:
    kwargs["head"] = dp_head

return LoadedPotential(
    name="deepmd",
    model_id=str(model_path),
    model_path=model_path,
    calculator_cls=DP,
    calculator_kwargs=kwargs,
)
```

## 8. 模块改造方案

### 8.1 `relax_structures.py`

改造内容：

- 增加通用势函数 CLI 参数。
- 删除或弱化 `require_chgnet()` 直接依赖。
- 使用 `load_potential()`。
- 将：

```python
atoms.calc = make_calculator(model, calculator_cls)
```

替换为：

```python
atoms.calc = potential.make_calculator()
```

预期收益：

- DeePMD 模型可直接用于结构弛豫。
- 原有 CHGNet 流程保持不变。

### 8.2 `phase_stability.py`

改造内容：

- 将 CHGNet 模型加载替换为通用势函数加载。
- 复用 `relax_structures.py` 的通用 calculator 构造。
- `--no-relax` 单点模式和弛豫模式均支持 DeePMD。

注意：

- 相稳定比较依赖模型能量零点。不同势函数之间不能直接比较数值，只能在同一势函数、同一训练基准下比较。

### 8.3 `mechanical_properties.py`

改造内容：

- 使用通用势函数加载。
- 保持应力读取逻辑：

```python
stress = atoms.get_stress(voigt=True) * EV_A3_TO_GPA
```

注意：

- DeePMD 模型必须支持 virial/stress 输出。
- 如果模型不支持 stress，应在运行初期给出明确错误，而不是生成空结果。

### 8.4 `stress_strain_curve.py`

改造内容：

- 同 `mechanical_properties.py`。
- 对 stress 不支持场景增加友好错误提示。

注意：

- 高应变区可能超出 DeePMD 模型训练域，应在输出中标注势函数类型和模型 ID。

### 8.5 `run_molecular_dynamics.py`

建议分两步改造。

第一步：

- CHGNet 继续走原 `MolecularDynamics`。
- DeePMD 走新增 `run_ase_md_for_structure()`。

第二步：

- CHGNet 也可切换到 ASE MD 路线，统一 MD 实现。

DeePMD ASE MD 推荐映射：

| ensemble | ASE 实现 |
|---|---|
| `nve` | `VelocityVerlet` |
| `nvt` | `NVTBerendsen` 或 `Langevin` |
| `npt` | `NPT` 或 `NPTBerendsen` |

输出格式保持现有：

- `md.traj`
- `md.log`
- `md_steps.csv`
- `*_md_final.POSCAR`
- `md_summary.csv`

## 9. Web UI 改造方案

涉及文件：

```text
web_code/templates/index.html
web_code/web_app.py
web_code/static/js/main.js
web_code/static/css/style.css
```

### 9.1 表单字段

将所有 “CHGNet Model” 改为：

```text
Potential
Model
Type Dict
```

建议 UI：

- `Potential`：下拉选择 `CHGNet` / `DeePMD-kit`
- `Model`：下拉或文本框，显示 registry 和自动发现模型
- `Type Dict`：仅 DeePMD 时显示
- `DP Head`：高级选项，可折叠

### 9.2 后端命令构造

`web_code/web_app.py` 中所有计算命令追加：

```python
append_flag(command, "--potential", data.get("potential", "chgnet"))
append_flag(command, "--model", data.get("model", ""))
append_flag(command, "--type-dict", data.get("type_dict", ""))
append_flag(command, "--dp-head", data.get("dp_head", ""))
append_flag(command, "--dp-backend", data.get("dp_backend", "auto"))
```

### 9.3 模型接口

`/api/models` 返回结构建议：

```json
{
  "models": [
    {
      "id": "chgnet-0.3.0",
      "label": "CHGNet 0.3.0 - MPtrj",
      "potential": "chgnet",
      "model": "0.3.0",
      "default": true
    }
  ]
}
```

前端按 `potential` 筛选模型列表。

## 10. 环境与依赖

### 10.1 Python 依赖

现有环境需要增加：

```text
deepmd-kit
```

实际安装方式需根据 CPU/GPU、TensorFlow/PyTorch 后端和系统平台确定。建议在独立 conda 环境中配置，避免与 CHGNet/PyTorch 依赖冲突。

### 10.2 运行前检查

新增环境检查：

```bash
python3 - <<'PY'
from deepmd.calculator import DP
print("DeePMD-kit ASE calculator OK")
PY
```

fallback 检查：

```bash
python3 - <<'PY'
from deepmd.tf.calculator import DP
print("DeePMD-kit TensorFlow ASE calculator OK")
PY
```

## 11. 验证计划

### 11.1 CHGNet 回归测试

必须确认原有功能不退化：

```bash
python3 modules/relax_structures.py \
  --input-dir calculator_runs/generated_structures \
  --output-dir calculator_runs/relaxed_results_test \
  --potential chgnet \
  --model 0.3.0 \
  --nsw 2
```

验收：

- 生成 `relax_summary.csv`
- 至少一个结构状态为 `ok`
- 输出 POSCAR、trajectory、optimizer log

### 11.2 DeePMD 单点能测试

```bash
python3 modules/phase_stability.py \
  --mode fixed \
  --elements "Hf Nb Ta Ti Zr" \
  --ratios "1 1 1 1 1" \
  --phases bcc \
  --samples-per-phase 1 \
  --no-relax \
  --potential deepmd \
  --model models/frozen_model.pb \
  --type-dict "Hf=0,Nb=1,Ta=2,Ti=3,Zr=4"
```

验收：

- 能输出 energy
- `status=ok`

### 11.3 DeePMD 弛豫测试

```bash
python3 modules/relax_structures.py \
  --input-dir calculator_runs/generated_structures \
  --output-dir calculator_runs/deepmd_relaxed_results \
  --input-format auto \
  --potential deepmd \
  --model models/frozen_model.pb \
  --type-dict "Hf=0,Nb=1,Ta=2,Ti=3,Zr=4" \
  --nsw 5
```

验收：

- FIRE 可运行
- 输出 `relaxation_steps.csv`
- 能量和最大力无异常 NaN

### 11.4 DeePMD 力学测试

```bash
python3 modules/mechanical_properties.py \
  --input-dir calculator_runs/deepmd_relaxed_results \
  --output-dir calculator_runs/deepmd_mechanical_results \
  --potential deepmd \
  --model models/frozen_model.pb \
  --type-dict "Hf=0,Nb=1,Ta=2,Ti=3,Zr=4" \
  --strain-max 0.003 \
  --num-points 3
```

验收：

- 能输出 `elastic_tensor_GPa.csv`
- stress 字段非空
- 无 NaN/inf

### 11.5 DeePMD MD 测试

```bash
python3 modules/run_molecular_dynamics.py \
  --input-dir calculator_runs/deepmd_relaxed_results \
  --output-dir calculator_runs/deepmd_md_results \
  --potential deepmd \
  --model models/frozen_model.pb \
  --type-dict "Hf=0,Nb=1,Ta=2,Ti=3,Zr=4" \
  --ensemble nvt \
  --temperature 1000 \
  --steps 10 \
  --loginterval 1
```

验收：

- 输出 `md.traj`
- 输出 `md_steps.csv`
- 温度、势能、动能字段合理

## 12. 风险与处理策略

| 风险 | 影响 | 处理策略 |
|---|---|---|
| DeePMD 未安装 | deepmd 任务无法运行 | 启动前检测并给出安装提示 |
| 元素类型顺序错误 | 结果物理意义错误 | 强制显示并记录 `type_dict` |
| 模型不支持 stress | 力学和应力-应变失败 | 预检查 `atoms.get_stress()` |
| MD 单位差异 | 温度/压力不合理 | 小体系基准测试并记录单位转换 |
| CHGNet 回归 | 原有用户流程受影响 | 默认仍为 `chgnet`，增加回归测试 |
| 不同势函数能量零点不同 | 横向比较误导 | 输出中标记 `potential` 和 `model_id` |

## 13. 输出结果元数据增强

建议所有 summary CSV 增加字段：

```text
potential
model_id
model_path
type_dict
dp_head
```

目的：

- 保证结果可追溯。
- 避免不同势函数结果混合比较。
- 方便 Web UI 按势函数筛选结果。

## 14. 实施里程碑

### Milestone 1：基础势函数抽象

范围：

- 新增 `modules/potential_loader.py`
- CHGNet 通过新接口运行
- DeePMD calculator 可加载

验收：

- `relax_structures.py --potential chgnet` 原样可用
- `relax_structures.py --potential deepmd` 可进入 ASE calculator

### Milestone 2：弛豫、相稳定、力学、应力-应变兼容

范围：

- 改造四个 ASE 友好模块
- 增加 stress 能力预检查
- summary CSV 增加势函数元数据

验收：

- CHGNet 回归通过
- DeePMD 单点和弛豫通过
- DeePMD 力学在支持 stress 的模型上通过

### Milestone 3：MD 通用化

范围：

- 新增 ASE MD 路线
- DeePMD 支持 NVE/NVT
- NPT 作为增强项

验收：

- DeePMD NVE/NVT 10 步测试通过
- 输出格式与现有 Web 预览兼容

### Milestone 4：Web UI 完整接入

范围：

- Potential 下拉框
- 模型按势函数筛选
- DeePMD type_dict 输入
- `/api/models` 支持多势函数 registry

验收：

- 网页可选择 CHGNet 或 DeePMD
- 运行任务命令包含正确参数
- 结果预览和下载保持可用

## 15. 推荐优先级

推荐实施顺序：

1. `potential_loader.py`
2. `relax_structures.py`
3. `phase_stability.py`
4. `mechanical_properties.py`
5. `stress_strain_curve.py`
6. `run_molecular_dynamics.py`
7. Web UI 和模型 registry

原因：

- 弛豫模块最容易验证 DeePMD calculator 是否接通。
- 相稳定复用弛豫逻辑，收益高。
- 力学和应力-应变依赖 stress 支持，应在基础链路稳定后改。
- MD 涉及积分器和单位细节，适合最后处理。

## 16. 最小可行版本定义

MVP 只需完成：

- CLI 支持 `--potential deepmd`
- DeePMD 支持结构弛豫
- Web UI 暂时不接入或只提供文本输入
- CHGNet 原有流程无回归

MVP 验收命令：

```bash
python3 modules/relax_structures.py \
  --input-dir calculator_runs/generated_structures \
  --output-dir calculator_runs/deepmd_relaxed_results \
  --potential deepmd \
  --model models/frozen_model.pb \
  --type-dict "Hf=0,Nb=1,Ta=2,Ti=3,Zr=4" \
  --nsw 5
```

MVP 验收标准：

- 成功生成 `relax_summary.csv`
- 至少一个结构 `status=ok`
- 输出 relaxed POSCAR
- 记录 `potential=deepmd` 和 `model_path`

