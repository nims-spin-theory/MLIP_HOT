[English](README.md) [日本語](README_ja.md)

# 基于 MLIP 的高通量优化与热力学分析工具（MLIP-HOT）

MLIP-HOT 是一个用于基于机器学习原子间势（Machine Learning Interatomic Potential, MLIP）计算的综合工具包，覆盖结构优化、形成能评估以及凸包（convex hull）分析等功能。该工具包专注于构建用于计算材料发现的高通量流程（pipeline）。其主要优势是易用性与高性能。

该方法已在我们的论文中描述并展示：
https://arxiv.org/abs/2508.20556 。如使用或扩展 MLIP-HOT，请引用该工作。


## 概览

本代码可以完成：
- **结构优化（Structural Optimization）**：使用多种 MLIP（CHGNet、MatterSim、eSEN-30M-OAM 等）对晶体结构进行优化
- **形成能计算（Formation Energy Calculation）**：使用 MLIP 计算形成能
- **凸包分析（Convex Hull Analysis）**：使用 MLIP 计算到凸包的距离

本仓库也包含一些实用脚本，用于：
- **高通量结构生成（HTP Structure Generation）**：从 POSCAR 或 CIF 输入生成用于筛选组分的结构
- **确定全局最低能结构（Determine Global Minimum）**：从多个局部最低能结构中确定全局最低能结构

#### 主要特性

- **MPI 并行**：通过分布式计算高效处理大规模数据集
- **灵活的任务分发**：可将数据集拆分为多个块并分别提交到不同计算资源
- **全局最低能判定**：从不同初始构型的多次优化结果中识别最低能结构
- **形成能计算**：使用 MLIP 得到的参考能量计算形成能
- **凸包距离分析**：使用 MLIP 得到的参考能量计算到凸包的距离
- **高质量参考结构**：使用来自 OQMD 的 DFT 优化结构作为参考能量计算的初始构型
- **从不同初始结构（应变扰动）进行弛豫**：在开始优化前对结构施加应变
- **原胞转换**：优化前可将结构转换为原胞以提升效率
- **无需 GPU**：使用预训练 MLIP，在 CPU 上也可高效运行


## 支持的 MLIP 模型

本工具包支持以下机器学习原子间势模型：

- **CHGNet**：`chgnet`
- **SevenNet 系列**：
  - `7net-0`
  - `7net-l3i5`
  - `7net-mf-ompa`
- **MatterSim**：`mattersim`
- **EquiformerV2（OMAT）**：
  - `eqV2_31M_omat`
  - `eqV2_86M_omat`
  - `eqV2_153M_omat`
  - `eqV2_31M_omat_mp_salex`
  - `eqV2_86M_omat_mp_salex`
  - `eqV2_153M_omat_mp_salex`
- **eSEN**：`esen_30m_oam`
- **HIENet**：`hienet`

关于 MLIP 的安装说明，请参阅下方 **MLIP 包安装（MLIP Package Installation）** 章节。

本工具包以模块化为设计目标，可将新的 MLIP 模型无缝集成到现有框架中。



## 前置条件

在使用本工具包之前，你需要在系统中安装 **Miniconda** 或 **Anaconda**。Miniconda 是 conda 的精简安装器，用于为不同 MLIP 模型创建隔离的 Python 环境。

**安装 Miniconda：**

1. 从官方网站下载 Miniconda：https://docs.conda.io/en/latest/miniconda.html
2. 选择与你的操作系统匹配的安装包（Linux、macOS 或 Windows）
3. 按照平台对应的安装说明完成安装

**验证安装：**
```bash
conda --version
```

安装好 conda 后，你可以按下方 **MLIP 包安装** 章节所述，为每个 MLIP 模型创建独立环境。


## 使用方法

MLIP-HOT 提供单一入口脚本（`scripts/MLIP_HOT.py`），可通过 YAML 配置运行完整流程（结构优化 → 形成能 → 距离凸包）或运行任意单独阶段。

我们提供了以下示例：

1. 快速开始：一次性运行三个任务的简单示例
2. 仅运行单个任务
3. 为提升效率，将任务分发到多节点
4. 使用多个初始结构确定全局最低能
5. 从 POSCAR 或 CIF 输入生成输入文件


### 1. 快速开始：一次性运行三个任务

#### 创建环境并安装 MLIP
本示例使用 `MatterSim` MLIP。创建 conda 环境并安装 MatterSim：

```bash
conda create -n MLIP_mattersim python=3.9
conda activate MLIP_mattersim
pip install mattersim
```

其他 MLIP 的安装说明见下方 **MLIP 包安装** 章节。我们建议每个 MLIP 包使用独立 conda 环境安装。下方示例均以 `mattersim` 为例。

#### 结构优化、形成能与凸包距离计算

`example` 目录包含一个包含 100 个化合物的示例数据集。该数据集来源于 [DXMag Computational HeuslerDB](https://www.nims.go.jp/group/spintheory/database/)。仓库也提供了预计算结果，便于你验证安装并对比输出。更多文件说明请查看 `example/README.md`。

```bash
# 若未激活环境请先激活
conda activate MLIP_mattersim

MLIP_HOT=../scripts/MLIP_HOT.py
python $MLIP_HOT -c pipeline.yaml
```

如果你将 example 文件夹复制到其他位置，或在真实项目中使用代码，请将 `MLIP_HOT=../scripts/MLIP_HOT.py` 改为你电脑上 `MLIP_HOT.py` 的绝对路径。

> 💡 快速修复：如果遇到“缺少模块”的报错，请安装相应模块。例如缺少 `pyyaml` 时，执行 `pip install pyyaml`。

所有设置都由配置文件 `pipeline.yaml` 控制。下面解释该配置文件各字段的含义：

```yaml
# Select task: pipeline | optimize | form | hull
task: pipeline
# Select interatomic potential model
model: mattersim
# Optional global MPI settings (per-task overrides supported)
mpi_nproc: 10

# Stage 1: Optimization
optimize:
  input:  ./example.csv         # input csv file
  output: example_result_task1  # directory where results will be written
# Stage 2: Formation energy; use default settings
form:
# Stage 3: Hull distance; use default settings
hull:
```

`input` 文件必须包含 `cell`、`positions` 和 `numbers` 三列，用于定义待弛豫的晶体结构。格式要求见示例输入 CSV。我们也提供了脚本，可从 POSCAR 或 CIF 生成输入 CSV（见示例 5）。

工具包会输出以下列：`optimized_formula`、`optimized_cell`、`optimized_positions`、`optimized_numbers`、`Energy (eV/atom)`、`Formation Energy (eV/atom)` 与 `Hull Distance (eV/atom)`。运行过程中会打印进度与细节。输出结果以新增列的方式追加，原有列会完整保留。建议在输入文件中包含标识列，例如 `formula`、`composition`、`ID` 等。

我们在 `example/results` 中提供了预计算结果与 DFT 结果，供你对比。

也可以不用配置文件，直接通过命令行（CLI）设置任务；其等价命令为：

```bash
# 在 example 文件夹中执行
conda activate MLIP_mattersim
MLIP_HOT=../scripts/MLIP_HOT.py

python $MLIP_HOT \
    --task pipeline \
    --model mattersim \
    --mpi_nproc 10 \
    --opt.input ./example.csv \
    --opt.output ./example_result_task1
```

优化结构、形成能与凸包距离会写入 `example_result` 中。

> 💡 提示：你可以同时使用配置文件与 CLI；CLI 参数会覆盖配置文件中的对应值。

### 2. 运行单个任务

每个阶段（optimize / formation energy / hull distance）都可以单独运行。示例配置如下：

```yaml
task: optimize
model: mattersim
mpi_nproc: 10
optimize:
  input:  ./example.csv         # input csv file
  output: example_result_task2  # directory where results will be written
```

```yaml
task: form
model: mattersim
form:
    input:  example_result_task1/structure_optimization_result.csv
    output: example_result_task2/form_result.csv
```

```yaml
task: hull
model: mattersim
mpi_nproc: 4
hull:
  input:  example_result_task2/form_result.csv
  output: example_result_task2/hull_result.csv
```

这些示例配置文件也在 `example` 文件夹中，可通过以下命令执行：

```bash
conda activate MLIP_mattersim
MLIP_HOT=../scripts/MLIP_HOT.py

python $MLIP_HOT -c config2_single_task_optimize.yaml
python $MLIP_HOT -c config2_single_task_form.yaml
python $MLIP_HOT -c config2_single_task_hull.yaml
```

同样的任务也可用等价 CLI 完成：

```bash
python $MLIP_HOT \
    --task optimize \
    --model mattersim \
    --mpi_nproc 4 \
    --optimize.input  ./example.csv \
    --optimize.output example_result_task2
```

```bash
python $MLIP_HOT \
    --task form \
    --model mattersim \
    --form.input  example_result_task1/structure_optimization_result.csv \
    --form.output example_result_task2/form_result.csv
```

```bash
python $MLIP_HOT \
    --task hull \
    --model mattersim \
    --mpi_nproc 4 \
    --hull.input  example_result_task2/form_result.csv \
    --hull.output example_result_task2/hull_result.csv
```

### 3. 为效率将任务分发到多节点

在高通量研究中，需要筛选的化合物数量通常非常大。在 `pipeline` 或 `optimize` 任务中，更高效的方式是将输入数据库拆分为多个块（chunks），在多个计算节点上分别运行，然后在末尾拼接结果。例如将数据库拆分为 20 个块，在 20 台机器上分别计算，最后汇总。

可以使用 `optimize.size` 与 `optimize.rank` 两个参数实现：`size` 指定要生成的块数；`rank` 指定当前计算要处理的块编号（从 $0$ 到 $N_{size}-1$）。

`example` 中提供了一个拆分为 3 个块的示例：

```bash
conda activate MLIP_mattersim
MLIP_HOT=../scripts/MLIP_HOT.py

python $MLIP_HOT -c config3_size_rank.yaml --optimize.size 3 --optimize.rank 0
python $MLIP_HOT -c config3_size_rank.yaml --optimize.size 3 --optimize.rank 1
python $MLIP_HOT -c config3_size_rank.yaml --optimize.size 3 --optimize.rank 2
```

注意：该配置文件与示例 1 相同，只是通过 CLI 增加了两个参数。

当所有块都计算完成后，可用脚本 `../script/concat_csv.py` 进行结果拼接：

```bash
# Concatenate results from multiple chunks
python ../scripts/concat_csv.py \
    -f "./example_result_task3" \
    -p "hull_distance_*.csv" \
    -o "example_result_task3/concat_result.csv"

#   -f, --folder:  Folder path containing CSV files to concatenate
#   -p, --pattern: Glob pattern to match files for concatenation
#                  (e.g., "structure_optimization_result_*.csv", "hull_distance_*.csv")
#   -o, --output:  Output CSV filename for concatenated results
```

该脚本可以拼接任意文件名遵循 `XX_{size}_{rank}.csv` 模式的文件。`concat_csv.py` 会打印被拼接的文件名，并指出是否存在未完成的块。

> 💡 提示：`concat_csv.py` 可用于 `pipeline` 任务与 `optimize` 任务的输出。

### 4. 使用多个初始结构确定全局最低能

一个化合物可能存在多个局部最低能（local minima），其中只有全局最低能（global minimum）才是真正的基态（ground state）。在这种情况下，不同的初始结构可能弛豫到不同局部最低能，对应不同能量。通过比较这些局部最低能并选取最低能者，即可识别基态。类似情况也会出现在基于 DFT 的优化中。

一种做法是准备多个包含不同初始结构的 CSV 文件，并对每个 CSV 分别执行结构优化。

另一种做法是在弛豫前对结构施加不同应变以生成不同初始结构。该代码可以通过 `strain` 参数便捷实现。应变既可以是标量（各向同性应变），也可以是 $3\times 3$ 矩阵（各向异性应变）。弛豫前生成的结构也会写入输出。

我们提供了一个简单示例：

```bash
MLIP_HOT=/Users/xiaoenda/WORK/y_git_repo/MLIP_HOT/scripts/MLIP_HOT.py
python $MLIP_HOT -c config4_strain.yaml --optimize.strain "0.1" --optimize.output example_result_task4/strain1

MLIP_HOT=/Users/xiaoenda/WORK/y_git_repo/MLIP_HOT/scripts/MLIP_HOT.py
python $MLIP_HOT -c config4_strain.yaml --optimize.strain "[[0.1, 0.1, 0.0], [0.1, -0.1, 0.0], [0.0, -0.1, 0.0]]" --optimize.output example_result_task4/strain2
```

> 💡 提示：该特性可与前面演示的 `size` 与 `rank` 组合使用。
> 即：对每个应变设置，使用 `size` 与 `rank` 拆分输入并在末尾拼接。

当所有计算完成后，可使用脚本 `find_global_minimum.py` 识别全局最低能：

```bash
# Find global minimum energies across multiple result files
python  /Users/xiaoenda/WORK/y_git_repo/MLIP_HOT/scripts/find_global_minimum.py \
    -i example_result_task4/strain1/hull_distance.csv \
       example_result_task4/strain2/hull_distance.csv \
    -o example_result_task4/global_min.csv \
    --energy-column "Energy (eV/atom)" \
    --group-by-column "composition"

# Flags:
#   -i, --input:       Multiple input CSV files to compare (space-separated list)
#   -o, --output:      Output file containing ground state structures
#   --energy-column:   Name of the column containing energy values (default: Energy (eV/atom))
#   --group-by-column: Column name used to identify the compound, (default: use index)
#                      i.e. entries with same value are regarded as the same compound.
```

更多参数说明请运行 `python ../scripts/find_global_minimum.py -h`。

> 💡 提示：该脚本可用于 pipeline、optimize、form、hull 任务的输出。

### 5. 从 POSCAR 或 CIF 生成输入文件

由于不同应用的结构生成方式差异较大，我们在 `example/generate_input` 中提供了一个简单示例，演示如何生成输入文件。该示例可从 POSCAR、CIF 或原子序数加载结构，然后通过替换元素生成新结构。生成的结构会保存到 CSV 中，可直接作为 MLIP-HOT 的输入。希望该示例能帮助你编写适配自身使用场景的脚本。

为保持本文档简洁，详细说明放在 notebook 中而非此处。一个重要点：对于使用 `pymatgen` 结构的用户，可以通过下方代码块，将结构列表导出为输入文件。熟悉 `ASE` 与 `phonopy` 的用户也可很容易将其结构转换为 pymatgen 格式。

```python
# structures_list contains pymatgen structures
data_list = []
for idx, modified_structure in enumerate(structures_list):
    cell      = str(modified_structure.lattice.matrix.tolist())
    positions = str(modified_structure.frac_coords.tolist())
    numbers   = str(list(modified_structure.atomic_numbers))
    composition    = str(modified_structure.composition.hill_formula.replace(" ", ""))

    data_list.append({
        'index': idx,
        'composition': composition,
        'cell':      cell,
        'positions': positions,
        'numbers':   numbers
    })

df_structures = pd.DataFrame(data_list)
output_csv_path = "generated_structures.csv"
df_structures.to_csv(output_csv_path, index=False)
```

## MLIP 包安装

本章节提供每个支持的 MLIP 模型的 conda 环境与安装说明。

### CHGNet

网站：https://chgnet.lbl.gov/

```bash
conda create -n MLIP_chgnet python=3.10
conda activate MLIP_chgnet
pip install chgnet
```

### SevenNet

网站：https://github.com/MDIL-SNU/SevenNet

```bash
conda create -n MLIP_7net python=3.10
conda activate MLIP_7net
pip install sevenn
```

### MatterSim

网站：https://github.com/microsoft/mattersim

```bash
conda create -n MLIP_mattersim python=3.9
conda activate MLIP_mattersim
pip install mattersim
```

### HIENet

网站：https://github.com/divelab/AIRS/tree/main/OpenMat/HIENet

```bash
conda create -n MLIP_HIENet python=3.9
conda activate MLIP_HIENet

pip install torch==2.1.2
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.1.2.html

git clone https://github.com/divelab/AIRS.git
cd AIRS/OpenMat/HIENet
pip install .
```

**排错（Troubleshooting）**：如果遇到错误 `OSError: /lib64/libstdc++.so.6: version 'GLIBCXX_3.4.29' not found`，请运行：

```bash
conda install -c conda-forge libstdcxx-ng
```

### EquiformerV2 与 eSEN

网站：https://github.com/facebookresearch/fairchem

网站：https://huggingface.co/facebook/OMAT24/tree/main

EquiformerV2 与 eSEN 模型在 FAIRChem 1.10.0 中实现，可按如下方式安装：

```bash
conda create -n MLIP_fairchem python=3.9
conda activate MLIP_fairchem
pip install fairchem-core==1.10.0
pip install torch_scatter torch_sparse torch_spline_conv torch_geometric
```

**注意**：对于 EquiformerV2 与 eSEN，训练好的模型权重（checkpoints）不包含在 FAIRChem 包中，需要从官网单独下载：https://huggingface.co/facebook/OMAT24/tree/main 。使用这些模型时，请通过 `--checkpoint_path` 指定权重路径：

```bash
mpirun -np 10 python ../scripts/MLIP_optimize.py \
    -d ./example/example_data.csv \
    -m "eqV2_31M_omat" \
    -o "opt_results" \
    --checkpoint_path ./fairchem_checkpoints/eqV2_31M_omat.pt
```


## 热力学稳定性指标：形成能与凸包

### 1. 形成能计算

**形成能（formation energy）** 是一种热力学量，描述化合物由其组成元素在标准参考态形成时的能量变化。它能反映材料的 **稳定性** —— 通常形成能越低（越负）意味着化合物越稳定。

$$ E_\text{form} = E_{\text{compound}} - \sum_i n_i \mu_i $$

其中：
- $E_{\text{compound}}$：化合物能量
- $n_i$：元素 $i$ 的原子数
- $\mu_i$：元素 $i$ 的化学势（通常是每原子能量）

### 2. 距离凸包（Distance above Convex Hull）

**到凸包的距离（distance to the convex hull）** 衡量某化合物的形成能相对于同一化学体系中所有可能竞争相所定义的热力学稳定边界的偏离程度。它量化了该化合物相对于同组分下最稳定相（或稳定相混合物）的不稳定程度。

$$ E_\text{hull} = E_\text{form} - E_\text{form}^\text{(hull)} $$

其中：
- $E_\text{form}$：该化合物的形成能
- $E_\text{form}^\text{(hull)}$：该组分在凸包上的形成能，即热力学稳定相（或稳定相混合物）的形成能


## 引用

如果你在研究中使用本工具包，请引用：

```bibtex
@misc{xiao2025accuratescreeningfunctionalmaterials,
  title={Accurate Screening of Functional Materials with Machine-Learning Potential and Transfer-Learned Regressions: Heusler Alloy Benchmark},
  author={Enda Xiao and Terumasa Tadano},
  year={2025},
  eprint={2508.20556},
  archivePrefix={arXiv},
  primaryClass={cond-mat.mtrl-sci},
  url={https://arxiv.org/abs/2508.20556}
}
```

另外，也请引用你所使用的具体 MLIP 模型。每个模型的引用信息请参考其官方文档与论文（见 **支持的 MLIP 模型** 章节）。


## 故障排查

#### GCC 版本问题

如果遇到与 GCC 版本过旧相关的错误，可以在 conda 环境中升级 GCC：

```bash
conda install -y -c conda-forge gcc=11.3.0
conda install -y -c conda-forge gxx=11.3.0
gcc --version
g++ --version
```

**注意**：运行这些命令前请确保已激活相应的 conda 环境。

## 【额外】通过 API 获取凸包化合物信息的脚本

为提升效率，我们预先从 OQMD 计算得到凸包上的化合物并用 MLIP 进行了计算， 因此可直接计算到凸包的距离。

由于数据库持续增长、参考文件更新可能存在滞后，我们也提供了脚本，可通过 API 从 OQMD 或 Materials Project（MP）获取凸包化合物。完整说明见 [docs/convex_hull_compounds.md](docs/convex_hull_compounds.md)
