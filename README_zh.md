[English](README.md) [日本語](README_ja.md)

# MLIP-based High-throughput Optimization and Thermodynamics (MLIP-HOT)
# 基于 uMLIP 的高通量结构优化与热力学稳定性计算（MLIP-HOT）

MLIP-HOT 是一个基于通用机器学习原子间势（Universal Machine Learning Interatomic Potential, uMLIP）的综合计算工具包，支持结构优化（structure optimization）、形成能（formation energy）计算以及到凸包距离（distance above convex hull）计算。本工具包旨在构建用于材料发现的高通量计算流程。

我们将本代码称为 MLIP-HOT，我们还提供了一个与其配套的工具 [MLIP-FTL](https://github.com/nims-spin-theory/MLIP_FTL)。MLIP-FTL 用于从零开始或通过迁移学习来训练并应用机器学习回归模型（MLRM）进行晶体性质的预测。

此方法的细节和应用展示包含在我们的论文中：[npj Computational Materials (2026) XX:XX](https://www.nature.com/articles/s41524-026-02013-0)。如果使用本代码或进行基于本代码的工作，请引用该论文以及使用的 uMLIP 模型。


## 概览

本工具包可以完成：
- **结构优化**：使用多种 uMLIP（CHGNet、MatterSim、eSEN-30M-OAM 等）对晶体结构进行优化
- **形成能计算**：使用基于 uMLIP 的参考能计算形成能
- **凸包分析**：使用基于 uMLIP 的参考能计算到凸包的距离

本工具包还包含一些实用脚本，用于：
- **高通量结构生成**：从 POSCAR、CIF 或描述晶格向量、原子位置和元素种类的数组生成用于筛选的结构
- **确定全局最低能**：在多个局域极小值中确定全局最低能结构
  
#### 关键特性

- **MPI 并行**：通过分布式计算高效处理大规模数据集
- **灵活的任务分发**：可将数据集分块后分别提交到多个计算节点上运行
- **全局最低能判定**：在不同初始构型的多次优化结果中识别最低能结构
- **预计算参考能**：用于形成能和凸包距离计算的参考能已预先计算，因此这两步的时间花费极小。参考能来自对 OQMD 数据库中的结构进行弛豫得到的结果。
- **通过施加应变从不同初始结构进行弛豫**
- **原胞转换**：可在优化前将结构转换为原胞以提升效率
- **无需 GPU**：本工具包使用预训练 uMLIP，可在 CPU 上高效运行


## 支持的 uMLIP 

本工具包支持以下机器学习原子间势模型：

- **CHGNet**：`chgnet` 
- **SevenNet**：
  - `7net-0` 
  - `7net-l3i5` 
  - `7net-mf-ompa` 
- **MatterSim**：`mattersim` 
- **EquiformerV2**：
  - `eqV2_31M_omat_mp_salex` 
  - `eqV2_86M_omat_mp_salex` 
  - `eqV2_153M_omat_mp_salex` 
- **eSEN**：`esen_30m_oam`
- **HIENet**：`hienet` 

关于 uMLIP 的安装说明，请参见下方的 **uMLIP 包安装** 章节。

新的 uMLIP 模型可以简单地添加到现有工具包中。



## 预先准备

在使用本工具包之前，需要在系统中安装 **Miniconda** 或 **Anaconda**。它们用于为不同 uMLIP 模型创建相互隔离的 Python 环境。

**安装 Miniconda：**

1. 从官方网站下载 Miniconda：https://docs.conda.io/en/latest/miniconda.html
2. 选择与你的操作系统（Linux、macOS 或 Windows）对应的安装包
3. 按照平台安装说明完成安装

**验证安装：**
```bash
conda --version
```

安装 conda 后，你就可以按照下方 **uMLIP 包安装** 章节中的说明为每个 uMLIP 模型创建独立环境。


## 使用方法

MLIP-HOT 提供了单一入口（`scripts/MLIP_HOT.py`），可通过 YAML 配置文件运行完整流程（结构优化 → 形成能 → 凸包距离），也可只运行其中任一阶段。

我们提供了以下示例：

1. 快速开始：一个一次性运行三个任务的简单示例
2. 只运行单个任务
3. 为提高效率，将任务拆分到多个节点上运行
4. 使用多个初始结构确定全局最低能
5. 从 POSCAR、CIF 或描述晶格向量、原子位置和元素种类的数组生成输入文件
6. 介绍一些有用的参数标志

所有输入文件均在 `example` 目录中提供。

### 1. 快速开始：一次运行三个任务的简单示例

#### 创建环境并安装 uMLIP
该示例使用 `MatterSim` uMLIP。创建 conda 环境并安装 MatterSim：

   ```bash
   conda create -n MLIP_mattersim python=3.9
   conda activate MLIP_mattersim
   pip install mattersim
   ```

其他 uMLIP 的安装说明见 **uMLIP 包安装** 章节。我们建议为每个 uMLIP 包使用单独的 conda 环境。

#### 结构优化、形成能和凸包距离计算

`example` 目录中提供了一个包含 10 个化合物的输入示例数据集。该数据集来自 [DXMag Computational HeuslerDB](https://www.nims.go.jp/group/spintheory/database/)。我们也提供了预先计算的结果，便于验证安装并对比输出。

进入 `example` 文件夹并执行如下示例计算。该示例可在个人电脑上几分钟内完成。

```bash
# 如果尚未激活环境，请先激活
conda activate MLIP_mattersim 

# 设置入口脚本路径
export MLIP_HOT=../scripts/MLIP_HOT.py
# 按配置文件启动计算
python $MLIP_HOT -c config1_pipeline.yaml 

# 计算完成
```

如果示例目录被复制到其他位置，或在实际工作中使用本代码，请将 `export MLIP_HOT=../scripts/MLIP_HOT.py` 改为你电脑上 `MLIP_HOT.py` 的绝对路径。

> 💡 可能的错误及解决方法
> 如果遇到提示缺少模块的错误信息，请安装相应的模块。
> 例如，如果缺少 pyyaml 或 mpi4py 模块，请执行：
> pip install pyyaml 或 pip install mpi4py

> 💡 可能的错误及解决方法
> 如果在执行 optimize 任务时遇到以下错误：
> Error processing structure: Optimizer.converged() missing 1 required positional argument: 'gradient'
> 这是由于 ASE 版本导致的问题，请安装以下版本：
> pip install ase==3.24.0

所有设置均由配置文件 `config1_pipeline.yaml` 控制。下面解释该配置文件的内容：

```yaml
# 选择任务：pipeline | optimize | form | hull
# pipeline 会依次执行 optimize、form 和 hull
task: pipeline 
# 选择原子间势模型
# 支持的模型见“支持的 uMLIP”章节
model: mattersim
# 可选：全局 MPI 设置（进程数）
mpi_nproc: 10

# 阶段 1：结构优化
optimize:
  input:  ./example.csv         # 输入 csv 文件
  output: example_result_task1  # 结果写入目录
# 阶段 2：形成能；使用默认设置
form:
# 阶段 3：凸包距离；使用默认设置
hull:
```

`input` 文件必须包含 `cell`、`positions` 和 `numbers` 三列，用于定义待弛豫的晶体结构：
- **cell**：3×3 矩阵列表 `[[a1,a2,a3], [b1,b2,b3], [c1,c2,c3]]`
- **positions**：Nx3 矩阵列表，分数坐标 `[[atom1x,atom1y,atom1z], [atom2x,atom2y,atom2z]...]`
- **numbers**：长度为 N 的原子序数列表 `[atom1,atom2,...]`
  
我们也提供了可从 POSCAR、CIF 或描述晶格向量、原子位置和元素种类的数组生成输入所需 CSV 数据集的脚本（见示例 5）。

工具包会将以下输出列写入 csv：`optimized_formula`、`optimized_cell`、`optimized_positions`、`optimized_numbers`、`Energy (eV/atom)`、`Formation Energy (eV/atom)` 和 `Hull Distance (eV/atom)`。执行过程中会显示进度与细节。在结果文件中，输入的数据集会全部保留，结果会以新的列的形式写入。建议在输入文件中包含诸如 `formula`、`composition` 或 `ID` 等标识列。

`example/more_input` 中有一个包含 100 个化合物的输入示例。使用 MatterSim 的计算结果在 `example/results` 中。结果文件的前 10 行对应上面 10 个化合物的示例。我们也提供了 DFT 结果以便对比。

下图为使用 Jupyter notebook `example/results/analysis.ipynb` 生成的 MatterSim uMLIP 与 DFT 结果的对比：

![alt text](example/results/dft_vs_ml_mattersim.png)

#### 命令行接口（CLI）

同一任务也可以通过命令行进行设置（不使用配置文件）；下面的CLI会进行相同的计算：
```bash
# 在 example 文件夹中执行
conda activate MLIP_mattersim 
export MLIP_HOT=../scripts/MLIP_HOT.py

python $MLIP_HOT \
    --task pipeline \
    --model mattersim \
    --mpi_nproc 10 \
    --opt.input ./example.csv \
    --opt.output ./example_result_task1 
```

> 💡提示：你可以同时使用配置文件和 CLI；CLI 参数将覆盖配置文件中的参数。

### 2. 只运行单个任务

每个任务（optimize / formation energy / hull distance）都可以单独执行。示例配置如下：

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

这些示例配置文件也包含在 `example` 目录中，可通过以下方式使用：

```bash
conda activate MLIP_mattersim 
export MLIP_HOT=../scripts/MLIP_HOT.py

python $MLIP_HOT -c config2_single_task_optimize.yaml
python $MLIP_HOT -c config2_single_task_form.yaml
python $MLIP_HOT -c config2_single_task_hull.yaml
```  

同样也可以用CLI进行相同的计算：

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
    --form.input  example_result_task2/structure_optimization_result.csv \
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

### 3. 为提高效率，将任务拆分到多个节点上运行

在高通量研究中，被筛选的化合物数量通常非常大。在 `pipeline` 或 `optimize` 任务中，更高效的方式是将输入数据库拆分为多个分块（chunks），在多个节点上分别运行各分块。例如，将输入拆分为 20 个分块，每个分块在一台计算机上运行，最后再把所有结果拼接起来。

这个方法可通过 `optimize.size` 与 `optimize.rank` 来进行设定：`size` 指定要生成的分块数量，`rank` 指定当前计算要处理的分块编号（从 $0$ 到 size 的值 $-1$）。

一个使用 3 个分块的示例：

```bash
conda activate MLIP_mattersim 
export MLIP_HOT=../scripts/MLIP_HOT.py

python $MLIP_HOT -c config3_size_rank.yaml --optimize.size 3 --optimize.rank 0
python $MLIP_HOT -c config3_size_rank.yaml --optimize.size 3 --optimize.rank 1
python $MLIP_HOT -c config3_size_rank.yaml --optimize.size 3 --optimize.rank 2
```

注意：该示例使用的配置文件与示例 1 相同，只是在 CLI 中额外加入了 `size` 和 `rank`。

当所有分块计算完成后，可以使用脚本 `../scripts/concat_csv.py` 拼接结果。

```bash
# 拼接多个分块的结果
python ../scripts/concat_csv.py \
    -f "./example_result_task3" \
    -p "hull_distance_*.csv" \
    -o "example_result_task3/concat_result.csv"

#   -f, --folder:  包含待拼接 CSV 的文件夹路径
#   -p, --pattern: 要拼接文件的名字的模式
#                  （例如 "structure_optimization_result_*.csv", "hull_distance_*.csv"）
#   -o, --output:  拼接后的输出 CSV 文件名
```

该脚本可拼接任意满足 `XX_{size}_{rank}.csv` 命名模式的文件。`concat_csv.py` 会打印已拼接的文件名，并指出尚未完成的分块。

> 💡 提示：脚本 `concat_csv.py` 同时适用于 `pipeline` 与 `optimize` 任务的输出。

### 4. 使用多个初始结构确定全局最低能

一个化合物可能存在多个具有局域极小能量的结构，而只有具有全局最低能量的结构才是真正的基态。因此，为了确定基态结构，我们需要对不同初始结构进行弛豫，并找出能量最低的结构。同样的情况也会出现在使用 DFT 的结构优化中。

一种方式是准备多个包含不同初始结构的 CSV 文件，对每个文件分别进行结构优化。

另一种方式是在弛豫前对结构施加不同应变，以生成不同初始结构。这个方法可通过 `strain` 参数完成。应变可以是标量（各向同性应变）或 3×3 矩阵（各向异性应变）。通过施加应变生成的结构也会写入输出的结果中。

我们提供一个简单的示例：

```bash
MLIP_HOT=/Users/xiaoenda/WORK/y_git_repo/MLIP_HOT/scripts/MLIP_HOT.py   
python $MLIP_HOT -c config4_strain.yaml --optimize.strain "0.1" --optimize.output example_result_task4/strain1

MLIP_HOT=/Users/xiaoenda/WORK/y_git_repo/MLIP_HOT/scripts/MLIP_HOT.py   
python $MLIP_HOT -c config4_strain.yaml --optimize.strain "[[0.1, 0.1, 0.0], [0.1, -0.1, 0.0], [0.0, -0.1, 0.0]]" --optimize.output example_result_task4/strain2
```

> 💡 提示：该功能可与前面演示的 `size` 和 `rank` 结合使用。
> 即：对每一种应变都用 `size`/`rank` 将输入拆分为多个分块，弛豫结束后再拼接结果。

当所有计算完成后，可使用脚本 `find_global_minimum.py` 识别全局最低能。

```bash
# 在多个结果文件中寻找全局最低能
python  ../scripts/find_global_minimum.py \
    -i example_result_task4/strain1/hull_distance.csv \
       example_result_task4/strain2/hull_distance.csv \
    -o example_result_task4/global_min.csv \
    --energy-column "Energy (eV/atom)" \
    --group-by-column "composition" 

# 参数：
#   -i, --input:       用于比较的多个 CSV 文件
#   -o, --output:      输出文件（包含全局最低能的结果）
#   --energy-column:   能量列名（默认：Energy (eV/atom)）
#   --group-by-column: 用于识别化合物的列名（默认：使用 index）
#                      即该列取值相同的条目会被视为同一个化合物
```

更多参数请运行 `python ../scripts/find_global_minimum.py -h`。

> 💡 提示：该脚本适用于 pipeline、optimize、form、hull 任务的输出。

### 5. 从 POSCAR、CIF 或描述晶格向量、原子位置和元素种类的数组生成输入文件

`example/generate_input` 中提供了一个简单示例，演示如何生成输入数据集。该示例从 POSCAR、CIF 或描述晶格向量、原子位置和元素种类的数组加载结构，然后通过用不同元素替换原子来生成新结构。生成的结构会保存为 CSV，可直接作为 MLIP-HOT 的输入。我们希望该示例能帮助编写符合自己需求的脚本。为保持本文档简洁，详细说明放在了示例中。

对于使用 `pymatgen` 的用户，可以用下方代码块从 `structure` 对象列表生成输入文件。对 `ASE` 或 `phonopy` 更熟悉的用户，也可使用 `pymatgen` 中的适配器函数将其结构转换为 Pymatgen 格式。

``` python
# structures_list 包含 pymatgen 的 Structure 对象
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

### 6. 一些有用参数简介

#### `fix_symmetry`

在结构优化的任务中，默认的设定是不保持对称性。如需保持对称性，请添加 `--optimize.fix_symmetry True`。

#### `print_col`

添加 `--optimize.print_col <column name>`后，程序会显示正在进行优化的行号和该列的值来显示进度。这对分析错误原因很有帮助。


## uMLIP 包安装

本节为每个支持的 uMLIP 模型提供 conda 环境配置说明。

### CHGNet

Website: https://chgnet.lbl.gov/

```bash
conda create -n MLIP_chgnet python=3.10
conda activate MLIP_chgnet
pip install chgnet
```

### SevenNet

Website: https://github.com/MDIL-SNU/SevenNet

```bash
conda create -n MLIP_7net python=3.10
conda activate MLIP_7net
pip install sevenn
```

注意：由于收敛问题，该模型参考文件中的数值是基于 OQMD 中的结构直接使用 uMLIP 计算得到的，没有使用 uMLIP 结构优化。


### HIENet

Website: https://github.com/divelab/AIRS/tree/main/OpenMat/HIENet

```bash
conda create -n MLIP_HIENet python=3.9
conda activate MLIP_HIENet

pip install torch==2.1.2
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.1.2.html

git clone https://github.com/divelab/AIRS.git
cd AIRS/OpenMat/HIENet
pip install .
```

注意：由于收敛问题，该模型参考文件中的数值是基于 OQMD 中的结构直接使用 uMLIP 计算得到的，没有使用 uMLIP 结构优化。

注意：如果遇到错误 `OSError: /lib64/libstdc++.so.6: version 'GLIBCXX_3.4.29' not found`，请运行：

```bash
conda install -c conda-forge libstdcxx-ng
```


### MatterSim

Website: https://github.com/microsoft/mattersim

```bash
conda create -n MLIP_mattersim python=3.9
conda activate MLIP_mattersim
pip install mattersim
```


### EquiformerV2 与 eSEN

Website: https://github.com/facebookresearch/fairchem
Website: https://huggingface.co/facebook/OMAT24/tree/main

EquiformerV2 与 eSEN uMLIP 包含在 FAIRChem 1.10.0 中，可按如下方式安装：

```bash
conda create -n MLIP_fairchem python=3.9
conda activate MLIP_fairchem
pip install fairchem-core==1.10.0
pip install torch==2.4.1  --index-url https://download.pytorch.org/whl/cpu
pip install torch-scatter torch-sparse torch-spline-conv -f https://data.pyg.org/whl/torch-2.4.1+cpu.html
pip install torch-cluster torch_geometric -f https://data.pyg.org/whl/torch-2.4.1+cpu.html
```

**注意**：对于 EquiformerV2 与 eSEN，训练好的模型不包含在 FAIRChem 包中，需要从官网单独下载：https://huggingface.co/facebook/OMAT24/tree/main。使用这些模型时，请用 `--opt.checkpoint_path` 指定这些模型的位置：

```bash
    --opt.checkpoint_path ./fairchem_checkpoints/eqV2_31M_omat.pt
```


## 热力学稳定性指标：形成能与凸包

### 1. 形成能计算

**形成能（formation energy）** 是衡量化合物由其组元元素在标准参考态形成时能量变化的热力学量，可用于评估材料的 **热力学稳定性**——通常形成能越低（越负），材料越稳定。

$$ E_\text{form} = (E_{\text{compound}} - \sum_i n_i \mu_i ) /n $$

其中：
- $E_{\text{compound}}$：化合物能量
- $n_i$：化合物中元素 $i$ 的原子数
- $\mu_i$：元素 $i$ 的能量（每原子）
- $n$：化合物总原子数，因此单位为 (eV/atom)

### 2. 距离凸包（Distance above Convex Hull）

**到凸包的距离** 衡量某化合物的形成能相对于其化学体系中所有可能竞争相所定义的稳定性极限（凸包）高出多少。它量化了该化合物相对于同组分下最稳定相（或相混合）的不稳定程度。数值越小通常越稳定：原则上热力学稳定的化合物满足 $E_\text{hull}=0$。实际中，有限温度效应与方法不确定性会引入偏差，因此通常会使用一个容差（例如此工作中将 $E_\text{hull} < 0.22 \text{eV/atom}$ 视为稳定：https://doi.org/10.1016/j.actamat.2025.121312）。

$$ E_\text{hull} = E_\text{form} - E_\text{form}^\text{(hull)} $$

其中：
- $E_\text{form}$：化合物的形成能
- $E_\text{form}^\text{(hull)}$：该组分下热力学稳定相（或相混合）的形成能，即凸包上的能量


## 问题与反馈

如果你有任何问题、故障或 bug，请在本仓库中提交 issue。提供环境信息与最小可复现示例将有助于我们更快定位问题。


## 引用

如果你在研究中使用或扩展本工具包，请引用：

```bibtex
@article{npjcm_2026accuratescreeningfunctionalmaterials,
	date = {2026/02/19},
	doi = {10.1038/s41524-026-02013-0},
	journal = {npj Computational Materials},
	title = {Accurate screening of functional materials with machine-learning potential and transfer-learned regressions: Heusler alloy benchmark},
	url = {https://doi.org/10.1038/s41524-026-02013-0},
	year = {2026},
}
```

请同时引用你所使用的具体 uMLIP 模型。相关信息见 **uMLIP 包安装** 章节中每个模型的官方文档。

如果进行了凸包距离计算，也请根据官方说明引用 OQMD 数据库：https://oqmd.org/documentation/publications。



## 故障排查

#### GCC 版本问题

如果你遇到与 GCC 版本过旧相关的错误，可以在 conda 环境中按如下命令升级 GCC：

```bash
conda install -y -c conda-forge gcc=11.3.0
conda install -y -c conda-forge gxx=11.3.0
gcc --version
g++ --version
```

**注意**：运行这些命令前请确保已激活对应 conda 环境。

## [附加] 通过 API 获取凸包化合物信息的脚本

为提升效率，我们预先计算了 OQMD 中凸包化合物的形成能，并将结果存储为参考文件，从而可直接计算凸包距离。

由于数据库会持续增长，参考文件的更新可能偶尔滞后，因此我们也提供脚本通过 OQMD 或 Materials Project（MP）的 API 获取凸包化合物。完整说明见 [docs/get_convex_hull_compounds.md](docs/get_convex_hull_compounds.md)。
