[English](README.md) [日本語](README_ja.md)

# 基于机器学习原子间相互作用势的高通量结构优化与热力学工具包（MLIP-HOT）
> **注**：本文档由 AI 生成，后由人工修正。
> 
用于机器学习原子间相互作用势（MLIP）计算的综合工具包，包括结构优化、形成能评估和凸包分析。本工具包专注于为计算材料发现构建高通量流程。该工具包的主要优势是使用简便和性能高效。

方法在我们的论文中进行了详细描述和演示：
https://arxiv.org/abs/2508.20556。如使用或扩展MLIP-HOT，请引用此工作。


## 概览

主要功能：
- **结构优化**：使用多种 MLIP（CHGNet、MatterSim、eSEN-30M-OAM 等）优化晶体结构
- **形成能计算**：基于 MLIP 评估形成能
- **凸包距离分析**：基于 MLIP 计算距凸包能差

我们还提供了一些其他实用的工具代码：
- **高通量结构生成**：从 POSCAR 或 CIF 生成筛选用结构
- **相图绘制**：生成所选化合物相图 （凸包图）
- **全局最低能确定**：对多初始结构的结果做能量筛选来获得全局最低能量结构

#### 关键特性

- **MPI 并行**：通过并行运算加速大规模数据处理
- **灵活任务拆分**：可将数据按块分发到多节点
- **全局最低能识别**：多初始构型、多次优化后自动选取最低能
- **形成能/凸包距离**：基于 MLIP 参考能量快速评估
- **高质量参考结构**：使用 OQMD 的 DFT 优化结构作为参考
- **预应变扰动**：优化前可对初始结构施加应变以探索构型
- **原胞转换**：可在优化前转为原胞以提升效率
- **CPU 友好**：无需 GPU


## 支持的 MLIP 模型

- **CHGNet**：`chgnet`
- **SevenNet 系列**：`7net-0`、`7net-l3i5`、`7net-mf-ompa`
- **MatterSim**：`mattersim`
- **EquiformerV2 (OMAT)**：`eqV2_31M_omat`、`eqV2_86M_omat`、`eqV2_153M_omat`、`eqV2_31M_omat_mp_salex`、`eqV2_86M_omat_mp_salex`、`eqV2_153M_omat_mp_salex`
- **eSEN**：`esen_30m_oam`
- **HIENet**：`hienet`

安装说明见文末 **MLIP 包安装** 章节。

本代码可较轻松的加入新模型。


## 前置环境

建议先安装 **Miniconda** 或 **Anaconda**，为各 MLIP 创建隔离环境避免相互影响。

安装步骤简述：
1. 下载：https://docs.conda.io/en/latest/miniconda.html
2. 按操作系统选择安装包并完成安装
3. 验证：
   ```bash
   conda --version
   ```

随后可按 **MLIP 包安装** 章节为每个模型创建独立环境。


## 用法

使用 `scripts/MLIP_HOT.py` 和 配置文件可以直接运行全流程（结构优化 → 形成能 → 凸包距离）或任一单独阶段。示例涵盖：
1. 快速上手：一次运行三步 （结构优化 → 形成能 → 凸包距离）
2. 单独运行某一任务
3. 多节点拆分任务提高速度
4. 多初始结构寻找全局最低能
5. 从 POSCAR 或 CIF 生成不同成分的筛选用结构

### 1. 快速上手：一次运行三步 （结构优化 → 形成能 → 凸包距离）

#### 创建环境并安装 MLIP
以 `MatterSim` 为例：
```bash
conda create -n MLIP_mattersim python=3.9
conda activate MLIP_mattersim
pip install mattersim
```
其他模型的安装见 **MLIP 包安装**。建议每个 MLIP 使用独立 conda 环境。

#### 结构优化 + 形成能 + 凸包距离

`example` 目录提供 100 个化合物示例，源自 [DXMag Computational HeuslerDB](https://www.nims.go.jp/group/spintheory/database/)，并附预计算结果用于对比。详见 `example/README.md`。

```bash
# 若尚未激活环境
conda activate MLIP_mattersim 

MLIP_HOT=../scripts/MLIP_HOT.py
python3 $MLIP_HOT -c pipeline.yaml 
```
若示例目录被拷贝至其他位置或用于实际数据，请将 `MLIP_HOT` 设置为本机 `MLIP_HOT.py` 的绝对路径。

> 快速排查
> - 缺失依赖时按提示安装，如缺少 `pyyaml`，执行 `pip install pyyaml`

所有设置均在 `pipeline.yaml` 控制：

```yaml
# 任务：pipeline | optimize | form | hull
task: pipeline 
# MLIP 模型
model: mattersim
# 可选 MPI，全局设置，nproc>1 启用 MPI
mpi_nproc: 4
# 阶段 1：结构优化
optimize:
    input: ./example_data.csv # 输入 CSV
    output: ./example_result  # 输出目录
# 阶段 2：形成能（默认参数）
form:
# 阶段 3：凸包距离（默认参数）
hull:
```

输入需包含 `cell`、`positions`、`numbers` 列描述初始结构。输出会在原列基础上追加 `optimized_formula`、`optimized_cell`、`optimized_positions`、`optimized_numbers`、`Energy (eV/atom)`、`Formation Energy (eV/atom)`、`Hull Distance (eV/atom)`。

亦可用命令行点式参数（无配置文件）：
```bash
# 在 example 目录
python3 $MLIP_HOT \
    --task pipeline \
    --model mattersim \
    --mpi_nproc 4 \
    --opt.input ./example_data.csv \
    --opt.output ./example_result 
```
> 提示：配置文件与 CLI 可共用，CLI 参数会覆盖配置文件。

### 2. 仅运行单个任务

形成能与凸包距离的示例配置：
```yaml
task: form 
model: mattersim
form:
    input:  ./example_data.csv
    output: ./example_data_formation_energy.csv
```
```yaml
task: hull 
model: mattersim
mpi_nproc: 4
hull:
    input:  ./example_data_formation_energy.csv
    output: ./example_data_hull_distance.csv
```
对应 CLI：
```bash
python3 $MLIP_HOT \
    --task form \
    --model mattersim \
    --form.input  ./example_data.csv \
    --form.output ./example_data_formation_energy.csv
```
```bash
python3 $MLIP_HOT \
    --task hull \
    --model mattersim \
    --mpi_nproc 4 \
    --hull.input  ./example_data_formation_energy.csv \
    --hull.output ./example_data_hull_distance.csv
```

### 3. 多节点拆分加速

大规模筛选可将数据按块拆分并分节点运行，通过 `size`（块数）与 `rank`（当前块编号，0 起）控制：
```yaml
# 任务：pipeline | optimize | form | hull
task: pipeline 
model: mattersim
mpi_nproc: 4
optimize:
    input: ./example_data.csv
    output: ./example_result 
    size: 4 # 块数
    rank: 0 # 块编号 0..size-1
form:
hull:
```
所有块完成后，用 `concat_csv.py` 合并：
```bash
python ../scripts/concat_csv.py \
    -f "./example_result" \
    -p "structure_optimization_result_*.csv" \
    -o example_data_structure_optimization_result.csv
```
脚本会列出参与合并的文件并提示缺失的块。更多功能见 `python ../scripts/concat_csv.py -h`。

> 提示：`concat_csv.py` 既适用于 `pipeline` 也适用于单独的 `optimize` 输出。

### 4. 多初始结构寻找全局最低能

与 DFT 类似，不同初始结构可能收敛到不同局域极小。可准备多份 CSV 分别优化，或在一次优化中通过 `strain` 施加预应变以生成不同起点。

示例：
```yaml
# 各向同性应变
task: pipeline
model: mattersim
mpi_nproc: 4
optimize:
    input: ./example_data.csv
    output: example_result_isotropic
    strain: 0.1
```
```yaml
# 各向异性应变（3x3 矩阵）
task: pipeline
model: mattersim
mpi_nproc: 4
optimize:
    input: ./example_data.csv
    output: example_result_anisotropic
    strain: [[0.1, 0.1, 0.0], [0.1, -0.1, 0.0], [0.0, -0.1, 0.0]]
```
应变后的结构存放在 `strained_cell`、`strained_positions`、`strained_numbers` 列。

用 `find_global_minimum.py` 识别全局最低能：
```bash
python ../scripts/find_global_minimum.py \
    -i example_data_result_test1.csv \
       example_data_result_test2.csv \
    -o example_data_result_global_min.csv \
    --labels composition
```
> 提示：可与前述 `size`/`rank` 块拆分策略组合使用。

### 5. 从 POSCAR/CIF 生成筛选结构

使用提供的脚本可将 POSCAR 或 CIF 转为筛选输入 CSV。常见流程：
1) 准备含目标化学计量的 POSCAR 或 CIF；
2) 运行脚本生成包含 `cell`、`positions`、`numbers` 列的 CSV；
3) 将生成的 CSV 作为 `optimize.input` 或 `pipeline` 的输入继续后续步骤。



## MLIP 包安装

### CHGNet
官网：https://chgnet.lbl.gov/
```bash
conda create -n MLIP_chgnet python=3.10
conda activate MLIP_chgnet
pip install chgnet
```

### SevenNet
官网：https://github.com/MDIL-SNU/SevenNet
```bash
conda create -n MLIP_7net python=3.10
conda activate MLIP_7net
pip install sevenn
```

### MatterSim
官网：https://github.com/microsoft/mattersim
```bash
conda create -n MLIP_mattersim python=3.9
conda activate MLIP_mattersim
pip install mattersim
```

### HIENet
官网：https://github.com/divelab/AIRS/tree/main/OpenMat/HIENet
```bash
conda create -n MLIP_HIENet python=3.9
conda activate MLIP_HIENet

pip install torch==2.1.2
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.1.2.html

git clone https://github.com/divelab/AIRS.git
cd AIRS/OpenMat/HIENet
pip install .
```
故障排查：若出现 `OSError: /lib64/libstdc++.so.6: version 'GLIBCXX_3.4.29' not found`，执行：
```bash
conda install -c conda-forge libstdcxx-ng
```

### EquiformerV2 与 eSEN
官网：https://github.com/facebookresearch/fairchem
模型权重：https://huggingface.co/facebook/OMAT24/tree/main

EquiformerV2 与 eSEN 集成于 FAIRChem 1.10.0，可按下述安装：
```bash
conda create -n MLIP_fairchem python=3.9
conda activate MLIP_fairchem
pip install fairchem-core==1.10.0
pip install torch_scatter torch_sparse torch_spline_conv torch_geometric
```
注意：需要单独下载模型 checkpoint，并在运行时通过 `--checkpoint_path` 指定，例如：
```bash
mpirun -np 10 python ../scripts/MLIP_optimize.py \
    -d ./example/example_data.csv \
    -m "eqV2_31M_omat" \
    -o "opt_results" \
    --checkpoint_path ./fairchem_checkpoints/eqV2_31M_omat.pt
```


## 热力学稳定性指标：形成能与凸包

### 1. 形成能
形成能衡量化合物由基态元素生成的能量变化，是判断稳定性的关键指标。形成能越低（越负），化合物通常越稳定。

$$ E_{\text{form}} = E_{\text{compound}} - \sum_i n_i \mu_i $$
其中：
- $E_{\text{compound}}$：化合物总能量
- $n_i$：元素 $i$ 的原子数
- $\mu_i$：元素 $i$ 的化学势（一般为每原子能量）

### 2. 距离凸包

距离凸包度量化合物形成能高于对应化学体系热力学稳定边界的幅度，反映其相对不稳定性。距离凸包越小， 化合物通常越稳定。

$$ E_{\text{hull}} = E_{\text{form}} - E_{\text{form}}^{(\text{hull})} $$

其中：
- $E_{\text{form}}$：化合物形成能
- $E_{\text{form}}^{(\text{hull})}$：在该化学计量下热力学稳定相（或相组合）的形成能，即凸包能量


## 引用

若本工具对您的研究有帮助，请引用：
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
并请根据所用 MLIP 模型引用对应文献。


## 常见问题

#### GCC 版本
若遇到 GCC 版本过旧，可在环境内升级：
```bash
conda install -y -c conda-forge gcc=11.3.0
conda install -y -c conda-forge gxx=11.3.0
gcc --version
g++ --version
```
注意执行前需激活对应环境。


## [附] 通过 API 获取凸包化合物

我们预先用 MLIP 评估了 OQMD 的凸包化合物并随仓库提供。若需更新或从 Materials Project 获取，可使用提供的脚本，详见 [docs/convex_hull_compounds.md](docs/convex_hull_compounds.md)。
