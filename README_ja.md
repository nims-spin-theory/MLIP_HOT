[English](README.md) [中文](README_zh.md)

# MLIP に基づく高スループット最適化と熱力学解析（MLIP-HOT）

MLIP-HOT は、機械学習原子間ポテンシャル（Machine Learning Interatomic Potential, MLIP）に基づく計算のための包括的なツールキットです。構造最適化、形成エネルギー評価、凸包（convex hull）解析を含みます。本ツールキットは計算材料探索に向けた高スループット・パイプラインの構築に焦点を当てています。主な利点は、使いやすさと高い性能です。

手法の説明とデモは以下の論文にあります：
https://arxiv.org/abs/2508.20556 。MLIP-HOT を使用または拡張する場合は、この研究を引用してください。


## 概要

このコードでできること：
- **構造最適化（Structural Optimization）**：各種 MLIP（CHGNet、MatterSim、eSEN-30M-OAM など）を用いた結晶構造最適化
- **形成エネルギー計算（Formation Energy Calculation）**：MLIP を用いた形成エネルギー計算
- **凸包解析（Convex Hull Analysis）**：MLIP を用いた凸包からの距離計算

このリポジトリには、以下の便利なスクリプトも含まれます：
- **高スループット構造生成（HTP Structure Generation）**：POSCAR または CIF 入力から、組成スクリーニング用の構造を生成
- **グローバル最小の決定（Determine Global Minimum）**：複数の局所最小からグローバル最小を決定

#### 主な機能

- **MPI 並列化**：分散計算により大規模データセットを効率的に処理
- **柔軟なジョブ分配**：データセットのチャンクを分割し、複数計算資源へ個別に投入可能
- **グローバル最小の決定**：異なる初期配置による複数の最適化結果から最小エネルギー構造を同定
- **形成エネルギー計算**：MLIP 由来の参照エネルギーを用いて形成エネルギーを計算
- **凸包距離解析**：MLIP 由来の参照エネルギーを用いて凸包距離を計算
- **高品質な参照構造**：OQMD 由来の DFT 最適化構造を参照エネルギー計算の初期構造として利用
- **異なる初期構造（ひずみ摂動）からの緩和**：最適化開始前に構造へひずみを付与
- **原始胞変換**：効率向上のため、最適化前に構造を原始胞へ変換可能
- **GPU 不要**：事前学習済み MLIP を使用し、CPU で効率的に実行可能


## 対応 MLIP モデル

本ツールキットは以下の機械学習原子間ポテンシャル（MLIP）モデルに対応しています：

- **CHGNet**：`chgnet`
- **SevenNet バリアント**：
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

MLIP のインストール手順は下記の **MLIP パッケージのインストール** セクションを参照してください。

本ツールキットはモジュール性を重視して設計されており、新しい MLIP モデルを既存フレームワークへシームレスに統合できます。



## 前提条件

このツールキットを使用する前に、システムへ **Miniconda** または **Anaconda** をインストールしてください。Miniconda は conda の最小インストーラで、異なる MLIP モデルごとに独立した Python 環境を作成するために使用します。

**Miniconda のインストール：**

1. 公式サイトから Miniconda をダウンロード：https://docs.conda.io/en/latest/miniconda.html
2. OS（Linux / macOS / Windows）に対応するインストーラを選択
3. プラットフォームごとの手順に従ってインストール

**インストール確認：**
```bash
conda --version
```

conda をインストールしたら、下記 **MLIP パッケージのインストール** セクションの説明に従い、各 MLIP モデル用に環境を分けて作成できます。


## 使い方

MLIP-HOT は単一のエントリポイント（`scripts/MLIP_HOT.py`）を提供します。YAML 設定によりフル・パイプライン（構造最適化 → 形成エネルギー → 凸包からの距離）を実行することも、各ステージのみを実行することも可能です。

以下の例を示します：

1. クイックスタート：3 つのタスクを一度に実行する簡単な例
2. 単一タスクの実行
3. 効率のため複数ノードへジョブを分割
4. 複数初期構造からグローバル最小を決定
5. POSCAR / CIF 入力から入力ファイルを生成


### 1. クイックスタート：3 つのタスクを一度に実行

#### 環境作成と MLIP のインストール
この例では `MatterSim` MLIP を使用します。conda 環境を作成し MatterSim をインストールします。

```bash
conda create -n MLIP_mattersim python=3.9
conda activate MLIP_mattersim
pip install mattersim
```

他の MLIP のインストール手順は **MLIP パッケージのインストール** セクションにあります。各 MLIP は別々の conda 環境にインストールすることを推奨します。以下の例は `mattersim` を使用します。

#### 構造最適化・形成エネルギー・凸包距離の計算

`example` ディレクトリに 100 化合物のサンプルを含みます。このデータセットは [DXMag Computational HeuslerDB](https://www.nims.go.jp/group/spintheory/database/) から取得しました。インストール確認および出力比較のため、事前計算結果も含まれています。含まれるファイルの詳細は `example/README.md` を参照してください。

```bash
# 環境が未アクティベートならアクティベート
conda activate MLIP_mattersim

MLIP_HOT=../scripts/MLIP_HOT.py
python $MLIP_HOT -c pipeline.yaml
```

example フォルダを別の場所にコピーした場合や実運用で使う場合は、`MLIP_HOT=../scripts/MLIP_HOT.py` をあなたの環境上の `MLIP_HOT.py` への絶対パスに変更してください。

> 💡 Quick Fix：不足モジュールのエラーが出た場合はインストールしてください。例：`pyyaml` が不足している場合は `pip install pyyaml`。

すべての設定は `pipeline.yaml` で制御されます。以下にこの設定ファイルの意味を示します。

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

`input` ファイルには、緩和対象の結晶構造を定義する `cell`、`positions`、`numbers` の列が必要です。フォーマット要件はサンプル入力 CSV を参照してください。POSCAR / CIF から入力 CSV を生成するスクリプトも提供しています（例 5）。

ツールキットは `optimized_formula`、`optimized_cell`、`optimized_positions`、`optimized_numbers`、`Energy (eV/atom)`、`Formation Energy (eV/atom)`、`Hull Distance (eV/atom)` を出力します。実行中に進捗と詳細が表示されます。出力は新しい列として追記され、元の列は保持されます。入力には `formula`、`composition`、`ID` などの識別列を含めることを推奨します。

`example/results` に事前計算結果および DFT による結果が含まれており、比較に利用できます。

設定ファイルを使わず CLI だけで同等に実行することもできます（等価な CLI 例）：

```bash
# example フォルダから実行
conda activate MLIP_mattersim
MLIP_HOT=../scripts/MLIP_HOT.py

python $MLIP_HOT \
    --task pipeline \
    --model mattersim \
    --mpi_nproc 10 \
    --opt.input ./example.csv \
    --opt.output ./example_result_task1
```

最適化構造、形成エネルギー、凸包距離は `example_result` に書き込まれます。

> 💡 Tip：設定ファイルと CLI を併用できます。CLI の値が設定ファイルを上書きします。

### 2. 単一タスクの実行

各ステージ（optimize / formation energy / hull distance）は個別に実行できます。例の設定は以下です：

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

これらの設定ファイルは `example` フォルダにも含まれており、次のように実行できます：

```bash
conda activate MLIP_mattersim
MLIP_HOT=../scripts/MLIP_HOT.py

python $MLIP_HOT -c config2_single_task_optimize.yaml
python $MLIP_HOT -c config2_single_task_form.yaml
python $MLIP_HOT -c config2_single_task_hull.yaml
```

同様のタスクは等価な CLI でも実行できます：

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

### 3. 効率のため複数ノードへジョブを分割

高スループット研究では、スクリーニング対象化合物数が非常に大きいことが一般的です。`pipeline` または `optimize` タスクでは、入力データベースを複数のチャンクに分割し、各チャンクを複数計算ノードで個別に実行して、最後に結果を結合する方が効率的です。例えばデータベースを 20 チャンクに分割し、それぞれを 1 台の計算機で実行し、最後に結合します。

これは `optimize.size` と `optimize.rank` で実現できます：`size` は生成するチャンク数、`rank` は現在の計算で処理するチャンク番号（$0$ から $N_{size}-1$）を指定します。

`example` に 3 チャンクの例があります：

```bash
conda activate MLIP_mattersim
MLIP_HOT=../scripts/MLIP_HOT.py

python $MLIP_HOT -c config3_size_rank.yaml --optimize.size 3 --optimize.rank 0
python $MLIP_HOT -c config3_size_rank.yaml --optimize.size 3 --optimize.rank 1
python $MLIP_HOT -c config3_size_rank.yaml --optimize.size 3 --optimize.rank 2
```

設定ファイルは例 1 と同じで、CLI に 2 つのフラグを追加しただけです。

すべてのチャンクが計算できたら、`../script/concat_csv.py` で結合できます。

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

このスクリプトは `XX_{size}_{rank}.csv` パターンに従う任意のファイルを結合できます。`concat_csv.py` は結合したファイル名を表示し、不完全なチャンクがあれば検出します。

> 💡 Tip：`concat_csv.py` は `pipeline` と `optimize` の出力に使用できます。

### 4. 複数初期構造からグローバル最小を決定

化合物には複数の局所最小（local minima）が存在する場合があり、真の基底状態（ground state）はグローバル最小（global minimum）です。この場合、異なる初期構造が異なる局所最小に緩和し、エネルギーが異なることがあります。これらの最小を比較して最も低エネルギーの構造を選ぶことで基底状態を同定できます。同様の状況は DFT による最適化でも起こります。

一つの方法は、異なる初期構造を含む複数の CSV を用意し、それぞれで構造最適化を行うことです。

別の方法は、緩和前に構造へ異なるひずみを与えて異なる初期構造を生成することです。本コードでは `strain` フラグで簡単に実行できます。ひずみはスカラー（等方ひずみ）または 3x3 行列（異方ひずみ）にできます。緩和前に生成された構造も出力へ書き出されます。

簡単な例：

```bash
MLIP_HOT=/Users/xiaoenda/WORK/y_git_repo/MLIP_HOT/scripts/MLIP_HOT.py
python $MLIP_HOT -c config4_strain.yaml --optimize.strain "0.1" --optimize.output example_result_task4/strain1

MLIP_HOT=/Users/xiaoenda/WORK/y_git_repo/MLIP_HOT/scripts/MLIP_HOT.py
python $MLIP_HOT -c config4_strain.yaml --optimize.strain "[[0.1, 0.1, 0.0], [0.1, -0.1, 0.0], [0.0, -0.1, 0.0]]" --optimize.output example_result_task4/strain2
```

> 💡 Tip：この機能は前述の `size` と `rank` と組み合わせられます。
> 例：各ひずみごとに `size` と `rank` で入力を分割し、最後に結合。

計算が完了したら、`find_global_minimum.py` でグローバル最小を特定できます。

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

詳しくは `python ../scripts/find_global_minimum.py -h` を実行してください。

> 💡 Tip：このスクリプトは pipeline / optimize / form / hull タスクの出力に対応します。

### 5. POSCAR / CIF から入力ファイルを生成

構造生成は用途により大きく異なるため、`example/generate_input` に入力ファイル生成の簡単な例を用意しました。この例では POSCAR、CIF、または原子番号から構造を読み込み、原子を異なる元素に置換して新しい構造を作成します。結果は MLIP-HOT への入力として直接使える CSV に保存されます。この例が各自の用途に合わせたスクリプト作成の助けになることを期待しています。

本ドキュメントを簡潔に保つため、詳細説明はここではなくノートブック内に記載しています。重要な点として、`pymatgen` 構造を使うユーザーは、以下のコードブロックで構造リストから入力ファイルを生成できます。`ASE` や `phonopy` に慣れているユーザーも、pymatgen 形式へ容易に変換できます。

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

## MLIP パッケージのインストール

このセクションでは、対応する各 MLIP モデルの conda 環境セットアップ手順を示します。

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

### MatterSim

Website: https://github.com/microsoft/mattersim

```bash
conda create -n MLIP_mattersim python=3.9
conda activate MLIP_mattersim
pip install mattersim
```

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

**Troubleshooting**：`OSError: /lib64/libstdc++.so.6: version 'GLIBCXX_3.4.29' not found` が出た場合は以下を実行してください。

```bash
conda install -c conda-forge libstdcxx-ng
```

### EquiformerV2 と eSEN

Website: https://github.com/facebookresearch/fairchem

Website: https://huggingface.co/facebook/OMAT24/tree/main

EquiformerV2 と eSEN の MLIP は FAIRChem 1.10.0 に実装されており、以下でインストールできます。

```bash
conda create -n MLIP_fairchem python=3.9
conda activate MLIP_fairchem
pip install fairchem-core==1.10.0
pip install torch_scatter torch_sparse torch_spline_conv torch_geometric
```

**Note**：EquiformerV2 と eSEN の学習済みチェックポイントは FAIRChem パッケージに含まれず、公式サイトから別途ダウンロードが必要です：https://huggingface.co/facebook/OMAT24/tree/main 。これらのモデルを使う場合は、`--checkpoint_path` フラグでチェックポイントパスを指定してください。

```bash
mpirun -np 10 python ../scripts/MLIP_optimize.py \
    -d ./example/example_data.csv \
    -m "eqV2_31M_omat" \
    -o "opt_results" \
    --checkpoint_path ./fairchem_checkpoints/eqV2_31M_omat.pt
```


## 熱力学的安定性指標：形成エネルギーと凸包

### 1. 形成エネルギー計算

**形成エネルギー（formation energy）** は、化合物が構成元素の標準参照状態から生成されるときのエネルギー変化を表す熱力学量です。材料の **安定性** を示す指標であり、一般に形成エネルギーが低い（より負）ほど安定な化合物であることを示します。

$$ E_\text{form} = E_{\text{compound}} - \sum_i n_i \mu_i $$

ここで：
- $E_{\text{compound}}$：化合物のエネルギー
- $n_i$：元素 $i$ の原子数
- $\mu_i$：元素 $i$ の化学ポテンシャル（通常は 1 原子あたりのエネルギー）

### 2. 凸包からの距離（Distance above Convex Hull）

**凸包からの距離（distance to the convex hull）** は、化合物の形成エネルギーが、同一化学系におけるすべての競合相から定義される熱力学的安定限界（凸包）からどれだけ上にあるかを測る指標です。同じ組成における最安定相（または相の混合）に対してどれだけ不安定かを定量化します。

$$ E_\text{hull} = E_\text{form} - E_\text{form}^\text{(hull)} $$

ここで：
- $E_\text{form}$：化合物の形成エネルギー
- $E_\text{form}^\text{(hull)}$：その組成で熱力学的に安定な相（または相混合）の形成エネルギー、すなわち凸包上のエネルギー


## 引用

研究で本ツールキットを使用する場合は、以下を引用してください：

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

さらに、使用した各 MLIP モデルについても引用してください。各モデルの公式ドキュメントおよび出版物を参照してください（**対応 MLIP モデル** セクション）。


## トラブルシューティング

#### GCC バージョンの問題

古い GCC バージョンに関連するエラーが発生した場合、conda 環境内で GCC をアップグレードできます。

```bash
conda install -y -c conda-forge gcc=11.3.0
conda install -y -c conda-forge gxx=11.3.0
gcc --version
g++ --version
```

**Note**：これらのコマンド実行前に、対象の conda 環境が有効化されていることを確認してください。

## 【Extra】API 経由で凸包化合物情報を取得するスクリプト

効率向上のため、OQMD から凸包上の化合物を事前計算し、MLIP で評価した結果を参照ファイルとして保存しています。これにより凸包距離を直接計算できます。

ただしデータベースは継続的に成長し、参照ファイルの更新が遅れる場合があります。そのため、OQMD または Materials Project（MP）から API を通じて凸包化合物を取得するスクリプトも提供しています。詳細は [docs/convex_hull_compounds.md](docs/convex_hull_compounds.md) を参照してください。
