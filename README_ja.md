[English](README.md) [中文](README_zh.md)

# MLIP-based High-throughput Optimization and Thermodynamics (MLIP-HOT)
# uMLIPを用いた高スループット構造最適化と熱力学安定性計算（MLIP-HOT）

MLIP-HOTは、汎用的機械学習原子間ポテンシャル（Universal Machine Learning Interatomic Potential, uMLIP）を用いた計算を支援するツールキットです。構造最適化、生成エネルギー計算、凸包からの距離（distance above convex hull）計算が可能です。

本ツール「MLIP-HOT」に加えて、機械学習回帰モデル（MLRM）の学習・適用（ゼロからの学習または転移学習）を行う相補的なツール [MLIP-FTL](https://github.com/nims-spin-theory/MLIP_FTL) も公開しています。ぜひご覧ください。

MLIP-HOTとMLIP-FTLの実装および応用の詳細は、[npj Computational Materials (2026) 12:133](https://www.nature.com/articles/s41524-026-02013-0) に記載しています。
MLIP-HOTを利用・拡張した内容を出版される際には、当該論文（および利用したuMLIP）を引用していただけますと幸いです。

## 概要

本ツールキットでできること：
- **構造最適化**：複数のuMLIP（CHGNet、MatterSim、eSEN-30M-OAMなど）を使った結晶構造最適化
- **生成エネルギー計算**：uMLIP由来の参照エネルギーを用いた生成エネルギーを計算
- **凸包解析**：uMLIP由来の参照エネルギーを用いて凸包からの距離を計算

また、以下の用途に役立つスクリプトも含まれています：
- **HTP 構造生成**：POSCAR、CIF、または格子ベクトル、原子位置、元素種を表す数値配列から、組成スクリーニング用の構造を生成
- **グローバル最小の決定**：複数の局所最小の中から全体の最小（最低エネルギー構造）を決定
  
#### 主な特徴

- **MPI 並列化**：分散計算により大規模データセットを効率的に処理
- **ジョブ分割**：データセットを分割して複数の計算資源に個別投入可能
- **グローバル最小の判定**：異なる初期構造での最適化結果から最低エネルギー構造を決定
- **参照エネルギーの事前計算**：生成エネルギーと凸包距離計算用の参照エネルギーは事前に計算済みのため、生成エネルギーと凸包距離の計算コストはほぼ無視できます。参照エネルギーは、OQMDデータベースの構造を複数のuMLIPで計算して得ています。
- **ひずみを与えた複数初期構造からの緩和**：結晶構造最適化前にひずみを与えることが可能です
- **プリミティブセルへの変換**：効率向上のため、最適化前にプリミティブセルへ変換できます
- **GPUは不要**：事前学習済みuMLIPを用いるため、CPU上で効率的に動作します


## 対応 uMLIP モデル

本ツールキットは以下のuMLIP（汎用機械学習原子間ポテンシャル）モデルをサポートしています。

- **CHGNet**：`chgnet` 
- **SevenNet 系**：
  - `7net-0` 
  - `7net-l3i5` 
  - `7net-mf-ompa` 
- **MatterSim**：`mattersim` 
- **EquiformerV2　系**：
  - `eqV2_31M_omat_mp_salex` 
  - `eqV2_86M_omat_mp_salex` 
  - `eqV2_153M_omat_mp_salex` 
- **eSEN**：`esen_30m_oam`
- **HIENet**：`hienet` 

uMLIPのインストール方法は、下の**uMLIPパッケージのインストール**を参照してください。

新しいuMLIPモデルを追加することも可能です。


## 前提条件

異なるuMLIPモデルごとに独立したPython環境を作成することを推奨します。以下では、環境管理ツールとして **Miniconda** を使用する方法を説明します。

**Miniconda のインストール：**

1. 公式サイトからMinicondaをダウンロード：https://docs.conda.io/en/latest/miniconda.html
2. OS（Linux、macOS、Windows）に合ったインストーラを選択
3. 各プラットフォームの手順に従ってインストール

**インストール確認：**
```bash
conda --version
```

condaの準備ができたら、下の **uMLIPパッケージのインストール** に従って各uMLIPモデル用の環境を作成してください。


## 使い方

MLIP-HOTでは単一のエントリポイント（`scripts/MLIP_HOT.py`）を提供しています。YAMLファイルに設定を記述することで、完全なパイプライン（構造最適化 → 生成エネルギー → 凸包距離）実行や任意の単一ステージ実行が可能です。

以下に実行コマンドの具体例を示します：

1. クイックスタート：3 タスクを一度に実行する簡単な例
2. 単一タスクのみ実行
3. 複数ノードへジョブ分割
4. 複数初期構造からの最安定構造決定
5. POSCAR / CIF / 格子ベクトル、原子位置、元素種を表す数値配列による入力ファイル生成
6. 便利なフラグの紹介

入力ファイルはすべて `example` に含まれています。

### 1. クイックスタート：3 タスクを一度に実行

#### 環境作成とuMLIP のインストール
この例では`MatterSim`を使用します。まずconda環境を作成してMatterSimをインストールします。

   ```bash
   conda create -n MLIP_mattersim python=3.9
   conda activate MLIP_mattersim
   pip install mattersim
   ```

他のuMLIPのインストール方法は、**uMLIPパッケージのインストール**に記載しています。各uMLIPパッケージは、競合を防ぐために別々のconda環境へインストールすることを推奨します。

#### 構造最適化・生成エネルギー・凸包距離の計算

`example`ディレクトリに、10化合物を含む入力例を用意しています。このデータセットは [DXMag Computational HeuslerDB](https://www.nims.go.jp/group/spintheory/database/) から取得しています。また、インストール後に出力結果を比較できるよう、計算結果の参照データも同梱しています。

`example`フォルダに移動して以下を実行してください。PCでも数分で完了するはずです。

```bash
# まだ有効化していない場合は環境を有効化
conda activate MLIP_mattersim 

# エントリスクリプトのパスを設定
export MLIP_HOT=../scripts/MLIP_HOT.py
# 設定ファイルに従って計算を開始
python $MLIP_HOT -c config1_pipeline.yaml 
```

`example`フォルダを別の場所へコピーした場合や実運用で使う場合は、`export MLIP_HOT=../scripts/MLIP_HOT.py`をあなたの環境にある`MLIP_HOT.py`の絶対パスに変更してください。

> 💡 発生しうる問題と対処法
> モジュールが見つからないというエラーメッセージが表示された場合は、そのモジュールをインストールしてください。
> 例えば、PyYAML や mpi4py モジュールが見つからない場合は、以下を実行してください。
> `pip install pyyaml`, `pip install mpi4py`

> 💡 発生しうる問題と対処法
> optimizeタスクで以下のエラーが発生した場合：
> Error processing structure: Optimizer.converged() missing 1 required positional argument: 'gradient'
> これはASEのバージョンに起因する問題です。以下のバージョンをインストールしてください。
> `pip install ase==3.24.0`

設定は`config1_pipeline.yaml`で制御します。内容は以下の通りです。

```yaml
# タスク選択：pipeline | optimize (構造最適化) | form (生成エネルギー)| hull (凸包距離)
# pipeline は optimize、form、hull を順に実行
task: pipeline 
# 原子間ポテンシャルモデルを選択
# 対応モデルは「対応 uMLIP モデル」セクションを参照
model: mattersim
# 任意：グローバル MPI 設定（プロセス数）
mpi_nproc: 10

# ステージ 1：最適化
optimize:
  input:  ./example.csv         # 入力 CSV ファイル
  output: example_result_task1  # 結果を書き出すディレクトリ
# ステージ 2：生成エネルギー（デフォルト設定）
form:
# ステージ 3：凸包距離（デフォルト設定）
hull:
```

入力ファイルには、緩和する結晶構造を定義する`cell`、`positions`、`numbers`の列が必要です。
- **cell**：3×3 行列（リスト）`[[a1,a2,a3], [b1,b2,b3], [c1,c2,c3]]`
- **positions**：N×3 行列（リスト、分率座標）`[[atom1x,atom1y,atom1z], [atom2x,atom2y,atom2z]...]`
- **numbers**：長さNの原子番号リスト `[atom1,atom2,...]`
  
POSCAR / CIF / 格子ベクトル、原子位置、元素種を表す数値配列 から入力 CSV を生成するスクリプトも提供しています（例 5）。

出力として、`optimized_formula`、`optimized_cell`、`optimized_positions`、`optimized_numbers`、`Energy (eV/atom)`、`Formation Energy (eV/atom)`、`Hull Distance (eV/atom)` の列をCSVに書き込みます。実行中は進捗の詳細が表示されます。出力は新しい列として追加され、元の列はすべて保持されます。入力に `formula`、`composition`、`ID` などの識別列を入れることを推奨します。

`example/more_input`には100化合物に対する入力例もあります。また、uMLIP計算の結果は`example/results`に格納しています。結果の先頭10行は、10化合物の例に対応しています。さらに、比較用にDFT計算の結果も同梱しています。

MatterSim uMLIP と DFT の比較プロットは、Jupyter Notebook `example/results/analysis.ipynb` で生成できます。

![alt text](example/results/dft_vs_ml_mattersim.png)

#### 同等のコマンドライン（CLI）

実行はCLIだけでも可能です（YAML設定ファイル不要）。上の例と同等なCLIは以下の通りです。

```bash
# example フォルダで実行
conda activate MLIP_mattersim 
export MLIP_HOT=../scripts/MLIP_HOT.py

python $MLIP_HOT \
    --task pipeline \
    --model mattersim \
    --mpi_nproc 10 \
    --opt.input ./example.csv \
    --opt.output ./example_result_task1 
```

> 💡Tip：設定ファイルとCLIを併用できますが、CLIの値が設定ファイルを上書きします。

### 2. 単一タスクのみ実行

各タスク（optimize / formation energy / hull distance）を個別に実行することもできます。以下に使用例を示します。

```yaml
task: optimize
model: mattersim
mpi_nproc: 10
optimize:
  input:  ./example.csv         # 入力 CSV ファイル
  output: example_result_task2  # 結果を書き出すディレクトリ
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

これらの設定ファイルは`example`フォルダに置いてあります。以下のコマンドで実際に実行可能です。

```bash
conda activate MLIP_mattersim 
export MLIP_HOT=../scripts/MLIP_HOT.py

python $MLIP_HOT -c config2_single_task_optimize.yaml
python $MLIP_HOT -c config2_single_task_form.yaml
python $MLIP_HOT -c config2_single_task_hull.yaml
```

同等のCLI：

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

### 3. 複数ノードへジョブ分割

実際のハイスループット計算では、対象とする化合物数が非常に多くなる場合があります。その場合、`pipeline`または特に時間のかかる`optimize`タスクでは、入力構造を複数チャンクに分割して複数ノードで並列実行し、最後に結果を結合するほうが効率的です。

この制御は`optimize.size`と`optimize.rank`で行います。`size`はチャンク数、`rank`は現在の計算で処理するチャンク番号（$0$ から $N_{size}-1$）です。

3チャンクの例：

```bash
conda activate MLIP_mattersim 
export MLIP_HOT=../scripts/MLIP_HOT.py

python $MLIP_HOT -c config3_size_rank.yaml --optimize.size 3 --optimize.rank 0
python $MLIP_HOT -c config3_size_rank.yaml --optimize.size 3 --optimize.rank 1
python $MLIP_HOT -c config3_size_rank.yaml --optimize.size 3 --optimize.rank 2
```

設定ファイルは例 1 と同じで、CLIで`size`と`rank`を追加しているだけです。

計算後、`../scripts/concat_csv.py` で結果を結合できます。

```bash
# 複数チャンクの結果を結合
python ../scripts/concat_csv.py \
    -f "./example_result_task3" \
    -p "hull_distance_*.csv" \
    -o "example_result_task3/concat_result.csv"

#   -f, --folder:  結合する CSV ファイルが入ったフォルダのパス
#   -p, --pattern: 結合対象ファイルの命名パターン
#                  （例："structure_optimization_result_*.csv", "hull_distance_*.csv"）
#   -o, --output:  結合後の出力 CSV ファイル名
```

このスクリプトは`XX_{size}_{rank}.csv`の命名パターンのファイルを検出し、内容を統合します。統合したファイル名を表示し、未完了チャンクを検出します。

> 💡 Tip：`concat_csv.py`は`pipeline`計算と`optimize`計算の出力統合のみに対応しています。

### 4. 複数初期構造からの最安定構造決定

化合物は複数の局所極小を持つ場合があり、基底状態に対応するのはそのうちの大域的最小エネルギー構造です。これを探索するには、異なる初期構造から複数回の構造緩和を行い、得られた構造の中で最もエネルギーの低いものを基底状態として選択する必要があります。同様の状況は、DFTに基づく構造最適化においても生じます。

方法1：初期構造が異なる複数のCSVを用意し、それぞれで最適化を実行します。

方法2：緩和前に構造に異なるひずみを加えて別の初期構造を生成します。MLIP-HOTでは`strain`フラグで簡単に実行できます。ひずみはスカラー値（等方的）または3×3行列（異方的）のいずれかで指定できます。生成した構造も出力ファイルへ書き込みます。

例：

```bash
MLIP_HOT=/Users/xiaoenda/WORK/y_git_repo/MLIP_HOT/scripts/MLIP_HOT.py   
python $MLIP_HOT -c config4_strain.yaml --optimize.strain "0.1" --optimize.output example_result_task4/strain1

MLIP_HOT=/Users/xiaoenda/WORK/y_git_repo/MLIP_HOT/scripts/MLIP_HOT.py   
python $MLIP_HOT -c config4_strain.yaml --optimize.strain "[[0.1, 0.1, 0.0], [0.1, -0.1, 0.0], [0.0, -0.1, 0.0]]" --optimize.output example_result_task4/strain2
```

> 💡 Tip：この機能は、前述の`size`と`rank`と組み合わせられます。
> つまり、各ひずみについて `size`/`rank` で分割し、緩和後に結果を統合することができます。

全計算後、`find_global_minimum.py` でグローバル最小エネルギー構造を求めます。

```bash
# 複数の結果ファイルからグローバル最小エネルギーを探索
python  ../scripts/find_global_minimum.py \
    -i example_result_task4/strain1/hull_distance.csv \
       example_result_task4/strain2/hull_distance.csv \
    -o example_result_task4/global_min.csv \
    --energy-column "Energy (eV/atom)" \
    --group-by-column "composition" 

# オプション：
#   -i, --input:       比較する複数の CSV ファイル
#   -o, --output:      グローバル最小を含む出力ファイル
#   --energy-column:   エネルギー列名（既定：Energy (eV/atom)）
#   --group-by-column: 化合物を識別する列名（既定：index を使用）
#                      つまり同じ値の行は同一化合物として扱います
```

詳細は `python ../scripts/find_global_minimum.py -h` を参照してください。

> 💡 Tip：このスクリプトは pipeline / optimize / form / hull の出力に対応します。

### 5. POSCAR / CIF / 格子ベクトル、原子位置、元素種を表す数値配列から入力ファイル生成

`example/generate_input` に、入力ファイル生成の簡単な例があります。POSCAR / CIF / 格子ベクトル、原子位置、元素種を表す数値配列 から構造を読み込み、元素置換で新しい構造を生成し、CSV に保存して MLIP-HOT の入力として使えます。用途に合わせたスクリプト作成の参考になれば幸いです。簡潔さを優先し、詳細な説明は例の中に記載しています。

`pymatgen` を使うユーザーは、`structure` オブジェクトのリストから以下のコードで入力 CSV を作れます。`ASE` や `phonopy` のユーザーも、`pymatgen` のアダプタ関数で変換できます。

``` python
# structures_list は pymatgen の Structure オブジェクトのリスト
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

### 6. 便利なフラグ

#### `fix_symmetry`

構造最適化では、デフォルトで対称性は保持されません。対称性保持を有効にするには `--optimize.fix_symmetry True` を追加してください。

#### `print_col`

`--optimize.print_col <column name>` を指定すると、ローカル行番号と指定列の値を表示して進捗を示します。デバッグに有用です。


## uMLIP パッケージのインストール

本節では対応 uMLIP モデルごとの conda 環境セットアップを示します。

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

注：収束性の問題のため、このモデルの参照ファイルの値は、OQMD の構造に対して uMLIP を直接適用して得たものであり、uMLIP ベースの構造最適化は行っていません。


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

注：収束性の問題のため、このモデルの参照ファイルの値は、OQMD の構造に対して uMLIP を直接適用して得たものであり、uMLIP ベースの構造最適化は行っていません。

注：`OSError: /lib64/libstdc++.so.6: version 'GLIBCXX_3.4.29' not found` というエラーが出た場合：

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


### EquiformerV2 と eSEN

Website: https://github.com/facebookresearch/fairchem
Website: https://huggingface.co/facebook/OMAT24/tree/main

EquiformerV2 と eSEN は FAIRChem 1.10.0 内で実装されています。以下の手順でインストールできます。

```bash
conda create -n MLIP_fairchem python=3.9
conda activate MLIP_fairchem
pip install fairchem-core==1.10.0
pip install torch==2.4.1  --index-url https://download.pytorch.org/whl/cpu
pip install torch-scatter torch-sparse torch-spline-conv -f https://data.pyg.org/whl/torch-2.4.1+cpu.html
pip install torch-cluster torch_geometric -f https://data.pyg.org/whl/torch-2.4.1+cpu.html
```

**注**：EquiformerV2 と eSEN の学習済みチェックポイントは FAIRChem パッケージに含まれず、公式サイトから別途ダウンロードが必要です：https://huggingface.co/facebook/OMAT24/tree/main 。使用時は `--opt.checkpoint_path` でパスを指定します：

```bash
    --opt.checkpoint_path ./fairchem_checkpoints/eqV2_31M_omat.pt
```


## 熱力学安定性指標：生成エネルギーと凸包

### 1. 生成エネルギー計算

**生成エネルギー（formation energy）** は、化合物が構成元素の標準参照状態から生成されるときのエネルギー変化を表す熱力学量です。材料の **熱力学的安定性** の指標となり、一般に生成エネルギーが低い（より負）ほど安定です。

$$ E_\text{form} = (E_{\text{compound}} - \sum_i n_i \mu_i ) /n $$

ここで：
- $E_{\text{compound}}$：化合物のエネルギー
- $n_i$：化合物中の元素 $i$ の原子数
- $\mu_i$：元素 $i$ のエネルギー（原子あたり）
- $n$：化合物の総原子数（単位は eV/atom）

### 2. 凸包からの距離（Distance above Convex Hull）

**凸包からの距離** は、化学系における競合相によって定まる熱力学的安定限界（凸包）より、化合物の生成エネルギーがどれだけ高いかを示します。同じ組成で最も安定な相（または相の混合）に対する不安定さの尺度です。小さいほど安定で、原理的に熱力学的に安定な化合物は $E_\text{hull}=0$ です。実際には有限温度効果や手法の不確かさによりずれが生じるため、しきい値を用いることがあります（例：本研究では $E_\text{hull} < 0.22 \text{eV/atom}$ を安定とみなす：https://doi.org/10.1016/j.actamat.2025.121312）。

$$ E_\text{hull} = E_\text{form} - E_\text{form}^\text{(hull)} $$

ここで：
- $E_\text{form}$：化合物の生成エネルギー
- $E_\text{form}^\text{(hull)}$：その組成における熱力学的安定相（または相混合）の生成エネルギー（凸包上のエネルギー）


## 質問・問題・バグ

質問、問題、バグがあれば、このリポジトリで Issue を作成してください。環境情報と最小再現例を添えていただけると、調査が早くなります。


## 引用

研究で本ツールキットを利用・拡張する場合は以下を引用してください：
> Xiao, E., Tadano, T. Accurate screening of functional materials with machine-learning potential and transfer-learned regressions: Heusler alloy benchmark. npj Comput Mater 12, 133 (2026). https://doi.org/10.1038/s41524-026-02013-0


使用する uMLIP モデルについても、各モデルの公式ドキュメントに従って引用してください（**uMLIP パッケージのインストール** 参照）。

凸包距離計算を行う場合は、OQMD データベースも公式に従って引用してください：https://oqmd.org/documentation/publications 。



## トラブルシューティング

#### GCC バージョンの問題

GCC が古いことに起因するエラーが出る場合は、conda 環境内で次のように GCC を更新できます：

```bash
conda install -y -c conda-forge gcc=11.3.0
conda install -y -c conda-forge gxx=11.3.0
gcc --version
g++ --version
```

**注**：これらのコマンドは該当の conda 環境を有効化してから実行してください。

## [Extra] API 経由で凸包化合物情報を取得するスクリプト

効率化のため、OQMD の凸包化合物の uMLIP 生成エネルギーを事前計算し、参照ファイルとして保存しています。これにより、凸包距離を直接計算できます。

データベースは継続的に成長し、参照ファイルの更新が遅れる場合があるため、OQMD または Materials Project（MP）の API から凸包化合物を取得するスクリプトも提供しています。手順の詳細は [docs/get_convex_hull_compounds.md](docs/get_convex_hull_compounds.md) を参照してください。
