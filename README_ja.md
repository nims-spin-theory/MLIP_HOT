# 機械学習原子間ポテンシャル（MLIP）に基づく高スループット最適化・熱力学ツールキット（MLIP-HOT）

> **注記**：本ドキュメントは AI により生成され、その後人により修正されています。

MLIP-HOT は、機械学習原子間ポテンシャル（MLIP）に基づく計算用の包括的なツールキットであり、構造最適化、形成エネルギー計算、凸包解析が含まれています。このツールキットは、高スループット材料探索パイプラインの構築に焦点を当てています。使いやすさと高い性能がこのツールキットの主な利点です。

方法の詳細と応用例については、以下の論文を参照してください：https://arxiv.org/abs/2508.20556。MLIP-HOT を使用または拡張する場合は、この論文を引用してください。


## 概要

このツールは以下の機能を提供します：
- **構造最適化**：様々な MLIP（CHGNet、MatterSim、eSEN-30M-OAM など）を使用して結晶構造を最適化
- **形成エネルギー計算**：MLIP を使用して形成エネルギーを計算
- **凸包距離解析**：MLIP を使用して凸包からの距離を計算

このリポジトリにはさらに以下の便利なスクリプトが含まれています：
- **高スループット構造生成**：POSCAR または CIF 入力から元素組成をスクリーニングするための構造を生成
- **凸包化合物情報の取得**：OQMD および Materials Project から API 経由で凸包化合物を取得
- **相図プロット**：選定された化合物の相図を生成
- **全体最小値の決定**：複数の初期構造からの最適化結果を比較し、全体最小値を決定
  
#### 主な機能

- **MPI 並列化**：分散コンピューティングを通じた大規模データセットの効率的処理
- **柔軟なジョブ配分**：複数の計算リソースにまたがるデータセットチャンク別のジョブ投入
- **全体最小値決定**：異なる初期構成による複数回の最適化実行から最低エネルギー構造を識別
- **形成エネルギー計算**：MLIP 導出の参照エネルギーを使用した形成エネルギー計算
- **凸包距離解析**：MLIP 導出の参照エネルギーを使用した凸包距離計算
- **高品質参照構造**：OQMD の DFT 最適化構造を参照エネルギー計算の初期構成として使用
- **歪み摂動による異なる初期構造からの緩和**：最適化開始前に構造に歪みを適用
- **原始格子への変換**：効率向上のため、最適化前に構造を原始格子に変換
- **GPU デバイスは不要**：本ツールキットは事前学習された MLIP を適用し、CPU 上で効率的に実行可能


## 対応 MLIP モデル

本ツールキットは以下の機械学習原子間ポテンシャルモデルに対応しています：

- **CHGNet**：`chgnet`
- **SevenNet 系統**：
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

MLIP インストール手順については、以下の **MLIP パッケージインストール** セクションを参照してください。

ツールキットはモジュール性を考慮して設計されており、新しい MLIP モデルを既存フレームワークにシームレスに統合できます。


## 前提条件

本ツールキットを使用する前に、システムに **Miniconda** または **Anaconda** がインストールされていることが必要です。Miniconda は conda の最小限インストーラーで、異なる MLIP モデル用の隔離された Python 環境を作成するために使用されます。

**Miniconda のインストール：**

1. 公式ウェブサイトから Miniconda をダウンロード：https://docs.conda.io/en/latest/miniconda.html
2. お使いのオペレーティングシステム（Linux、macOS、Windows）用のインストーラーを選択
3. プラットフォームのインストール手順に従う

**インストールの確認：**
```bash
conda --version
```

conda がインストールされたら、以下の **MLIP パッケージインストール** セクションに記載されているように、各 MLIP モデル用の独立した環境を作成できます。


## 使用方法

MLIP-HOT は、単一のエントリーポイント（`scripts/MLIP_HOT.py`）を提供し、完全なパイプライン（構造最適化 → 形成エネルギー → 凸包距離）またはYAML 設定を使用した個別ステージを実行できます。

以下の例を示します：

1. クイックスタート：すべてのタスクを一度に実行する簡単な例
2. 単一タスクの実行
3. 複数ノード間でのジョブ分割による効率化
4. 複数の初期構造を使用した全体最小値の決定
5. POSCAR または CIF からの入力ファイル生成


### 1. クイックスタート：すべてのタスクを一度に実行

#### 環境作成と MLIP のインストール
この例では `MatterSim` MLIP を使用します。conda 環境を作成し MatterSim をインストールするには、以下を実行します。

```bash
conda create -n MLIP_mattersim python=3.9
conda activate MLIP_mattersim
pip install mattersim
```
他の MLIP のインストール手順は、**MLIP パッケージインストール** セクションに記載されています。各 MLIP パッケージを独立した conda 環境にインストールすることを推奨します。以下の例で `mattersim` を使用する場合：

#### 構造最適化、形成エネルギー、凸包距離計算

`example` ディレクトリには 100 個の化合物を含む例が用意されています。このデータセットは [DXMag Computational HeuslerDB](https://www.nims.go.jp/group/spintheory/database/) から取得されています。インストール検証と出力比較を支援するための事前計算結果も含まれています。含まれるファイルの詳細については、`example/README.md` を確認してください。

```bash
# 環境がまだ有効になっていない場合
conda activate MLIP_mattersim 

MLIP_HOT=../scripts/MLIP_HOT.py
python3 $MLIP_HOT -c pipeline.yaml 
```
例フォルダが別の場所にコピーされたか、実際のデータに使用される場合は、`MLIP_HOT=../scripts/MLIP_HOT.py` をお使いのコンピュータの `MLIP_HOT.py` への絶対パスに変更してください。

> クイックフィックス
> - モジュール欠失エラーの場合、インストールしてください
> - 例えば、`pyyaml` モジュールが欠失している場合は、`pip install pyyaml` を実行してください


すべての設定は設定ファイル `pipeline.yaml` で制御されます。ここで、この設定ファイルの意味を説明します。

```yaml
# タスク選択: pipeline | optimize | form | hull
task: pipeline 
# MLIP モデル選択
model: mattersim
# オプション：グローバル MPI 設定、nproc>1 で MPI 有効化
mpi_nproc: 4
# ステージ 1：最適化
optimize:
    input: ./example_data.csv # 入力 CSV ファイル
    output: ./example_result  # 出力ディレクトリ
# ステージ 2：形成エネルギー（デフォルト設定を使用）
form:
# ステージ 3：凸包距離（デフォルト設定を使用）
hull:
```

`input` ファイルには、緩和の初期結晶構造を定義する `cell`、`positions`、`numbers` 列が含まれている必要があります。形式要件については、例の入力 CSV を参照してください。出力は新しい列として追加され、すべての元の入力列は保持されます。`formula`、`composition`、`ID` などの識別子列を入力ファイルに追加することをお勧めします。

ツールキットは以下の出力列を書き込みます：`optimized_formula`、`optimized_cell`、`optimized_positions`、`optimized_numbers`、`Energy (eV/atom)`、`Formation Energy (eV/atom)`、`Hull Distance (eV/atom)`。進捗と詳細は実行中に出力されます。


ジョブはコマンドラインインターフェース（CLI）（設定ファイルなし）でドット記号フラグを使用して実行することもできます。同等の CLI は以下の通りです：

```bash
# example フォルダから
python3 $MLIP_HOT \
    --task pipeline \
    --model mattersim \
    --mpi_nproc 4 \
    --opt.input ./example_data.csv \
    --opt.output ./example_result 
```

最適化された構造、形成エネルギー、凸包距離は `example_result` のファイルに書き込まれます。

> ヒント
> - 設定ファイルと CLI オーバーライド両方を使用できます。CLI 値は設定ファイル値を上書きします。

### 2. 単一タスクの実行

各ステージ（最適化／形成エネルギー／凸包距離）を個別に実行できます。例えば、単一の形成エネルギーまたは凸包距離計算を実行する場合、以下の設定を使用できます。

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
同等の CLI は以下の通りです：

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

### 3. 複数ノード間でのジョブ分割による効率化

高スループット研究では、スクリーニング対象の化合物数が非常に多いことがあります。データベースをいくつかのチャンクに分割し、複数の計算ノード上で各チャンクの最適化を個別に実行する方が効率的です。例えば、データベースを 20 個のチャンクに分割し、各チャンクを 1 台のコンピュータで実行し、最後にすべての結果を連結します。

これは `size` および `rank` フラグで制御されます。`size` は生成するチャンク数を指定し、`rank` は現在の計算で処理するチャンクを指定します。

```yaml
# タスク選択: pipeline | optimize | form | hull
task: pipeline 
# MLIP モデル選択
model: mattersim
# オプション：グローバル MPI 設定、nproc>1 で MPI 有効化
mpi_nproc: 4
# ステージ 1：最適化
optimize:
    input: ./example_data.csv # 入力 CSV ファイル
    output: ./example_result  # 出力ディレクトリ
    size: 4 # チャンク数
    rank: 0 # チャンク番号は 0 から size-1
# ステージ 2：形成エネルギー（デフォルト設定を使用）
form:
# ステージ 3：凸包距離（デフォルト設定を使用）
hull:
```

すべてのチャンクが計算されたら、`concat_csv.py` を使用して結果を連結できます。

```bash
# 複数チャンクの結果を連結
python ../scripts/concat_csv.py \
    -f "./example_result" \
    -p "structure_optimization_result_*.csv" \
    -o example_data_structure_optimization_result.csv

# フラグ：
#   -f: 連結する CSV ファイルを含むフォルダパス
#   -p, --pattern: 特定ファイルにマッチするグロブパターン（例："*.csv"、"data_*.csv"）
#   -o, --output: 連結結果の出力 CSV ファイル名
```

スクリプト `concat_csv.py` は連結対象ファイルの名前を出力し、不完全なチャンクを識別します。このスクリプトの追加機能については、`python ../scripts/concat_csv.py -h` を実行してください。

> ヒント
> - スクリプト `concat_csv.py` は `pipeline` タスクまたは `optimize` タスクの出力に対応しています。

### 4. 複数の初期構造を使用した全体最小値の決定

DFT ベースの最適化と同様に、異なる初期構造から開始した最適化は、異なるエネルギーを持つ異なる局所極小値に収束することがあります。基底状態は、これらの極小値を比較し、最低エネルギー構造を選択することで識別されます。

これを行う 1 つの方法は、異なる初期構造を持つ複数の CSV ファイルを準備し、各ファイルに対して構造最適化を実行することです。

ツールキットは、`strain` フラグを使用して異なる初期構造構成を探索するために、最適化前に歪み摂動を適用することもできます。

設定ファイルの例：

```yaml
# 等方性歪み
task: pipeline
model: mattersim
mpi_nproc: 4
optimize:
    input: ./example_data.csv
    output: example_result_isotropic
    strain: 0.1
```

```yaml
# 異方性歪み（3x3 行列）
task: pipeline
model: mattersim
mpi_nproc: 4
optimize:
    input: ./example_data.csv
    output: example_result_anisotropic
    strain: [[0.1, 0.1, 0.0], [0.1, -0.1, 0.0], [0.0, -0.1, 0.0]]
```
歪み後の構造は、列 `strained_cell`、`strained_positions`、`strained_numbers` に格納されます。

スクリプト `find_global_minimum.py` を使用して全体最小値を識別できます。

```bash
# 複数の結果ファイル全体で全体最小エネルギーを検出
python ../scripts/find_global_minimum.py \
    -i example_data_result_test1.csv \
       example_data_result_test2.csv \
    -o example_data_result_global_min.csv \
    --labels composition
# フラグ：
#   -i, --input: 比較する複数の入力 CSV ファイル（スペース区切りリスト）
#   -o, --output: 全体最小エネルギーを持つ構造を含む出力ファイル
#   --labels: 各入力ファイルのオプションカスタムラベル（入力ファイルの順序に一致）
```
このスクリプトの追加機能については、`python ../scripts/find_global_minimum.py -h` を実行してください。

> ヒント
> - この機能は前述の `size` および `rank` 機能と組み合わせることができます。


### 5. POSCAR または CIF からのスクリーニング構造生成

提供されたスクリプトを使用すると、POSCAR または CIF をスクリーニング入力 CSV に変換できます。一般的なワークフロー：
1) 目的の化学計量法を含む POSCAR または CIF を準備
2) スクリプトを実行して `cell`、`positions`、`numbers` 列を含む CSV を生成
3) 生成された CSV を `optimize.input` または `pipeline` の入力として後続ステップに進む


## MLIP パッケージインストール

このセクションでは、サポートされている各 MLIP モデルの conda 環境セットアップ手順を提供します。

### CHGNet

ウェブサイト：https://chgnet.lbl.gov/

```bash
conda create -n MLIP_chgnet python=3.10
conda activate MLIP_chgnet
pip install chgnet
```

### SevenNet

ウェブサイト：https://github.com/MDIL-SNU/SevenNet

```bash
conda create -n MLIP_7net python=3.10
conda activate MLIP_7net
pip install sevenn
```

### MatterSim

ウェブサイト：https://github.com/microsoft/mattersim

```bash
conda create -n MLIP_mattersim python=3.9
conda activate MLIP_mattersim
pip install mattersim
```

### HIENet

ウェブサイト：https://github.com/divelab/AIRS/tree/main/OpenMat/HIENet

```bash
conda create -n MLIP_HIENet python=3.9
conda activate MLIP_HIENet

pip install torch==2.1.2
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.1.2.html

git clone https://github.com/divelab/AIRS.git
cd AIRS/OpenMat/HIENet
pip install .
```

**トラブルシューティング**：`OSError: /lib64/libstdc++.so.6: version 'GLIBCXX_3.4.29' not found` エラーが発生した場合は、以下を実行してください：

```bash
conda install -c conda-forge libstdcxx-ng
```

### EquiformerV2 および eSEN

ウェブサイト：https://github.com/facebookresearch/fairchem
ウェブサイト：https://huggingface.co/facebook/OMAT24/tree/main

EquiformerV2 および eSEN MLIP は FAIRChem バージョン 1.10.0 内に実装されており、以下のようにインストールできます：

```bash
conda create -n MLIP_fairchem python=3.9
conda activate MLIP_fairchem
pip install fairchem-core==1.10.0
pip install torch_scatter torch_sparse torch_spline_conv torch_geometric
```

**注意**：EquiformerV2 および eSEN MLIP の場合、学習済みモデルのチェックポイントは FAIRChem パッケージに含まれておらず、公式ウェブサイトから個別にダウンロードする必要があります：https://huggingface.co/facebook/OMAT24/tree/main。これらのモデルを使用する場合は、`--checkpoint_path` フラグでチェックポイントパスを指定します：

```bash
mpirun -np 10 python ../scripts/MLIP_optimize.py \
    -d ./example/example_data.csv \
    -m "eqV2_31M_omat" \
    -o "opt_results" \
    --checkpoint_path ./fairchem_checkpoints/eqV2_31M_omat.pt
```


## 熱力学的安定性指標：形成エネルギーと凸包

### 1. 形成エネルギー計算

化合物の **形成エネルギー** は、標準参照状態にある構成元素から化合物が形成される際のエネルギー変化を測定する熱力学量です。材料の **安定性** に関する洞察を提供します — 形成エネルギーが低い（より負である）ほど、通常、化合物はより安定です。

$$ E_\text{form} = E_{\text{compound}} - \sum_i n_i \mu_i $$
ここで：
- $E_{\text{compound}}$：化合物のエネルギー
- $n_i$：化合物内の元素 $i$ の原子数
- $\mu_i$：元素 $i$ の化学ポテンシャル（通常は原子当たりのエネルギー）

### 2. 凸包からの距離

**凸包からの距離** は、化学体系内のすべての可能な競合相によって定義される熱力学的安定性限界の上にある化合物の形成エネルギーがどの程度離れているかを測定します。これは、同じ組成での最も安定した相の組み合わせに対する化合物の相対的な不安定性を定量化します。

$$ E_\text{hull} = E_\text{form} - E_\text{form}^\text{(hull)} $$

ここで：
- $E_\text{form}$：化合物の形成エネルギー
- $E_\text{form}^\text{(hull)}$：その組成での熱力学的に安定な相（または相の混合物）の形成エネルギー、すなわち凸包上のエネルギー


## 引用

本ツールキットを研究に使用する場合は、以下を引用してください：

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

さらに、研究で使用する特定の MLIP モデルを引用してください。**対応 MLIP モデル** セクションに記載された各モデルの公式ドキュメントと発表を参照してください。


## トラブルシューティング

#### GCC バージョン問題

GCC バージョンの更新に関連するエラーが発生した場合は、以下のコマンドを使用して conda 環境内で GCC をアップグレードできます：

```bash
conda install -y -c conda-forge gcc=11.3.0
conda install -y -c conda-forge gxx=11.3.0
gcc --version
g++ --version
```

**注意**：これらのコマンドを実行する前に、conda 環境が有効になっていることを確認してください。

## [補足] API 経由で凸包化合物情報を取得するスクリプト

効率を改善するため、OQMD の凸包化合物を事前計算し、MLIP を使用して評価しました。これらの結果は参照ファイルとして保存されているため、凸包距離を直接計算できます。

データベースが継続的に増加し、参照ファイル更新がときどき遅延する可能性があるため、OQMD または Materials Project（MP）から API 経由で凸包化合物を取得するスクリプトも提供しています。完全な手順については、[docs/convex_hull_compounds.md](docs/convex_hull_compounds.md) を参照してください。
