# Financial Time Series Forecasting with Deep Learning : A Systematic Literature Review

## Project ID
proj_f3f0f0a0

## Taxonomy
Other

## Current Cycle
2

## Objective
Implement, validate, and iteratively improve the paper's approach with production-quality standards.


## Design Brief
### Problem
This paper is a systematic literature review of 94 papers on deep learning for financial time series forecasting published between 2017 and 2020. It does not propose a new model but rather summarizes the state-of-the-art, common methodologies, challenges (e.g., non-stationarity, low signal-to-noise ratio), and future research directions. The problem is not to reproduce a single result, but to implement a representative framework that embodies the common practices and addresses the key challenges identified in the review. This project will involve implementing a standard deep learning model (LSTM) for a typical forecasting task (daily price direction prediction), evaluating it within a robust walk-forward framework with transaction costs, and systematically analyzing its performance and limitations, reflecting the critical perspective of the review paper.

### Datasets
BTC-USD daily data from yfinance. API call: `yfinance.download('BTC-USD', start='2017-01-01', end='2023-12-31', interval='1d')`

### Targets
Predict the direction of the next day's price change (binary classification: 1 for close > open, 0 otherwise).

### Model
The core model will be a Long Short-Term Memory (LSTM) network, a representative choice for time series forecasting frequently cited in the review paper. The model will take a sequence of historical data (e.g., 30 days of returns and technical indicators) as input and use a sigmoid output layer to predict the probability of the next day's price increase. This represents a common approach discussed in the literature for financial forecasting.

### Training
Training will be conducted using a walk-forward validation scheme with expanding windows. For each window, the model will be trained on the training portion, with early stopping based on a validation set carved out from the end of the training data. The trained model is then evaluated on the subsequent out-of-sample test block. This process is repeated across all windows to generate a continuous series of out-of-sample predictions.

### Evaluation
Evaluation will use both classification and financial metrics. Classification metrics include Accuracy, F1-Score, and Precision. Financial metrics, calculated on the out-of-sample predictions, include Cumulative Net Returns, Sharpe Ratio, and Maximum Drawdown. A transaction cost of 5 basis points (0.05%) per trade will be applied. The model's performance will be compared against two baselines: a random guess strategy and a buy-and-hold strategy.


## データ取得方法（共通データ基盤）

**合成データの自作は禁止。以下のARF Data APIからデータを取得すること。**

### ARF Data API
```bash
# OHLCV取得 (CSV形式)
curl -o data/aapl_1d.csv "https://ai.1s.xyz/api/data/ohlcv?ticker=AAPL&interval=1d&period=5y"
curl -o data/btc_1h.csv "https://ai.1s.xyz/api/data/ohlcv?ticker=BTC/USDT&interval=1h&period=1y"
curl -o data/nikkei_1d.csv "https://ai.1s.xyz/api/data/ohlcv?ticker=^N225&interval=1d&period=10y"

# JSON形式
curl "https://ai.1s.xyz/api/data/ohlcv?ticker=AAPL&interval=1d&period=5y&format=json"

# 利用可能なティッカー一覧
curl "https://ai.1s.xyz/api/data/tickers"
```

### Pythonからの利用
```python
import pandas as pd
API = "https://ai.1s.xyz/api/data/ohlcv"
df = pd.read_csv(f"{API}?ticker=AAPL&interval=1d&period=5y")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.set_index("timestamp")
```

### ルール
- **リポジトリにデータファイルをcommitしない** (.gitignoreに追加)
- 初回取得はAPI経由、以後はローカルキャッシュを使う
- data/ディレクトリは.gitignoreに含めること



## Preflight チェック（実装開始前に必ず実施）

**Phase の実装コードを書く前に**、以下のチェックを実施し結果を `reports/cycle_2/preflight.md` に保存すること。

### 1. データ境界表
以下の表を埋めて、未来データ混入がないことを確認:

```markdown
| 項目 | 値 |
|---|---|
| データ取得終了日 | YYYY-MM-DD (今日以前であること) |
| Train期間 | YYYY-MM-DD 〜 YYYY-MM-DD |
| Validation期間 | YYYY-MM-DD 〜 YYYY-MM-DD |
| Test期間 | YYYY-MM-DD 〜 YYYY-MM-DD |
| 重複なし確認 | Yes / No |
| 未来日付なし確認 | Yes / No |
```

### 2. Feature timestamp 契約
- 全ての特徴量は時刻 t の予測に t-1 以前のデータのみを使用しているか？ → Yes / No
- Scaler / Imputer は train データのみで fit しているか？ → Yes / No
- Centered rolling window を使用していないか？ → Yes / No (使用していたら修正)

### 3. Paper spec 差分表
論文の主要パラメータと現在の実装を比較:

```markdown
| パラメータ | 論文の値 | 現在の実装 | 一致? |
|---|---|---|---|
| ユニバース | (論文の記述) | (実装の値) | Yes/No |
| ルックバック期間 | (論文の記述) | (実装の値) | Yes/No |
| リバランス頻度 | (論文の記述) | (実装の値) | Yes/No |
| 特徴量 | (論文の記述) | (実装の値) | Yes/No |
| コストモデル | (論文の記述) | (実装の値) | Yes/No |
```

**preflight.md が作成されるまで、Phase の実装コードに進まないこと。**

## ★ 今回のタスク (Cycle 2)


### Phase 2: 実データパイプラインの構築 [Track ]

**Track**:  (A=論文再現 / B=近傍改善 / C=独自探索)
**ゴール**: yfinanceからBTC-USDの日足データを取得し、モデルが利用できる形式に前処理するパイプラインを構築する。

**具体的な作業指示**:
1. `src/data_loader.py`を作成し、`load_btc_data`関数を実装します。この関数は`yfinance.download('BTC-USD', start='2017-01-01', end='2023-12-31', interval='1d')`を使用してデータを取得し、`data/raw/btc_usd.csv`に保存します。2. `src/features.py`に`build_features`関数を作成します。この関数は、OHLCVデータから以下の特徴量とターゲットを計算します：a) 終値のリターン b) RSI(14) c) MACD(12, 26, 9) d) ターゲット変数（翌日の終値 > 始値なら1、そうでなければ0）。3. `main.py`を修正し、これらの関数を呼び出して前処理済みデータを生成し、`data/processed/features.pkl`として保存する`process-data`サブコマンドを追加します。

**期待される出力ファイル**:
- src/data_loader.py
- src/features.py
- data/processed/features.pkl

**受入基準 (これを全て満たすまで完了としない)**:
- `python main.py process-data`が成功する。
- `data/processed/features.pkl`が生成され、期待される形状と列を持つDataFrameが含まれている。
- データにNaN値が含まれていないことを確認するテストを追加する。




## データ問題でスタックした場合の脱出ルール

レビューで3サイクル連続「データ関連の問題」が指摘されている場合:
1. **データの完全性を追求しすぎない** — 利用可能なデータでモデル実装に進む
2. **合成データでのプロトタイプを許可** — 実データが不足する部分は合成データで代替し、モデルの基本動作を確認
3. **データの制約を open_questions.md に記録して先に進む**
4. 目標は「論文の手法が動くこと」であり、「論文と同じデータを揃えること」ではない







## 全体Phase計画 (参考)

✓ Phase 1: コアLSTMモデルの実装 — 合成データ上で動作する基本的なLSTM分類モデルを実装する。
→ Phase 2: 実データパイプラインの構築 — yfinanceからBTC-USDの日足データを取得し、モデルが利用できる形式に前処理するパイプラインを構築する。
  Phase 3: ウォークフォワード評価フレームワークの実装 — 時系列の性質を尊重したウォークフォワード検証の仕組みを実装する。
  Phase 4: 財務評価とコストモデルの実装 — 取引コストを考慮した財務パフォーマンス指標を計算し、ベースライン戦略と比較する。
  Phase 5: ハイパーパラメータ最適化 — LSTMモデルの主要なハイパーパラメータを論文の近傍値で最適化する。
  Phase 6: ロバスト性検証 — ウォークフォワードの分割数を増やし、モデルパフォーマンスの安定性を評価する。
  Phase 7: 代替特徴量セットの評価 — より多くのテクニカル指標を含む拡張特徴量セットの影響を評価する。
  Phase 8: 代替モデル（LightGBM）との比較 — 深層学習モデルの有効性を評価するため、強力な非DL系ベースラインモデルと比較する。
  Phase 9: 市場レジーム別パフォーマンス分析 — ブル相場、ベア相場など異なる市場環境でモデルのパフォーマンスがどう変化するかを分析する。
  Phase 10: 特徴量重要度の分析 — モデルの予測にどの特徴量が最も寄与しているかを分析する。
  Phase 11: 包括的レポートの生成 — 全フェーズの結果を統合し、発見事項をまとめた技術レポートを生成する。
  Phase 12: エグゼクティブサマリーと最終化 — 非技術者向けの要約を作成し、コードの品質を向上させる。


## ベースライン比較（必須）

戦略の評価には、以下のベースラインとの比較が**必須**。metrics.json の `customMetrics` にベースライン結果を含めること。

| ベースライン | 実装方法 | 意味 |
|---|---|---|
| **1/N (Equal Weight)** | 全資産に均等配分、月次リバランス | 最低限のベンチマーク |
| **Vol-Targeted 1/N** | 1/N にボラティリティターゲティング (σ_target=10%) を適用 | リスク調整後の公平な比較 |
| **Simple Momentum** | 12ヶ月リターン上位50%にロング | モメンタム系論文の場合の自然な比較対象 |

```python
# metrics.json に含めるベースライン比較
"customMetrics": {
  "baseline_1n_sharpe": 0.5,
  "baseline_1n_return": 0.05,
  "baseline_1n_drawdown": -0.15,
  "baseline_voltarget_sharpe": 0.6,
  "baseline_momentum_sharpe": 0.4,
  "strategy_vs_1n_sharpe_diff": 0.1,
  "strategy_vs_1n_return_diff": 0.02,
  "strategy_vs_1n_drawdown_diff": -0.05,
  "strategy_vs_1n_turnover_ratio": 3.2,
  "strategy_vs_1n_cost_sensitivity": "論文戦略はコスト10bpsで1/Nに劣後"
}
```

「敗北」の場合、**どの指標で負けたか** (return / sharpe / drawdown / turnover / cost) を technical_findings.md に明記すること。

## 評価原則
- **主指標**: Sharpe ratio (net of costs) on out-of-sample data
- **Walk-forward必須**: 単一のtrain/test splitでの最終評価は不可
- **コスト必須**: 全メトリクスは取引コスト込みであること
- **安定性**: Walk-forward窓の正の割合を報告
- **ベースライン必須**: 必ずナイーブ戦略と比較

## 再現モードのルール（論文忠実度の維持）

このプロジェクトは**論文再現**が目的。パフォーマンス改善より論文忠実度を優先すること。

### パラメータ探索の制約
- **論文で既定されたパラメータをまず実装し、そのまま評価すること**
- パラメータ最適化を行う場合、**論文既定パラメータの近傍のみ**を探索（例: 論文が12ヶ月なら [6, 9, 12, 15, 18] ヶ月）
- 論文と大きく異なるパラメータ（例: 月次論文に対して日次10営業日）で良い結果が出ても、それは「論文再現」ではなく「独自探索」
- 独自探索で得た結果は `customMetrics` に `label: "implementation-improvement"` として記録し、論文再現結果と明確に分離

### データ条件の忠実度
- 論文のデータ頻度（日次/月次/tick）にできるだけ合わせる
- ユニバース規模が論文より大幅に小さい場合、その制約を `docs/open_questions.md` に明記
- リバランス頻度・加重方法も論文に合わせる



## 禁止事項

### データ・特徴量の禁止パターン（具体的）
- `scaler.fit(full_data)` してから split → **禁止**。`scaler.fit(train_data)` のみ
- `df.rolling(window=N, center=True)` → **禁止**。`center=False` (デフォルト) を使用
- データの `end_date` が今日以降 → **禁止**。`end_date` を明示的に過去に設定
- `merge` で未来のタイムスタンプを持つ行が特徴量に混入 → **禁止**
- ラベル生成後に特徴量を合わせる（ラベルの存在を前提に特徴量を選択）→ **禁止**

### 評価・報告の禁止パターン
- コストなしのgross PnLだけで判断しない
- テストセットでハイパーパラメータを調整しない
- 時系列データにランダムなtrain/test splitを使わない
- README に metrics.json と異なる数値を手書きしない
- APIキーやクレデンシャルをコミットしない
- **新しい `scripts/run_cycle_N.py` や `scripts/experiment_cycleN.py` を作成しない。既存の `src/` 内ファイルを修正・拡張すること**
- **合成データを自作しない。必ずARF Data APIからデータを取得すること**
- **「★ 今回のタスク」以外のPhaseの作業をしない。1サイクル=1Phase**
- **論文が既定するパラメータから大幅に逸脱した探索を「再現」として報告しない**

## Git / ファイル管理ルール
- **データファイル(.csv, .parquet, .h5, .pkl, .npy)は絶対にgit addしない**
- `__pycache__/`, `.pytest_cache/`, `*.pyc` がリポジトリに入っていたら `git rm --cached` で削除
- `git add -A` や `git add .` は使わない。追加するファイルを明示的に指定する
- `.gitignore` を変更しない（スキャフォールドで設定済み）
- データは `data/` ディレクトリに置く（.gitignore済み）
- 学習済みモデルは `models/` ディレクトリに置く（.gitignore済み）

## 出力ファイル
以下のファイルを保存してから完了すること:
- `reports/cycle_2/preflight.md` — Preflight チェック結果（必須、実装前に作成）
- `reports/cycle_2/metrics.json` — 下記スキーマに従う（必須）
- `reports/cycle_2/technical_findings.md` — 実装内容、結果、観察事項

### metrics.json 必須スキーマ（Single Source of Truth）
```json
{
  "sharpeRatio": 0.0,
  "annualReturn": 0.0,
  "maxDrawdown": 0.0,
  "hitRate": 0.0,
  "totalTrades": 0,
  "transactionCosts": { "feeBps": 10, "slippageBps": 5, "netSharpe": 0.0 },
  "walkForward": { "windows": 0, "positiveWindows": 0, "avgOosSharpe": 0.0 },
  "customMetrics": {}
}
```
- 全フィールドを埋めること。Phase 1-2で未実装のメトリクスは0.0/0で可。
- `customMetrics`に論文固有の追加メトリクスを自由に追加してよい。

### レポート生成ルール（重要: 数値の一貫性）
- **`metrics.json` が全ての数値の唯一のソース (Single Source of Truth)**
- README や technical_findings に書く数値は **必ず metrics.json から引用** すること
- **手打ちの数値は禁止**。metrics.json に含まれない数値を README に書かない
- technical_findings.md で数値に言及する場合も metrics.json の値を参照
- README.md の Results セクションは metrics.json を読み込んで生成すること

### テスト必須
- `tests/test_data_integrity.py` のテストを実装状況に応じて有効化すること
- 新しいデータ処理や特徴量生成を追加したら、対応する leakage テストも追加
- `pytest tests/` が全パスしない場合、サイクルを完了としない

### その他の出力
- `docs/open_questions.md` — 未解決の疑問と仮定
- `README.md` — 今回のサイクルで変わった内容を反映して更新
- `docs/open_questions.md` に以下も記録:
  - ARF Data APIで問題が発生した場合
  - CLAUDE.mdの指示で不明確な点や矛盾がある場合
  - 環境やツールの制約で作業が完了できなかった場合

## 標準バックテストフレームワーク

`src/backtest.py` に以下が提供済み。ゼロから書かず、これを活用すること:
- `WalkForwardValidator` — Walk-forward OOS検証のtrain/test split生成
- `calculate_costs()` — ポジション変更に基づく取引コスト計算
- `compute_metrics()` — Sharpe, 年率リターン, MaxDD, Hit rate算出
- `generate_metrics_json()` — ARF標準のmetrics.json生成

```python
from src.backtest import WalkForwardValidator, BacktestConfig, calculate_costs, compute_metrics, generate_metrics_json
```

## Key Commands
```bash
pip install -e ".[dev]"
pytest tests/
python -m src.cli run-experiment --config configs/default.yaml
```

Commit all changes with descriptive messages.
