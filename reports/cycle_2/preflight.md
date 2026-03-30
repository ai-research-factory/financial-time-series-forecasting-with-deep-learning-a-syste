# Preflight Check — Cycle 2 (Phase 2: Data Pipeline)

## 1. データ境界表

| 項目 | 値 |
|---|---|
| データ取得終了日 | 2023-12-31 (今日 2026-03-30 以前) |
| Train期間 | 2017-01-01 〜 (walk-forward で動的に決定、Phase 3で実装) |
| Validation期間 | (walk-forward で動的に決定、Phase 3で実装) |
| Test期間 | (walk-forward で動的に決定、Phase 3で実装) |
| 重複なし確認 | Yes — timestampをインデックスとして重複排除済み |
| 未来日付なし確認 | Yes — end_date=2023-12-31で明示的にフィルタ |

**注意**: ARF Data APIのBTC-USDデータは2016-03-30が最古。論文指定の2017-01-01開始は取得可能。

## 2. Feature timestamp 契約

- 全ての特徴量は時刻 t の予測に t-1 以前のデータのみを使用しているか？ → **Yes**
  - リターン: `close.pct_change()` は t と t-1 のデータを使用（t時点で利用可能）
  - RSI(14): 過去14期間のリターンから計算（`center=False`）
  - MACD(12,26,9): EMAベースで過去データのみ使用
  - ターゲット: 翌日の `close > open` を使用（予測対象なのでOK）
- Scaler / Imputer は train データのみで fit しているか？ → **N/A** (Phase 2ではスケーリング未実装、Phase 3で実装時に遵守)
- Centered rolling window を使用していないか？ → **Yes** (使用していない、全て `center=False`)

## 3. Paper spec 差分表

| パラメータ | 論文の値 | 現在の実装 | 一致? |
|---|---|---|---|
| ユニバース | BTC-USD | BTC-USD | Yes |
| ルックバック期間 | 30日 | 30日 (シーケンス長、Phase 3で使用) | Yes |
| リバランス頻度 | 日次 | 日次 | Yes |
| 特徴量 | リターン, RSI(14), MACD(12,26,9) | リターン, RSI(14), MACD(12,26,9) | Yes |
| コストモデル | 5 bps per trade | 5 bps (Phase 4で実装) | Yes |
| データ期間 | 2017-01-01 〜 2023-12-31 | 2017-01-01 〜 2023-12-31 | Yes |
| ターゲット | 翌日 close > open → 1 | 翌日 close > open → 1 | Yes |
