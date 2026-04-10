# 開発者ガイド

このドキュメントは、README に載せない実装寄りの情報をまとめるためのものです。  
利用者向けの手順や制約は、まず `README.md` を参照してください。

## 最初に把握したいこと

- 起動の入口は `app.py`
- アプリ全体の流れは `src/app.py`
- 画面操作まわりは `src/region_selector.py`
- RO 自動実行モードは `src/ro_auto.py`
- OCR と CSV 出力の中心は `src/pipeline.py`
- 認識モデルの扱いは `src/ocr_engine.py`

## モジュール構成

- `app.py`
  - 起動用の入口
- `src/app.py`
  - GUI 起動、CLI オプション解釈、OCR 実行フロー
- `src/region_selector.py`
  - ROI 選択画面、パン / ズーム / 最終確認
- `src/ro_auto.py`
  - Ragnarok Online 用の自動実行設定
  - 最新スクリーンショット群の検出
  - ステージング用フォルダ作成
  - 処理済みバッチの状態保存
- `src/pipeline.py`
  - 画像処理全体、行分割、OCR、CSV 行生成
  - 価格別数量グラフ PNG と数量集計 TXT の生成も担当
- `src/ocr_engine.py`
  - PaddleOCR の認識エンジン
- `src/preprocess.py`
  - 列別の前処理
- `src/normalize.py`
  - `item_name` / `quantity` / `price` の正規化
- `src/csv_writer.py`
  - CSV 出力
- `src/settings.py`
  - 設定 JSON 構造
- `src/ui_helpers.py`
  - プレビュー描画、デバッグ保存

## 設定と依存

- 依存管理は `pyproject.toml` と `uv.lock`
- `uv` のクールダウン設定として `exclude-newer = "1 week"` を使う
- ローカルユーザー設定は `config/region.json`
- 共有用設定例は `config/region.sample.json`
- RO 自動実行のローカル設定は `config/ro_auto.json`
- RO 自動実行の共有用設定例は `config/ro_auto.sample.json`
- RO 自動実行の状態保存は `config/ro_auto_state.json`

## 開発ツール

このプロジェクトでは、開発用ツールとして `ruff`、`mypy`、`poethepoet` を使います。

- `ruff`
  - Python のリントと整形を担当する
  - 現在は `E` / `F` / `I` を有効にして、未使用 import、基本的な構文上の問題、import 順の乱れを見つける
  - `src/ocr_engine.py` だけは、`warnings.filterwarnings(...)` を import 前に置く都合で `E402` を除外している
- `mypy`
  - 静的型検査を段階的に導入するために使う
  - まだ既存コードには OpenCV / Paddle まわりの型の粗い部分があるため、`ignore_missing_imports = true` と一部の error code 無効化で運用している
  - 目的は「まず継続運用できる型検査入口を作ること」で、将来的に徐々に厳しくしていく
- `poethepoet`
  - 開発用コマンドを `pyproject.toml` にまとめるためのタスクランナー
  - `uv run poe ...` で共通コマンドを呼べる

ローカル開発環境を作るときは、利用者向け README の `uv sync --no-dev` ではなく、開発用依存も含めて同期する。

```powershell
uv venv
.venv\Scripts\Activate.ps1
uv sync
```

よく使うタスク:

```powershell
uv run poe format
uv run poe lint
uv run poe lint_fix
uv run poe typecheck
uv run poe check
uv run poe run
uv run poe run_ro_auto
```

## 起動モード

- 通常モード
  - `uv run python app.py`
  - GUI で入力フォルダ選択、ROI 確認、設定保存を行う
- RO 自動実行モード
  - `uv run python app.py --ro-auto`
  - GUI は使わず、`config/region.json` を使って無人実行する
  - 入力元は既定で `C:\Gravity\Ragnarok\ScreenShot`
  - 最新画像から `batch_window_seconds` 秒以内の画像を 1 バッチとして拾う
  - 選ばれた画像は `input/ro_auto/<batch_id>/` へコピーしてから処理する
  - 出力は `output/ro_auto/<batch_id>/` にまとまる
  - 同じバッチを再処理したいときだけ `--ro-auto-force` を使う

ROI を変えたい場合は、必ず通常モードを起動して `config/region.json` を更新する。

## RO 自動実行設定

`config/ro_auto.json` が無い場合でも、コード内の既定値で動く。
値を変えたいときは `config/ro_auto.sample.json` を元にローカル設定を作る。

主な項目:

- `source_dir`
  - スクリーンショット保存先
- `filename_regex`
  - 自動収集対象にするファイル名パターン
  - 既定値は `screenNoatun000.jpg` のような連番ファイル向け
- `batch_window_seconds`
  - 最新画像から何秒前までを同一バッチとして扱うか
- `staging_root`
  - 入力用にコピーした画像を置く場所
- `output_root`
  - 自動実行モードの出力先
- `debug_root`
  - 自動実行モードのデバッグ出力先
- `ocr_mode`
  - `accuracy` または `speed`
- `skip_already_processed_batch`
  - 前回と同じバッチならスキップするか

## `modelscope` スタブについて

このプロジェクトには `modelscope/__init__.py` があります。  
これは通常のアプリ機能ではなく、依存関係の回避用ファイルです。

理由:

- `paddleocr -> paddlex -> modelscope` の import 連鎖がある
- `modelscope` は環境によって `torch` を import しようとする
- しかしこのツールでは `torch` の機能を使わない
- `torch` は import だけでも Windows 環境依存のエラー源になりやすい

そのため、`modelscope` の初期化だけローカルで最小上書きして、不要な `torch` 依存を回避しています。

注意:

- 将来 `modelscope` の実機能を使う場合はこのスタブを見直す必要がある
- 現状はローカルキャッシュ済みの Paddle モデルを使う前提

## OCR の現状

- `item_name`
  - 日本語向け認識モデル
- `quantity`
  - 数字向け認識モデル
- `price`
  - 数字向け認識モデル + 専用前処理

`price_suspect` は、価格を目視確認しやすくするための補助列です。

現在の判定は主に 2 段階です。

1. 前後行との並び順を見る局所判定
   - 昇順前提から大きく外れる値を検知する
   - ただし隣の 1 行だけが明らかに壊れている場合は、その巻き添えで現在行を疑いすぎないようにしている
2. 同一アイテム名内での外れ値判定
   - `item_name_normalized` ごとに価格を集める
   - 価格を対数化してから median / MAD ベースで外れ値を見る
   - modified z-score が 3.5 以上なら `price_suspect=1`
   - ただし、前後の同一アイテム行に近い価格がある場合は、局所的な自然変動とみなして `price_suspect` を外す

対数化を使う理由:

- 価格は差分より倍率で見た方が自然
- OCR の桁飛び誤認識は対数空間で目立ちやすい
- median / MAD は平均 / 標準偏差より外れ値に強い

実装上の注意:

- 同一アイテム名の価格が 4 件未満なら外れ値判定はしない
- `MAD` が 0 に近い場合は無理に判定しない
- 前後の同一アイテム行との価格比が `0.85` 以上なら、近い価格の支えがあるとみなす

## 出力物の整理

- CSV
  - `output/ocr_result_YYYYMMDD_HHMMSS.csv`
  - OCR の生結果と正規化結果、`price_suspect` を保存する
- グラフ PNG
  - `output/ocr_result_YYYYMMDD_HHMMSS_price_quantity_chart.png`
  - `price_normalized` ごとに `quantity_normalized` を合算して棒グラフ化する
  - タイトルは先頭行のアイテム名と `captured_at` を使う
- 集計 TXT
  - `output/ocr_result_YYYYMMDD_HHMMSS_summary.txt`
  - 全価格帯の合計数量と、最安値の 1.5 倍までの価格帯に含まれる数量合計を保存する

`src/app.py` の完了ダイアログは、この 3 つの出力先を表示します。  
また、どこかの行に `price_suspect` があれば、完了後に追加の警告ポップアップを出します。

## 補足

- `README.md`
  - 利用者向けの導入手順と使い方
- `pyproject.toml`
  - 依存、`uv` 設定、`ruff` / `mypy` / `poe` タスク定義
