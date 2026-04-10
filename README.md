# Market Screenshot OCR Tool

このツールは、ゲーム内の露店検索画面をスクリーンショットで保存しておき、あとからまとめて CSV に変換したい人のための Windows ローカルツールです。  
露店検索画面に並んでいる「アイテム名」「数量」「価格」を読み取り、表形式のデータとして保存できます。

たとえば、複数ページぶんの露店検索結果をあとから見返したいときや、価格比較・整理・集計をしたいときに使えます。

## この README で使う言葉

- OCR
  - 画像の中の文字を読み取ってテキストにする処理です
- ROI
  - 画像のどこを読み取るか指定する範囲のことです
  - このツールでは、露店検索画面の「アイテム名」「数量」「価格」の列ごとに範囲を指定します

## 何ができるか

- スクリーンショットフォルダを選んで一括処理
- 最初の 1 枚だけで読み取り範囲を指定
- 残り画像に同じ範囲を適用
- 露店検索画面の `item_name`、`quantity`、`price` を CSV 出力
- 価格ごとの数量合計を棒グラフ PNG で出力
- 全価格帯数量と最安値 1.5 倍以内の数量合計を TXT で出力
- 前回設定の再利用
- 末尾の空行の自動除外
- Ragnarok Online 用の自動実行モード

## 動作環境

- Windows
- Python 3.10 以上
- 同一解像度のスクリーンショット一式

## セットアップ

このプロジェクトでは、初回セットアップに `uv` を使って `.venv` を作成します。
以降の実行例は、`.venv` を有効化した前提で `python` / `poe` を使います。

```powershell
uv venv
.venv\Scripts\Activate.ps1
uv sync --no-dev
```

GPU を使う場合は、CUDA 対応 GPU と、それに対応した Paddle GPU パッケージが必要です。  
このリポジトリで確認している構成は以下です。

```powershell
uv pip install --index-url https://www.paddlepaddle.org.cn/packages/stable/cu129/ paddlepaddle-gpu==3.2.2
```

GPU を使わず CPU だけで動かす場合:

```powershell
uv pip install "paddlepaddle>=3.0.0"
```

依存は `.venv` に入るため、グローバル Python 環境を汚しにくい構成です。

普段の起動やコマンド実行前には、先に仮想環境を有効化してください。

```powershell
.venv\Scripts\Activate.ps1
```

## 使い方

1. ゲーム内で露店検索画面を開く
2. 露店検索結果をページごとにスクリーンショット保存する
3. スクリーンショットを撮る間は、露店検索ウィンドウの位置や大きさを変えない
4. 処理したいスクリーンショットを 1 つのフォルダにまとめる
5. ターミナルでこのプロジェクトのフォルダを開く
6. アプリを起動する
7. 入力フォルダとして、そのスクリーンショットフォルダを選ぶ
8. 必要なら前回設定を再利用する
9. 最初の 1 枚で、露店検索画面の「アイテム名」「数量」「価格」の範囲を順に確認または選択する
10. 最終確認画面で 3 つの範囲を確認し、OCR を実行する
11. `output/` に出力された CSV / グラフ / 集計 TXT を確認する
12. `price_suspect` が付いた行があった場合は、完了後の警告ポップアップを確認する

実行:

```powershell
python app.py
```

Ragnarok Online 用の自動実行:

```powershell
python app.py --ro-auto
```

または:

```powershell
poe run_ro_auto
```

## Ragnarok Online 自動実行モード

`--ro-auto` は、Ragnarok Online のスクリーンショット保存先から最新のスクリーンショット群を見つけて、
入力フォルダ準備から OCR 実行、出力作成までを無人で行うモードです。

動作の流れ:

1. `C:\Gravity\Ragnarok\ScreenShot` を見る
2. `screenNoatun000.jpg` のような連番ファイルを候補にする
3. 最新画像からさかのぼって、撮影時刻が一定秒数以内の画像を 1 バッチとして選ぶ
4. 選ばれた画像を `input/ro_auto/<batch_id>/` にコピーする
5. 保存済みの `config/region.json` を使って OCR を実行する
6. `output/ro_auto/<batch_id>/` に CSV / グラフ / 集計 TXT / ログを出力する

このモードでは GUI は出ません。  
読み取り範囲を変えたいときは、通常モードを起動して ROI を保存し直してください。

デフォルト設定:

- スクリーンショット保存先
  - `C:\Gravity\Ragnarok\ScreenShot`
- バッチ判定時間
  - 最新画像から 60 秒以内
- OCR モード
  - `accuracy`
- デバッグ画像
  - 保存しない

同じ最新バッチをすでに自動処理済みの場合は、再処理せず終了します。  
同じバッチをもう一度処理したい場合:

```powershell
python app.py --ro-auto --ro-auto-force
```

自動実行用の設定を変えたい場合は、`config/ro_auto.sample.json` を元に `config/ro_auto.json` を作成してください。  
`batch_window_seconds` を変えると、最新画像から何秒前までを同一バッチに含めるか調整できます。

## 範囲指定のコツ

- アイテム名
  - 行頭から商品名の末尾までを囲います
- 数量
  - 数字部分だけをできるだけぴったり囲います
- 価格
  - 数字部分だけを囲い、右端の通貨単位 `Z` はできるだけ含めないようにします

このツールは、指定した範囲を縦に等間隔で分割して 1 行ずつ読み取ります。  
そのため、上下に余白が多いと読み取り精度が落ちやすくなります。

## 画面操作

- `Enter`
  - 次へ進む / OCR 実行
- `Esc`
  - キャンセルして終了
- `V`
  - 移動モード / 範囲選択モードの切替
- `N` / `Tab`
  - 対象列の切替
- `R`
  - 現在の範囲をクリア
- `P`
  - 前処理プレビュー ON/OFF
- `S`
  - 設定保存
- `C`
  - `output/` と `debug/` の生成物を掃除
- `0`
  - ズームと表示位置をリセット
- `マウスホイール`
  - 拡大 / 縮小
- ドラッグ
  - 移動モード: 表示位置を移動
  - 範囲選択モード: 範囲を指定

## 出力

出力ファイル:

- `output/ocr_result_YYYYMMDD_HHMMSS.csv`
  - OCR の元データ
- `output/ocr_result_YYYYMMDD_HHMMSS_price_quantity_chart.png`
  - `price_normalized` を横軸、同価格を合算した `quantity_normalized` を縦軸にした棒グラフ
  - グラフタイトルは先頭行のアイテム名とキャプチャ時刻を使います
- `output/ocr_result_YYYYMMDD_HHMMSS_summary.txt`
  - 全価格帯の合計数量
  - 最安値の 1.5 倍までの価格帯に含まれる合計数量
  - あわせて、実際にどの価格まで集計したかを記録します

`--ro-auto` の場合は、これらの出力が `output/ro_auto/<batch_id>/` の下にまとまって作成されます。  
処理対象として選ばれた元画像のコピーは `input/ro_auto/<batch_id>/` に保存されます。

各列の意味:

- `source_file`
  - どのスクリーンショットから読み取った行か
- `captured_at`
  - その画像の作成日時
  - 取得できない場合は更新日時相当を使います
- `row_index`
  - 画面内で上から何行目か
- `item_name_raw`
  - 画像から読み取ったアイテム名そのまま
- `item_name_normalized`
  - 空白や記号だけの値を整理したアイテム名
- `quantity_raw`
  - 画像から読み取った数量そのまま
- `quantity_normalized`
  - 数字だけに整理した数量
- `price_raw`
  - 画像から読み取った価格そのまま
- `price_normalized`
  - 記号や単位を除いて数字として扱いやすくした価格
- `price_suspect`
  - 価格の並びや読み取り結果が怪しいときに `1` が入る補助列

`price_suspect` が 1 件でもある場合は、CSV / PNG / TXT の出力完了後に警告ポップアップを表示します。

## 設定ファイル

- ローカル設定: `config/region.json`
- 共有用サンプル: `config/region.sample.json`
- RO 自動実行のローカル設定: `config/ro_auto.json`
- RO 自動実行の共有用サンプル: `config/ro_auto.sample.json`
- RO 自動実行の状態保存: `config/ro_auto_state.json`

`config/region.json` はローカル専用で、Git には含めません。
`config/ro_auto.json` と `config/ro_auto_state.json` もローカル専用です。

## 制約

- 同一解像度画像のみ想定
- 解像度不一致画像はスキップ
- 行分割は等間隔分割
- 初回のモデルロード時は少し待つことがあります
- 価格の補助判定は、画面内で価格が昇順に並ぶ前提を使っています

## 開発者向け情報

実装や依存回避の理由は以下を参照してください。

- [開発者ガイド](docs/developer_guide.md)
