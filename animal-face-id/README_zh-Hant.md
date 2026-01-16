🌐 Languages:
[English](./README.md) | [繁體中文](./README_zh-Hant.md)

# 動物臉部辨識

本專案包含一個使用 PyTorch 實現的動物臉部辨識概念驗證（PoC）流程。它專為「封閉集合辨識」（closed-set identification），即辨識已知的個體而設計，並內建了未來支援「開放集合辨識」（open-set）和「註冊」（enrollment）工作流程的元件。

目前的實作專注於使用 [Chimpanzee Faces](https://github.com/cvjena/chimpanzee_faces) 資料集來辨識黑猩猩個體。

## GUI 總覽

這裡是對 GUI 應用程式 (`tools/chimp_gui_app.py`) 的快速視覺化指南。

### 1. 初始畫面
載入模型前的初始介面。顯示了設定檔路徑、模型權重路徑、裝置選擇器（CUDA/CPU），以及空的「辨識」分頁和其拖放上傳區。

![載入模型前的初始 GUI](./GUI-demo1.png)

### 2. 註冊新個體
「註冊」工作流程允許使用者將新個體新增至辨識索引中。使用者提供新個體的名稱/ID，上傳一張或多張已裁切的臉部圖片，然後點擊「新增至索引」。索引會自動更新並儲存。

![註冊工作流程：將新個體新增至索引](./GUI-demo2.png)

### 3. 辨識已知個體
此為辨識已知個體的結果。模型的信心度與圖庫相似度均超過了各自的閾值。系統回傳「已知個體（信心度高於閾值）」並列出最匹配的候選者。

![辨識結果：高信心度的已知個體](./GUI-demo3.png)

### 4. 辨識未知個體（開放集合）
此為辨識新個體或未知個體的結果。雖然模型的分類信心度很高，但與圖庫中最接近的臉孔相似度低於閾值（例如 < 0.75）。這觸發了開放集合邏輯，系統顯示「可能是新個體（觸發開放集合辨識）」。

![辨識結果：因新個體觸發開放集合辨識](./GUI-demo4.png)

## 功能

- **端到端工作流程**：涵蓋從資料準備、訓練、評估到推論的完整流程。
- **設定檔驅動**：所有實驗均透過簡單的 YAML 設定檔進行控制。
- **高效能模型**：包含使用 ResNet 骨幹網路和 ArcFace 損失函數的設定，這是臉部辨識任務的標準組合。
- **可重現性**：提供腳本和固定的隨機種子，確保資料分割和訓練過程可重現。

---

## 專案狀態：高效能模型已訓練完成

**截至 2025 年 11 月，一個高效能的黑猩猩辨識模型已成功訓練完成。**

- **模型**：`ResNet50` 骨幹網路搭配 `ArcFace` 頭部。
- **訓練**：在 `min10` 資料集上進行完整訓練（200 個 epoch，設定檔 `configs/train_chimp_min10_resnet50_arc_full.yaml`）。
- **結果**：最佳模型權重：`artifacts/chimp-min10-resnet50-arcface-full_best.pt`。已準備好進行評估和推論。

## 資料集說明 (為何使用 “min10”)

- `annotations_merged_all.txt`: 7,187 張圖片，102 個 ID，包含樣本數極少的類別。
- `annotations_merged_min10.txt`: 7,150 張圖片，87 個 ID，每個 ID 至少有 10 張圖片。
- 我們在 **min10** 上進行訓練/評估，以避免樣本數極少的類別，改善類別平衡，並穩定 ArcFace 的訓練/評估。只有當您明確希望包含長尾類別時，才使用 `annotations_merged_all.txt`（預期會有更嚴重的類別不平衡和較差的單 ID 指標）。

---

## 概念總覽

本專案涉及幾個關鍵的深度學習概念。關於專案架構的詳細解釋和常見問題的解答，請閱讀我們的新指南：

**➡️ [概念總覽與常見問題](./docs/CONCEPTS.md)**

本指南回答了以下問題：

- 模型如何在不完全重新訓練的情況下「記住」新的臉孔？
- GPU 在訓練中的角色與 CPU 在推論中的角色有何不同？
- 為何這個模型是「黑猩猩專家」？它有哪些限制？

---

## 文件

本專案整理成一系列詳細的指南。請從設定您的環境開始，並依序遵循步驟。

| #   | 指南                                                              | 描述                                                         |
| --- | ------------------------------------------------------------------ | ------------------------------------------------------------------- |
| 1   | **[環境設定](./docs/SETUP.md)**                           | 如何在 Windows、WSL 或 macOS 上設定您的 Python 環境。 |
| 2   | **[資料準備](./docs/DATA_PREPARATION.md)**                 | 如何下載、驗證和準備用於訓練的資料集。    |
| 3   | **[模型訓練](./docs/TRAINING.md)**                           | 如何運行訓練腳本並理解其輸出。          |
| 4   | **[評估與推論](./docs/EVALUATION_AND_INFERENCE.md)** | 如何評估您訓練好的模型並預測新圖片。          |

---

## 🚀 首次設定：5 步驟快速入門

**剛接觸本專案嗎？** 請遵循以下步驟設定您的環境並運行 GUI 應用程式：

### 步驟 1：建立虛擬環境

**在 Linux/macOS/WSL 上：**

```bash
python3 -m venv .venv
```

**在 Windows (命令提示字元或 PowerShell) 上：**

```cmd
python -m venv .venv
```

### 步驟 2：啟動虛擬環境

**在 Linux/macOS/WSL 上：**

```bash
source .venv/bin/activate
```

**在 Windows (命令提示字元) 上：**

```cmd
.venv\Scripts\activate.bat
```

**在 Windows (PowerShell) 上：**

```powershell
.venv\Scripts\Activate.ps1
```

### 步驟 3：升級 pip

```bash
pip install --upgrade pip
```

### 步驟 4：安裝依賴套件

```bash
pip install -r requirements.txt
```

### 步驟 5：運行 GUI 應用程式

```bash
python tools/chimp_gui_app.py
```

### 步驟 6：在瀏覽器中開啟

打開您的瀏覽器並前往：

```
http://127.0.0.1:7860
```

🎉 **您已準備就緒！** GUI 允許您辨識黑猩猩和註冊新個體。

> **注意：** 為確保 GUI 正常運作，請確認您已取得預訓練模型和資料集（見下文）。

---

## 快速入門：使用預訓練模型和資料集

**不想從頭開始訓練？** 您可以透過取得預先準備好的資源來更快地開始：

### 選項 A：取得預訓練模型 (artifacts/)

- **聯絡 Jones** 以索取訓練好的模型 ZIP 檔案
- 解壓縮並覆蓋 `artifacts/` 目錄
- 這包括最佳模型權重 (`chimp-min10-resnet50-arcface-full_best.pt`) 和圖庫索引

### 選項 B：取得資料集 (data/)

對於黑猩猩資料集，您有兩個選項：

1. **直接從官方來源下載**：[cvjena/chimpanzee_faces](https://github.com/cvjena/chimpanzee_faces)
2. **聯絡 Jones** 以索取準備好的資料集 ZIP 檔案，以便更輕鬆地設定

取得其中一項或兩項後，您可以直接跳到下面的評估/推論步驟。

---

## 如何運行：完整工作流程

以下是從全新複製專案到進行預測的完整指令序列。

### 1. 設定與資料準備

_請先確保您已完成 [環境設定](./docs/SETUP.md) 和 [資料準備](./docs/DATA_PREPARATION.md) 指南中的步驟。_

```bash
# 啟動您的虛擬環境 (例如，在 Linux/WSL/macOS 上)
source .venv/bin/activate

# 1. 驗證您的資料集結構是否正確
python validate_dataset.py

# 2. 建立訓練/驗證/測試分割檔案 (只需運行一次)
python scripts/prepare_chimpanzee_splits.py
```

### 2. 訓練模型

_詳情請參閱 [模型訓練](./docs/TRAINING.md) 指南。_

```bash
# 使用高效能設定進行完整訓練
python -m src.training.train --config configs/train_chimp_min10_resnet50_arc_full.yaml
```

### 3. 建立圖庫並進行預測

_詳情請參閱 [評估與推論](./docs/EVALUATION_AND_INFERENCE.md) 指南。_

```bash
# 1. 從您訓練好的模型建立 k-NN 圖庫索引
python -m src.inference.build_gallery --config configs/train_chimp_min10_resnet50_arc_full.yaml --device cuda

# 2. 預測新圖片的 ID
 python -m src.inference.predict --image /path/to/your/chimp_face.png --config configs/train_chimp_min10_resnet50_arc_full.yaml --device cpu
```

### 4. 在測試分割上進行最終評估

```bash
python tools/run_final_eval.py \
  --config configs/train_chimp_min10_resnet50_arc_full.yaml \
  --ckpt artifacts/chimp-min10-resnet50-arcface-full_best.pt \
  --device cuda
```

輸出會儲存到 `artifacts/final_eval/` 和 `FINAL_EVAL_REPORT.md`。

### 5. GUI (實驗性)

```bash
python tools/chimp_gui_app.py
```

使用上述的預設設定檔/模型權重；如果 `artifacts/index/chimp_index` 存在，則會使用該索引。辨識分頁顯示模型 + 圖庫 top-k 結果；註冊分頁可將新個體新增至索引。

### 6. 從標註自動建立圖庫索引

```bash
python tools/build_chimp_index_from_annotations.py \
  --max-per-id 10 \
  --device cuda \
  --prefix artifacts/index/chimp_min10_auto
```

- 自動選取 min10 標註，使用 train+val 作為圖庫（test 集被保留），限制每個 ID 的樣本數，並使用完整模型批次處理嵌入。
- 將索引儲存至 `artifacts/index/chimp_min10_auto_*` 和 entries CSV；如果存在，GUI 將自動載入此索引。

### 7. GUI 中的開放集合提示
- 辨識分頁會同時使用模型 top-1 信心度和圖庫 top-1 相似度來顯示「開放集合狀態」。
- 預設閾值（可透過滑桿調整）：模型機率 0.5，圖庫相似度 0.75。如果任一項低於閾值，GUI 會警告「可能是新個體」，您可以將最後辨識的圖片直接傳送到註冊分頁而無需重新上傳。

---

## 專案結構

```
.
├── artifacts/              # 模型的輸出資料夾 (.pt) 和日誌 (.csv)
│                           # 💡 提示：聯絡 Jones 索取預訓練模型 ZIP 以跳過訓練
├── configs/                # 訓練運行的 YAML 設定檔
├── data/
│   ├── chimpanzee_faces/
│   │   ├── annotations/    # 標註檔案和生成的 splits.json
│   │   └── raw/            # 下載的圖片資料集位置 (被 Git 忽略)
│   │                       # 💡 提示：從 https://github.com/cvjena/chimpanzee_faces 取得
│   │                       #        或聯絡 Jones 索取準備好的資料集 ZIP
├── docs/                   # 詳細的文件指南
├── scripts/                # 輔助腳本 (例如，用於準備資料分割)
├── src/                    # 主要原始碼
│   ├── datasets/           # 資料載入器
│   ├── inference/          # 推論腳本 (預測、建立圖庫)
│   ├── models/             # 模型定義 (骨幹網路、頭部、損失函數)
│   └── training/           # 訓練和評估邏輯
├── tools/                  # 獨立工具 (例如，最終評估腳本)
└── README.md               # 本檔案
```

---

## 🧭 路線圖 / 下一步

### 開放集合辨識

**目前支援：**

- ✅ **已知個體辨識** (模型 top-k)
- ✅ **圖庫最近鄰搜尋** (索引 kNN)
- ✅ **註冊新個體** (註冊分頁)
- ✅ **開放集合提示** (當模型機率 < 0.5 或圖庫相似度 < 0.75 時，辨識分頁會發出警告；可將最後一張圖片傳送至註冊分頁) 

**尚不支援：** 辨識前的臉部偵測/裁切 (目前的 GUI 假設輸入為已裁切的臉部)。

### 🎯 下一個里程碑
- **辨識前的臉部偵測 + 裁切 (高優先級)：** 新增一個偵測器階段 (例如，RetinaFace)，使流程能處理真實世界的照片 (如陷阱相機、動物園閉路電視、研究人員拍攝的照片)，透過自動尋找臉部，然後重複使用現有的嵌入 + 索引堆疊。
- **可選：** 在目前的 M1 開放集合提示基礎上，建立更進階的開放集合功能 (校準、日誌記錄、「模糊」緩衝區/審查流程)。
