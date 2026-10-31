# HW2 多元線性回歸作業 Spec

日期：2026-10-19（Asia/Taipei）  
狀態：設計草案，供使用者檢閱；尚未安裝環境、下載資料、訓練模型或產出繳交成果。  
專案根目錄：`week6-note/hw_2/`。本文件依 Superpowers brainstorming 整理；後續詳細 implementation plan 以核可的 spec 為依據。

## 1. 目的與資料來源

完成一份可重現、可解釋的個人多元線性回歸作業，以 CRISP-DM 六階段展示從問題定義、資料理解、前處理、建模、評估到新資料預測的完整流程。優先滿足評分要求，而非只追求 Kaggle 排名。

已讀來源：

1. 根目錄的 `hw_2 作業評分要求.docx`：已讀文字及全部 7 張內嵌圖片。評分資訊主要來自圖片 1、2，補充方法來自圖片 3–6。圖片 7 的重疊文字不可可靠辨識，不據此推定新規定。
2. [HW 2 Multiple Linear Regression](https://app.notion.com/p/HW-2-Multiple-Linear-Regression-3f9c47529811809caba6d4116dcc5f6d)：已經側邊瀏覽器閱讀頁面正文，包括資料集推薦與 CRISP-DM／特徵選擇建議。
3. [AIoT-DA L5](https://app.notion.com/p/AIoT-DA-L5-From-Simple-Linear-Regression-to-Auto-Regression-2026-9-30-3ebc4752981180e0b701cae5f72972f3)：已讀頁面正文、線性回歸各形式與範例。頁內另附 notebook、PDF、CSV，本次未下載閱讀其附件內容。

來源衝突與處理：

- 正式作業截圖寫 Kaggle 資料集 **10–20 個特徵**；補充與 HW2 Notion 寫 **14–39 個輸入特徵，不含 target**。
- 本設計選 **20 個原始輸入特徵**，並比較選出 **14 個特徵**的版本，同時落在兩者交集。ID 不列為特徵，也不計入特徵數。
- CSV 尚未取得，20 欄目前是教材描述，不是假裝已驗證的資料事實。下載後必須實際確認欄位。
- 截圖顯示截止日期 **2026-11-04**、個人作業、占課程 10%；這是文件快照，繳交前仍須核對課程平台最新公告與時刻。

## 2. 評分要求與交付對照

| 評分項目 | 比例 | 必須留下的證據 |
|---|---:|---|
| CRISP-DM 完整且邏輯清楚 | 25% | Notebook 與 PDF 報告均有六階段，問題、方法、結果彼此對應 |
| GPT 對話與 NotebookLM 摘要 | 15% | 對話 PDF、NotebookLM 實際研究結果及至少 100 字摘要，明標「GPT 輔助內容」「NotebookLM 摘要」 |
| 資料集來源與研究脈絡 | 10% | Kaggle 確切網址、檔案來源、欄位表、背景與引用 |
| 模型可執行、特徵選擇與評估 | 25% | 從乾淨環境重跑、選取前後比較、無資料洩漏、指標與設定記錄 |
| 結果合理、美觀、有說服力 | 15% | 清楚圖表、解讀與限制，包含信賴區間與預測區間 |
| Kaggle 名次（若有）／預測結果評估 | 10% | 預測圖、評估表；若實際參賽才記錄名次與日期，不虛構排行榜結果 |

必要繳交：`<學號>_hw2.py` 或 `<學號>_hw2.ipynb`、PDF 報告、GPT 對話 PDF、相關資料，壓縮為 `<學號>_hw2.zip`。實際學號在打包前提供，不沿用老師示範學號。GitHub／Colab 為選用；若使用，提供連結與 README。

## 3. 方案比較與推薦

| 方案 | 優點 | 代價 | 決策 |
|---|---|---|---|
| Notebook + 少量 Python 模組 + venv | 適合呈現課程分析，也能測試與重現 | 必須隨 notebook 一起交付模組 | **推薦** |
| 全部放單一 Notebook／Colab | 最接近課堂操作，起步快 | 隱藏執行狀態與套件版本較難控制 | 可行替代 |
| 純 Python CLI | 自動重跑方便 | 敘事和圖表需另外整合報告 | 不作主要交付形式 |

預設題目：教材推薦的 **Predict the Flood**，target 為 `FloodProbability`，以實際 CSV 核對名稱。優先評估教材連結 `https://www.kaggle.com/competitions/predictheflood`；另一個教材候選是 `https://www.kaggle.com/competitions/playground-series-s4e5`。兩者不可當作同一份資料混用；正式下載時選定其一並記錄來源、版本、授權／取得條件與 SHA-256。

若首選無法取得，先檢查另一洪水資料來源；若其欄位不符合 14–20 個有效輸入特徵，應修訂 spec 後再選題。Istanbul 房價涉及類別編碼與特徵數爭議；Human–AI 資料涉及合成資料解讀，暫不列主線。

## 4. env 與可重現環境

### 4.1 三種概念分開

- **`.venv/`**：這份作業自己的 Python 與套件環境，避免跟 week3 或其他作業互相影響。
- **`.env`**：本機路徑等環境變數設定；不是虛擬環境，也不應放模型研究參數。預設採手動下載資料，不需要 API 金鑰。
- **`config.json`**：可交付且可追蹤的實驗設定，例如 seed、split、特徵選擇與模型參數。

環境設計：Python 3.11 作為初始目標，實作時確認本機可用版本。主要依賴 pandas、numpy、scipy、scikit-learn、statsmodels、matplotlib、seaborn、joblib、python-dotenv；Notebook 使用 jupyterlab、ipykernel、nbconvert；測試使用 pytest。這是依賴需求清單，不宣稱任何尚未驗證的版本組合已可用。

保存直接依賴於 `requirements.in`，在乾淨 `.venv` 完成安裝與驗證後產生精確版本的 `requirements-lock.txt`；README 記錄 Python patch 版本、OS、重建指令。不得直接 freeze 使用者全域環境。

預定 Windows PowerShell 建立流程（本次未執行）：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m ipykernel install --user --name hw2-regression --display-name "Python (HW2)"
.\.venv\Scripts\python.exe -m jupyter lab
```

首次建置尚無 lock 時，先依 requirements.in 安裝與驗證，再產生 lock；後續重現才用上述 lock 安裝。直接使用 `.venv` 的 interpreter，無須為 Activate.ps1 修改全機執行政策。

`.env.example` 預計提供：

```dotenv
HW2_DATA_DIR=./data/raw
HW2_OUTPUT_DIR=./outputs
```

路徑相對於專案根目錄解析；未建立 `.env` 時也使用上述預設值。系統環境變數優先於 `.env`。資料夾不存在或設定無效時，顯示明確錯誤。

`.gitignore` 與打包排除規則必須排除 `.venv/`、`.env`、憑證、快取與 `.ipynb_checkpoints/`。交付 `.env.example`、lock 與 config；每次執行保存已解析的非敏感設定、版本、seed 與輸入檔案 hash。

## 5. 分析流程與模型設計

### Business Understanding

說明以資料特徵估計洪水機率的研究問題、潛在用途與限制。資料若為合成資料，須明示；不可宣稱可直接作為真實防災預警。`FloodProbability` 雖然名稱包含 probability，仍按連續目標進行回歸，不轉成分類準確率作主要評分。

### Data Understanding

列出樣本數、20 個特徵、target、ID、資料型態、描述統計、缺失、重複列與分布。確認是否有群組／時間相依資料；若有，必須改用相應切分並修訂以下隨機切分設計。保留原始資料不修改。

無 target 的 Kaggle test.csv 僅供外部預測，不能拿來計算 MAE 或 R²。缺失 target 的有標籤資料列須記錄並排除，不補 target。ID、target 派生欄位及預測時不可取得的資訊不得當特徵。

### Data Preparation

預設使用獨立樣本：固定 `random_state=42`，先保留 20% holdout；其餘 80% 訓練集內做 5-fold CV（shuffle=True、random_state=42）。切分索引保留供所有模型共用。

中位數補值、StandardScaler、特徵選擇及模型必須放在同一 Pipeline；每個 CV fold 僅 fit 該 fold 的訓練部分。所有用來決定模型的探索與特徵排名只用訓練資料。重複紀錄不得跨 train/test 造成洩漏，須先記錄並處理。

目前預期全數為數值欄位；遇到意外文字欄位先報錯並檢視來源，不自動將字串編碼後繼續。異常值不因 test 表現不好而刪除。

### Modeling／Feature Selection

核心實驗，使用一致資料切分與主要選模指標 CV RMSE：

1. DummyRegressor(mean)：建立最低比較基準。
2. LinearRegression：全部 20 個特徵。
3. SelectKBest(f_regression, k=14) + LinearRegression。
4. RFE(LinearRegression, n_features_to_select=14) + LinearRegression。
5. Ridge：全部特徵，alpha 候選 `[0.01, 0.1, 1, 10, 100]`。
6. Lasso：全部特徵，alpha 候選 `[0.00001, 0.0001, 0.001, 0.01, 0.1]`，記錄非零係數數量與收斂狀態。

至少實作兩種明確的特徵選擇方法並比較保留名單與效果。Lasso 非零數可能少於 14，僅作正則化對照，不能冒稱其特徵數符合主模型規定。CV 選定的 OLS 版本（20 或 14 特徵）作主要可解釋模型；Ridge/Lasso 作補充比較，即使其效果較好也保留 OLS 的完整分析。

教材另提 Sequential Feature Selection、permutation importance；視時間作延伸，不宣稱正式評分要求五種都要實作。20 個原始特徵不能選出 25 或 30 個。AR 只有在真正時間序列題目才適用，不因 L5 標題包含 AR 就強行加進本題。

### Evaluation

- 指標：MAE、MSE、RMSE、R²；同時呈現 train、CV mean/std、holdout 結果，避免只報訓練分數。
- OLS 提供 Adjusted R²，清楚標示所用資料與 `n`、`p`，公式 `1-(1-R²)*(n-1)/(n-p-1)`，`n <= p+1` 不計算。Ridge/Lasso 不直接套普通 OLS 自由度作無條件比較。
- CV 決定特徵與超參數後才評估 holdout；不根據 holdout 結果反覆調整。若模型未優於 dummy，保留真實結果並分析原因，不編造最低 R² 門檻。
- 輸出 actual-vs-predicted（含 y=x）、殘差-vs-fitted、殘差分布、Q-Q、相關熱圖、係數／選取特徵圖。

五項回歸假設診斷：

| 假設 | 檢視方式 | 解讀限制 |
|---|---|---|
| 線性 | 殘差圖、各特徵與目標關係 | 結構性曲線可能表示模型不足 |
| 誤差獨立性 | 先研究樣本取得方式；有合理順序時可看 Durbin–Watson／ACF | 任意 CSV 排序的統計量不能證明獨立 |
| 同質變異 | 殘差圖、Breusch–Pagan | 違反時討論區間可信度及穩健方法 |
| 誤差常態性 | Q-Q、殘差分布、適當檢定 | 大樣本檢定很敏感，不只看 p 值 |
| 無嚴重多重共線性 | 相關係數與 VIF | VIF 奇異或無限大要報告，不能靜默略過 |

區間要求：對已選定的 OLS 使用相同訓練資料、補值／縮放結果及特徵順序，透過 statsmodels OLS（含 intercept）建立 **95% 平均反應信賴區間**和 **95% 單筆觀測預測區間**。檢查其點預測與 sklearn OLS 一致。模型選取後的傳統區間屬條件式、未含選模不確定性，須在報告說明。

多元回歸不畫成一條假裝涵蓋所有變數的迴歸線。可將 holdout 按預測值排序，繪實際值、點預測、兩種區間，橫軸明標「依預測值排序的樣本」，不能稱時間序列。對機率目標，OLS 或區間超出 [0,1] 時呈現並說明，不為美觀靜默裁切。

### Deployment

儲存完整前處理與模型 Pipeline、輸入欄位順序、選取特徵及 metadata。提供讀取新 CSV 的本機預測流程，驗證必要欄位、數值型態，輸出原始 row ID 和預測值。讀回模型後的結果須與存檔前一致。Deployment 以可重跑的新資料預測示範完成，不增加雲端網站需求。

## 6. 預定檔案結構

```text
hw_2/
  hw_2 作業評分要求.docx
  docs/superpowers/specs/2026-10-19-hw2-regression-design.md
  docs/superpowers/plans/                # spec 核可後撰寫詳細計畫
  <學號>_hw2.ipynb                      # 六階段分析、圖與解讀
  hw2lib/
    __init__.py
    data.py                            # 讀取、欄位驗證與切分
    modeling.py                        # Pipeline、CV 與選模
    evaluation.py                      # 指標、診斷、區間、圖表
    predict.py                         # 新樣本預測
  config.json
  requirements.in
  requirements-lock.txt
  .env.example
  .gitignore
  README.md
  data/raw/                            # 原始 CSV
  data/README.md                       # 來源、版本、授權、hash、欄位
  outputs/                            # metrics、圖、模型、metadata
  reports/                            # 報告來源與最終 PDF
  evidence/                           # GPT PDF、NotebookLM 紀錄、研究引用
  tests/                              # 關鍵正確性測試
```

這是預定結構，並非本次已建立的程式。Notebook 不依賴使用者機器的絕對路徑，所有必要模組跟著 ZIP 交付。

## 7. 實作順序與每階段驗收

1. **環境與資料契約**：建立獨立 env、鎖定依賴、核對來源與 20 特徵、保存 hash；驗收為乾淨環境可讀 CSV，錯誤欄位會清楚失敗。
2. **資料理解與 baseline**：完成資料字典、切分、訓練集 EDA、dummy 與完整 OLS；驗收為切分無交集、seed 可重現、無 target/ID 洩漏。
3. **特徵選擇與比較**：完成 SelectKBest、RFE、Ridge、Lasso 與 CV 比較；驗收為 selector 只 fit fold-train，保留特徵與分數有檔案紀錄。
4. **診斷與區間**：完成 holdout 指標、五項假設、95% CI/PI 與圖；驗收為 sklearn/statsmodels 點預測一致、一般 OLS 下 PI 不窄於 CI、異常統計值有說明。
5. **新資料預測與重跑**：存取模型、欄位驗證、Restart & Run All；驗收為全新 kernel 從頭成功，不靠手動先跑某格。
6. **報告與證據**：完成六階段 PDF、網路方法比較、實際 NotebookLM 摘要、對話匯出；驗收為每項評分有對應位置，圖表與輸出數值一致。
7. **打包與獨立驗收**：複製至乾淨位置重建 env 執行，檢查 ZIP 內容、命名與檔案大小；不包含 `.venv`、`.env` 或秘密資訊。

後續詳細計畫應將各階段拆為檔案、函式介面、測試與指令；目前只核定設計方向，不在此預先產生產品程式。

## 8. 關鍵測試與失敗處理

- 資料不足、缺 target、非數值目標、特徵數錯誤、重複欄名、缺必要預測欄位：明確停止並說明，不能悄悄換資料或欄位。
- train/test 索引不得重疊；用測試集極端值檢驗 imputer/scaler 不受 test 影響。
- 以已知 y/y_pred 驗證指標；constant target 與不足樣本的 R²／Adjusted R² 顯示限制。
- 對純線性小型合成資料驗證回歸與區間介面；奇異設計矩陣與全缺失欄位有明確處理，不以合成資料充當作業成果。
- 模型序列化前後預測數值一致；中文／含空白路徑與不同工作目錄仍能正確解析專案路徑。
- Notebook 全新 kernel 可執行；圖表中文字、圖例、區間與座標標示可讀。
- 套件錯誤不可透過修改全域 Python 修補；依 env 重現並更新 lock 與說明。

## 9. 報告與 AI 使用證據

PDF 報告章節：摘要 → Business Understanding → Data Understanding → Data Preparation → Modeling → Evaluation → Deployment → 網路方法比較 → GPT 輔助內容 → NotebookLM 摘要 → 限制與參考資料。

NotebookLM 摘要必須來自實際使用 NotebookLM 研究該主題／網路解法，保留來源與紀錄，再撰寫至少 100 字摘要；不能由本次助手自行寫一段就稱為 NotebookLM 產出。整理選用 OLS、Ridge、Lasso 與特徵選擇方法的外部來源，逐項說明與本作業流程的異同。

保留所有與本作業相關的 GPT／Codex 對話，依老師要求以 pdfCrowd 或其他方式匯出 PDF；報告含對話過程說明與附件索引，若平台要求單一 PDF 則併入附錄。工具無法自動取得的對話匯出或 NotebookLM 操作，由使用者完成後再整合，不能標記為已完成。

## 10. 檢閱項目與完成定義

目前預設：洪水預測、Notebook + 小型模組、Windows 本機 `.venv`、`.env` 只放本機設定、20 特徵基準與 14 特徵比較。本次尚無使用者選題核可，也尚無實際 CSV；這些前提須在開始實作時確立。

打包前另需實際學號、資料可再散布條件、最新課程繳交規定。若資料不能重打包，須先處理與「附資料」要求的衝突，不能自行宣稱僅交下載網址已符合。

完成定義：三份來源要求有可追溯對應、在乾淨 env 可重現、模型與特徵選擇無資料洩漏、評估與區間可解釋、AI 使用紀錄真實、最終 PDF 與學號命名 ZIP 齊全。本專案目錄目前不是 Git repository；此 spec 僅儲存為檔案，未建立或提交 Git repository。
