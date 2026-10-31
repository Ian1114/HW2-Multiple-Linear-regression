# 資料集變更紀錄：WHO 平均餘命

2026-11-01 使用者指定改用 Kaggle Life Expectancy (WHO)，取代原設計的洪水資料。既有洪水設計保留作歷史紀錄，以下為本次優先規則。

- 實際檔案：`dataset/Life Expectancy (WHO)/Life Expectancy Data.csv`，2,938 × 22；國家 193、年度 2000–2015。目標 `Life expectancy` 去除首尾空格後作欄名，缺失目標 10 筆排除。
- Country 僅用於分組，Year 與二元 Status 加 18 個數值欄位合計 20 原始輸入；預測目標為同年度平均餘命，沒有聲稱未來或因果預測能力。
- 同國多年度相關，改採 GroupShuffleSplit(seed=42, test_size=0.2) 保留國家，訓練資料用 GroupKFold(5)。這是跨國泛化評估；不是時間外推。補值、標準化、選擇與模型均在 pipeline 中。
- 20 特徵 OLS 與 KBest、RFE 各 14 特徵使用分組 CV RMSE 選主要 OLS；Ridge/Lasso 使用相同分組 CV 比較候選 alpha，最後再評估保留集。
- 平均反應 CI 與單筆 PI 為條件於選模及標準 OLS 假設的區間；同國相依、異質變異與選模不確定性可能影響有效性。
- 實際檔案雜湊與套件版本由 Notebook 輸出記錄；授權、NotebookLM、對話 PDF、學號和最終打包仍需確認。
