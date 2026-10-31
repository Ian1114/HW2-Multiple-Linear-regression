# HW2 洪水回歸實作計畫

**Goal:** 依設計文件完成可重現的 CRISP-DM 多元線性回歸研究。
**Architecture:** 以 Notebook 呈現六階段，將資料驗證、模型與診斷拆為 Python 模組；資料與輸出維持分離。
**Tech Stack:** Python、pandas、scikit-learn、statsmodels、Jupyter。
**Spec:** `docs/superpowers/specs/2026-10-19-hw2-regression-design.md`

## 實作與驗證

- [ ] 取得指定競賽的實際資料，確認授權、欄位及 SHA-256；不可用其他競賽資料冒充。
- [ ] 建立隔離環境、安裝依賴並鎖定實際版本。
- [ ] 測試資料契約、缺值與重複紀錄處理、切分無洩漏，再完成資料載入。
- [ ] 測試並實作共同切分、五折 CV、基準、OLS、兩種 14 特徵選擇與正則化對照。
- [ ] 測試 sklearn 與 statsmodels 預測一致、區間寬度、模型重載與新資料欄位驗證。
- [ ] 完成六階段 Notebook、圖表及假設診斷；重新啟動 kernel 從頭執行。
- [ ] 根據實測結果製作 PDF 報告；保留 NotebookLM、對話匯出、學號及授權待確認事項，不捏造。
- [ ] 檢驗乾淨環境與交付清單，再依學號打包。
