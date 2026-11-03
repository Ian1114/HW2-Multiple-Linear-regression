# GPT 輔助內容與 NotebookLM 摘要

姓名：李易軒  
學號：5115056026  
作業：HW2 多元線性回歸

## GPT 輔助內容

本作業使用 GPT／Codex 協助整理多元線性回歸的分析流程、檢查程式與模型評估方法、解讀圖表及研究限制，並協助組織 CRISP-DM 報告與網站的說明文字。資料集選擇、研究問題及最終呈現由我確認；報告中的樣本數、特徵選擇結果、MAE、RMSE、R² 與預測區間，均以實際執行 Notebook 或分析程式所產生的結果核對。針對 AI 提出的解釋，我也檢查程式與資料來源，沒有將 AI 回答直接當作實驗結果或研究文獻。

## NotebookLM 摘要

我使用 NotebookLM 研究 Furkan Kazan 在 Kaggle 發表的 WHO 平均餘命線性回歸 Notebook，並與本作業比較。兩者都使用 WHO 平均餘命資料建立線性回歸模型，但處理缺值與評估模型的方式不同。Kaggle Notebook 刪除部分有缺值的資料列，以中位數補入 GDP 的缺值，並採用隨機資料列切分；本作業的平均餘命分析只刪除缺少預測目標的 10 筆紀錄，其餘特徵在訓練流程中補值，並依國家分組，以未參與訓練的國家進行評估。本作業也比較完整模型與兩種特徵選擇方法。

比較時，我發現 Kaggle Notebook 的線性模型在切分測試集之前，已使用全部資料訓練，因此其後顯示的線性回歸 R² 不能視為獨立測試結果。兩份研究雖然都報告 R²，仍不能直接以數值高低判斷模型優劣；若要公平比較，必須使用相同的資料處理、特徵及測試方式。這次研究讓我了解，解讀模型結果時，評估流程與分數同樣重要。

## 研究來源

- Furkan Kazan，[Life_Expectancy_LinearRegression](https://www.kaggle.com/code/furkankazan/life-expectancy-linearregression)，版本 6；程式第 19–22、48–52 格。
- [本作業平均餘命報告](https://ian1114.github.io/HW2-Multiple-Linear-regression/report.pdf)。

## 佐證文件

NotebookLM 實際使用紀錄與 GPT／Codex 對話 PDF 應另行保存或附上，以供查核。本文的 NotebookLM 使用過程依作業者提供的內容記錄。

