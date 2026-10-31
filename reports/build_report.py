from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import pandas as pd, json
pdfmetrics.registerFont(TTFont('MSung-Light', 'C:/Windows/Fonts/kaiu.ttf'))
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='Chinese',fontName='MSung-Light',fontSize=10,leading=16,spaceAfter=7))
styles.add(ParagraphStyle(name='TitleZH',fontName='MSung-Light',fontSize=19,leading=26,alignment=TA_CENTER,spaceAfter=15))
styles.add(ParagraphStyle(name='HeadZH',fontName='MSung-Light',fontSize=14,leading=21,textColor=colors.HexColor('#14566b'),spaceBefore=8,spaceAfter=7))
P=lambda x:Paragraph(x,styles['Chinese'])
H=lambda x:Paragraph(x,styles['HeadZH'])
metric=pd.read_csv('outputs/metrics.csv'); cv=pd.read_csv('outputs/cv_results.csv'); sel=json.load(open('outputs/selected_features.json',encoding='utf8'))
rows=[['模型','CV RMSE','保留集 MAE','保留集 RMSE','保留集 R²']]
for name in ['dummy','ols','kbest','rfe','ridge_1','lasso_0.01']:
    a=cv[cv.model==name].iloc[0]; b=metric[(metric.model==name)&(metric['set']=='holdout')].iloc[0]
    rows.append([name,f'{a.cv_rmse:.3f}',f'{b.MAE:.3f}',f'{b.RMSE:.3f}',f'{b.R2:.3f}'])
table=Table(rows,repeatRows=1,colWidths=[3.2*cm,2.5*cm,3*cm,3.2*cm,2.5*cm]); table.setStyle(TableStyle([('FONTNAME',(0,0),(-1,-1),'MSung-Light'),('FONTSIZE',(0,0),(-1,-1),9),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#dceef0')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f3f7f7')]),('GRID',(0,0),(-1,-1),.3,colors.grey),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]))
story=[Paragraph('WHO 平均餘命：多元線性回歸分析',styles['TitleZH']),P('CRISP-DM 作業報告草稿｜資料與程式執行日期：2026-11-01｜姓名：李易軒｜學號：5115056026｜尚待補入 AI 使用證據'),H('摘要'),P('以國家年度健康與社經指標估計平均餘命。原始檔有 2,938 筆；排除 10 筆缺少目標後，依國家切分訓練與保留資料。以訓練國家的五折交叉驗證選擇 OLS 主模型；20 特徵 OLS 的保留集 MAE 為 3.121 年、RMSE 為 4.063 年，R² 為 0.823。資料中的同年度死亡率等欄位可能與目標存在概念重疊，本研究不能解讀為因果效果或未來預測。'),H('1. Business Understanding｜問題理解'),P('研究問題：能否利用國家年度的健康與社經資訊估計未見國家的平均餘命？成功標準為優於訓練資料平均值基準，並檢查誤差、特徵選擇與線性模型限制。此資料為國家層級觀測，不能推論個人壽命。'),H('2. Data Understanding｜資料理解'),P('資料來源：Kaggle「Life Expectancy (WHO)」：https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who 。本機 CSV 包含 193 國、2000–2015 年、22 欄；國家及年份組合沒有重複。目標為 Life expectancy；20 個輸入包含 Year、Status 與 18 個健康、人口及社經數值指標。原始 CSV 的 SHA-256：872125dd1dd0f9140fbead61df20585a815f5cf47db68f08bf54efaf87963b11。'),P('缺失較多的欄位包括 Population（652）、Hepatitis B（553）及 GDP（448）；目標缺失 10 筆直接排除。資料來源描述、版本與再散布條件須於提交前核對。'),PageBreak(),H('3. Data Preparation｜資料準備'),P('國家名稱不進模型，只作分組依據。Status 以 Developing=0、Developed=1 編碼。保留 37 個國家共 592 筆，其餘 146 個國家共 2,336 筆訓練；資料分組以 seed 42 固定。訓練集內使用五折 GroupKFold，同一國家不會跨訓練折與驗證折。中位數補值與標準化放入 pipeline，由各折訓練資料擬合。此切分評估跨國泛化，並不驗證對未來年份的預測。'),H('4. Modeling｜建模'),P('候選包括平均值基準、20 特徵 OLS、SelectKBest 與 RFE 各選 14 特徵，以及 Ridge、Lasso；正則化強度以相同分組交叉驗證比較。以訓練 CV RMSE 選 OLS 主模型，結果為全部 20 特徵；CV 選參後的最佳分數可能樂觀，保留集用作最終評估。'),table,Spacer(1,10),P('OLS、RFE 的 CV RMSE 幾乎相同，兩者的差距不應被解讀為明確的實務優勢。選取清單儲存在 outputs/selected_features.json；Lasso 最終保留 19 個非零係數，屬補充比較。'),H('5. Evaluation｜評估'),P('保留集的 OLS 調整後 R² 為 0.817；平均值基準 RMSE 為 9.949 年。實際值與預測值及殘差圖如下。不同國家、年度的資料量不一，這些指標按資料列計算，不是每國等權的平均誤差。'),Image('outputs/holdout_diagnostics.png',width=17*cm,height=4.5*cm),PageBreak(),H('回歸診斷與區間'),P('訓練資料 Breusch–Pagan LM 檢定 p 值約 3.88e-58，提示殘差變異可能不一致；同國觀測相依使傳統檢定的推論受到限制。infant deaths 與 under-five deaths 的 VIF 約為 189 與 188，顯示嚴重共線性。殘差圖與 Q-Q 圖可協助判斷線性及常態偏離；同國跨年度相依性不能從任意排序的 Durbin–Watson 判定。'),P('以下是依預測值排序的部分保留樣本，帶狀區分 95% 平均反應信賴區間與單筆觀測預測區間。區間依傳統 OLS 假設計算，未納入模型選擇的不確定性；異質變異與國家內相關可能使覆蓋率偏離 95%。'),Image('outputs/intervals.png',width=17*cm,height=6.5*cm),H('6. Deployment｜部署'),P('完整前處理與選定的 OLS 已存為 joblib；Notebook 驗證重載前後預測一致。可使用 hw2lib.predict 接收新 CSV，檢查必要欄位、數值與 Status 類別，再輸出預測。此研究模型不應直接用於政策或個人判斷。'),H('限制與後續'),P('這是觀察性、國家層級、歷史資料；不可宣稱因果。資料可能存在指標定義變更、缺失機制不隨機、估計值誤差及同國年度相依。部分死亡率與平均餘命在概念或推估流程可能重疊，需確認發布時間與來源後才能主張事前預測。OLS 係數受嚴重共線性影響，係數方向及大小不能直接解讀為政策效果。'),H('AI 使用與待補項目'),P('Codex 協助分析程式、解讀及報告整理；應附真實對話 PDF。NotebookLM 至少 100 字摘要必須來自實際使用與來源紀錄，目前尚未取得，不以本報告代稱。最終交付前需核對教授最新作業規範及資料授權，並確認所有結果與 Notebook 一致。')]
def footer(canvas,doc):
    canvas.setFont('MSung-Light',8); canvas.drawRightString(19*cm,1.2*cm,str(doc.page))
SimpleDocTemplate('reports/life_expectancy_report.pdf',pagesize=(21*cm,29.7*cm),rightMargin=2*cm,leftMargin=2*cm,topMargin=1.7*cm,bottomMargin=1.7*cm).build(story,onFirstPage=footer,onLaterPages=footer)


