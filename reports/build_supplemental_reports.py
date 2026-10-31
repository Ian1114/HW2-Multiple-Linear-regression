from pathlib import Path
import json
import pandas as pd
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT=Path(__file__).resolve().parent.parent
pdfmetrics.registerFont(TTFont('Kaiu','C:/Windows/Fonts/kaiu.ttf'))
s=getSampleStyleSheet();s.add(ParagraphStyle(name='BodyTC',fontName='Kaiu',fontSize=10,leading=16,spaceAfter=8,wordWrap='CJK'));s.add(ParagraphStyle(name='TitleTC',fontName='Kaiu',fontSize=19,leading=27,alignment=TA_CENTER,spaceAfter=14));s.add(ParagraphStyle(name='HeadingTC',fontName='Kaiu',fontSize=14,leading=20,textColor=colors.HexColor('#245746'),spaceBefore=7,spaceAfter=7));s.add(ParagraphStyle(name='SmallTC',fontName='Kaiu',fontSize=8.6,leading=14,spaceAfter=6,wordWrap='CJK'))
P=lambda t:Paragraph(t,s['BodyTC']); H=lambda t:Paragraph(t,s['HeadingTC']); SP=lambda t:Paragraph(t,s['SmallTC'])
configs={'paris':{'title':'巴黎房價：多元線性回歸分析','url':'https://www.kaggle.com/datasets/mssmartypants/paris-housing-price-prediction','target':'price','unit':'資料集價格單位','problem':'以住宅面積、房間、設施及屋況等欄位估計資料集中的價格，並與平均值基準及特徵選擇版本比較。','context':'資料集作者描述為虛構住宅價格，故並非真實巴黎交易樣本。cityCode 被作為數值輸入，將代碼視作連續量有解讀限制；price 單位依原始資料描述，未自行假設貨幣。','limitation':'資料是虛構生成的，接近 1 的 R² 只能說明此樣本的線性結構，不能外推真實房市。資料未提供同一住宅、區域或時間的群組關係，隨機切分結果可能無法衡量跨區或未來泛化。','purpose':'本機表單輸入 16 個原始特徵，後端讀取原有完整 OLS 管線回傳預測。','compare':'KBest 根據單一變數的線性關聯排序，RFE 逐步移除標準化後係數較小的特徵；使用者可查看兩者名單及 CV 差異，不以保留集挑選模型。'},'student':{'title':'學生成績：多元線性回歸分析','url':'https://www.kaggle.com/datasets/larsen0966/student-performance-data-set','target':'G3','unit':'分（0–20）','problem':'使用學生背景、學習資訊與第一、二期成績，估計葡萄牙語課程最終成績 G3，並比較平均值基準及特徵選擇版本。','context':'本研究使用指定 Kaggle 下載的 student-por.csv：葡萄牙語課程資料。原始檔 33 欄，研究依既定網站模型選用 20 個輸入欄位，目標 G3 不計入特徵。','limitation':'G1、G2 已包含前期成績，模型只適合已取得這些成績的情境，不是入學前預測。分數與社會人口資料屬觀察關係，不能作因果或單獨教育決策；線性預測可能超出 0–20，未靜默裁切。隨機切分不保證跨學校或跨時間泛化。','purpose':'本機表單輸入 20 個原始欄位，後端讀取原有完整 OLS 管線回傳預測。','compare':'KBest 在訓練折中依單變數關聯選 14 欄，RFE 在訓練折中遞迴移除特徵。資料中此處類別皆為二元，獨熱編碼各產生一個欄位；選取數對應 14 個原始特徵。'}}

def footer(canvas,doc):
    canvas.setFont('Kaiu',8);canvas.drawRightString(19*cm,1.2*cm,str(doc.page))

def build(key):
    cfg=configs[key]; d=json.loads((ROOT/'outputs'/f'{key}_report_analysis.json').read_text(encoding='utf8')); m=pd.read_csv(ROOT/'outputs'/f'{key}_report_metrics.csv'); hold=m[m['set']=='holdout']; full=hold[hold.model=='OLS all'].iloc[0]; train=m[(m['set']=='train')&(m.model=='OLS all')].iloc[0]
    rows=[['模型','CV RMSE ± SD','訓練 RMSE','測試 RMSE','測試 R²']]
    for name in ['dummy','OLS all','KBest 14','RFE 14']:
        r=hold[hold.model==name].iloc[0]; tr=m[(m['set']=='train')&(m.model==name)].iloc[0]
        rows.append([name,f'{r.cv_rmse:,.3f} ± {r.cv_std:,.3f}',f'{tr.rmse:,.3f}',f'{r.rmse:,.3f}',f'{r.r2:.8f}'])
    tab=Table(rows,colWidths=[2.65*cm,4.0*cm,2.7*cm,2.7*cm,3.3*cm],repeatRows=1);tab.setStyle(TableStyle([('FONTNAME',(0,0),(-1,-1),'Kaiu'),('FONTSIZE',(0,0),(-1,-1),8.2),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e2efe7')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f5f8f5')]),('GRID',(0,0),(-1,-1),.3,colors.grey),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)]))
    def fig(name,w=16.7,h=4.7):return Image(str(ROOT/'outputs'/f'{key}_{name}.png'),width=w*cm,height=h*cm)
    story=[Paragraph(cfg['title'],s['TitleTC']),P('CRISP-DM 作業報告草稿｜姓名：李易軒｜學號：5115056026｜分析日期：2026-11-01'),H('摘要'),P(f"以 {len(d['features'])} 個原始輸入特徵估計 {cfg['target']}，使用固定種子切分及訓練集內五折交叉驗證。完整 OLS 在保留測試集的 MAE 為 {full.mae:,.3f}、MSE 為 {full.mse:,.3f}、RMSE 為 {full.rmse:,.3f}、R² 為 {full.r2:.8f}。單位：{cfg['unit']}。"),H('1. Business Understanding｜問題理解'),P(cfg['problem']),H('2. Data Understanding｜資料理解'),P(cfg['context']),P(f"原始檔共 {d['raw_rows']:,} 筆、{d['raw_columns']} 欄；全表缺失值 {d['missing_total']} 個，目標缺失 {d['missing_target']} 筆，完全重複列 {d['duplicates']} 筆。目標欄位為 {cfg['target']}。"),SP('資料集來源：<link href="'+cfg['url']+'" color="blue">'+cfg['url']+'</link>'),SP('本機原始檔 SHA-256：'+d['sha256']),H('3. Data Preparation｜資料準備'),P(f"使用 random_state=42 隨機切分為訓練 {d['train_rows']:,} 筆及保留測試 {d['test_rows']:,} 筆。數值欄位採中位數補值與標準化；類別欄位以訓練資料眾數補值及獨熱編碼。上述步驟、特徵選擇與模型都在 pipeline 內，訓練集以打亂的五折交叉驗證擬合，避免驗證折資訊洩漏。"),P('完整模型輸入：'+', '.join(d['features'])+'。'),PageBreak(),H('4. Modeling｜建模與特徵選擇'),P('比較平均值基準、完整 OLS，以及 SelectKBest 和 RFE 各 14 特徵版本。完整 OLS 是原網站已部署的模型；額外比較不根據保留測試集更改已使用的網站模型。CV 指標為五折平均，標準差為各折的標準差。'),tab,Spacer(1,9),P(cfg['compare']),SP('SelectKBest 14 欄：'+', '.join(d['selection']['KBest 14'])+'。'),SP('RFE 14 欄：'+', '.join(d['selection']['RFE 14'])+'。'),H('5. Evaluation｜評估'),P(f"完整 OLS 訓練 RMSE 為 {train.rmse:,.3f}；五折 CV RMSE 為 {full.cv_rmse:,.3f} ± {full.cv_std:,.3f}；保留測試 RMSE 為 {full.rmse:,.3f}。測試集調整後 R² 為 {full.adj_r2:.8f}（n={int(full.n)}，p={int(full.p)}）。"),fig('diagnostics'),P('左圖比較實際與預測，虛線為理想位置；中圖檢視殘差與預測值；右圖為測試殘差分布。'),PageBreak(),H('回歸假設與診斷'),P(f"訓練集 Breusch–Pagan 檢定 LM={d['bp_lm']:.3f}，p={d['bp_p']:.3e}，顯示常態參考下可能存在異質變異。最大 VIF 為 {d['max_vif']:.3f}（{d['max_vif_feature']}）；VIF 衡量線性共線性，不能單獨證明其他假設成立。設計矩陣 rank={d['design_rank']}，欄數（含截距）={d['design_columns']}。"),P('線性與等變異可從殘差圖觀察；常態性可參考下方 Q-Q 圖。獨立性取決於抽樣與資料產生方式，任意檔案排序下的 Durbin–Watson 不能證明獨立。資料集的抽樣與生成限制應納入判斷。'),fig('qq',w=9.5,h=6.7),H('95% 信賴與預測區間'),P('以下按預測值排序並抽樣顯示保留集：窄帶為平均反應信賴區間，寬帶為單筆觀測預測區間。statsmodels 與網站 OLS 的點預測已核對一致。傳統區間依 OLS 假設計算，未納入特徵選擇不確定性；異質變異或資料相依可能使涵蓋率偏離 95%。'),fig('intervals',w=16.7,h=5.7),PageBreak(),H('6. Deployment｜部署'),P(cfg['purpose']),P('模型與前處理一起保存；展示系統不儲存輸入。請確認訓練來源與模型版本一致，本地展示不代表已公開部署。'),H('限制與後續'),P(cfg['limitation']),P('測試結果只適用於此固定切分；要評估結果穩定性，應另行研究重複切分或適合資料取得方式的外部驗證。OLS 的相關性不代表因果。'),H('參考資料'),SP('資料集：<link href="'+cfg['url']+'" color="blue">'+cfg['url']+'</link>'),SP('scikit-learn：線性模型與交叉驗證文件 https://scikit-learn.org/stable/modules/linear_model.html ； https://scikit-learn.org/stable/modules/cross_validation.html'),SP('statsmodels：OLS 預測區間 https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLSResults.get_prediction.html'),H('AI 使用與待補項目'),P('Codex 協助程式、分析與報告整理。真實對話 PDF 與實際 NotebookLM 研究紀錄、至少 100 字摘要仍須由使用者補充；不可將此段冒充 NotebookLM 產出。提交前請核對教授最新規範與資料授權。')]
    SimpleDocTemplate(str(ROOT/'reports'/f'{key}_report.pdf'),pagesize=(21*cm,29.7*cm),leftMargin=2*cm,rightMargin=2*cm,topMargin=1.7*cm,bottomMargin=1.7*cm).build(story,onFirstPage=footer,onLaterPages=footer)
    print(key,'created')

if __name__=='__main__':
    for key in configs:build(key)
