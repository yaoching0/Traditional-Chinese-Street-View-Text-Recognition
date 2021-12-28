# AI CUP 2021 繁體中文場景文字辨識競賽－高階賽：複雜街景之文字定位與辨識
這是Private Leaderboard第一名的實作程式碼，[比賽連結](https://tbrain.trendmicro.com.tw/Competitions/Details/19)

## 系統架構圖
![Architecture](https://github.com/yaoching0/Traditional-Chinese-Street-View-Text-Recognition/blob/main/data/Architecture.jpg)

## 系統開發環境
作業系統：Windows 10

程式語言：Python 3.8

## 可使用如下指令安裝全部套件
`!conda create -n trad_ch python=3.8`

`!conda activate trad_ch`

`!pip install -r [path to repo]/requirements.txt`
## Getting Started
**下載repository**

`git clone https://github.com/yaoching0/Traditional-Chinese-Street-View-Text-Recognition.git`

`cd Traditional-Chinese-Street-View-Text-Recognition`

**下載各個模型的權重檔案：[Google Drive](https://drive.google.com/file/d/1-NUQxovnON0DlgDbFG3s-SCR5XtW6h95/view?usp=sharing)**

將權重存至 [path to repo]/weights/，執行

`python inference.py` 



**基本參數：**

--yolo                       Yolo模型權重路徑

--cls_model_eff       EfficientNetV2模型權重路徑

--cls_model_vit       Vision Transfomer模型權重路徑

--cls_model_nf        NF-Net模型權重路徑          

--dataset_path         分類模型資料集路徑

--Ensemble_weight 融合模型權重路徑

--image_file_path    需要預測的image存放的資料夾路徑

--bbx_label_path     需要預測的image之比賽方提供的bouding box資訊csv檔路徑

  以上參數皆以預設完畢，只要相關檔案和主程式放置同一個資料夾即可直接執行，無須另外調整。

  唯若要另外指定其他資料夾中的image作預測，可利用--image_file_path和--bbx_label_path。  
 
**最終輸出會存為：…/繁體中文比賽/submission.csv**

**關於資料處理及模型訓練相關程式介紹請參考報告第三、四章**
