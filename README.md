# AI CUP 2021 繁體中文場景文字辨識競賽－高階賽：複雜街景之文字定位與辨識
(2021/12/24)這是Private Leaderboard第一名的實作程式碼，[比賽連結](https://tbrain.trendmicro.com.tw/Competitions/Details/19)


(2022/01/13)本系統已獲得比賽 **冠軍**，[獲獎名單](https://www.aicup.tw/post/%EF%BC%81%EF%BC%81%E7%B9%81%E9%AB%94%E4%B8%AD%E6%96%87%E6%96%87%E5%AD%97%E5%A0%B4%E6%99%AF%E9%AB%98%E9%9A%8E%E7%AB%B6%E8%B3%BD%E5%BE%97%E7%8D%8E%E5%90%8D%E5%96%AE%EF%BC%81%EF%BC%81)
## 系統架構圖
![Architecture](https://github.com/yaoching0/Traditional-Chinese-Street-View-Text-Recognition/blob/main/data/Architecture.jpg)

## 系統開發環境
作業系統：Windows 10

程式語言：Python 3.8

注：本系統至少需要8GB顯存

## 可使用如下指令安裝全部套件
`!conda create -n trad_ch python=3.8`

`!conda activate trad_ch`

`!pip install -r requirements.txt`
## Getting Started
**下載repository**

`!git clone https://github.com/yaoching0/Traditional-Chinese-Street-View-Text-Recognition.git`

`!cd Traditional-Chinese-Street-View-Text-Recognition`

**下載各個模型的權重檔案：[Google Drive](https://drive.google.com/file/d/1-NUQxovnON0DlgDbFG3s-SCR5XtW6h95/view?usp=sharing)**

將權重存至 [path to repo]/weights/，執行以下指令可對街道圖片中的文字進行偵測及識別

`!python inference.py` 

系統默認偵測 **[path to repo]/input_images** 中的圖片，inference.py會自動讀取當前資料夾路徑，故不需再做任何路徑設定，結果會自動存至 **[path to repo]/submission.csv** 中。

## Inference重要參數

雖然使用默認參數可直接執行，但也可以自行指定

'--image-file-path', type=str, 要偵測的圖片資料夾路徑

'--output-path', type=str, 存放預測結果submission.csv路徑

'--yolo-string', type=str, 偵測圖片中字串的yolov5權重路徑

'--yolo-character', type=str, 偵測圖片中字元的yolov5權重路徑

'--cls-model-eff', type=str, EfficientNetV2分類模型權重路徑

'--cls-model-vit', type=str, Vision Transformer分類模型權重路徑

'--cls-model-nf', type=str, Nf Net分類模型權重路徑

## 訓練分類模型(EfficientNetV2/NF-Net/Vision Transformer)

**分類模型資料集下載：[Google Drive](https://drive.google.com/file/d/1qbEOJeWvy-fejHR2JupT6Sah5L7XQanv/view?usp=sharing)**

三個模型皆是使用timm訓練，可以參照其[github repo](https://github.com/rwightman/pytorch-image-models)，有完整的範例。

本系統訓練時實作了三重隨機性來逼近模擬真實情況，如下圖
![data_aug](https://github.com/yaoching0/Traditional-Chinese-Street-View-Text-Recognition/blob/main/data/data_aug.jpg)

該資料增強程式存放在 **[path to github repo]/data/classifier_data_augmentation.py**

## 訓練Res-EnsembleNet
本團隊在競賽中(進階賽)提出了一個專用於分類模型融合之網路，架構圖如下
![Res-EnsembleNet](https://github.com/yaoching0/Traditional-Chinese-Street-View-Text-Recognition/blob/main/data/Res-EnsembleNet.png)

訓練資料集生成程式 **[path to github repo]/data/generate_feature_for_model_Ensemble_model.py**

訓練程式 **[path to github repo]/trainer/EnsembleNet-train.py**

注：此二程式檔暫未包成自動獲取路徑，需進入程式檔手動替換相關路徑。
## 訓練Bert(三分類/Masked Language Modeling)
三分類和Masked Language Modeling兩個任務都是使用同一個資料集，存放路徑 **[path to repo]/data/tfer-dataset.csv**

Token list 存放路徑 **[path to repo]/data/tokens.txt** 

三分類Bert訓練程式檔：**[path to github repo]/trainer/3-class-bert-train.py**

Masked Language Modeling訓練程式檔：**[path to github repo]/trainer/bert-train-normal-LM.py**

注：此二程式檔暫未包成自動獲取路徑，需進入程式檔手動替換相關路徑。

#### 其中三分類用於判斷[MASK]位置是數字、英文還是中文，如下圖
![bert_3_class](https://github.com/yaoching0/Traditional-Chinese-Street-View-Text-Recognition/blob/main/data/BERT-3-CLASS.PNG)

#### 而另一個Bert則是做Masked Language Modeling，如下圖
![bert_mlm](https://github.com/yaoching0/Traditional-Chinese-Street-View-Text-Recognition/blob/main/data/BERT-MAKSED-LM.PNG)
