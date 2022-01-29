# 번개장터 CTR(클릭율) 향상을 위한 신규 게시글 등록 가이드

## Overview
### 프로젝트 기간
2021년 12월 13일 ~ 2022년 1월 24일
### Team
|                            김동영                            |                            심태양                            |                            선명한                            |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| [![Avatar](https://avatars.githubusercontent.com/u/89237850?v=4)](https://github.com/dongyoung0) | [![Avatar](https://avatars.githubusercontent.com/u/89237873?v=4)](https://github.com/taeyang1224) | [![Avatar](https://avatars.githubusercontent.com/u/89237880?v=4)](https://github.com/Sunmyunghan) |
| [Github](https://github.com/dongyoung0) | [Github](https://github.com/taeyang1224) | [Github](https://github.com/Sunmyunghan) |

### Environments
`https://github.com/shenweichen/DeepCTR-Torch`

### 프로젝트 목적

### 파이프라인

![image](https://user-images.githubusercontent.com/89237850/151507927-8e9942b4-72b5-4b9e-a3d1-23b163eadcfd.png)

### CTR 예측 모델
####  LightGBM
####  DeepFM

### 결과


## Code Structure
```
├── notebooks/
│   ├── 데이터 수집 및 전처리/
│   │   ├── 전처리 및 병합.ipynb
│   │   ├── 신규 게시글 크롤링.ipynb
│   │   ├── image/
│   │   │       background_OpenCV.ipynb
│   │   │       image_background.ipynb
│   │   │       image2vec_practice.ipynb
│   │   │
│   │   └── text/
│   │           brand.txt
│   │           FastText.ipynb
│   │           train_text2vec.ipynb
│   │
│   ├── CTR Prediction/
│   │   ├── LightGBM/
│   │   ├── DeepFM/
│   │   └── CTR prediction.ipynb
│   │
│   ├── 특징 분석/
│   │       모델 비교 EDA.ipynb
│   │       정형 데이터 분석.ipynb
│   │       이미지 분석.ipynb
│   │       텍스트 분석.ipynb
│   │
│   ├── SQL.py
│   └── SQL_업로드.py
│
├── code/
│   ├── data/
│   │
│   ├── deepctr_torch/
│   │
│   ├── requirements.txt
│   ├── preprocess.py
│   ├── feature_extractor.py
│   ├── predictor.py
│   ├── app.py
│   └── README.md
│
├── documents
│   ├── 기획안
│   └── 발표자료
│
└── README.md
```


## Web demo
```
$ cd code
pip install -r requirements.txt
```

```
$ streamlit run app.py
```
