# 번개장터 CTR(클릭율) 향상을 위한 신규 게시글 등록 가이드

### 🗓️ 프로젝트 기간
2021년 12월 13일 ~ 2022년 1월 24일
### Team
|                            김동영                            |                            심태양                            |                            선명한                            |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| [![Avatar](https://avatars.githubusercontent.com/u/89237850?v=4)](https://github.com/dongyoung0) | [![Avatar](https://avatars.githubusercontent.com/u/89237873?v=4)](https://github.com/taeyang1224) | [![Avatar](https://avatars.githubusercontent.com/u/89237880?v=4)](https://github.com/Sunmyunghan) |
| [Github](https://github.com/dongyoung0) | [Github](https://github.com/taeyang1224) | [Github](https://github.com/Sunmyunghan) |

### Notion: [Final_project](https://www.notion.so/SSAC-Final_Project_3-2d335c0493614a0d87c78fa52a0ee01c)

### ✔️ 프로젝트 목적
**번개장터**에서 제공받은 **광고 로그 데이터**를 기반으로 게시글들의 **클릭율을 예측**하여 클릭율이 높은 게시글들의 **특징을 분석**한 후 신규 게시글 등록시에 **가이드를 제공**해주는 프로젝트입니다.

### 🛠️ 사용 기술 및 라이브러리
- CTR 예측 : LightGBM, DeepFM(Pytorch)
- 이미지 분석 : OpenCV, ResNet, DeepLab v3+
- 텍스트 분석 : KoNLPy, Gensim(FastText)
- 프로토타입 : Streamlit

### 파이프라인

![image](https://user-images.githubusercontent.com/89237850/151507927-8e9942b4-72b5-4b9e-a3d1-23b163eadcfd.png)

### 🏃🏻‍♂️ 진행 사항

1. **문제 정의**
    - CTR을 예측하고 특징 분석을 통한 신규 게시물 가이드 제공   
      → 매력적인 상품을 쉽게 발견
2.  **데이터 수집** 
    - 번개장터에서 제공한 광고 로그 데이터와 번개장터 API를 통해 게시글 크롤링  
     
3.  **데이터 전처리**
    - Text **:** 게시글 정보로 FastText 학습 후 텍스트를 벡터로 임베딩, 유사도 계산
    - Image : ResNet으로 이미지 벡터 임베딩, DeepLab v3+로 배경과 상품 분리 및 특징 추출  
    
4.  **CTR 예측 모델 학습** 
    - 제공받은 광고 로그 데이터로 LightGBM, DeepFM 모델 학습 및 비교
    - 최종 모델로 DeepFM 선정  
     
5.  **CTR 예측** 
    - 최종 학습시킨 모델을 통해 크롤링한 신규 게시글들의 CTR 예측  
     
6.  **이미지 및 텍스트 특징 분석** 
    - 클릭율이 높다고 예측된 게시글들의 특징 분석 및 인사이트 도출
        - 이미지 : 배경과 상품 분리 후 색, 밝기, 노이즈등 특징 추출 및 분석
        - 텍스트 : 제목-키워드 유사도 분석, 클릭율이 높은 키워드 랭킹 분석   
        
7.  **프로토타입 구현** 
    - 신규 게시글 입력 시 이미지, 텍스트 가이드와 예상 클릭율을 제공하는 웹 구현


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

