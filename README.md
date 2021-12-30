# Final_Project
- 주제: 번개장터 데이터를 활용한 머신러닝 프로젝트

## Team
|                            김동영                            |                            심태양                            |                            선명한                            |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| [![Avatar](https://avatars.githubusercontent.com/u/89237850?v=4)](https://github.com/dongyoung0) | [![Avatar](https://avatars.githubusercontent.com/u/89237873?v=4)](https://github.com/taeyang1224) | [![Avatar](https://avatars.githubusercontent.com/u/89237880?v=4)](https://github.com/Sunmyunghan) |
| [Github](https://github.com/dongyoung0) | [Github](https://github.com/taeyang1224) | [Github](https://github.com/Sunmyunghan) |

## 프로젝트 소개
- 작업환경: Colab, Git-hub, Jupyter-Notebook
- 사용언어: python
- 작업 계획
<img src="https://user-images.githubusercontent.com/89237873/147722365-d30be98c-32d3-4a08-82fc-bce058c66511.png" width="400" height="400">


## Requirements
[![Python-package GitHub Actions Build Status](https://github.com/microsoft/LightGBM/workflows/Python-package/badge.svg?branch=master)](https://github.com/microsoft/LightGBM/actions)
- colab
- DeepCTR
- SQL

## Structure
```
├── 0.crawling_bunjang
│   ├── ad_content_crawling.ipynb
│   ├── advertiser_title 크롤링(21.12.15).ipynb
│   └── crawling.md
│
├── 1.processing_code
│   ├── age_ch_processing.ipynb
│   ├── b_pay_rate_processing.ipynb
│   ├── EDA_category-emergency(21.12.16).ipynb
│   ├── image_to_feature.ipynb
│   ├── merge_data.ipynb
│   ├── place.py
│   ├── 전처리_및_병합.py
│   └── processing.md
│
├── 2.model_code
│   ├── deepctr_torch
│   │   │
│   │   ├── layers
│   │   │   ├── __init__.py
│   │   │   ├── activation.py
│   │   │   ├── core.py
│   │   │   ├── interaction.py
│   │   │   ├── sequence.py
│   │   │   └── utils.py
│   │   │
│   │   ├── models
│   │   │   ├── __init__.py
│   │   │   ├── basemodel.py
│   │   │   └── deepfm.py
│   │   │
│   │   ├── __init__.py
│   │   ├── callbacks.py
│   │   ├── inputs.py
│   │   └── utils.py
│   │
│   ├── lgb_model
│   │    └── grid_100_result.csv
│   │
│   ├── DeepFM(예시).ipynb
│   ├── DeepFM_1227_동영.ipynb
│   ├── DeepFM_태양.ipynb
│   ├── LightGBM(예시).ipynb
│   ├── LightGBM_display.ipynb
│   └── Model.md
│
├── documents
│   └── documents.md
│
├── img
│
├── .gitignore
├── DeepFM_동영.ipynb
├── introduce.md
├── LICENSE
└── README.md
```

## 모델
|모델|AUC|RIG|
|--|--|---|
|LGB|0.7813|0.1186|
|DeepFM|0.7487|0.0856|

## 특징분석
