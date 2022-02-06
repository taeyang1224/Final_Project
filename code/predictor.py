
import numpy as np
import pandas as pd
import json
import joblib
import torch
from itertools import product
from preprocess import Preprocesser
from feature_extractor import TextFeatExtractor, ImageFeatExtractor

class Predictor():
    def __init__(self):
        self.preprocesser = Preprocesser()
        self.image_processer = ImageFeatExtractor()
        self.text_processer = TextFeatExtractor()

    def load_feature(self):
        self.preprocesser.load_dict()
        self.categorical = self.preprocesser.pre_dict['features']['categorical']
        self.continuous = self.preprocesser.pre_dict['features']['continuous']

    def make_candidates(self):
        # 유저 정보 (median 사용)
        users = {'bid_price': [50],
                'viewer_gender': [1,2],
                'viewer_age': range(10, 80),
                'viewer_chat_count': [0.0],
                'viewer_following_count': [1.0],
                'viewer_parcel_post_count': [0.0],
                'viewer_pay_count': [0.0],
                'viewer_transfer_count': [0.0]
                }

        adver = {'adv_follower_count': 6,
                'adv_item_count': 60,
                'adv_review_count': 9,
                'content_likes': 1,
                'content_comment_count': 1,
                'content_views': 200
                }

        # 유저 생성
        user_list = []
        for i, values in enumerate(list(product(*users.values()))):
            user = dict(zip(users.keys(), values))
            user_list.append({**user, **adver})

        return pd.DataFrame(user_list)

    def prepare(self):
        print('예측 준비')
        self.load_feature()
        # Label Encoder
        with open('data/cat_1_dict.json', 'r') as f:
            self.cat_label = json.load(f)
        # Standard Scaler
        self.scaler = joblib.load('model/standard_scaler.pkl')
        # 학습된 DeepFM 모델
        self.model_dfm = torch.load('model/deepfm_final.h5')
        self.candidates = self.make_candidates()
        print('완료')


    def predict_ctr(self, input_features, image_vec, text_vec):
        '''
        입력받은 게시글 정보와 생성한 가상 유저 정보를 
        DeepFM에 넣어서 CTR 예측
        '''
        candidates_df = self.candidates.copy()
        feature_names = self.categorical + self.continuous
        continuous = self.continuous
        scaler = self.scaler

        num_cands = 140
        for name, value in input_features.items():
            candidates_df[name] = value
        
        candidates_df[continuous] = scaler.transform(candidates_df[continuous])

        candidates_input = {name: candidates_df[name] for name in feature_names}
        candidates_input['image_vec'] = np.array([list(image_vec) for i in range(num_cands)]) 
        candidates_input['text_vec'] = np.array([text_vec for i in range(num_cands)])
        
        CTR_dfm = self.model_dfm.predict(candidates_input, 512)

        return CTR_dfm