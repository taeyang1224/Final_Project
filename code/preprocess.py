# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import torch
import joblib
import json
from tqdm.notebook import tqdm

class Preprocesser():
    '''
    크롤링 및 새로운 데이터를 전처리하는 클래스
    '''
    def __init__(self):
        self.file_path = 'data/'

    def load_dict(self):
        with open(self.file_path + 'preprocess_dict.json', 'r') as f:
            self.pre_dict = json.load(f)

    def map_place(self, place):
        # 전처리 dict에서 place 전처리 dict 꺼내기
        place_dict = self.pre_dict['place']
        # place dict에서 key(서울)가 포함되면 value(서울특별시)로 mapping
        # 예) 서울특별시 송파구 잠실동 -> 서울특별시
        if isinstance(place, str):
            for name in place_dict.keys():
                if name in place:
                    place = place_dict[name]
        # 결측 및 예외들은 전국으로 처리
        if place not in place_dict.values():
            place = '전국'
        return place

    def preprocess_df(df_):
        df = df_.copy()
        rename_dict = self.pre_dict['rename']
        df = df.rename(rename_dict, axis=1)
        # 중복 제거
        df = df.drop_duplicates('content_id')
        df = df.reset_index(drop=True)
        if isinstance(df['category_name'].iloc[0], str):
            # 대분류
            df['content_cat_1'] = df['category_name'].apply(lambda x : x.split("'")[3])
            # 중분류
            df['content_cat_2'] = df['category_name'].apply(lambda x : x.split("'")[7] if len(x.split("'"))>6 else x.split("'")[3])
            # 소분류
            df['content_cat_3'] = df['category_name'].apply(lambda x : x.split("'")[-2])
        else:
            df['content_cat_1'] = df['category_name'].apply(lambda x: list(x.values)[0])
            df['content_cat_2'] = df['category_name'].apply(lambda x: list(x.values)[1])
            df['content_cat_3'] = df['category_name'].apply(lambda x: list(x.values)[2])

        # delivery_fee -> train 데이터와 같게
        df['content_delivery_fee'] = df['content_delivery_fee'].apply(lambda x : 1*x)
        # b_pay -> train 데이터와 같게
        df['content_b_pay'] = df['content_b_pay'].apply(lambda x : 1 - 1*x)
        # 중고 여부를 train 데이터와 같게
        df['content_used'] = df['content_used'].apply(lambda x: 0 if x==2 else 1)
        # 지역명을 시/도 형식에 맞게
        df['content_place'] = df['content_place'].apply(map_place)
        # pay option에서 필요한 부분만 사용
        df['pay_option_in_person'] = df['pay_option'].apply(lambda x : x.split("'")[4][2:-2])
        df['pay_option_bun_pay_filter_enabled'] = df['pay_option'].apply(lambda x : x.split("'")[6][2:-2])
        # unix time을 실제 시간으로
        df['update_time'] = pd.to_datetime(df['update_time'], unit='s')
        df['access_time'] = pd.to_datetime(df['access_time'], unit='s')
        df['join_date'] = pd.to_datetime(df['join_date'], unit='s')
        
        df['seller_name'] = df['badges'].apply(lambda x : x[94:97])

        # 필요 없는 컬럼 drop 및 보기 좋게 정렬
        sort_col = self.pre_dict['sort']
        df = df[sort_col]
        return df
