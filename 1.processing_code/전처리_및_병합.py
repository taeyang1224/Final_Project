# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import datetime
from datetime import timedelta


# impression, view logs
def drop_duplicate_time(df):
    dup_id = df['imp_id'].value_counts()[df['imp_id'].value_counts() > 1].index.tolist()
    drop_dup = df.set_index('imp_id').loc[dup_id].reset_index().groupby('imp_id').min().reset_index()
    no_dup = df.set_index('imp_id').drop(dup_id).reset_index()
    df = pd.concat([no_dup, drop_dup])
    return df


def merge_logs(impression_df, view_df):
    # 중복 제거
    impression_df = impression_df.drop_duplicates()
    view_df = view_df.drop_duplicates()
    
    # impression 데이터에 log가 없는 view들의 id
    # 노출된 기록이 없는데 클릭한 기록만 있는 것들
    no_imp_id = list(set(view_df['imp_id']) - set(impression_df['imp_id']))
    # impression 데이터에 log가 없는 view들 모음
    no_imp_view = view_df.set_index('imp_id').loc[no_imp_id].reset_index()
    
    # 기존 view_df에서 impression과 겹치지 않는 id 제거
    view_df = view_df.set_index('imp_id').drop(no_imp_id).reset_index()
    # impression_df의 imp_id, user_id 기준으로 view_df의 view들에 user_id 붙여줌
    imp_user = impression_df[['imp_id', 'user_id', 'content_id']]
    view_df = view_df.merge(imp_user, 'left').drop_duplicates()
    # 라벨 생성
    # - 광고를 클릭했으면 1, 노출만 됐으면 0을 부여해서 병합
    view_df['label'] = 1
    impression_df['label'] = 0
    # 같은 impression이 한 유저에게 여러번 나온 경우 한개만 사용
    impression_df = drop_duplicate_time(impression_df)
    view_df = drop_duplicate_time(view_df)
    
        
    # 시간 전처리
    view_df['server_time_kst'] = view_df['server_time_kst'].apply(lambda x : x[11:-6])
    impression_df['server_time_kst'] = impression_df['server_time_kst'].apply(lambda x : x[11:-6])

    # view_df 에서 노출된 시간, 클릭한 시간을 따로 저장
    view_df['imp_time'] = impression_df.set_index('imp_id').loc[view_df['imp_id']]['server_time_kst'].tolist()
    view_df = view_df.rename({'server_time_kst': 'view_time'}, axis=1)
    
    # 클릭된 광고들의 정보는 view_df에 들어갔으므로, impression_df에는 클릭이 되지 않은 노출 데이터만 남기기
    impression_df = impression_df.set_index('imp_id').drop(view_df['imp_id']).reset_index()
    impression_df = impression_df.rename({'server_time_kst': 'imp_time'}, axis=1)


    # view, imp 병합
    merged_df = pd.concat([view_df, impression_df])
    return merged_df

# Ad 병합
def merge_crawling(ad_df, ad_crawling):
    ad_df = pd.concat([ad_df, ad_crawling], axis=1)
    ad_df = ad_df.drop('id', axis=1)
    return ad_df

def merge_content(df, ad_df):
    # 중복 제거
    ad_df = ad_df.drop_duplicates()
    # 컬럼명 변경
    ad_dict = {'user_id': 'advertiser_id', 'name': 'content_name', 'keyword': 'content_keyword', 
               'price': 'content_price', 'flag_used': 'content_used', 
               'category_id_1': 'content_cat_1', 'category_id_2': 'content_cat_2', 'category_id_3': 'content_cat_3', 
              'emergency_cnt': 'content_emergency_count', 'comment_cnt': 'content_comment_count',
              'interest': 'content_views', 'pfavcnt': 'content_likes', 
              'status': 'content_status', 'b_pay': 'content_b_pay', 'place': 'content_place',
              'text': 'content_text', 'delivery_fee': 'content_delivery_fee'}
    ad_df = ad_df.rename(ad_dict, axis=1)
    
    # 카테고리 한글로
    cat2kor_1 = {220: '지역 서비스', 310: '여성 의류', 320: '남성 의류', 
                 400: '남성 악세사리', 405: '신발', 410: '미용품', 420: '시계/쥬얼리', 430: '가방',
                500: '유아동/출산', 600: '전자제품', 700: '스포츠/레저', 750: '차량/오토바이', 800: '생활/가공식품',
                810: '가구/인테리어', 900: '도서/티켓/문구', 910: '스타굿즈', 920: '음반/악기', 930: '키덜트',
                 980: '반려동물용품', 990: '예술/희귀/수집품',999: '기타'
                }
    ad_df['content_cat_1'] = ad_df['content_cat_1'].apply(lambda x: cat2kor_1[x])
    
    # 중고 여부 3인 것들 1로 변경
    ad_df['content_used'] = ad_df['content_used'].apply(lambda x : x % 2)
    # image url 생성
    ad_df['content_img_url'] = ad_df['content_id'].apply(lambda x: f'https://media.bunjang.co.kr/product/{x}_...')

    merged_df = df.merge(ad_df, on='content_id')
    return merged_df

### Advertiser.csv
def merge_adv(df, adv_df):
    # 중복 제거
    adv_df = adv_df.drop_duplicates()
    # 컬럼명 변경
    adv_name_dict = {'user_id': 'advertiser_id','comment_count': 'adv_comment_count', 'follower_count' : 'drop', 'pay_count': 'adv_pay_count', 
                'parcel_post_count': 'adv_parcel_post_count', 'transfer_count': 'adv_transfer_count', 'chat_count': 'adv_chat_count',
                    'grade': 'adv_grade', 'favorite_count': 'adv_follower_count', 'review_count': 'adv_review_count', 'interest': 'adv_views',
                     'title': 'adv_title', 'item_count': 'adv_item_count'}
    adv_df = adv_df.rename(adv_name_dict, axis=1)
       
    merged_df = df.merge(adv_df, on='advertiser_id')
    merged_df = merged_df.rename({'advertiser_id': 'adv_id'}, axis=1)
    return merged_df


# Viewer
def merge_viewer(df, viewer_df):
    # 중복 제거
    viewer_df = viewer_df.drop_duplicates()
    # 컬럼명 변경
    viewer_name_dict = {'comment_count': 'viewer_comment_count', 'following_cnt' : 'viewer_following_count', 'pay_count': 'viewer_pay_count', 
                'parcel_post_count': 'viewer_parcel_post_count', 'transfer_count': 'viewer_transfer_count', 'chat_count': 'viewer_chat_count',
                       'gender': 'viewer_gender', 'age': 'viewer_age'}
    viewer_df = viewer_df.rename(viewer_name_dict, axis=1)
    # 병합
    merged_df = df.merge(viewer_df, on='user_id')
    return merged_df

# 정리
def drop_trash(df):
    df = df.drop('drop', axis=1)
    df = df.drop('device_type', axis=1)
    df = df.drop('category', axis=1)
    
    df = drop_buyer(df)
    return df

def drop_buyer(df):
    ad_df = df[['content_id', 'content_name']]
    ad_df = ad_df.drop_duplicates()
    drop_name_list = []
    for c_name in ad_df['content_name']:
        if '매입' in c_name:
            drop_name_list.append(c_name)
        elif '삽니다' in c_name:
            drop_name_list.append(c_name)
        
    drop_name_id = ad_df.set_index('content_name').loc[drop_name_list]['content_id'].tolist()
    df = df.set_index('content_id').drop(drop_name_id).reset_index()
    return df

### age feature 전처리
def age_ch_prepro(df):
    # 새로운 column 생성
    df["viewer_age_ch"] = df["viewer_age"]
    # 80세 이상을 0세로 변경
    df.loc[df["viewer_age_ch"] > 80, "viewer_age_ch"] = 0
    return df



# +
### place feature 전처리
def sido(place):
    '''시/도 양식에 맞게 데이터 변환'''
    place = place[:6]
    if '서울' in place:
        place = '서울특별시'
    elif '강남' in place:
        place = '서울특별시'
    elif '경기도' in place:
        place = '경기도'
    elif '대구' in place:
        place = '대구광역시'
    elif '대전' in place:
        place = '대전광역시'
    elif '인천' in place:
        place = '인천광역시'
    elif '경상북도' in place:
        place = '경상북도'
    elif '경상남도' in place:
        place = '경상남도'
    elif '충청북도' in place:
        place = '충청북도'
    elif '강원도' in place:
        place = '강원도'
    return place

def preprocess_place(df):
    # 데이터프레임에서 광고주 id, 지역만 따로 추출
    ad_df = df[['adv_id', 'content_place']].copy()
    # 각 광고주마다 지역(최빈값)을 추출
    user_place = {}
    no_user = []
    for uid in ad_df['adv_id'].unique():
        try:
            modd = ad_df[ad_df['adv_id'] ==uid]['content_place'].mode().values[0]
            user_place[uid] = modd
        except:
            # 결측치는 '전국'으로 처리
            user_place[uid] = '전국'
    user_place_df = pd.DataFrame(user_place.values(), index = user_place.keys(), columns=['content_place'])
    user_place_df['content_place'] = user_place_df['content_place'].apply(sido)
    user_place_df = user_place_df.reset_index().rename({'index':'adv_id'}, axis=1)
    df = df.drop('content_place', axis=1)
    df = pd.merge(df, user_place_df)
    return df


# -

### title 길이 추가
def title_len(df):
    df['title_len'] = df['content_name'].apply(lambda x: len(x))
    return df

### B_pay feature 전처리
def cat_bpay_prepro(df):
    # 전체 중의 각 카테고리별 b_pay 사용 비율
    b_pay_rate = df.groupby(['content_cat_1']).sum()['content_b_pay'] / df.groupby(['content_cat_1']).count()['content_b_pay']
    # column명 변경
    b_pay_rate = pd.DataFrame(b_pay_rate).rename(columns={"content_b_pay" : "b_pay_rate"})
    # 원래 데이터프레임과의 병합
    b_pay_merge = pd.merge(df, b_pay_rate, how='left', on='content_cat_1')
    return b_pay_merge

### time feature 전처리
def time_len_prepro(cnt_T, col):

    for i in range(len(cnt_T)):
        if len(cnt_T[col][i]) < 8:
            cnt_T[col][i] = pd.to_datetime(cnt_T[col][i], format="%H:%M")
        elif len(cnt_T[col][i]) < 12:
            cnt_T[col][i] = pd.to_datetime(cnt_T[col][i], format="%H:%M:%S")
        else:
            cnt_T[col][i] = pd.to_datetime(cnt_T[col][i], format="%H:%M:%S.%f")
    return cnt_T

def time_prepro(df):
    # view_time에 있는 Nan값 구분하여 새로운 column 생성
    cnt_F = df[df['view_time'].isnull()].copy()
    cnt_T = df[df['view_time'].notnull()].copy()
    cnt_F[['click', 'delay']] = 0
    cnt_F.index=range(len(cnt_F))
    cnt_T['click'] = 1
    cnt_T.index=range(len(cnt_T))

    # 각 형식이 다르기 때문에 길이와 형식을 맞춰줌
    cnt_T = time_len_prepro(cnt_T, col = "imp_time")
    cnt_T = time_len_prepro(cnt_T, col = "view_time")     
    
    cnt_T['delay'] = cnt_T['view_time'] - cnt_T['imp_time']
    
    # 초단위 계산
    sec = cnt_T["delay"].dt.seconds
    sec = sec.apply(lambda x: x-86400 if (x > 80000) else x)
    # "sec" 컬럼 생성
    cnt_T["sec"] = sec

    bcnt_time = pd.concat([cnt_T,cnt_F])

    # 필요없는 click, delay 컬럼 drop
    bcnt_time.drop(["click", "delay"], axis = 1, inplace = True)
    bcnt_time["sec"] = bcnt_time["sec"].fillna(0)

    return bcnt_time

def preprocessing(df):
    df = age_ch_prepro(df)
    df = preprocess_place(df)
    df = cat_bpay_prepro(df)
    df = time_prepro(df)
    df = title_len(df)
    return df

# +
# 전처리 및 병합 진행

# +
impression_df = pd.read_csv('data/raw/impression_log.csv')
view_df = pd.read_csv('data/raw/view_log.csv')
df = merge_logs(impression_df, view_df)

ad_df = pd.read_csv('data/ad_new.csv')
df = merge_content(df, ad_df)

adv_df = pd.read_csv('data/crawling/advertiser_title.csv')
df = merge_adv(df, adv_df)

viewer_df = pd.read_csv('data/raw/viewer.csv')
df = merge_viewer(df, viewer_df)

df = drop_trash(df)
df_notext = df.drop(['content_keyword', 'content_text', 'adv_title'], axis=1)

df_new = preprocessing(df_notext)
df_full = preprocessing(df)
