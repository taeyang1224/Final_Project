# 각 광고주마다 지역(최빈값)을 추출
# 결측치는 '전국'으로 처리
def usr_place_list(ad_df):
    user_place = {}
    no_user = []
    for uid in ad_df['user_id'].unique():
        try:
            modd = ad_df[ad_df['user_id'] ==uid]['place'].mode().values[0]
            user_place[uid] = modd
        except:
            user_place[uid] = '전국'
            
    return pd.DataFrame(user_place.values(), index = user_place.keys(), columns=['place'])

# 시/도 양식에 맞게 데이터 변환
def preprocess_place(place):
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

# 광고 데이터 불러와서
ad_df = pd.read_csv('data/ad_new.csv')
# 각 광고주별 가장 많이 나온 지역 데이터 프레임 만들어서
user_place_df =  usr_place_list(ad_df)
# 시/도 양식에 맞게 바꿔주고
user_place_df['place'] = user_place_df['place'].apply(preprocess_place)
# 기존 df와 병합하기 위해 각 광고주 id를 adv_id 컬럼에 추가해주고
user_place_df = user_place_df.reset_index().rename({'index':'adv_id'}, axis=1)

# 기존 df파일 불러와서
df = pd.read_csv('data/merged_data_notext.csv')
# adv_id 기준으로 병합
df = pd.merge(df, user_place_df)
# 기존 place 삭제, 새로운 place로 대체
df = df.drop('content_place', axis=1)
df = df.rename({'place':'content_place'}, axis=1)