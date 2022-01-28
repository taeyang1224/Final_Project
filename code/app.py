
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
import json
from feature_extractor import TextFeatExtractor, ImageFeatExtractor
from predictor import Predictor

def prepare_predict():
    predictor = Predictor()
    predictor.prepare()

    TFE = TextFeatExtractor()
    TFE.load_model()
    TFE.load_brand_dict()

    IFE = ImageFeatExtractor()
    IFE.prepare()
    return predictor, TFE, IFE

def load_image(image_file):
	img = Image.open(image_file)
	return img

def main():
    # st.title('매력적인 광고를 위한 가이드')
    menu = ["새 게시글","Dashboard"]
    choice = st.sidebar.selectbox("Menu",menu)
    predictor, TFE, IFE = prepare_predict()
    df_cat = pd.read_csv('data/카테고리별_CTR.csv')
    df_price = pd.read_csv('data/가격별_CTR.csv')

    if choice == '새 게시글':
        st.subheader("새 게시글 등록")
        content_dict = {}
        image_file = st.file_uploader("이미지", type=["png","jpg","jpeg"])
        if image_file is not None:            
            img = load_image(image_file)
            
            mask = IFE.get_mask(img)
            img_dict = IFE.image_features(np.array(img), mask)

            # col1, col2, col3 = st.columns(3)
            col1, col2 = st.columns(2)
            img = img.convert('RGB')
            img_re = img.resize((224,224))
            col1.image(img_re)            
   
            bright = img_dict['bg_rgb'].max()
        
            if img_dict['bg_noise'] > 0.05:
                col2.warning(':bulb: 좀 더 단순한 배경을 사용해보세요')
            if bright < 30:
                col2.error("배경이 너무 어두워요 :disappointed_relieved:")
            elif bright < 200:
                col2.warning(":bulb: 좀 더 밝은 배경을 사용해보세요")
            else:
                col2.success('좋아요 :thumbsup:')
            
            # col1, col2 = st.columns(2)
            # col1.image(img * mask[:,:,np.newaxis])
            # col2.write(img_dict) 
        
        c_name = st.text_input('상품명', placeholder = '상품명을 입력해주세요 ~')
        content_dict['content_name'] = TFE.re_kor_brand(c_name)
        
        cat_1 = st.selectbox('대분류', ['남성의류', '여성의류'])
        if cat_1 == '남성의류':
            cat_2 = st.selectbox('중분류', ['맨투맨', '후드', '패딩', '코트', '셔츠', '가디건', '바지', '청바지', '반바지', '자켓', '정장', '트레이닝', '언더웨어'])
        else:
            cat_2 = st.selectbox('중분류', ['원피스', '코트', '맨투맨', '셔츠', '바지', '청바지', '반바지', '치마', '가디건', '니트', '자켓', '정장', '점프수트', '트레이닝', '언더웨어'])
    
        name_cat = content_dict['content_name'] + [cat_1, cat_2]
        
        tag_df = pd.read_csv('data/키워드_랭킹.csv')
        tag_df = tag_df[tag_df.name.apply(lambda x: isinstance(x, str))]
        tag_list = tag_df[tag_df.cnt>10].name
        content_dict['keyword'] = st.multiselect('# 태그', options=tag_list)
        if len(content_dict['keyword']) > 1:
            sim_score = TFE.text_similarity(name_cat, content_dict['keyword'])
            if sim_score > 0.8:
                st.success('잘 어울리는 태그에요~😉')
            elif sim_score > 0.5:
                st.warning('조금 더 어울리는 태그를 사용해보세요~ :slightly_smiling_face:')
            else:
                st.error('태그가 적합하지 않아요 :disappointed_relieved:')

        sentences = name_cat + content_dict['keyword']

        content_dict['price'] = st.number_input('가격', min_value=100, max_value=100*10000,
                                                value=10000, step=1000, help = '최소 100원 이상으로 설정해주세요')
        
        with st.expander("옵션 설정"):
            content_dict['content_delivery_fee'] = st.checkbox('배송비 포함 여부', value = False)
            # option    
            c_used = st.radio('중고여부', ['중고', '새 상품'])
            if c_used == '중고':
                content_dict['content_used'] = 1
            else:
                content_dict['content_used'] = 0
            content_dict['qty'] = st.number_input('수량', min_value=1, max_value=100, value=1, step=1)
            # content_dict['tradable']
    
            content_dict['content_b_pay'] = st.checkbox('B-Pay 사용', value =True)

        content_dict['description'] = st.text_area('상품 상세 설명', placeholder = '상품 설명을 입력해주세요 : ')    

        if st.button('미리보기'):
            input_features = {'content_cat_1': 3, 
                  'content_price': content_dict['price'],
                  'content_b_pay': content_dict['content_b_pay']*1,
                  'content_delivery_fee': content_dict['content_delivery_fee'] * 1,
                  'content_used': content_dict['content_used'], 
                  }
            text_vec = TFE.text2vec(sentences)
            image_vec = IFE.image2vec(img)

            CTR = predictor.predict_ctr(input_features, image_vec, text_vec)
            avg_ctr = CTR.mean()
            # st.write(avg_ctr)


            col1, col2, col3 = st.columns(3)
        
#             col1.subheader('썸네일 미리보기')
            col1.image(img)
            col1.write(c_name)
            col1.markdown(f"** {content_dict['price']}원 **")

            median_ctr_cat = df_cat[(df_cat['content_cat_1'] == cat_1) & (df_cat['content_cat_2'] == cat_2)].ctr_median.values[0]
            cat_ratio = (avg_ctr - median_ctr_cat)/median_ctr_cat
            col2.metric('', '', '')
            col2.metric('동일 카테고리 대비', cat_2, f'{round(100*cat_ratio, 2)}%')

            if content_dict['price'] < 10*10000:
                price10 = content_dict['price'] // 10000
            else:
                price10 = (content_dict['price'] // 100000)*10
            median_ctr_price = df_price[df_price.price == price10].ctr_median.values[0]
            price_ratio = (avg_ctr - median_ctr_price)/median_ctr_price
            col3.metric('', '', '')
            col3.metric('비슷한 금액의 상품 대비', f'{price10}만원대', f'{round(100*price_ratio, 2)}%')

if __name__ == '__main__':
    main()