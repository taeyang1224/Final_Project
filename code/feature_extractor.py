
import numpy as np
import pandas as pd
import json
import torch
from torchvision import transforms

import requests
import cv2
from PIL import Image
from img2vec_pytorch import Img2Vec
from skimage.io import imread, imshow
from skimage.util import img_as_ubyte
from skimage.filters.rank import entropy
from skimage.morphology import disk
import matplotlib.pyplot as plt

import re
from konlpy.tag import Kkma, Okt
from gensim.models import FastText

from tqdm.notebook import tqdm
tqdm.pandas()


class ImageFeatExtractor():
    '''
    이미지를 불러와서(번개장터 크롤링) 전처리, 벡터로 변환, 배경 분리를 진행하고
    색상, 노이즈 등 feature를 뽑아내는 클래스
    '''
    def __init__(self):
        self.file_path = 'data/'

    def load_deeplab(self):
        '''
        모델 불러오기(deeplab)
        '''
        self.deeplab = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
        self.deeplab.eval()
    
    def load_img2vec(self):
        '''
        img2vec(resnet) 불러오기
        '''
        self.img2vec = Img2Vec(cuda=True)

    def load_color_dict(self):
        with open(self.file_path + 'color_dict.json', 'r') as f:
            self.color_dict = json.load(f)
        
    def prepare(self):
        print('Preparing Image Feature Extractor')
        self.load_deeplab()
        self.load_img2vec()
        self.load_color_dict()
        print('Complete')
        
    def cid2img(self, cid):
        '''
        번개장터 content_id를 통해 이미지를 크롤링해오는 함수
        '''
        url = f'https://media.bunjang.co.kr/product/{cid}_...'
        image_nparray = np.asarray(bytearray(requests.get(url).content), dtype=np.uint8)
        image = cv2.imdecode(image_nparray, cv2.IMREAD_COLOR)  
        image = Image.fromarray(image)
        image = image.convert("RGB")
        return image

    def image2vec(self, image):
        '''
        이미지를 벡터로 변환하는 함수(resnet)
        '''
        # 전처리(resize 등), 형식 확인해서 맞춰주는 코드 추가 필요

        img_vec = self.img2vec.get_vec(image)

        return img_vec

    def rgb_to_color(self, rgb):
        '''
        rgb값을 가장 가까운 대표색으로 변환하는 함수
        '''
        color_dict = self.color_dict
        color_distance = [np.linalg.norm(np.array(rgb)-col) for col in color_dict.values()]
        return list(color_dict.keys())[np.argmin(color_distance)]

    def get_entropy(self, image_gray):
        '''
        image를 입력받아 entropy(노이즈)를 계산하는 함수
        '''
        scaled_entropy = image_gray / image_gray.max()
        entropy_image = entropy(img_as_ubyte(scaled_entropy), disk(6))
        scaled_entropy = entropy_image / entropy_image.max()
        mask_ent = scaled_entropy > 0.8
        return mask_ent

    def get_mask(self, image):
        '''
        image에서 배경, 상품 분리하는 함수(DeepLab v3+)
        '''
        # input이 array인 경우 PIL image로 변환
        if isinstance(image, np.ndarray):
            input_image = Image.fromarray(image)
        else:
            input_image = image
        input_image = input_image.convert("RGB")
        
        # 이미지 전처리
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # input image를 tensor로
        input_tensor = preprocess(input_image)
        # 미니 배치 생성
        input_batch = input_tensor.unsqueeze(0) 

        # GPU 사용
        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
            self.deeplab.to('cuda')
        
        # DeepLab으로 mask 예측
        with torch.no_grad():
            output = self.deeplab(input_batch)['out'][0]
        output_predictions = output.argmax(0)

        # mask를 tensor에서 numpy array로 변환
        try: 
            mask = output_predictions.cpu().numpy()
        except:
            mask = output_predictions.numpy()

        # 1보다 크면 1, 0이면 0
        mask = np.where(mask>0, 1, 0)
        return mask

    def image_features(self, image, mask):
        '''
        이미지, deeplab으로 얻은 mask를 통해 feature들을 추출하는 함수
        '''
        # mask = self.get_mask(image)
        # 배경 mask를 통해 상품:배경 비율 계산
        background_ratio = ((1-mask).sum())/(image.shape[0]*image.shape[1])
        # 배경, 상품의 rgb 값을 array로 표현(각 마스크에 해당하지 않는 부분을 버림)
        bg_color = [np.delete(image[:,:,i], (mask.reshape(-1)==1)) for i in range(3)]
        nobg_color = [np.delete(image[:,:,i], (mask.reshape(-1)==0)) for i in range(3)]

        # 배경 색상 추출 (중앙값)
        rgb_bg = np.array([int(np.median(temp)) for temp in bg_color])
        color_bg = self.rgb_to_color(rgb_bg)
        # 상품 색상 추출 (중앙값)
        rgb_nobg = np.array([int(np.median(temp)) for temp in nobg_color])
        color_nobg = self.rgb_to_color(rgb_nobg)

        # 노이즈 계산
        # 이미지를 gray scale로 변환
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 배경 노이즈 계산
        bg_gray = image_gray * (1-mask)
        bg_ent = self.get_entropy(bg_gray)
        bg_noise = bg_ent.sum() / (1-mask).sum()
        # 상품 노이즈 계산
        nobg_gray = image_gray * mask
        nobg_ent = self.get_entropy(nobg_gray)
        nobg_noise = nobg_ent.sum() / mask.sum()

        # HSV -> 사람 구분
        upper_human = (20, 150, 200)
        lower_human = (0, 50, 60)
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        human_mask = cv2.inRange(image_hsv, lower_human, upper_human)
        human_ratio = np.count_nonzero(mask * human_mask) / mask.sum()

        # 결과 저장
        image_feature_dict = {'background_ratio': background_ratio,
        'bg_rgb': rgb_bg, "nobg_rgb": rgb_nobg, 
        "bg_color": color_bg, "nobg_color": color_nobg, 
        "bg_noise": bg_noise, "nobg_noise": nobg_noise,
        "human_ratio": human_ratio}

        return image_feature_dict

    def visualize_mask(self, image, how='both'):
        '''
        이미지를 배경, 상품으로 분리해서 시각화하는 함수
        '''
        mask = self.get_mask(image)
        # 이미지 배경, 상품 분리
        image_nobg = image * mask[:, :, np.newaxis]
        image_bg = image * (1-mask)[:, :, np.newaxis]
        # how = both 이면 배경, 상품 둘 다 시각화
        if how.lower() == 'both':
            f, ax = plt.subplots(1, 2, figsize=(10,10))
            ax[0].imshow(image_nobg)
            ax[1].imshow(image_bg)
            ax[0].axis('off') # 축 지우기
            ax[1].axis('off')
        # 배경만 시각화
        elif how.lower() == 'background':
            plt.imshow(image_bg)
            plt.axis('off')
        # 상품만 시각화
        elif how.lower() == 'content':
            plt.imshow(image_nobg)
            plt.axis('off')


class TextFeatExtractor():
    def __init__(self):
        self.file_path = 'data/'
        
    def load_model(self):
        self.model_ft = FastText.load('model/ft.model')
        self.kkma = Kkma()

    def load_brand_dict(self):
        with open(self.file_path + 'brand_dict.json', 'r') as f:
            self.brand_dict = json.load(f)

    def load_cat_dict(self):
        with open(self.file_path + 'cat_3_dict.json', 'r') as f:
            self.cat_dict = json.load(f)

    def brand_(self, name, brand_dict):
        for key in self.brand_dict.keys():
            for value in self.brand_dict[key]:
                if value in name.split(' '):
                    name = name.replace(value, key)
        return name

    def re_kor_brand(self, s):
        s = self.brand_(s, self.brand_dict)
        pre = re.compile('[^ 가-힣]')
        result = pre.sub(' ', s) 
        while '  ' in result:
            result = result.replace('  ', ' ')
        result = result.split(' ')
        result = [r for r in result if r!='']
        return result

    def get_voca_cnt(self, voca_list):
        total_vocabs = []
        for t in voca_list:
            total_vocabs.extend(t)
        total_unique, total_count = np.unique(total_vocabs, return_counts=True)
        vocab_cnt = pd.DataFrame({'name': total_unique, 'cnt': total_count})
        return vocab_cnt.sort_values('cnt', ascending=False)

    def re_kkma_voca(self, sentence, total_vocabs):
        voca_list1 = kkma.nouns(sentence)
        voca_list2 = re_kor_brand(sentence)
        voca_list = voca_list1 + voca_list2
        return [voca for voca in voca_list if voca in total_vocabs.name.tolist()]

    def tokenize_name(self, name):
        tok_name = self.re_kor_brand(name)
        return tok_name

    def tokenize_keyword(self, keyword):
        tok_keyword = keyword.split(',')
        return tok_keyword
    
    def tokenize_category(self, category):
        cat1 = [cat.replace('기타', 'etc').replace(' ', '') for cat in category.content_cat_1.split('/')]
        cat2 = category.content_cat_2.apply(lambda x: x.replace('/', ' ')).apply(lambda x: cat_dict[x] if x in cat_dict.keys() else x).apply(lambda x: x.split(' '))
        cat3 = category.content_cat_3.apply(lambda x: x.replace('/', ' ')).apply(lambda x: cat_dict[x] if x in cat_dict.keys() else x).apply(lambda x: x.split(' '))
        return cat1 + cat2 + cat3

    def tokenize_decription(self, description):        
        return self.kkma.nouns(description)

    def get_total_voca(self):
        name_cnt = get_voca_cnt(name_list)
        keyword_cnt = get_voca_cnt(keyword_list)
        total_vocab = pd.merge(name_cnt, keyword_cnt,how='outer', on='name').fillna(0)
        total_vocab['cnt'] = (total_vocab['cnt_x'] + total_vocab['cnt_y']).astype(int)
        total_vocab = total_vocab[['name', 'cnt']]
        return total_vocab
        
    # 텍스트
    def text2vec(self, text):
        total_vec = np.zeros(100,)
        cnt = 0
        for t in text:
            try:
                total_vec += self.model_ft.wv.word_vec(t)
                cnt += 1
            except:
                pass
        total_vec /= cnt
        return list(total_vec)

    def text_similarity(self, T1, T2):
        sim = self.model_ft.wv.n_similarity(T1, T2)
        return sim