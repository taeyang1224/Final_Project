from tqdm import tqdm
import numpy as np
import pandas as pd
import requests
import os
import urllib.request
import time

dy = [310, 320, 400, 405, 410, 420, 500]
ty = [600, 700, 750, 800, 810, 900]
mh = [910, 920, 930, 980, 990, 999]

cat_list = dy

total_info = []
for cat_id in cat_list:
    infos = []
    for page_num in tqdm(range(1,300)):
        url = f"https://api.bunjang.co.kr/api/1/find_v2.json?f_category_id={cat_id}&page={page_num}&order=date&req_ref=category&stat_device=w&n=100&version=4"
        response = requests.get(url)
        ids = [item['pid'] for item in response.json()['list']]
        time.sleep(1 + 2*np.random.rand(1))
        for pid in ids:
            time.sleep(np.random.rand(1))
            try: 
                url = f'https://api.bunjang.co.kr/api/1/product/{pid}/detail_info.json?version=4'
                res = requests.get(url)
                item_info = res.json()['item_info']
                item_info.pop('description')
                item_info.pop('description_for_detail')
                seller_info = res.json()['seller_info']
                item_info.update(seller_info)
                infos.append(item_info)
            except:
                pass
    pd.DataFrame(infos).to_csv(f'{cat_id}.csv', index=False)
    total_info.append(infos)
   