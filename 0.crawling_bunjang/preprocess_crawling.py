import pandas as pd

rename_dict = {'pid': 'content_id', 
               'uid': 'adv_id', 
               'name': 'content_name',
               'status': 'content_status',
               'price': 'content_price',
               'num_faved': 'content_likes',
               'num_item_view': 'content_views',
               'num_comment': 'content_comment_count', 
               'user_name': 'adv_name',
               'free_shipping': 'content_delivery_fee',
               'used_code': 'content_used',
               'location': 'content_place',
               'need_induce_bun_pay_filter': 'content_b_pay',
               'num_item': 'adv_item_count',
               'num_grade_avg': 'adv_grade',
               'num_follower': 'adv_follower_count',
               'num_review': 'adv_review_count',
           }

drop_list = ['contact_enabled', 'naver_shopping_url', 'bizseller', 'groups', 'is_withdraw',
            'category_id', 'ordernow_token_required', 'only_neighborhood', 'warning', 'is_identification',
            'is_adult', 'style', 'group_ids', 'ordernow_label', 'checkout', 'is_free_sharing',
            'is_blocked', 'extended_spec', 'contact_hope', 'used', 'comment_enabled',
            'restriction_status', 'is_buncar', 'badges', 'category_name', 'product_image', 'pay_option']

def preprocess_crawling(df_):
    df = df_.copy()
    df = df.rename(rename_dict, axis=1)
    df = df.reset_index().drop('index', axis=1)

    # 대분류
    df['content_cat_1'] = df['category_name'].apply(lambda x : x.split("'")[3])
    # 중분류
    df['content_cat_2'] = df['category_name'].apply(lambda x : x.split("'")[7] if len(x.split("'"))>6 else x.split("'")[3])
    # 소분류
    df['content_cat_3'] = df['category_name'].apply(lambda x : x.split("'")[-2])
    # pay option에서 필요한 부분만 사용
    df['pay_option_in_person'] = df['pay_option'].apply(lambda x : x.split("'")[4][2:-2])
    df['pay_option_bun_pay_filter_enabled'] = df['pay_option'].apply(lambda x : x.split("'")[6][2:-2])
    # unix time을 실제 시간으로
    df['update_time'] = pd.to_datetime(df['update_time'], unit='s')
    df['access_time'] = pd.to_datetime(df['access_time'], unit='s')
    df['join_date'] = pd.to_datetime(df['join_date'], unit='s')

    df['seller_name'] = df['badges'].apply(lambda x : x[94:97])

    df = df.drop(drop_list, axis=1)
    return df

df = pd.read_csv('data/new_total_data.csv')
df = preprocess_crawling(df)

sort_col = ['content_id', 'content_name', 'adv_id', 'adv_name', 'keyword', 'content_status',
       'content_price', 'content_used', 'content_likes', 'content_views',
       'content_comment_count', 'content_delivery_fee',
       'content_cat_1', 'content_cat_2', 'content_cat_3',
       'content_b_pay', 
       'adv_item_count', 'adv_grade', 'adv_follower_count', 'adv_review_count',
       'tradable', 'qty', 
       'content_place', 'latitude', 'longitude', 'is_location_confirm', 'address_id',
       'image_count', 'image_file_list_for_edit', 'image_source', 'profile_image',
       'neighborhood_option', 'is_ad', 'bunpay', 'pay_option_in_person', 'pay_option_bun_pay_filter_enabled', 
       'update_time', 'access_time', 'join_date', 
       'seller_name']

df = df[sort_col]