"""
Metrics

"""
import numpy as np
import pandas as pd

def prefilter_items(data_train, item_features=None, take_n_popular=5000):
#     # Уберем самые популярные товары (их и так купят)
#     popularity = data_train.groupby('item_id')['user_id'].nunique().reset_index() / data_train['user_id'].nunique()
#     popularity.rename(columns={'user_id': 'share_unique_users'}, inplace=True)
    
#     top_popular = popularity[popularity['share_unique_users'] > 0.5].item_id.tolist()
#     data = data[~data['item_id'].isin(top_popular)]
    
#     # Уберем самые НЕ популярные товары (их и так НЕ купят)
#     top_notpopular = popularity[popularity['share_unique_users'] < 0.01].item_id.tolist()
#     data = data[~data['item_id'].isin(top_notpopular)]
    
    data_train = data_train.copy()
    
    popularity = data_train.groupby('item_id')['quantity'].sum().reset_index()
    popularity.rename(columns={'quantity': 'n_sold'}, inplace=True)

    top_5000 = popularity.sort_values('n_sold', ascending=False).head(5000).item_id.tolist()
    
    data_train.loc[~data_train['item_id'].isin(top_5000), 'item_id'] = 999999
    
    # Уберем товары, которые не продавались за последние 12 месяцев
    
    # Уберем не интересные для рекоммендаций категории (department)
    
    # Уберем слишком дешевые товары (на них не заработаем). 1 покупка из рассылок стоит 60 руб. 
    
    # Уберем слишком дорогие товарыs
    
    # ...
    
    return data_train
    