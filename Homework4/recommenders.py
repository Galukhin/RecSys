import pandas as pd
import numpy as np

# Для работы с матрицами
from scipy.sparse import csr_matrix

# Матричная факторизация
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender  # нужен для одного трюка
from implicit.nearest_neighbours import bm25_weight, tfidf_weight

import os, sys

module_path = os.path.abspath(os.path.join(os.pardir))
if module_path not in sys.path:
    sys.path.append(module_path)

from utils import prefilter_items


class MainRecommender:
    """Рекоммендации, которые можно получить из ALS
    
    Input
    -----
    user_item_matrix: pd.DataFrame
        Матрица взаимодействий user-item
    """
    
    def __init__(self, data, weighting=True):
        
        # your_code. Это не обязательная часть. Но если вам удобно что-либо посчитать тут - можно это сделать
        
        self.user_item_matrix = self.prepare_matrix(data)  # pd.DataFrame
        self.id_to_itemid, self.id_to_userid, self.itemid_to_id, self.userid_to_id = self.prepare_dicts(self.user_item_matrix)
        
        if weighting:
            self.user_item_matrix = bm25_weight(self.user_item_matrix.T).T 
        
        self.model = self.fit(self.user_item_matrix)
        self.own_recommender = self.fit_own_recommender(self.user_item_matrix)
     
    @staticmethod
    def prepare_matrix(data):
        
        # your_code
        
        data = prefilter_items(data, take_n_popular=5000)
        
        user_item_matrix = pd.pivot_table(data, 
                                          index='user_id', columns='item_id', 
                                          values='quantity', # Можно пробовать другие варианты
                                          aggfunc='count', 
                                          fill_value=0
                                         )

        user_item_matrix = user_item_matrix.astype(float) # необходимый тип матрицы для implicit
        
        return user_item_matrix
    
    @staticmethod
    def prepare_dicts(user_item_matrix):
        """Подготавливает вспомогательные словари"""
        
        userids = user_item_matrix.index.values
        itemids = user_item_matrix.columns.values

        matrix_userids = np.arange(len(userids))
        matrix_itemids = np.arange(len(itemids))

        id_to_itemid = dict(zip(matrix_itemids, itemids))
        id_to_userid = dict(zip(matrix_userids, userids))

        itemid_to_id = dict(zip(itemids, matrix_itemids))
        userid_to_id = dict(zip(userids, matrix_userids))
        
        return id_to_itemid, id_to_userid, itemid_to_id, userid_to_id
     
    @staticmethod
    def fit_own_recommender(user_item_matrix):
        """Обучает модель, которая рекомендует товары, среди товаров, купленных юзером"""
    
        own_recommender = ItemItemRecommender(K=1, num_threads=4)
        own_recommender.fit(csr_matrix(user_item_matrix).T.tocsr())
        
        return own_recommender
    
    
    @staticmethod
    def fit(user_item_matrix, n_factors=20, regularization=0.001, iterations=15, num_threads=4):
        """Обучает ALS"""
        
        model = AlternatingLeastSquares(factors=n_factors, 
                                             regularization=regularization,
                                             iterations=iterations,  
                                             num_threads=num_threads,
                                       random_state=42)
        model.fit(csr_matrix(user_item_matrix).T.tocsr())
        
        return model

    def get_similar_items_recommendation(self, user, N=5):
        """Рекомендуем товары, похожие на топ-N купленных юзером товаров"""

        # your_code
        # Практически полностью реализовали на прошлом вебинаре
        
        # Получим id всех купленных юзером товаров по user_item_matrix. Можно получить по data_train, но user_item_matrix
        # у нас в атрибутах объекта
        selected_items = pd.DataFrame(self.user_item_matrix.toarray())
        selected_items = selected_items.loc[selected_items.index == self.userid_to_id[user]]
        selected_items = selected_items.loc[:, (selected_items != 0).any(axis=0)].columns.tolist()
        
        # Получим топ-N купленных юзером товаров
        top_n_items = self.model.rank_items(userid=self.userid_to_id[user],
                                            user_items=csr_matrix(self.user_item_matrix).tocsr(),
                                            selected_items=selected_items,
                                            recalculate_user=False)[:N]

        # N товаров, похожих на топ-N товаров, купленных юзером, найдем следующим образом:
        # для каждого товара найдем один самый похожий товар и порекомендуем его.
        # Это можно сделать и другим способом, например для каждого товара найти топ-N похожих и отранжировать их все 
        # относительно данного юзера.
        similar_items = [self.id_to_itemid[self.model.similar_items(itm, N=2)[1][0]] for itm, score in top_n_items]

        assert len(similar_items) == N, 'Количество рекомендаций != {}'.format(N)
        return similar_items
    
    def get_similar_users_recommendation(self, user, N=5):
        """Рекомендуем топ-N товаров, среди купленных похожими юзерами"""
    
        # your_code
        # Найдем топ-N юзеров, похожих на данного юзера
        similar_users = [u for u, s in self.model.similar_users(self.userid_to_id[user], N=N+1) if u != self.userid_to_id[user]]
        
        # От каждого похожего юзера возьмем товар с наивысшим скором и порекоммендуем его.
        # Как и в прошлой функции, эту функцию можно реализовать по разному.
        top_items = []
        for u in similar_users:
            selected_items = pd.DataFrame(self.user_item_matrix.toarray())
            selected_items = selected_items.loc[selected_items.index == u]
            selected_items = selected_items.loc[:, (selected_items != 0).any(axis=0)].columns.tolist()
            
            top_1_item = self.model.rank_items(userid=u,
                                                user_items=csr_matrix(self.user_item_matrix).tocsr(),
                                                selected_items=selected_items,
                                                recalculate_user=False)[0][0]
            
            top_items.append(self.id_to_itemid[top_1_item])


        assert len(top_items) == N, 'Количество рекомендаций != {}'.format(N)
        return top_items