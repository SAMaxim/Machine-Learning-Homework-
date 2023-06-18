#!/usr/bin/env python
# coding: utf-8

# In[12]:


import math
import pickle
import random
from typing import List, Tuple

import numpy as np
import torch
from catboost.datasets import msrank_10k
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from tqdm.auto import tqdm


# In[13]:


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# In[14]:


def dcg(ys_true: torch.FloatTensor, ys_pred: torch.FloatTensor, gain_scheme: str ="exp2", k: int = None) -> float:
    k = k or ys_true.size(dim=0)
    if k > ys_true.size(dim=0):
        k = ys_true.size(dim=0)
    idx = torch.argsort(ys_pred, descending=True)
    true_sorted = ys_true[idx].to(torch.float64)
    steps = torch.arange(2, k + 2, dtype=torch.float64)
    steps = torch.log2(steps)
    gains = true_sorted.apply_(lambda x: compute_gain(x, gain_scheme))[0:k]
    return float(torch.sum(gains / steps))

def ndcg(ys_true: torch.FloatTensor, ys_pred: torch.FloatTensor, gain_scheme: str = 'exp2', k: int = None) -> float:
    k = k or ys_true.size(dim=0)
    if k > ys_true.size(dim=0):
        k =  ys_true.size(dim=0)
    dcg_score = dcg(ys_true, ys_pred, gain_scheme, k)
    true_sorted, _ = torch.sort(ys_true, descending=True)
    ideal_dcg = dcg(true_sorted, true_sorted, gain_scheme, k)
    return float(dcg_score / ideal_dcg)


# In[15]:


def compute_gain(y_value: float, gain_scheme: str) -> float:
    if gain_scheme == 'const':
        return float(y_value)
    elif gain_scheme == 'exp2':
        return float(2 ** y_value - 1)
    return float("inf")


# In[16]:


class Solution:
    def __init__(self, n_estimators: int = 100, lr: float = 0.5, ndcg_top_k: int = 10,
                 subsample: float = 0.6, colsample_bytree: float = 0.9,
                 max_depth: int = 5, min_samples_leaf: int = 8):
        self.X_train = None
        self.ys_train = None
        self.X_test = None
        self.ys_test = None
        self._prepare_data()

        self.ndcg_top_k = ndcg_top_k
        self.n_estimators = n_estimators  # количество деревьев
        self.lr = lr  # Learning Rate, коэффициент, на который умножаются предсказания каждого нового дерева
        self.max_depth = max_depth  # максимальная глубина
        self.min_samples_leaf = min_samples_leaf  # минимальное количество термальных листьев

        self.subsample = subsample  # доля объектов от выборки
        self.colsample_bytree = colsample_bytree  # доля признаков от выборки

        self.trees: List[DecisionTreeRegressor] = []  # все деревья
        self.idxs_array = []
        self.all_ndcg: List[float] = []
        self.best_ndcg = float(0.0)

    def _get_data(self) -> List[np.ndarray]:
        train_df, test_df = msrank_10k()

        X_train = train_df.drop([0, 1], axis=1).values
        y_train = train_df[0].values
        query_ids_train = train_df[1].values.astype(int)

        X_test = test_df.drop([0, 1], axis=1).values
        y_test = test_df[0].values
        query_ids_test = test_df[1].values.astype(int)

        return [X_train, y_train, query_ids_train, X_test, y_test, query_ids_test]

    def _prepare_data(self) -> None:
        (X_train, y_train, self.query_ids_train,
         X_test, y_test, self.query_ids_test) = self._get_data()
        self.X_train = torch.FloatTensor(self._scale_features_in_query_groups(X_train, self.query_ids_train))
        self.ys_train = torch.FloatTensor(y_train)
        self.X_test = torch.FloatTensor(self._scale_features_in_query_groups(X_test, self.query_ids_test))
        self.ys_test = torch.FloatTensor(y_test)
        
    def _scale_features_in_query_groups(self, inp_feat_array: np.ndarray,
                                        inp_query_ids: np.ndarray) -> np.ndarray:
        for id in np.unique(inp_query_ids):
            scaler = StandardScaler()
            idxs = inp_query_ids == id
            inp_feat_array[idxs] = scaler.fit_transform(inp_feat_array[idxs])
        return inp_feat_array

    def _train_one_tree(self, cur_tree_idx: int, train_preds: torch.FloatTensor) -> Tuple[DecisionTreeRegressor, np.ndarray]:
        """
        Метод для тренировки одного дерева.
        @cur_tree_idx: номер текущего дерева, который предлагается использовать в качестве random_seed для того,
        чтобы алгоритм был детерминирован.
        @train_preds: суммарные предсказания всех предыдущих деревьев (для расчёта лямбд).
        @return: это само дерево и индексы признаков, на которых обучалось дерево
        """
        # Устанавливаем seed для случайных операций
        set_seed(cur_tree_idx)

        # Создаем словарь с индексами для каждого query_id
        query_indices = {}
        for i, query_id in enumerate(self.query_ids_train):
            if query_id not in query_indices:
                query_indices[query_id] = []
            query_indices[query_id].append(i)

        # Создаем массив лямбда-значений для каждого примера в обучающей выборке
        lambdas = torch.zeros_like(train_preds)

        # Рассчитываем лямбда-значения для каждого блока данных с одинаковым query_id
        for query_id, indices in query_indices.items():
            lambdas_query = self._compute_lambdas(self.ys_train[indices], train_preds[indices])
            lambdas[indices] = lambdas_query.squeeze()

        # Случайным образом выбираем подмножества примеров и признаков для обучения дерева
        samples_count = self.X_train.size(dim=0)
        features_count = self.X_train.size(dim=1)
        samples_indices = torch.full((samples_count,), False)
        feature_indices = torch.full((features_count,), False)
        for i in range(samples_count):
            if np.random.rand() < self.subsample:
                samples_indices[i] = True
        for i in range(features_count):
            if np.random.rand() < self.colsample_bytree:
                feature_indices[i] = True

        # Выбираем подмножество данных и обучаем дерево решений
        sub = self.X_train[samples_indices]
        sub = sub[:, feature_indices]
        lambdas_sub = lambdas[samples_indices]
        dtr = DecisionTreeRegressor(max_depth=self.max_depth, random_state=cur_tree_idx)
        dtr.fit(sub, lambdas_sub)

        # Сохраняем дерево и индексы признаков
        return dtr, torch.where(feature_indices)[0]

    def _calc_data_ndcg(self, queries_list: np.ndarray,
                        true_labels: torch.FloatTensor, preds: torch.FloatTensor) -> float:
       
        score = []
        for id in np.unique(queries_list):
            idxs = queries_list == id
            score.append(ndcg(true_labels[idxs], preds[idxs], gain_scheme="exp2", k=15))
        return np.array(score).mean()

    def fit(self):
        # Устанавливаем seed для воспроизводимости результатов
        set_seed(0)

        # Создаем массивы для предсказанных значений для обучающей и тестовой выборок
        predicted_train = torch.zeros_like(self.ys_train)
        predicted_test = torch.zeros_like(self.ys_test)

        # Обучаем n_estimators деревьев
        for k in tqdm(range(self.n_estimators)):
            # Обучаем одно дерево и сохраняем его
            tree, feature_indices = self._train_one_tree(k, predicted_train)
            self.trees.append(tree)
            self.idxs_array.append(feature_indices)

            # Применяем обученное дерево к обучающей и тестовой выборкам
            prediction_train = tree.predict(self.X_train[:, feature_indices])
            prediction_test = tree.predict(self.X_test[:, feature_indices])

            # Обновляем предсказанные значения для обучающей и тестовой выборок
            predicted_train -= self.lr * prediction_train
            predicted_test -= self.lr * prediction_test

            # Вычисляем NDCG на тестовой выборке и сохраняем его
            ndcg = self._calc_data_ndcg(self.query_ids_test, self.ys_test, predicted_test)
            self.all_ndcg.append(ndcg)

            # Если текущее значение NDCG лучше, чем лучшее значение, сохраняем его
            if ndcg > self.best_ndcg:
                self.best_ndcg = ndcg
    
        # Выбираем лучшее дерево и удаляем оставшиеся
        last = self.all_ndcg.index(self.best_ndcg)
        self.trees = self.trees[0:last+1]
        self.idxs_array = self.idxs_array[0:last+1]

        # Выводим лучший результат NDCG
        print(f'Total NDCG score {self.best_ndcg}')

    def predict(self, data: torch.FloatTensor) -> torch.FloatTensor:
        # Выбираем признаки, используемые каждым деревом
        feature_indices = torch.tensor(self.idxs_array, dtype=torch.long)
        data_subset = torch.index_select(data, dim=1, index=feature_indices)

        # Получаем предсказания на всех деревьях
        tree_preds = torch.stack([dt.predict(data_subset) for dt in self.trees])

        # Вычисляем сумму предсказаний всех деревьев и умножаем на learning rate
        ans = -self.lr * torch.sum(tree_preds, dim=0)

        return ans

    def _compute_lambdas(self, y_true: torch.FloatTensor, y_pred: torch.FloatTensor) -> torch.FloatTensor:
        ndcg_scheme = "exp2"
        ideal_dcg = dcg(y_true, y_true, ndcg_scheme)
        N = 0
        if ideal_dcg != 0:
            N = 1 / ideal_dcg
            
        y_true_temp = y_true.reshape(-1, 1)
        y_pred_temp = y_pred.reshape(-1, 1)
        
        _, rank_order = torch.sort(y_true_temp, descending=True, axis=0)
        rank_order += 1
        with torch.no_grad():
            # получаем все попарные разницы скоров в батче
            pos_pairs_score_diff = 1.0 + torch.exp((y_pred_temp - y_pred_temp.t()))

            # поставим разметку для пар, 1 если первый документ релевантнее
            # -1 если второй документ релевантнее
            Sij = self._compute_labels_in_batch(y_true_temp)
            # посчитаем изменение gain из-за перестановок
            gain_diff = self._compute_gain_diff(y_true_temp, ndcg_scheme)

            # посчитаем изменение знаменателей-дискаунтеров
            decay_diff = (1.0 / torch.log2(rank_order + 1.0)) - (1.0 / torch.log2(rank_order.t() + 1.0))
            # посчитаем непосредственное изменение nDCG
            delta_ndcg = torch.abs(N * gain_diff * decay_diff)
            # посчитаем лямбды
            lambda_update =  (0.5 * (1 - Sij) - 1 / pos_pairs_score_diff) * delta_ndcg
            lambda_update = torch.sum(lambda_update, dim=1, keepdim=True)

            return lambda_update
    
    def _compute_labels_in_batch(self, y_true: torch.FloatTensor):
        # разница релевантностей каждого с каждым объектом
        rel_diff = y_true - y_true.t()

        # 1 в этой матрице - объект более релевантен
        pos_pairs = (rel_diff > 0).type(torch.float32)

        # 1 тут - объект менее релевантен
        neg_pairs = (rel_diff < 0).type(torch.float32)
        Sij = pos_pairs - neg_pairs
        return Sij


    def _compute_gain_diff(self, y_true: torch.FloatTensor, gain_scheme: str):
        if gain_scheme == "exp2":
            gain_diff = torch.pow(2.0, y_true) - torch.pow(2.0, y_true.t())
        elif gain_scheme == "diff":
            gain_diff = y_true - y_true.t()
        else:
            raise ValueError(f"{gain_scheme} method not supported")
        return gain_diff
    
    def _ndcg_k(self, ys_true, ys_pred, ndcg_top_k) -> float:
        try:
            return ndcg(ys_true, ys_pred, gain_scheme='exp2', k=ndcg_top_k)
        except ZeroDivisionError:
            return float(0)

    def save_model(self, path: str):
        pickle.dump(self, open('%s.lmart' % path, "wb"), protocol=2)
                        
    def load_model(self, path: str):
        model = pickle.load(open(path, "rb"))
        self.X_train = model.X_train
        self.ys_train = model.ys_train
        self.X_test = model.X_test
        self.ys_test = model.ys_test
        self.ndcg_top_k = model.ndcg_top_k
        self.n_estimators = model.n_estimators 
        self.lr = model.lr
        self.max_depth = model.max_depth
        self.min_samples_leaf = model.min_samples_leaf 
        self.subsample = model.subsample
        self.colsample_bytree = model.colsample_bytree
        self.trees = model.trees
        self.idxs_array = model.idxs_array
        self.all_ndcg = model.all_ndcg
        self.best_ndcg = model.best_ndcg

