import pandas as pd
import numpy as np
import pickle

import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from sklearn.datasets import make_classification


from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


from sklearn.metrics import RocCurveDisplay, auc

from collections import Counter

import warnings
warnings.filterwarnings('ignore')



class KNN:
    def __init__(self, k, metric):
        self.k = k
        self.metric = metric
    def fit(self,X,y):
        self.X_train = X
        self.y_train = y
        
        
    def euclidean_distance(self, x1,x2):
        return np.sqrt(np.sum((x1-x2)**2))

    def cosine_distance(self, x1,x2):
        return 1 - np.dot(x1,x2)/(np.linalg.norm(x1)*np.linalg.norm(x2))


    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions


    def _predict(self, x):
        if self.metric == 'euclidean':
            distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]
        elif self.metric == 'cosine':
            distances = [self.cosine_distance(x, x_train) for x_train in self.X_train]
        else:
            raise ValueError("Métrica inválida. Escolha 'euclidean' ou 'cosine'.")
        
        # Gets the indices of the k closest points.
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        most_common = Counter(k_nearest_labels).most_common()
        return most_common[0][0]



def load_data(file_path="data/mini_gm_public_v0.1.p"):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data

def structure_data(data):
    embeddings, syndromes, subject_ids, image_ids = [], [], [], []

    for syndrome_id, subjects in data.items():
        for subject_id, images in subjects.items():
            for image_id, embedding in images.items():
                embeddings.append(embedding)
                syndromes.append(syndrome_id)
                subject_ids.append(subject_id)
                image_ids.append(image_id)

    embeddings = np.array(embeddings)
    syndrome_df = pd.DataFrame({"syndrome": syndromes})
    embedding_df = pd.DataFrame(
        embeddings,
        columns=[f"emb_{i}" for i in range(embeddings.shape[1])]
    )

    return pd.concat([syndrome_df, embedding_df], axis=1)

def kk_for_knn(X_train, X_test, y_train, y_test):
    cosine_values = {}
    euclidean_values = {}

    for m in ['euclidean', 'cosine']:
        for i in range(1,16):
            
            

            clf = KNN(k=i, metric=m)
            clf.fit(X_train, y_train)
            predictions = clf.predict(X_test)

            acc = np.sum(predictions == y_test) / len(y_test)


            if m == 'euclidean':
                euclidean_values[i] = acc
            elif m == 'cosine':
                cosine_values[i] = acc

    idx_euc, acc_euc = max(euclidean_values.items(), key=lambda x: x[1])
    idx_cos, acc_cos = max(cosine_values.items(), key=lambda x: x[1])

    return idx_euc, idx_cos




def main():
    
    data = load_data()
    df = structure_data(data)
    
    X = df.drop("syndrome", axis=1).values
    y = df["syndrome"].values

    tsne = TSNE(n_components=2, perplexity=97, learning_rate=570, max_iter=3000)
    X_tsne = tsne.fit_transform(X)
    
    #Normalize with int values
    index_arr = {syndrome: idx for idx, syndrome in enumerate(np.unique(y))}
    y = np.array([index_arr.get(val, -1) for val in y], dtype=int)
    
    X_train, X_test, y_train, y_test = train_test_split(X_tsne, y, test_size=0.2)
    k_euclidian, acc_euclidian, k_cosine, acc_cosine = kk_for_knn(X_train, X_test, y_train, y_test)
    
    print(k_euclidian, acc_euclidian)    
    print(k_cosine, acc_cosine)    


if __name__ == "__main__":
    main()
        
    