import pandas as pd
import numpy as np
import seaborn as sns
import scipy.cluster.hierarchy as clusterizacao_hierarquica
import scipy.stats as estatistica
from scipy.stats import zscore
from scipy.spatial.distance import pdist
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
import plotly.express as px 
import plotly.io as pio
import matplotlib.pyplot as plt

pio.renderers.default='browser'

def aplica_zscore(dados):
    return dados.apply(zscore, ddof=1)

def remove_coluna(dados, colunas:list):
    return dados.drop(columns=colunas)

def calcula_distancias(dados, metrica:str = 'euclidean'):
    return pdist(dados, metric=metrica)

def gera_modelo_clusterizacao(numero_de_clsuter:int, metrica:str='euclidean', metodo_agregacao:str='single'):
    return AgglomerativeClustering(n_clusters=numero_de_clsuter,
                                    affinity=metrica,
                                   linkage=metodo_agregacao)

def gera_dendograma(dados, rotulos, metodo, metrica):
    plt.figure(figsize=(16,8))
    dados_dendograma = clusterizacao_hierarquica.linkage(dados, method = metodo, metric = metrica)
    clusterizacao_hierarquica.dendrogram(dados_dendograma,  labels = rotulos)
    plt.title('Dendrograma', fontsize=16)
    plt.ylabel('Distância Euclidiana', fontsize=16)
    plt.show()

PATH = 'Crop_recommendation.csv'
LABEL = 'label'
N_CLUSTERS = 3
METRICA = 'euclidean'
# Opções para as distâncias ("metric"):
    ## euclidean
    ## sqeuclidean
    ## cityblock
    ## chebyshev
    ## canberra
    ## correlation

METODO = 'average'
# Opções para o método de encadeamento ("method"):
    ## single
    ## complete
    ## average

dados_brutos = pd.read_csv(PATH)

dados = dados_brutos[dados_brutos['label']=='rice']
dados_metricas = remove_coluna(dados, [LABEL])
dados_metricas_padronizado = aplica_zscore(dados_metricas)
distancia = calcula_distancias(dados_metricas, metrica=METRICA)
modelo = gera_modelo_clusterizacao(N_CLUSTERS, METRICA, METODO)
classificacao_cluster = modelo.fit_predict(dados_metricas_padronizado)
dados['Cluster_single'] = classificacao_cluster

gera_dendograma(dados_metricas_padronizado, list(dados.index), METODO, METRICA)

print(dados)