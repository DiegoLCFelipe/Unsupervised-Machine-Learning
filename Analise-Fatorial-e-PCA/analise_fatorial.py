import pandas as pd
import numpy as np
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
import plotly.graph_objects as go

def ler_excel(caminho):
    return pd.read_excel(caminho)
# incluir função para selecionar apenas as variáveis métricas
def gera_matriz_de_correlacoes(dados):
    return dados.corr()

def aplica_teste_de_esfericidade_de_bartlett(dados, significancia):
    bartlett, p_valor = calculate_bartlett_sphericity(dados)
    print(f'Qui² Bartlett: {round(bartlett,2)}\n'
          f'p-valor: {round(p_valor, 4)}\n')
    if p_valor < significancia:
        print(f'Rejeita-se a hipótese nula de que a matriz de correlações é igual a matriz identidade,\n'
              f' logo a análise fatorial pode ser aplicada')
    else:
        print(f'Aceita-se a hipótese nula de que a matriz de correlações é igual a matriz identidade,\n'
              f' logo a análise fatorial não pode ser aplicada')
        
def analisa_os_fatores(dados, numero_de_fatores):
    return FactorAnalyzer(n_factors=numero_de_fatores, method='principal', rotation=None).fit(dados)

def calcula_autovalores(dados, numero_de_fatores):
    analise_fatorial = analisa_os_fatores(dados, numero_de_fatores)
    return analise_fatorial.get_factor_variance()

def calcula_cargas_fatoriais(dados, numero_de_fatores):
    analise_fatorial = analisa_os_fatores(dados, numero_de_fatores)
    return analise_fatorial.loadings_

def calcula_comunalidades(dados, numero_de_fatores):
    analise_fatorial = analisa_os_fatores(dados, numero_de_fatores)
    return analise_fatorial.get_communalities()

def extrai_fatores_para_as_observacoes(dados, numero_de_fatores):
    fatores = pd.DataFrame(analisa_os_fatores(dados, numero_de_fatores).transform(dados))
    fatores.columns =  [f"Fator {i+1}" for i, v in enumerate(fatores.columns)]
    return fatores

def aplicar_criterio_de_kaiser(dados, numero_de_fatores):
    return len([i for i in calcula_autovalores(dados, numero_de_fatores)[0] if i>=1])

def mostra_tabela_de_autovalores(dados, numero_de_fatores):
    autovalores_dos_fatores = calcula_autovalores(dados, numero_de_fatores)
    tabela_autovalores = pd.DataFrame(autovalores_dos_fatores)
    tabela_autovalores.columns = [f"Fator {i+1}" for i, v in enumerate(tabela_autovalores.columns)]
    tabela_autovalores.index = ['Autovalor','Variância', 'Variância Acumulada']
    tabela_autovalores = tabela_autovalores.T
    print(tabela_autovalores)

def mostra_tabela_de_cargas(dados, numero_de_fatores):
    cargas_fatoriais = pd.DataFrame(calcula_cargas_fatoriais(dados, numero_de_fatores))
    cargas_fatoriais.columns = [f"Fator {i+1}" for i, v in enumerate(cargas_fatoriais.columns)]
    cargas_fatoriais.index = dados.columns
    print(cargas_fatoriais)

def mostra_tabela_de_comunalidades(dados, numero_de_fatores):
    comunalidades = pd.DataFrame(calcula_comunalidades(dados, numero_de_fatores))
    comunalidades.columns = ['Comunalidades']
    comunalidades.index = dados.columns
    print(comunalidades)

def gera_mapa_de_calor(dados):
    fig = go.Figure(
    go.Heatmap(
        x=dados.index,
        y=dados.index,
        z=dados,
        text=dados.values,
        colorscale='Viridis'))
    
    fig.update_layout(
        title='Matriz de Correlações',
        xaxis_nticks=36)
    fig.show()

notas = ler_excel('Analise-Fatorial-e-PCA/notas_fatorial.xlsx')
notas_pca = notas[['finanças', 'custos', 'marketing', 'atuária']]
correlacoes = gera_matriz_de_correlacoes(notas_pca)
SIGNIFICANCIA = 0.05
#gera_mapa_de_calor(correlacoes)   
#aplica_teste_de_esfericidade_de_bartlett(notas_pca, 0.05)
#mostra_tabela_de_autovalores(notas_pca, 4)
#mostra_tabela_de_cargas(notas_pca, 4)
#mostra_tabela_de_comunalidades(notas_pca, 4)
#print(extrai_fatores_para_as_observacoes(notas_pca, 4))
#print(notas)
aplica_teste_de_esfericidade_de_bartlett(notas_pca, 0.05)
numero_de_criterios = aplicar_criterio_de_kaiser(notas_pca, 4)

