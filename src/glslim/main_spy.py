#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess
import pandas as pd
import numpy as np

caminho_projeto = "/home/bob/Documents/mestrado/sist_recomendacao/tp_final"
dir_dados = caminho_projeto + "/data"
dir_cluto_dados = caminho_projeto + "/data/cluto"
dir_cluto = caminho_projeto + "tools/cluto"
dir_matriz_cluto = "/run/media/bob/Pablo/ufmg/mestrado/sistemas_recomendacao/tp_final/arquivos_netflix/matriz_cluto_netflix.txt.teste"
slim = caminho_projeto + "tools/slim/build/examples"
cluto = caminho_projeto + dir_cluto + "Linux/vcluster"

#parametros
num_clusters = 5
iteracao = 1
nome_arq_cluto = "netflix.clustering." + str(num_clusters) + "." + str(iteracao)

### inicio_algoritmo *************
clusters_cluto = pd.read_csv(dir_cluto_dados + "/" + nome_arq_cluto, header=None)

numero_usuarios = clusters_cluto.size

qu = 0.5 * np.ones((1, numero_usuarios))

### cria matriz de avaliação binária S
import re

arq_matriz = open(dir_matriz_cluto, 'r')
linha = arq_matriz.readline()
match = re.match('(\d+) (\d+) (\d+)', linha)
num_usuarios = int(match.group(1))
num_itens = int(match.group(2))
num_avaliacoes = int(match.group(3))

S = np.zeros((num_usuarios, num_itens))
S_bin = np.zeros((num_usuarios, num_itens))

for id_usuario in range(num_usuarios):
    linha = arq_matriz.readline()
    matches = re.findall(r'(\d+\s\d+)', linha)
    for match in matches:
        match_col_av = re.match('(\d+)\s(\d+)', match)
        coluna = int(match_col_av.group(1))
        avaliacao = int(match_col_av.group(2))
        # no arquivo colunas 1-coluna_max, em S, 0-coluna_max-1
        S[id_usuario, coluna - 1] = avaliacao
        S_bin[id_usuario, coluna - 1] = 1    

arq_matriz.close()

### Cria matriz Su

Su_cluster = [None for i in range(num_clusters)]
cluster = 0

for cluster in range(num_clusters):
    Su_cluster[cluster] = np.copy(S_bin)
    linhas_para_zerar = ~clusters_cluto.isin([cluster])
    contador_linha = 0
    for linha in linhas_para_zerar[0]:
        if linha:
            Su_cluster[cluster][contador_linha,:] = Su_cluster[cluster][contador_linha,:] * 0
        contador_linha = contador_linha + 1
    escreve_matriz_em_formato_csr(Su_cluster[cluster], cluster)

for cluster in range(num_clusters):
    subprocess.call([slim, "-train_file=%s"%(get_nome_matriz_su(cluster)), 
                     "-model_file=%s"%(), "-starti=%d"%(), "-endi=%d"%(),
                     "-lambda=%d"%(), "-beta=%d"%(), "-optTol=%d"%(), ])
subprocess.call(cluto)

###./slim_learn -topn=10 -train_file=../../examples
### [bob@localhost tp_final]$                                                  │/train.mat -test_file=../../examples/test.mat -model_file=modelo_teste_1


#### 
def get_nome_matriz_su(cluster):
    return caminho_m_su + "/su." + str(cluster)

def escreve_matriz_em_formato_csr(matriz, cluster):
   caminho_m_su = dir_dados + "/matriz_su"       
   arquivo = open(caminho_m_su + "/su." + str(cluster), 'w') 
   
   for u in range(num_usuarios):
       for i in range(num_itens):
           if matriz[u,i] == 1:
               arquivo.write("%d %d"%(i,1))
       arquivo.write("\n")        
   arquivo.close()
