import numpy as np
import subprocess
import threading
import re

import settings
from preprocessors import *


class calcula_submatrizes_pu_thread(threading.Thread):
    def __init__(self, cluster):
        threading.Thread.__init__(self)
        self.cluster = cluster
    def run(self):
        calcula_submatriz_Pu(self.cluster)


class atualiza_estrutura_dados_su_cluster_thread(threading.Thread):
    def __init__(self, cluster):
        threading.Thread.__init__(self)
        self.cluster = cluster
    def run(self):
        settings.su_cluster[self.cluster] = le_matriz_em_formato_csr(
            "%s/%s.local.%d.csr"%(settings.dir_saida_slim_learn,settings.nome_dataset,self.cluster),
            (settings.num_itens,settings.num_itens))
        return


# Todos os metodos esperam df como numpy matrix

def escreve_em_formato_csr(nome_arquivo, df, binario=True, cabecalho=False):
    if not cabecalho:
        saida = open(nome_arquivo, 'w')
    else:
        saida = open("temp", 'w')

    contador_nao_nulos = 0

    num_linhas, num_colunas = df.shape
    for linha in range(num_linhas):
        for coluna in range(num_colunas):
            if df[linha, coluna] != 0:
                contador_nao_nulos = contador_nao_nulos + 1
                if not binario:
                    saida.write("%d %f "%(coluna + 1, df[linha, coluna]))
                else:    
                    saida.write("%d %d "%(coluna + 1, 1))
        saida.write("\n")   
    saida.close()

    if cabecalho:
        arq = open("temp", 'r')
        saida = open(nome_arquivo, 'w')
        saida.write("%d %d %d\n"%(num_linhas, num_colunas, contador_nao_nulos))
        saida.write(arq.read())
        saida.close()
        arq.close()
        subprocess.call(["rm","temp"])

    return        


def le_matriz_em_formato_csr(dir_matriz, dimensao_matriz, cabecalho=False):
    arq = open(dir_matriz, 'r')
    matriz = np.zeros(dimensao_matriz)
    linha = arq.readline()
    contador_linha = 0
    while linha:
        matches = re.findall(r'[\d\.]+\s+[\d\.]+', linha)
        for match in matches:
            match_col_av = re.match('([\d\.]+)\s+([\d\.]+)', match)
            coluna = int(match_col_av.group(1))
            valor_coluna = float(match_col_av.group(2))
            # no arquivo colunas 1-coluna_max, em S, 0-coluna_max-1
            matriz[contador_linha, coluna - 1] = valor_coluna
        contador_linha = contador_linha + 1
        linha = arq.readline()
    arq.close()
    return matriz


def calcula_submatriz_Pu(cluster):
    settings.Ru_cluster[cluster] = np.copy(settings.R_global)
    linhas_para_zerar = ~settings.vetor_clusters_usuarios.isin([cluster])
    contador_linha = 0
    for linha in linhas_para_zerar:
        if linha:
            settings.Ru_cluster[cluster][contador_linha,:] = settings.Ru_cluster[cluster][contador_linha,:] * 0
        contador_linha = contador_linha + 1
    escreve_em_formato_csr(
        "%s/%s.ru.%d.bin.csr"%(settings.dir_entrada_slim_learn,settings.nome_dataset,cluster),
        settings.Ru_cluster[cluster], 
        binario=True, cabecalho=False)
    return


def calcula_submatrizes_Pu():
    settings.Ru_cluster = [None for cluster in range(settings.num_clusters)]

    for cluster in range(settings.num_clusters):
        calcula_submatriz_Pu(cluster)
    return


def calcula_submatrizes_Pu_paralelizado():
    settings.Ru_cluster = [None for cluster in range(settings.num_clusters)]
    threads = []
    for cluster in range(settings.num_clusters):
        thread = calcula_submatrizes_pu_thread(cluster)
        thread.start()
        threads.append(thread)
    
    for thread in threads:
        thread.join()

    return


# atualiza estrutura de su de cada cluster atualizado no passo anterior (diretorio out_slim)
def atualiza_estrutura_dados_su_cluster_paralelizado():
    if settings.su_cluster == None:
        settings.su_cluster = [None for cluster in range(settings.num_clusters)]
    
    threads = []
    for cluster in range(settings.num_clusters):
        thread = atualiza_estrutura_dados_su_cluster_thread(cluster)
        thread.start()
        threads.append(thread)
    
    for thread in threads:
        thread.join()
    return


def cria_matriz_R_binaria_a_partir_matriz_avaliacoes():
    # A matriz R_global eh binaria: 1 se usuario avaliou o item, 0 caso contrario
    settings.R_global = arq_avaliacoes.copy()
    settings.R_global[settings.R_global > 0] = 1
    # Numpy faster than Pandas
    settings.R_global = settings.R_global.as_matrix()
    escreve_em_formato_csr(
        settings.R_global,
        "%s/%s.R.global.bin.csr"%(settings.dir_entrada_slim_learn,settings.nome_dataset),
        binario=True, header=False)
    return
