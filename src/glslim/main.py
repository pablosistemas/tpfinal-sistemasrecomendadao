#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Diretorio matrizes de avaliações: ./data/ratings/
Diretorio matrizes binarias entrada CLUTO: ./data/in_cluto/
Diretorio matrizes saida do CLUTO: ./data/out_cluto/
Diretorio matrizes entrada do SLIM_LEARN: ./data/in_slim/
Diretorio matrizes saida SLIM_LEARN: ./data/out_slim/
'''

import pandas as pd
import numpy as np
import subprocess
import threading
import operator
import cProfile
import random
import time
import sys
import re

# CAMINHOS
caminho_projeto = ''
dir_dados = ''
dir_entrada_cluto = ''
dir_saida_cluto = ''
dir_entrada_slim_learn = ''
dir_saida_slim_learn = ''
dir_cluto = ''
nome_dataset = ''
caminho_matriz_cluto = ''
caminho_matriz_cluto = ''
dir_m_su = ''
caminho_matriz_cluto_treinamento = ''
caminho_matriz_cluto_teste = ''
slim = ''
cluto = ''
dir_matriz_avaliacoes = ''

# MATRIZES E VETORES
vetor_clusters_usuarios = None
arq_avaliacoes = None
novo_vetor_clusters_usuarios = None
Ru_cluster = None
su_cluster = None
su_global = None
R_global = None
gu = None

# VARIAVEIS DE CONTROLE. Ex: num_usuarios, iteracões, num_clusters, etc
num_usuarios = 0
num_itens = 0
tolerancia_treinamento = 1e-5
N = 10
hit_rate = None
average_reciprocal_hit_rate = None

class slim_thread(threading.Thread):
    def __init__(self, cluster, tolerancia_treinamento):
        threading.Thread.__init__(self)
        self.cluster = cluster
        self.tolerancia_treinamento = tolerancia_treinamento
    def run(self):
        estima_matriz_S_com_sim_learn("%s/%s.ru.%d.bin.csr"%(dir_entrada_slim_learn,nome_dataset,self.cluster),
                "%s/%s.local.%d.csr"%(dir_saida_slim_learn,nome_dataset,self.cluster),optTol=self.tolerancia_treinamento)
        return


class calcula_submatrizes_pu_thread(threading.Thread):
    def __init__(self, cluster):
        threading.Thread.__init__(self)
        self.cluster = cluster
    def run(self):
        calcula_submatriz_Pu(self.cluster)


class glslim_thread(threading.Thread):
    def __init__(self, usuario):
        threading.Thread.__init__(self)
        self.usuario = usuario
    def run(self):
        glslim_treinamento_usuario(self.usuario)


class atualiza_estrutura_dados_su_cluster_thread(threading.Thread):
    def __init__(self, cluster):
        threading.Thread.__init__(self)
        self.cluster = cluster
    def run(self):
        global su_cluster
        su_cluster[self.cluster] = le_matriz_em_formato_csr_sem_cabecalho("%s/%s.local.%d.csr"%(dir_saida_slim_learn,nome_dataset,self.cluster),(num_itens,num_itens))


def inicializa_variaveis_globais():
    global caminho_projeto, dir_dados, dir_entrada_cluto, dir_saida_cluto, dir_entrada_slim_learn, \
        dir_saida_slim_learn, dir_cluto, nome_dataset, caminho_matriz_cluto, dir_m_su, \
        caminho_matriz_cluto_treinamento, caminho_matriz_cluto_teste, slim, cluto, num_clusters, \
        dir_matriz_avaliacoes
    
    caminho_projeto = define_caminhos_absolutos()
    dir_dados       = caminho_projeto + "/data/"
    dir_entrada_cluto = dir_dados + "/in_cluto/"
    dir_saida_cluto = "%s/out_cluto/"%(dir_dados)
    dir_entrada_slim_learn = "%s/in_slim/"%(dir_dados)
    dir_saida_slim_learn = "%s/out_slim/"%(dir_dados)
    dir_cluto       = caminho_projeto + "/tools/cluto/"
    nome_dataset    = "movielens1m"
    caminho_matriz_cluto = "/%s/%s.cluto.csr" %(dir_saida_cluto,nome_dataset)
    dir_m_su        = "%s/matriz_su/"%(dir_dados)              
    caminho_matriz_cluto_treinamento = "%s/tools/slim/examples/train.mat"%(caminho_projeto)
    caminho_matriz_cluto_teste = "%s/tools/slim/examples/test.mat"%(caminho_projeto)
    slim            = "%s/tools/slim/build/examples/slim_learn"%(caminho_projeto)
    cluto           = "%s/Linux/vcluster"%(dir_cluto)
    num_clusters    = get_num_clusters()
    dir_matriz_avaliacoes = "%s/ratings/%s.csv"%(dir_dados,nome_dataset)


def get_num_clusters():
    if sys.argv.__len__() > 2:
        return sys.argv[2]
    else:
        return 5


def define_caminhos_absolutos():
    if sys.argv.__len__() > 1:
        return sys.argv[1]
    else:
        return "./"


def calcula_clusters_com_cluto(caminho_matriz_cluto, nome_arq_saida_cluto, num_clusters):
    subprocess.call(["rm", "-f", nome_arq_saida_cluto])
    subprocess.call([cluto, caminho_matriz_cluto, str(num_clusters), "-clustfile=" + nome_arq_saida_cluto, "-colmodel=none"])
    global vetor_clusters_usuarios
    vetor_clusters_usuarios = pd.read_csv(nome_arq_saida_cluto, header=None)[0]
    return


def estima_matriz_S_com_sim_learn(caminho_matriz_treinamento, caminho_matriz_saida_modelo,
    coluna_inicial=0, coluna_final=None, optTol=1e-2, lambda_p=1, beta_p=5):
    parametros_bash = [slim, '-train_file=%s'%(caminho_matriz_treinamento), 
                    '-optTol=%f'%(optTol),
                    '-model_file=%s'%(caminho_matriz_saida_modelo),
                    '-lambda=%f'%(lambda_p),
                    '-beta=%f'%(beta_p)]
    if coluna_final:
        parametros_bash.append("-starti=%d"%(coluna_inicial))
        parametros_bash.append("-endi=%d"%(coluna_final))
    subprocess.call(["rm", "-f", caminho_matriz_saida_modelo])
    subprocess.call(parametros_bash)
    return


def cria_matriz_R_binaria_a_partir_matriz_avaliacoes():
    # A matriz R_global eh binaria: 1 se usuario avaliou o item, 0 caso contrario
    global R_global
    R_global = arq_avaliacoes.copy()
    R_global[R_global > 0] = 1
    # Numpy faster than Pandas
    R_global = R_global.as_matrix()
    escreve_matriz_em_formato_csr(R_global,"%s/%s.R.global.bin.csr"%(dir_entrada_slim_learn,nome_dataset))
    return


def calcula_submatriz_Pu(cluster):
    global Ru_cluster
    Ru_cluster[cluster] = np.copy(R_global)
    linhas_para_zerar = ~vetor_clusters_usuarios.isin([cluster])
    contador_linha = 0
    for linha in linhas_para_zerar:
        if linha:
            Ru_cluster[cluster][contador_linha,:] = Ru_cluster[cluster][contador_linha,:] * 0
        contador_linha = contador_linha + 1
    escreve_matriz_em_formato_csr(Ru_cluster[cluster], "%s/%s.ru.%d.bin.csr"%(dir_entrada_slim_learn,nome_dataset,cluster))


def calcula_submatrizes_Pu_paralelizado():
    global Ru_cluster
    Ru_cluster = [None for cluster in range(num_clusters)]
    threads = []
    for cluster in range(num_clusters):
        thread = calcula_submatrizes_pu_thread(cluster)
        thread.start()
        threads.append(thread)
    
    for thread in threads:
        thread.join()


def calcula_submatrizes_Pu():
    global Ru_cluster
    Ru_cluster = [None for cluster in range(num_clusters)]

    for cluster in range(num_clusters):
        calcula_submatriz_Pu(cluster)
    return


def retorna_lista_itens_avaliados_pelo_usuario(usuario):
    #return R_global.ix[usuario,:].index[R_global.ix[usuario,:] == 1].tolist()
    return np.where(R_global[usuario,:] == 1)[0]


# FIXME:
def avalia_modelo():
    global su_global
    for usuario in range(num_usuarios):
        indices_itens_avaliados_pelo_usuario = retorna_lista_itens_avaliados_pelo_usuario(usuario)
        itens_treino, item_teste = divide_itens_treino_teste(indices_itens_avaliados_pelo_usuario)
        # Ajusta matriz R_global para teste
        R_global[usuario, item_teste] = 0

        estima_modelo_slim_global()        
        su_global = le_matriz_em_formato_csr_sem_cabecalho("%s/%s.global.csr"%(dir_saida_slim_learn,nome_dataset),(num_itens,num_itens))

        estima_modelo_slim_para_todos_clusters_paralelizado()
        atualiza_estrutura_dados_su_cluster_paralelizado()

        # top_n = calcula_top_n(usuario, indices_itens_avaliados_pelo_usuario)

        # desfaz ajuste matriz R_global
        R_global.ix[usuario, item_teste] = 1


def divide_itens_treino_teste(lista_itens):
    itens_treino = lista_itens[:]
    item_teste = selecionar_item_aleatoriamente(itens_treino)
    return item_teste


def calcula_top_n(usuario, indices_itens_avaliados_pelo_usuario):
    predicao = [None for item in range(num_itens)]
    for item in range(num_itens):
        predicao[item] = calcula_predicao_usuario_item(usuario, vetor_clusters_usuarios[usuario], item, gu[usuario], indices_itens_avaliados_pelo_usuario)
    ranking = [item[0] for item in sorted(enumerate(predicao), key=lambda x:x[1])]
    if num_itens < N:
        return ranking[0:num_itens]
    else:
        return ranking[0:N]


def selecionar_item_aleatoriamente(lista_itens):
    return random.sample(lista_itens,1)[0]


def estima_modelo_slim_global():
    # Estima modelo global (matrix R completa)
    estima_matriz_S_com_sim_learn("%s/%s.R.global.bin.csr"%(dir_entrada_slim_learn,nome_dataset),
            "%s/%s.global.csr"%(dir_saida_slim_learn,nome_dataset),optTol=tolerancia_treinamento)
    return


# calcula modelo SLIM para cada submatriz de clusters
def estima_modelo_slim_para_todos_clusters_paralelizado():
    threads = []
    for cluster in range(num_clusters):
        thread = slim_thread(cluster, tolerancia_treinamento)
        thread.start()
        threads.append(thread)
    
    for thread in threads:
        thread.join()

    
def estima_modelo_slim_para_todos_clusters():
    for cluster in range(num_clusters):
        # Estima modelo global (matrix R completa)
        estima_matriz_S_com_sim_learn("%s/%s.ru.%d.bin.csr"%(dir_entrada_slim_learn,nome_dataset,cluster),
                "%s/%s.local.%d.csr"%(dir_saida_slim_learn,nome_dataset,cluster),optTol=tolerancia_treinamento)
    return


# atualiza estrutura de su de cada cluster atualizado no passo anterior (diretorio out_slim)
def atualiza_estrutura_dados_su_cluster_paralelizado():
    global su_cluster
    if su_cluster == None:
        su_cluster = [None for cluster in range(num_clusters)]
    
    threads = []
    for cluster in range(num_clusters):
        thread = atualiza_estrutura_dados_su_cluster_thread(cluster)
        thread.start()
        threads.append(thread)
    
    for thread in threads:
        thread.join()
    return


def atualiza_estrutura_dados_su_cluster():
    global su_cluster
    if su_cluster == None:
        su_cluster = [None for cluster in range(num_clusters)]
    for cluster in range(num_clusters):
        su_cluster[cluster] = le_matriz_em_formato_csr_sem_cabecalho("%s/%s.local.%d.csr"%(dir_saida_slim_learn,nome_dataset,cluster),(num_itens,num_itens))
    return


def glslim_treinamento_usuario(usuario):
    global gu, novo_vetor_clusters_usuarios
    erros_treinamento = []
    gu_por_cluster = []
    # indices das colunas dos itens avaliados pelo usuario (=1)
    indices_itens_avaliados_pelo_usuario = retorna_lista_itens_avaliados_pelo_usuario(usuario)
    # calcula gus para todos os clusters. Ao fim, escolhe o cluster de menor erro e atribui gu correspondente
    for cluster in range(num_clusters):
        gu_cluster = calcula_gu(usuario, cluster, indices_itens_avaliados_pelo_usuario)
        
        # em alguns casos, gu sai do intervalo [0,1], work-around:
        gu_cluster = max(0, gu_cluster)
        gu_cluster = min(1, gu_cluster)

        gu_por_cluster.append(gu_cluster)

        erro = calcula_erro_predicao(usuario, cluster, gu_cluster, indices_itens_avaliados_pelo_usuario)
        erros_treinamento.append(erro)
    # escolhe novo cluster do usuario
    min_idx, min_valor = min(enumerate(erros_treinamento), key=operator.itemgetter(1))
    # se houver empate, nao atualiza
    if (min_valor < erros_treinamento[vetor_clusters_usuarios[usuario]]):
        novo_vetor_clusters_usuarios[usuario] = min_idx
        gu[usuario] = gu_por_cluster[min_idx]


def calcula_hit_rate_e_average_reciprocal_hit_rate(itens_teste):
    hit = 0
    arhr = 0
    for usuario in range(num_usuarios):
        indices_itens_avaliados_pelo_usuario = retorna_lista_itens_avaliados_pelo_usuario(usuario)
        top_n = calcula_top_n(usuario, indices_itens_avaliados_pelo_usuario)
        if itens_teste[usuario] in top_n:
            hit = hit + 1
            arhr = arhr + 1.0/(top_n.index(itens_teste[usuario]) + 1)
    return float(hit)/num_usuarios, float(arhr)/num_usuarios


def leave_one_out_cross_validation(k_cross_validation=1):
    global su_global, hit_rate, average_reciprocal_hit_rate, R_global, vetor_clusters_usuarios

    R_global = le_matriz_em_formato_csr_sem_cabecalho("%s/%s.R.global.bin.csr"%(dir_entrada_slim_learn,nome_dataset),(num_usuarios,num_itens))
    calcula_clusters_com_cluto('%s/%s.R.csr'%(dir_entrada_cluto,nome_dataset),\
        '%s/%s.%d.csr'%(dir_saida_cluto,nome_dataset,num_clusters), num_clusters)
    
    vetor_clusters_usuarios_backup = vetor_clusters_usuarios
    
    hit_rate = average_reciprocal_hit_rate = [0 for k in range(k_cross_validation)]
    
    for k in range(k_cross_validation):
        print('BEGINNING %d CROSS VALIDATION at %s'%(k, time.strftime("%d/%m/%Y %H:%M:%S")))
        itens_teste = [-1 for usuario in range(num_usuarios)]

        # remove item de teste da matriz de treinamento
        for usuario in range(num_usuarios):
            lista_usuario = retorna_lista_itens_avaliados_pelo_usuario(usuario)
            itens_teste[usuario] = divide_itens_treino_teste(lista_usuario)
            R_global[usuario,itens_teste[usuario]] = 0

        estima_modelo_slim_global()        
        su_global = le_matriz_em_formato_csr_sem_cabecalho("%s/%s.global.csr"%(dir_saida_slim_learn,nome_dataset),(num_itens,num_itens))
    
        print('BEGIN %d-esimo GLSLIM: %s'%(k, time.strftime("%d/%m/%Y %H:%M:%S")))
        glslim()
        print('END %d-esimo GLSLIM: %s'%(k, time.strftime("%d/%m/%Y %H:%M:%S")))
    
        hit_rate[k], average_reciprocal_hit_rate[k] = calcula_hit_rate_e_average_reciprocal_hit_rate

        # insere novamente o item removido para treinamento
        for usuario in range(num_usuarios):
            R_global[usuario,itens_teste[usuario]] = 1
        
        # vetor de cluster inicial foi modificado pelo glslim, restaura inicial para novo treinamento
        vetor_clusters_usuarios = vetor_clusters_usuarios_backup
    
    # calcular media e variancia
    print('HIT RATE MEDIO: %f\mHIT RATE DESVIO PADRAO: %f\n'%(np.mean(hit_rate), np.std(hit_rate)))
    print('AVERAGE HIT RATE RECIPROCO  MEDIO: %f\mAVERAGE HIT RATE RECIPROCO DESVIO PADRAO: %f\n'%(np.mean(average_reciprocal_hit_rate), np.std(average_reciprocal_hit_rate)))


def executa_nucleo_glslim_paralelo():
    threads = []
    for usuario in range(num_usuarios):
        thread = glslim_thread(usuario)
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()


def executa_nucleo_glslim():
    for usuario in range(num_usuarios):
        glslim_treinamento_usuario(usuario)
    

def glslim():
    global gu, R_global, su_global, su_cluster, vetor_clusters_usuarios, novo_vetor_clusters_usuarios
    gu = 0.5 * np.ones((num_usuarios))
    
    percentual_mudancas = 1
    while percentual_mudancas > 0.01:
        
        calcula_submatrizes_Pu_paralelizado()
        novo_vetor_clusters_usuarios = vetor_clusters_usuarios.tolist()
        
        estima_modelo_slim_para_todos_clusters_paralelizado()
        atualiza_estrutura_dados_su_cluster_paralelizado()

        executa_nucleo_glslim()

        percentual_mudancas = retorna_percentual_mudancas()

        # atualiza vetor de usuario/cluster
        vetor_clusters_usuarios = pd.Series(novo_vetor_clusters_usuarios)
        print('Percentual de mudancas: %f\n'%(percentual_mudancas))
    
    return


# numero de clusters diferentes na iteracao
def retorna_percentual_mudancas():
    mudanca_cluster = 0
    for usuario in range(num_usuarios):
        if vetor_clusters_usuarios[usuario] != novo_vetor_clusters_usuarios[usuario]:
            mudanca_cluster = mudanca_cluster + 1
    return float(mudanca_cluster)/num_usuarios


def calcula_predicao_usuario_item(usuario, cluster, item, gu_cluster, indices_itens_avaliados_pelo_usuario):
    predicao = 0
    for item_avaliado in indices_itens_avaliados_pelo_usuario:
        predicao = predicao + gu_cluster * su_global[item_avaliado,item]  + (1 - gu_cluster) * su_cluster[cluster][item_avaliado,item]
    
    return predicao
    

# calcula erro para todos os itens nao avaliados pelo usuario
def retorna_itens_nao_avaliados_pelo_usuario(indices_itens_avaliados_pelo_usuario):
    indices_itens_nao_avaliados_pelo_usuario = set(range(num_itens))
    indices_itens_nao_avaliados_pelo_usuario.difference_update(indices_itens_avaliados_pelo_usuario)
    return indices_itens_nao_avaliados_pelo_usuario


def calcula_erro_predicao(usuario, cluster, gu_cluster, indices_itens_avaliados_pelo_usuario):
    erro = 0
    indices_itens_nao_avaliados_pelo_usuario = retorna_itens_nao_avaliados_pelo_usuario(indices_itens_avaliados_pelo_usuario)
    for item in indices_itens_nao_avaliados_pelo_usuario:
        predicao = calcula_predicao_usuario_item(usuario, cluster, item, gu_cluster, indices_itens_avaliados_pelo_usuario)
        erro = erro + (arq_avaliacoes[usuario,item] - predicao) ** 2
    
    erro = erro/num_itens
    return erro

    
def calcula_gu(usuario, cluster, indices_itens_avaliados_pelo_usuario):
    # u eh o usuario da iteracao
    num = den = 0
    for item in range(num_itens):
        soma_sli = 0
        soma_suli = 0
        for item_similar in indices_itens_avaliados_pelo_usuario:
            soma_sli = soma_sli + su_global[item_similar,item]
            soma_suli = soma_suli + su_cluster[cluster][item_similar,item]
            
        num = num + (soma_sli - soma_suli) * (R_global[usuario,item] - soma_suli)
        den = den + (soma_sli - soma_suli)**2

    # evita NaN
    if den == 0.0:
        return 0.0
    return num/den
        
        
def get_nome_matriz_su(cluster):
    return "%s/su.%d.csr" %(dir_m_su, cluster)


def escreve_matriz_em_formato_csr(matriz, nome_matriz):
   arquivo = open(nome_matriz, 'w') 
   
   for u in range(num_usuarios):
       for i in range(num_itens):
           if matriz[u,i] == 1:
               # i + 1 pois formato .csr admite colunas começando em 1...N
               arquivo.write("%d %d "%(i+1,1))
       arquivo.write("\n")        
   arquivo.close()


# SLIM
def le_matriz_em_formato_csr_sem_cabecalho(dir_matriz, dimensao_matriz):
    arq = open(dir_matriz, 'r')
    matriz = np.zeros(dimensao_matriz)
    linha = arq.readline()
    contador_linha = 0
    while linha:
        matches = re.findall(r'[\d\.]+\s[\d\.]+', linha)
        for match in matches:
            match_col_av = re.match('([\d\.]+)\s([\d\.]+)', match)
            coluna = int(match_col_av.group(1))
            valor_coluna = float(match_col_av.group(2))
            # no arquivo colunas 1-coluna_max, em S, 0-coluna_max-1
            matriz[contador_linha, coluna - 1] = valor_coluna
        contador_linha = contador_linha + 1
        linha = arq.readline()
    arq.close()
    return matriz


def teste_numpy():
    global arq_avaliacoes_np
    linhas, colunas = arq_avaliacoes_np.shape
    for i in range(linhas):
        for j in range(colunas):
            arq_avaliacoes_np[i,j] = i*j


def teste_pandas(arq_avaliacoes):
    linhas, colunas = arq_avaliacoes.shape
    for i in range(linhas):
        for j in range(colunas):
            arq_avaliacoes.loc[i,j] = i*j


def main():
    global arq_avaliacoes, num_usuarios, num_itens
    # le arquivo de avaliações NxM
    arq_avaliacoes = pd.read_csv(dir_matriz_avaliacoes, header=None)
    arq_avaliacoes = arq_avaliacoes.as_matrix()

    #cProfile.run('teste_pandas(arq_avaliacoes)')
    #cProfile.run('teste_numpy()')

    num_usuarios, num_itens = arq_avaliacoes.shape
    leave_one_out_cross_validation(1)
    #avalia_modelo()
    return


if __name__ == "__main__":
    # reproducibilidade
    random.seed(1234)
    inicializa_variaveis_globais()
    main()
        