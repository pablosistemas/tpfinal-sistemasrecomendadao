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
import operator
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
Ru_cluster = None
su_cluster = None
su_global = None
R_global = None
gu = None

# VARIAVEIS DE CONTROLE. Ex: num_usuarios, iteracões, num_clusters, etc
num_usuarios = 0
num_itens = 0
tolerancia_treinamento = 1e-2

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


def estimar_matriz_S_com_sim_learn(caminho_matriz_treinamento, caminho_matriz_saida_modelo,
    coluna_inicial=0, coluna_final=None, optTol=1e-2):
    parametros_bash = [slim, "-train_file=%s"%(caminho_matriz_treinamento), 
                    '-optTol=%f'%(optTol),
                    "-model_file=%s"%(caminho_matriz_saida_modelo)]
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
    escreve_matriz_em_formato_csr(R_global.as_matrix(),"%s/%s.R.global.bin.csr"%(dir_entrada_slim_learn,nome_dataset))
    return


def calcula_submatrizes_Pu():
    global Ru_cluster
    Ru_cluster = [None for cluster in range(num_clusters)]

    for cluster in range(num_clusters):
        Ru_cluster[cluster] = np.copy(R_global)
        linhas_para_zerar = ~vetor_clusters_usuarios.isin([cluster])
        contador_linha = 0
        for linha in linhas_para_zerar:
            if linha:
                Ru_cluster[cluster][contador_linha,:] = Ru_cluster[cluster][contador_linha,:] * 0
            contador_linha = contador_linha + 1
        escreve_matriz_em_formato_csr(Ru_cluster[cluster], "%s/%s.ru.%d.bin.csr"%(dir_entrada_slim_learn,nome_dataset,cluster))
    return


def avalia_modelo():
    pass
    

def glslim():
    global gu, R_global, su_global, su_cluster, vetor_clusters_usuarios
    gu = 0.5 * np.ones((num_usuarios))
    # cria_matriz_R_binaria_a_partir_matriz_avaliacoes()
    R_global = le_matriz_em_formato_csr_sem_cabecalho("%s/%s.R.global.bin.csr"%(dir_entrada_slim_learn,nome_dataset),(num_usuarios,num_itens))
    calcula_clusters_com_cluto('%s/%s.R.csr'%(dir_entrada_cluto,nome_dataset),\
        '%s/%s.%d.csr'%(dir_saida_cluto,nome_dataset,num_clusters), num_clusters)
    calcula_submatrizes_Pu()

    percentual_mudancas = 1
    while percentual_mudancas > 0.01:
        novo_cluster = [None for usuario in range(num_usuarios)]
        # Estima modelo global (matrix R completa)
        estimar_matriz_S_com_sim_learn("%s/%s.R.global.bin.csr"%(dir_entrada_slim_learn,nome_dataset),
                "%s/%s.global.csr"%(dir_saida_slim_learn,nome_dataset),optTol=1e-3)
        
        # carrega modelos su (em out_slim) de cada cluster
        su_global = le_matriz_em_formato_csr_sem_cabecalho("%s/%s.global.csr"%(dir_saida_slim_learn,nome_dataset),(num_itens,num_itens))
    
        # calcula modelo SLIM para cada submatriz de clusters
        for cluster in range(num_clusters):
            # Estima modelo global (matrix R completa)
            estimar_matriz_S_com_sim_learn("%s/%s.ru.%d.bin.csr"%(dir_entrada_slim_learn,nome_dataset,cluster),
                    "%s/%s.local.%d.csr"%(dir_saida_slim_learn,nome_dataset,cluster),optTol=1e-3)
        
        # atualiza estrutura de su de cada cluster atualizado no passo anterior
        su_cluster = [None for cluster in range(num_clusters)]
        for cluster in range(num_clusters):
            su_cluster[cluster] = le_matriz_em_formato_csr_sem_cabecalho("%s/%s.local.%d.csr"%(dir_saida_slim_learn,nome_dataset,cluster),(num_itens,num_itens))

        for usuario in range(num_usuarios):
            erros_treinamento = []
            gu_por_cluster = []
            # indices das colunas dos itens avaliados pelo usuario (=1)
            indices_itens_avaliados_pelo_usuario = R_global.ix[usuario,:].index[R_global.ix[usuario,:] == 1].tolist()
            # calcula gus para todos os clusters. Ao fim, escolhe o cluster de menor erro e atribui gu correspondente
            for cluster in range(num_clusters):
                gu_cluster = calcula_gu(usuario, cluster, indices_itens_avaliados_pelo_usuario)
                # em alguns casos, gu sai do intervalo [0,1], workaround:
                gu_cluster = max(0, gu_cluster)
                gu_cluster = min(1, gu_cluster)
                
                gu_por_cluster.append(gu_cluster)

                erro = calcula_erro_predicao(usuario, cluster, gu_cluster, indices_itens_avaliados_pelo_usuario)
                erros_treinamento.append(erro)
            # escolhe novo cluster do usuario
            min_idx, min_valor = min(enumerate(erros_treinamento), key=operator.itemgetter(1))
            # se houver empate, nao atualiza
            if (min_valor < erros_treinamento[vetor_clusters_usuarios[usuario]]):
                novo_cluster[usuario] = min_idx
                gu[usuario] = gu_por_cluster[min_idx]
        
        # numero de clusters diferentes na iteracao
        mudanca_cluster = 0
        for usuario in range(num_usuarios):
            if vetor_clusters_usuarios[usuario] != novo_cluster[usuario]:
                mudanca_cluster = mudanca_cluster + 1
        percentual_mudancas = float(mudanca_cluster)/num_usuarios
        # atualiza vetor de usuario/cluster
        vetor_clusters_usuarios = pd.Series(novo_cluster)
        print('Percentual de mudancas: %f\n'%(percentual_mudancas))
        
    return


def main():
    global arq_avaliacoes, num_usuarios, num_itens
    # le arquivo de avaliações NxM
    arq_avaliacoes = pd.read_csv(dir_matriz_avaliacoes, header=None)
    num_usuarios, num_itens = arq_avaliacoes.shape
    glslim()
    return


def verifica_clustering():
    #mock
    return True


def calcula_predicao_usuario_item(usuario, cluster, item, gu_cluster, indices_itens_avaliados_pelo_usuario):
    predicao = 0
    for item_avaliado in indices_itens_avaliados_pelo_usuario:
        predicao = predicao + gu_cluster * su_global.ix[item_avaliado,item]  + (1 - gu_cluster) * su_cluster[cluster].ix[item_avaliado,item]
    
    return predicao
    

def calcula_erro_predicao(usuario, cluster, gu_cluster, indices_itens_avaliados_pelo_usuario):
    predicao = [0 for item in range(num_itens)]
    erro = 0
    for item in range(num_itens):
        predicao[item] = calcula_predicao_usuario_item(usuario, cluster, item, gu_cluster, indices_itens_avaliados_pelo_usuario)
        erro = erro + (arq_avaliacoes.ix[usuario,item] - predicao[item]) ** 2
    
    erro = erro/num_itens
    return erro

    
def calcula_gu(usuario, cluster, indices_itens_avaliados_pelo_usuario):
    # u eh o usuario da iteracao
    num = den = 0
    for item in range(num_itens):
        soma_sli = 0
        soma_suli = 0
        for item_similar in indices_itens_avaliados_pelo_usuario:
            soma_sli = soma_sli + su_global.ix[item_similar,item]
            soma_suli = soma_suli + su_cluster[cluster].ix[item_similar,item]
            
        num = num + (soma_sli - soma_suli) * (R_global.ix[usuario,item] - soma_suli)
        den = den + (soma_sli - soma_suli)**2

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
    return pd.DataFrame(matriz)


if __name__ == "__main__":
    inicializa_variaveis_globais()
    main()
    