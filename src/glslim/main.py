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
import math
import time
import sys
import re

from preprocessors.shared import *
from evaluation.main import *
from utils.main import *
from slim.main import *
import settings


class glslim_thread(threading.Thread):
    def __init__(self, usuario):
        threading.Thread.__init__(self)
        self.usuario = usuario
    def run(self):
        glslim_treinamento_usuario(self.usuario)


def glslim_treinamento_usuario(usuario):
    erros_treinamento = []
    gu_por_cluster = []
    # indices das colunas dos itens avaliados pelo usuario (=1)
    indices_itens_avaliados_pelo_usuario = retorna_lista_itens_avaliados_pelo_usuario(usuario)
    # calcula gus para todos os clusters. Ao fim, escolhe o cluster de menor erro e atribui gu correspondente
    for cluster in range(settings.num_clusters):
        gu_cluster = calcula_gu(usuario, cluster, indices_itens_avaliados_pelo_usuario)
        gu_por_cluster.append(gu_cluster)

        erro = calcula_erro_predicao(usuario, cluster, gu_cluster, indices_itens_avaliados_pelo_usuario)
        erros_treinamento.append(erro)
    # escolhe novo cluster do usuario
    min_idx, min_valor = min(enumerate(erros_treinamento), key=operator.itemgetter(1))
    # se houver empate, nao atualiza
    if (min_valor < erros_treinamento[settings.vetor_clusters_usuarios[usuario]]):
        settings.novo_vetor_clusters_usuarios[usuario] = min_idx
        settings.gu[usuario] = gu_por_cluster[min_idx]
        # mantem historico
        atualiza_historico_usuario_gu(usuario, settings.gu[usuario])
    return


def executa_nucleo_glslim_paralelo():
    threads = []
    for usuario in range(num_usuarios):
        thread = glslim_thread(usuario)
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()


def executa_nucleo_glslim():
    for usuario in range(settings.num_usuarios):
        # mantem historico
        atualiza_historico_usuario_gu(usuario, settings.gu[usuario])
        glslim_treinamento_usuario(usuario)
    return


def glslim():
    # contribuicao dos modelos local e global iguais para todos os usuarios no inicio do treinamento
    settings.gu = 0.5 * np.ones((settings.num_usuarios))
    percentual_mudancas = 1
    num_iteracoes = 0
    while percentual_mudancas > 0.01 and num_iteracoes < settings.max_num_iteracoes:
        estima_modelo_slim_global()        
        settings.su_global = le_matriz_em_formato_csr(
            "%s/%s.global.csr"%(settings.dir_saida_slim_learn,settings.nome_dataset),
            (settings.num_itens,settings.num_itens))

        calcula_submatrizes_Pu_paralelizado()
        settings.novo_vetor_clusters_usuarios = settings.vetor_clusters_usuarios.tolist()
        
        estima_modelo_slim_para_todos_clusters_paralelizado()
        atualiza_estrutura_dados_su_cluster_paralelizado()

        executa_nucleo_glslim()

        percentual_mudancas = retorna_percentual_mudancas()

        # atualiza vetor de usuario/cluster
        settings.vetor_clusters_usuarios = pd.Series(settings.novo_vetor_clusters_usuarios)
        print('Percentual de mudancas: %f\n'%(percentual_mudancas))
        print('Numero iteracoes: %d\n'%(num_iteracoes))
        num_iteracoes = num_iteracoes + 1

    return
