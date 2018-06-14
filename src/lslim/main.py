import pandas as pd
import numpy as np
import operator

from preprocessors.shared import *
from evaluation.main import *
from utils.main import *
from slim.main import *
import settings


def executa_nucleo_lslim():
    for usuario in range(settings.num_usuarios):
        lslim_treinamento_usuario(usuario)
    return

    
def lslim_treinamento_usuario(usuario):
    erros_treinamento = []
    # indices das colunas dos itens avaliados pelo usuario (=1)
    indices_itens_avaliados_pelo_usuario = retorna_lista_itens_avaliados_pelo_usuario(usuario)
    # calcula gus para todos os clusters. Ao fim, escolhe o cluster de menor erro e atribui gu correspondente
    for cluster in range(settings.num_clusters):
        erro = calcula_erro_predicao(usuario, cluster, settings.gu[usuario], indices_itens_avaliados_pelo_usuario)
        erros_treinamento.append(erro)
    # escolhe novo cluster do usuario
    min_idx, min_valor = min(enumerate(erros_treinamento), key=operator.itemgetter(1))
    # se houver empate, nao atualiza
    if (min_valor < erros_treinamento[settings.vetor_clusters_usuarios[usuario]]):
        settings.novo_vetor_clusters_usuarios[usuario] = min_idx
    return


def lslim():
    settings.gu = np.zeros((settings.num_usuarios))
    percentual_mudancas = 1
    
    num_iteracoes = 0
    while percentual_mudancas > 0.01 and num_iteracoes < settings.max_num_iteracoes:
        calcula_submatrizes_Pu_paralelizado()
        settings.novo_vetor_clusters_usuarios = settings.vetor_clusters_usuarios.tolist()
        
        estima_modelo_slim_para_todos_clusters_paralelizado()
        atualiza_estrutura_dados_su_cluster_paralelizado()

        executa_nucleo_lslim()
        percentual_mudancas = retorna_percentual_mudancas()

        # atualiza vetor de usuario/cluster
        settings.vetor_clusters_usuarios = pd.Series(settings.novo_vetor_clusters_usuarios)
        print('Percentual de mudancas: %f\n'%(percentual_mudancas))
        print('Numero iteracoes: %d\n'%(num_iteracoes))
        num_iteracoes = num_iteracoes + 1

    return
