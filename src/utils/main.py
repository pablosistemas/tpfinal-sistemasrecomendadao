import numpy as np
import random

import settings


def retorna_lista_itens_avaliados_pelo_usuario(usuario):
    return np.where(settings.R_global[usuario,:] > 0)[0]


def divide_itens_treino_teste(lista_itens):
    itens_treino = lista_itens[:]
    item_teste = selecionar_item_aleatoriamente(itens_treino)
    return item_teste


def selecionar_item_aleatoriamente(lista_itens):
    return random.sample(lista_itens,1)[0]


def teste_numpy():
    global arq_avaliacoes_np
    linhas, colunas = arq_avaliacoes_np.shape
    for i in range(linhas):
        for j in range(colunas):
            arq_avaliacoes_np[i,j] = i*j
    return


def teste_pandas(arq_avaliacoes):
    linhas, colunas = arq_avaliacoes.shape
    for i in range(linhas):
        for j in range(colunas):
            arq_avaliacoes.loc[i,j] = i*j
    return


# calcula erro para todos os itens nao avaliados pelo usuario
def retorna_itens_nao_avaliados_pelo_usuario(indices_itens_avaliados_pelo_usuario):
    indices_itens_nao_avaliados_pelo_usuario = set(range(settings.num_itens))
    indices_itens_nao_avaliados_pelo_usuario.difference_update(indices_itens_avaliados_pelo_usuario)
    return indices_itens_nao_avaliados_pelo_usuario


def atualiza_historico_usuario_gu(usuario, gu):
    settings.historico_gu_usuario[usuario].append(gu)


def grava_historico(modelo):
    arq = open("%s/resultados/%s.hist.gu"%(settings.dir_dados,modelo),'w')    
    for usuario in range(settings.num_usuarios):
        arq.write("usuario #%d\n"%(usuario))
        for gu in historico_gu_usuario[usuario]:
            arq.write("%f\n"%(gu))
    arq.close()
    return
