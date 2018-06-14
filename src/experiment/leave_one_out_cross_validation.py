import numpy as np

from glslimr0.main import *
from lslimr0.main import *
from glslim.main import *
from gslim.main import *
from lslim.main import *
from cluto.main import *
from utils.main import *
import settings

def leave_one_out_cross_validation(k_cross_validation=1, modelo='gslsim'):
    settings.R_global = le_matriz_em_formato_csr(
            "%s/%s.R.global.bin.csr"%(settings.dir_entrada_slim_learn,settings.nome_dataset), (settings.num_usuarios,settings.num_itens))

    calcula_clusters_com_cluto(
        '%s/%s.R.csr'%(settings.dir_entrada_cluto,settings.nome_dataset),
        '%s/%s.%d.csr'%(settings.dir_saida_cluto,settings.nome_dataset,settings.num_clusters), 
        settings.num_clusters)
    
    settings.hit_rate = settings.average_reciprocal_hit_rate = [0 for k in range(k_cross_validation)]

    vetor_clusters_usuarios_backup = np.copy(settings.vetor_clusters_usuarios)
    
    for k in range(k_cross_validation):
        print('BEGINNING %d CROSS VALIDATION at %s'%(k, time.strftime("%d/%m/%Y %H:%M:%S")))
        itens_teste = [-1 for usuario in range(settings.num_usuarios)]

        # remove item de teste da matriz de treinamento
        for usuario in range(settings.num_usuarios):
            lista_usuario = retorna_lista_itens_avaliados_pelo_usuario(usuario)
            itens_teste[usuario] = divide_itens_treino_teste(lista_usuario)
            settings.R_global[usuario,itens_teste[usuario]] = 0

        escreve_em_formato_csr('%s/%s.test.csr'%(settings.dir_entrada_slim_teste, settings.nome_dataset), settings.R_global)

        print('BEGIN %d-esimo %s: %s'%(k, modelo.upper(), time.strftime("%d/%m/%Y %H:%M:%S")))
        
        if modelo == 'glslim':
            glslim()
        elif modelo == 'glslimr0':
            glslimr0()
        elif modelo == 'lslim':
            lslim()
        elif modelo == 'lslimr0':
            lslimr0()
        elif modelo == 'gslim':
            gslim()
        else:
            raise 'Modelo %s nao implementado'%(modelo)

        print('END %d-esimo %s: %s'%(k, modelo.upper(), time.strftime("%d/%m/%Y %H:%M:%S")))
    
        settings.hit_rate[k], settings.average_reciprocal_hit_rate[k] = calcula_hit_rate_e_average_reciprocal_hit_rate(itens_teste, utilizar_slim_predict=True)

        # insere novamente o item removido para treinamento
        for usuario in range(settings.num_usuarios):
            settings.R_global[usuario,itens_teste[usuario]] = 1

        # vetor de cluster inicial foi modificado pelo glslim, restaura inicial para novo treinamento
        settings.vetor_clusters_usuarios = vetor_clusters_usuarios_backup
    
    # calcular media e variancia
    print('HIT RATE MEDIO: %f\mHIT RATE DESVIO PADRAO: %f\n'%(np.mean(settings.hit_rate), np.std(settings.hit_rate)))
    print('AVERAGE HIT RATE RECIPROCO  MEDIO: %f\mAVERAGE HIT RATE RECIPROCO DESVIO PADRAO: %f\n'%(np.mean(settings.average_reciprocal_hit_rate), np.std(settings.average_reciprocal_hit_rate)))

    return
