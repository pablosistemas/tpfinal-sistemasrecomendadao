import numpy as np

import settings


def executa_nucleo_glslimr0():
    erros = []
    for usuario in range(settings.num_usuarios):
        erro = glslimr0_treinamento_usuario(usuario)
        erros.append(erro)
    return erros

    
def glslimr0():
    #
    settings.gu = 0.5 * np.ones((settings.num_usuarios))

    diff = 1
    erro_anterior = 0
    num_iteracoes = 1
    while diff > 0.01 and num_iteracoes < settings.max_num_iteracoes:
        estima_modelo_slim_global()        
        settings.su_global = le_matriz_em_formato_csr(
            "%s/%s.global.csr"%(settings.dir_saida_slim_learn,nome_dataset),
            (settings.num_itens,settings.num_itens))

        calcula_submatrizes_Pu_paralelizado()
        settings.novo_vetor_clusters_usuarios = settings.vetor_clusters_usuarios.tolist()

        estima_modelo_slim_para_todos_clusters_paralelizado()
        atualiza_estrutura_dados_su_cluster_paralelizado()

        erros = executa_nucleo_glslimr0()
        # calcula erro medio
        erro_atual = sum(erros)/len(erros)
        diff = (erro_anterior - erro_atual)
        
        print('Numero iteracoes: %d\n'%(num_iteracoes))
        num_iteracoes = num_iteracoes + 1
    return 


def glslimr0_treinamento_usuario(usuario):
    global gu
    # indices das colunas dos itens avaliados pelo usuario (=1)
    indices_itens_avaliados_pelo_usuario = retorna_lista_itens_avaliados_pelo_usuario(usuario)
    cluster = vetor_clusters_usuarios[usuario]
    gu_cluster = calcula_gu(usuario, cluster, indices_itens_avaliados_pelo_usuario)
    gu[usuario] = gu_cluster
    erro = calcula_erro_predicao(usuario, cluster, gu_cluster, indices_itens_avaliados_pelo_usuario)
    return erro
