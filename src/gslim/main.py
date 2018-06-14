from preprocessors.shared import *
from evaluation.main import *
from utils.main import *
from slim.main import *
import settings


def gslim_treinamento_usuario(usuario):
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


def gslim():
    # contribuicao dos modelos local e global iguais para todos os usuarios no inicio do treinamento
    settings.gu = np.ones((settings.num_usuarios))

    estima_modelo_slim_global()        
    settings.su_global = le_matriz_em_formato_csr(
        "%s/%s.global.csr"%(settings.dir_saida_slim_learn,settings.nome_dataset),
        (settings.num_itens,settings.num_itens))

    #executa_nucleo_gslim()

    # atualiza vetor de usuario/cluster
    settings.vetor_clusters_usuarios = pd.Series(settings.novo_vetor_clusters_usuarios)
    print('Percentual de mudancas: %f\n'%(percentual_mudancas))
    print('Numero iteracoes: %d\n'%(num_iteracoes))

    return
