from preprocessors.shared import *
from utils.main import *
import settings


def retorna_ranking(vetor_predicoes):
    return [item[0] for item in sorted(enumerate(vetor_predicoes), key=lambda x:x[1], reverse=True)]
    

def calcula_top_n(usuario, indices_itens_avaliados_pelo_usuario):
    predicao = [None for item in range(settings.num_itens)]
    indices_itens_nao_avaliados_pelo_usuario = retorna_itens_nao_avaliados_pelo_usuario(indices_itens_avaliados_pelo_usuario)
    for item in indices_itens_nao_avaliados_pelo_usuario:
        predicao[item] = calcula_predicao_usuario_item(usuario, settings.vetor_clusters_usuarios[usuario], item, settings.gu[usuario], indices_itens_avaliados_pelo_usuario)
    ranking = retorna_ranking(predicao)
    if settings.num_itens < settings.N:
        return ranking[0:settings.num_itens]
    else:
        return ranking[0:settings.N]


def calcular_predicoes_com_slim_predict(
    nome_arquivo_treinamento_r_global_ou_local,
    nome_arquivo_modelo_out_slim_global_ou_local,
    nome_arquivo_teste_sem_item_cross_validacao,
    nome_arquivo_predicoes_saida):
    subprocess.call(["rm", "-f", nome_arquivo_predicoes_saida])
    parametros_slim_predict = [
        settings.slim_predict,        
        '-train_file=%s'%(nome_arquivo_treinamento_r_global_ou_local),
        '-model_file=%s'%(nome_arquivo_modelo_out_slim_global_ou_local),
        '-test_file=%s'%(nome_arquivo_teste_sem_item_cross_validacao),
        '-pred_file=%s'%(nome_arquivo_predicoes_saida),
        '-topn=%d'%(settings.N)]

    subprocess.call(parametros_slim_predict)
    return le_matriz_em_formato_csr(nome_arquivo_predicoes_saida, (settings.num_usuarios, settings.num_itens))


def calcula_predicao_usuario_item(usuario, cluster, item, gu_cluster, indices_itens_avaliados_pelo_usuario):
    predicao = 0
    for item_avaliado in indices_itens_avaliados_pelo_usuario:
        if gu_cluster > 0:
            predicao = predicao + gu_cluster * settings.su_global[item_avaliado,item]
        predicao = predicao + (1 - gu_cluster) * settings.su_cluster[cluster][item_avaliado,item]
    
    return predicao


def calcula_hit_rate_e_average_reciprocal_hit_rate(itens_teste, utilizar_slim_predict=False):
    hit = 0
    arhr = 0
    if not utilizar_slim_predict:
        for usuario in range(settings.num_usuarios):
            indices_itens_avaliados_pelo_usuario = retorna_lista_itens_avaliados_pelo_usuario(usuario)
            top_n = calcula_top_n(usuario, indices_itens_avaliados_pelo_usuario)
            if itens_teste[usuario] in top_n:
                hit = hit + 1
                arhr = arhr + 1.0/(top_n.index(itens_teste[usuario]) + 1)
    else:
        # nome_arquivo_treinamento_r_global_ou_local
        # nome_arquivo_modelo_out_slim_global_ou_local
        # nome_arquivo_teste_sem_item_cross_validacao
        # nome_arquivo_predicoes_saida
        # se utiliza modelo local
        avaliacoes = np.zeros((settings.num_usuarios, settings.num_itens))
        if settings.su_cluster is not None:
            for cluster in range(settings.num_clusters):
                avaliacao_usuario = calcular_predicoes_com_slim_predict(
                    '%s/%s.ru.%d.bin.csr'%(settings.dir_entrada_slim_learn, settings.nome_dataset, cluster),
                    '%s/%s.local.%d.csr'%(settings.dir_saida_slim_learn, settings.nome_dataset, cluster),
                    '%s/%s.test.csr'%(settings.dir_entrada_slim_teste, settings.nome_dataset),
                    '%s/%s.pred.csr'%(settings.dir_saida_predicoes, settings.nome_dataset)
                )
                avaliacoes = avaliacoes + avaliacao_usuario
        
        if settings.modelo in ['glslim', 'glslimr0']:
            avaliacao_usuario = calcular_predicoes_com_slim_predict(
                    '%s/%s.R.global.bin.csr'%(settings.dir_entrada_slim_learn, settings.nome_dataset),
                    '%s/%s.global.csr'%(settings.dir_saida_slim_learn, settings.nome_dataset),
                    '%s/%s.test.csr'%(settings.dir_entrada_slim_teste, settings.nome_dataset),
                    '%s/%s.pred.csr'%(settings.dir_saida_predicoes, settings.nome_dataset)
                )
            avaliacoes = avaliacoes + avaliacao_usuario
        
        for usuario in range(settings.num_usuarios):
            top_n = retorna_ranking(avaliacoes[usuario,:])[:settings.N]
            if itens_teste[usuario] in top_n:
                hit = hit + 1
                arhr = arhr + 1.0/(top_n.index(itens_teste[usuario]) + 1)

    return float(hit)/settings.num_usuarios, float(arhr)/settings.num_usuarios   
 
            
# numero de clusters diferentes na iteracao
def retorna_percentual_mudancas():
    mudanca_cluster = 0
    for usuario in range(settings.num_usuarios):
        if settings.vetor_clusters_usuarios[usuario] != settings.novo_vetor_clusters_usuarios[usuario]:
            mudanca_cluster = mudanca_cluster + 1
    return float(mudanca_cluster)/settings.num_usuarios


def calcula_erro_predicao(usuario, cluster, gu_cluster, indices_itens_avaliados_pelo_usuario):
    erro = 0
    indices_itens_nao_avaliados_pelo_usuario = retorna_itens_nao_avaliados_pelo_usuario(indices_itens_avaliados_pelo_usuario)
    for item in indices_itens_nao_avaliados_pelo_usuario:
    #for item in range(num_itens):
        predicao = calcula_predicao_usuario_item(usuario, cluster, item, gu_cluster, indices_itens_avaliados_pelo_usuario)
        #erro = erro + (arq_avaliacoes[usuario,item] - predicao) ** 2
        erro = erro + (settings.R_global[usuario,item] - predicao) ** 2
    
    erro = erro/settings.num_itens
    return erro

    
def calcula_gu(usuario, cluster, indices_itens_avaliados_pelo_usuario):
    # u eh o usuario da iteracao
    num = den = 0
    for item in range(settings.num_itens):
        soma_sli = 0
        soma_spuli = 0
        for item_similar in indices_itens_avaliados_pelo_usuario:
            soma_sli = soma_sli + settings.su_global[item,item_similar]
            soma_spuli = soma_spuli + settings.su_cluster[cluster][item,item_similar]
            
        num = num + (soma_sli - soma_spuli) * (settings.R_global[usuario,item] - soma_spuli)
        den = den + (soma_sli - soma_spuli)**2

    # evita NaN
    if den == 0.0:
        return 0.0
    # casos triviais onde o soma modelo local eh 0
    if soma_spuli == 0.0:
        return 1.0

    gu_cluster = num/den

    arq = open('%s/resultados/distribuicao_gu'%(settings.dir_dados),'a')
    arq.write('%f\n'%(gu_cluster))
    arq.close()

    # em alguns casos, gu sai do intervalo [0,1], work-around:
    gu_cluster = max(0, gu_cluster)
    gu_cluster = min(1, gu_cluster)
    return gu_cluster


def slim_predict():
    pass


'''
# FIXME:
def avalia_modelo():
    global su_global
    for usuario in range(num_usuarios):
        indices_itens_avaliados_pelo_usuario = retorna_lista_itens_avaliados_pelo_usuario(usuario)
        itens_treino, item_teste = divide_itens_treino_teste(indices_itens_avaliados_pelo_usuario)
        # Ajusta matriz R_global para teste
        R_global[usuario, item_teste] = 0

        estima_modelo_slim_global()        
        su_global = le_matriz_em_formato_csr("%s/%s.global.csr"%(dir_saida_slim_learn,nome_dataset),(num_itens,num_itens))

        estima_modelo_slim_para_todos_clusters_paralelizado()
        atualiza_estrutura_dados_su_cluster_paralelizado()

        # top_n = calcula_top_n(usuario, indices_itens_avaliados_pelo_usuario)

        # desfaz ajuste matriz R_global
        R_global.ix[usuario, item_teste] = 1
'''