# Compartilha variaveis globais entre modulos

caminho_projeto = ''
dir_dados = ''
dir_entrada_cluto = ''
dir_saida_cluto = ''
dir_entrada_slim_learn = ''
dir_entrada_slim_teste = ''
dir_saida_slim_learn = ''
dir_saida_predicoes = ''
dir_cluto = ''
nome_dataset = ''
caminho_matriz_cluto = ''
caminho_matriz_cluto = ''
dir_m_su = ''
caminho_matriz_cluto_treinamento = ''
caminho_matriz_cluto_teste = ''
slim = ''
cluto = ''
slim_predict = ''
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
historico_gu_usuario = None

# VARIAVEIS DE CONTROLE. Ex: num_usuarios, iteracoes, num_clusters, etc
modelo = ''
num_usuarios = 0
num_itens = 0
tolerancia_treinamento = 1e-5
N = 10
hit_rate = None
average_reciprocal_hit_rate = None
max_num_iteracoes = 1


def inicializa_variaveis_globais(caminho_absoluto='.'):
    global caminho_projeto, dir_dados, dir_entrada_cluto, dir_saida_cluto, dir_entrada_slim_learn, dir_saida_predicoes, \
        dir_entrada_slim_teste, dir_saida_slim_learn, dir_cluto, nome_dataset, caminho_matriz_cluto, dir_m_su, \
        caminho_matriz_cluto_treinamento, caminho_matriz_cluto_teste, slim, cluto, slim_predict, num_clusters, \
        dir_matriz_avaliacoes
    
    caminho_projeto = caminho_absoluto
    dir_dados       = caminho_projeto + "/data/"
    dir_entrada_cluto = dir_dados + "/in_cluto/"
    dir_saida_cluto = "%s/out_cluto/"%(dir_dados)
    dir_entrada_slim_learn = "%s/in_slim/"%(dir_dados)
    dir_entrada_slim_teste = "%s/in_slim_test"%(dir_dados)
    dir_saida_predicoes    = "%s/predicoes"%(dir_dados)
    dir_saida_slim_learn = "%s/out_slim/"%(dir_dados)
    dir_cluto       = caminho_projeto + "/tools/cluto/"
    nome_dataset    = "jester"
    caminho_matriz_cluto = "/%s/%s.cluto.csr" %(dir_saida_cluto,nome_dataset)
    dir_m_su        = "%s/matriz_su/"%(dir_dados)              
    caminho_matriz_cluto_treinamento = "%s/tools/slim/examples/train.mat"%(caminho_projeto)
    caminho_matriz_cluto_teste = "%s/tools/slim/examples/test.mat"%(caminho_projeto)
    slim            = "%s/tools/slim/build/examples/slim_learn"%(caminho_projeto)
    cluto           = "%s/Linux/vcluster"%(dir_cluto)
    slim_predict    = 'slim_predict'
    num_clusters    = get_num_clusters()
    dir_matriz_avaliacoes = "%s/ratings/%s.csv"%(dir_dados,nome_dataset)


def get_num_clusters():
    return 5