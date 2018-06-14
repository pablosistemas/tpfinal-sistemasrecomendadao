import numpy as np

from preprocessors.shared import *
from slim.main import *
import settings


def executa_nucleo_lslimr0():
    for usuario in range(settings.num_usuarios):
        lslimr0_treinamento_usuario(usuario)
    return


def lslimr0():
    # apenas modelo local
    settings.gu = np.zeros((settings.num_usuarios))

    calcula_submatrizes_Pu_paralelizado()
    settings.novo_vetor_clusters_usuarios = settings.vetor_clusters_usuarios.tolist()
    
    estima_modelo_slim_para_todos_clusters_paralelizado()
    atualiza_estrutura_dados_su_cluster_paralelizado()

    return
