from experiment.leave_one_out_cross_validation import *
import settings

import pandas as pd
import random
import sys
import os

    
def main():
    # reproducibilidade
    random.seed(1234)
    settings.inicializa_variaveis_globais()

    # le arquivo de avaliações NxM
    settings.arq_avaliacoes = pd.read_csv(settings.dir_matriz_avaliacoes, header=None)
    settings.arq_avaliacoes = settings.arq_avaliacoes.as_matrix()
    settings.num_usuarios, settings.num_itens = settings.arq_avaliacoes.shape
    
    for modelo in ['glslim', 'glslimr0', 'lslim', 'lslimr0']:
        # prepara arranjos de historico
        settings.modelo = modelo
        settings.historico_gu_usuario = [[] for usuario in range(settings.num_usuarios)]
        leave_one_out_cross_validation(1, modelo=modelo)

    return


if __name__ == "__main__":
    sys.path.append(os.getcwd())
    main()
