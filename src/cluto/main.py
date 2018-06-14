import settings

import pandas as pd
import subprocess


def calcula_clusters_com_cluto(caminho_matriz_cluto, nome_arq_saida_cluto, num_clusters):
    subprocess.call(["rm", "-f", nome_arq_saida_cluto])
    subprocess.call([settings.cluto, caminho_matriz_cluto, str(num_clusters), "-clustfile=" + nome_arq_saida_cluto, "-colmodel=none"])
    settings.vetor_clusters_usuarios = pd.read_csv(nome_arq_saida_cluto, header=None)[0]
    return