import settings

import subprocess
import threading


class slim_thread(threading.Thread):
    def __init__(self, cluster, tolerancia_treinamento):
        threading.Thread.__init__(self)
        self.cluster = cluster
        self.tolerancia_treinamento = tolerancia_treinamento
    def run(self):
        estima_matriz_S_com_slim_learn(
            "%s/%s.ru.%d.bin.csr"%(settings.dir_entrada_slim_learn,settings.nome_dataset,self.cluster),
                "%s/%s.local.%d.csr"%(settings.dir_saida_slim_learn,settings.nome_dataset,self.cluster),optTol=self.tolerancia_treinamento)
        return


# Estima modelo global (matrix R completa)
def estima_modelo_slim_global():
    estima_matriz_S_com_slim_learn("%s/%s.R.global.bin.csr"%(settings.dir_entrada_slim_learn,settings.nome_dataset),
            "%s/%s.global.csr"%(settings.dir_saida_slim_learn,settings.nome_dataset),optTol=settings.tolerancia_treinamento)
    return


def estima_matriz_S_com_slim_learn(
    caminho_matriz_treinamento, 
    caminho_matriz_saida_modelo,
    coluna_inicial=0, coluna_final=None, optTol=1e-2, lambda_p=1, beta_p=5):
    parametros_bash = [settings.slim, '-train_file=%s'%(caminho_matriz_treinamento), 
                    '-optTol=%f'%(optTol),
                    '-model_file=%s'%(caminho_matriz_saida_modelo),
                    '-lambda=%f'%(lambda_p),
                    '-beta=%f'%(beta_p)]
    if coluna_final:
        parametros_bash.append("-starti=%d"%(coluna_inicial))
        parametros_bash.append("-endi=%d"%(coluna_final))
    subprocess.call(["rm", "-f", caminho_matriz_saida_modelo])
    subprocess.call(parametros_bash)
    return


# calcula modelo SLIM para cada submatriz de clusters
def estima_modelo_slim_para_todos_clusters_paralelizado():
    threads = []
    for cluster in range(settings.num_clusters):
        thread = slim_thread(cluster, settings.tolerancia_treinamento)
        thread.start()
        threads.append(thread)
    
    for thread in threads:
        thread.join()

    return


def estima_modelo_slim_para_todos_clusters():
    for cluster in range(settings.num_clusters):
        # Estima modelo global (matrix R completa)
        estima_matriz_S_com_slim_learn(
            "%s/%s.ru.%d.bin.csr"%(settings.dir_entrada_slim_learn,nome_dataset,cluster),
                "%s/%s.local.%d.csr"%(settings.dir_saida_slim_learn,nome_dataset,cluster),
                optTol=settings.tolerancia_treinamento)
    return
