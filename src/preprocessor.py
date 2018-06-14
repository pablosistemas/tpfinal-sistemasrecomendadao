from preprocessors.movielens100k.main import *
from preprocessors.jester.main import *
from preprocessors.shared import *
from settings import *

def jester():
    dir_raiz = "/home/bob/Documents/mestrado/sist_recomendacao/tp_final"
    dir_dataset = dir_raiz + '/data/datasets/'
    dir_ratings = dir_raiz + '/data/ratings/'
    dir_destino_matriz_csr = dir_raiz + '/data/in_cluto/'
    dir_in_slim_csr_binario = dir_raiz + '/data/in_slim/'
    arq_jester = dir_dataset + 'jester-data-1.csv'

    df = le_csv(arq_jester)
    # descarta primeira coluna q contem numero de piadas avaliadas por usuario
    df = df.loc[:,1:]
    df = normaliza_valor_avaliacao(df)
    df = remove_entradas_nao_avaliadas(df)
    df_subamostrado = subamostra_dataset(df, 500, 0)

    escreve_em_formato_csr('%s/jester.R.global.bin.csr'%(dir_in_slim_csr_binario), df_subamostrado.as_matrix(), binario=True, cabecalho=False)
    escreve_em_formato_csr('%s/jester.R.csr'%(dir_destino_matriz_csr), df_subamostrado.as_matrix(), binario=True, cabecalho=True)
    
    # escreve dataframe de avaliacoes diretamente no arquivo
    df_subamostrado.to_csv("%s/jester.csv"%(dir_ratings), index=False, header=False)
    return
    

def main():
    jester()


if __name__ == "__main__":
    main()