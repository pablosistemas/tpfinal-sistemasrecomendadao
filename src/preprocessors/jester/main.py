import numpy as np
import pandas as pd
import subprocess as sbp

import os
import sys


# espera objeto com pandas DataFrame
def normaliza_valor_avaliacao(df):
    shifted =  10 + df
    maximo = 20
    minimo = 0
    scaled_up = 10 * (shifted - minimo)/(maximo - minimo)
    return scaled_up


# entradas com valor 99 representam null = not rated
def remove_entradas_nao_avaliadas(df, valor_a_ser_removido=99):
    valor_a_ser_removido_escalado = 10*float(10 + valor_a_ser_removido - 0)/(20)
    df[df == valor_a_ser_removido_escalado] = 0
    return df


def le_csv(caminho):
    return pd.read_csv(caminho, header=None)


# primeira coluna apresenta o numero de piadas avaliadas por cada usuario
# as proximas 100 sao as avaliacoes
def subamostra_dataset(df, num_amostras, axis=0):
    return df.loc[:,1:].sample(n=num_amostras, axis=0)
    

def retorna_esparsidade(df):
    return float(df[df != 0].count().sum())/df.size
