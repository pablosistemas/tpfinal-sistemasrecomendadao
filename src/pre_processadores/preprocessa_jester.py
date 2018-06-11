#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
PREPROCESSA JESTER
'''

import numpy as np
import pandas as pd
import subprocess as sbp

def normaliza_valor_avaliacao():
    shifted =  10 + jester.ix[:,1:100]
    maximo = 20
    minimo = 0
    scaled_up = 10 * (shifted - minimo)/(maximo - minimo)
    jester.ix[:,1:100] = scaled_up


# entradas com valor 99 representam null = not rated
def remove_entradas_not_rated():
    num_avaliacoes_por_usuario = jester[0]
    valor_a_ser_removido = 99
    valor_a_ser_removido_escalado = 10*(10 + valor_a_ser_removido - 0)/(20)
    jester[jester == valor_a_ser_removido_escalado] = 0
    # caso alguma entrada da primeira coluna tenha sido substituida
    jester[0] = num_avaliacoes_por_usuario

dir_raiz = "/home/bob/Documents/mestrado/sist_recomendacao/tp_final"
dir_dataset = dir_raiz + '/data/datasets/'
dir_ratings = dir_raiz + '/data/ratings/'
dir_destino_matriz_csr = dir_raiz + '/data/in_cluto/'
dir_in_slim_csr_binario = dir_raiz + '/data/in_slim/'
arq_jester = dir_dataset + 'jester-data-1.csv'

jester = pd.read_csv(arq_jester, header=None)

jester_bkp = jester.copy()

# primeira coluna apresenta o número de piadas avaliadas pelo usuario
# as proximas 100 sao as avaliações

normaliza_valor_avaliacao()

remove_entradas_not_rated()


def escreve_em_formato_csr_com_cabecalho(nome_arquivo, df, binario=True):
    arq = open("temp", 'w')
    contador_nao_nulos = 0
    
    for linha in df.index:
        num_coluna = 1
        for coluna in df.columns:
            if df.ix[linha, coluna] != 0:
                contador_nao_nulos = contador_nao_nulos + 1
                if binario:
                    arq.write("%d %d "%(num_coluna, 1))
                else:
                    arq.write("%d %f "%(num_coluna, df.ix[linha, coluna]))
            num_coluna = num_coluna + 1
        arq.write("\n")        
    arq.close()
    
    arq = open("temp", 'r')
    saida = open(nome_arquivo, 'w')
    saida.write("%d %d %d\n"%(df.shape[0], df.shape[1], contador_nao_nulos))
    saida.write(arq.read())
    saida.close()
    arq.close()
    sbp.call(["rm","temp"])
   
    
def escreve_em_formato_csr(nome_arquivo):
    arq = open("temp", 'w')
    contador_nao_nulos = 0
    
    for linha in range(jester.shape[0]):
        for coluna in range(1,jester.shape[1]):
            if jester.ix[linha, coluna] != 0:
                contador_nao_nulos = contador_nao_nulos + 1
                arq.write("%d %f "%(coluna, jester.ix[linha, coluna]))
        arq.write("\n")        
    arq.close()
    
    arq = open("temp", 'r')
    saida = open(nome_arquivo, 'w')
    saida.write("%d %d %d\n"%(jester.shape[0], jester.shape[1]-1, contador_nao_nulos))
    saida.write(arq.read())
    saida.close()
    arq.close()
    sbp.call(["rm","temp"])

def escreve_em_formato_binario_csr_sem_cabecalho(nome_arquivo, df):
    saida = open(nome_arquivo, 'w')
    
    for linha in df.index:
        for coluna in df.columns:
            if df.ix[linha, coluna] != 0:
                saida.write("%d %d "%(coluna, 1))
        saida.write("\n")        
    saida.close()
    
def escreve_em_formato_nao_binaria_csr_sem_cabecalho(nome_arquivo, df):
    saida = open(nome_arquivo, 'w')
    
    for linha in df.index:
        for coluna in df.columns:
            if df.ix[linha, coluna] != 0:
                saida.write("%d %f "%(coluna, df.at[linha, coluna]))
        saida.write("\n")        
    saida.close()    

## SUBAMOSTRA DATASET
num_amostras = 400
jester_sampled = jester.ix[:,1:].sample(n=num_amostras, axis=0)
esparsidade_dataset_sampled = float(jester_sampled[jester_sampled != 0].count().sum())/jester_sampled.size


escreve_em_formato_csr('%s/jester.R.global.bin.csr'%(dir_in_slim_csr_binario))
escreve_em_formato_binario_csr_sem_cabecalho('%s/jester.R.global.bin.csr'%(dir_in_slim_csr_binario), jester_sampled)
escreve_em_formato_nao_binaria_csr_sem_cabecalho('%s/jester.R.global.bin.csr'%(dir_in_slim_csr_binario), jester_sampled)
escreve_em_formato_csr_com_cabecalho('%s/jester.R.csr'%(dir_destino_matriz_csr), jester_sampled, False)
jester_sampled.to_csv("%s/jester.csv"%(dir_ratings), index=False, header=False)
# escreve matriz sem row labels e sem header labels
jester.ix[:,1:].to_csv("%s/jester.csv"%(dir_ratings), index=False, header=False)

''' TESTE SANIDADE PADRONIZACAO'''
'''
idxs = jester.idxmax(0,False)
for coluna in range(1,101):
    #print(jester.ix[idxs[coluna],coluna])
    if jester.ix[idxs[coluna],coluna] > 10:
        print("ERRO")     
'''        
    