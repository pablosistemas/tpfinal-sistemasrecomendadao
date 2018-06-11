#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import subprocess
import re

diretorio_dados = '/home/bob/Documents/mestrado/sist_recomendacao/tp_final/data/'
caminho_arquivo_movies_lens = '%s/datasets/ml-1m/ratings.dat'%(diretorio_dados)
diretorio_matriz_avaliacoes = '%s/ratings/'%(diretorio_dados)
diretorio_matriz_ru_em_csr = '%s/in_slim/'%(diretorio_dados)
diretorio_matriz_cluto_em_csr = '%s/in_cluto/'%(diretorio_dados)
nome_dataset = 'movielens1m'

movie_lens = pd.DataFrame(np.zeros((6040,3952)))

arquivo = open(caminho_arquivo_movies_lens, 'r')

linha = arquivo.readline()

while linha:
    matches = re.match('(\d+)::(\d+)::(\d+).*\n', linha)
    usuario_id = int(matches.group(1))
    item_id = int(matches.group(2))
    rating = float(matches.group(3))
    movie_lens.loc[usuario_id-1, item_id-1] = rating
    linha = arquivo.readline()
        
arquivo.close()

## REMOVE LINHAS COM TODOS OS VALORES ZERO
linhas_diferentes_de_zero = movie_lens.apply(sum, axis=1) != 0
movie_lens = movie_lens.loc[linhas_diferentes_de_zero,:]

## REMOVE COLUNAS COM TODOS OS VALORES ZERO
colunas_diferentes_de_zero = movie_lens.apply(sum, axis=0) != 0
movie_lens = movie_lens.loc[:,colunas_diferentes_de_zero]

## ESCREVE MATRIZ AVALIACOES EM CSV NO DIRETORIO DE RATINGS

esparsidade_dataset = float(movie_lens[movie_lens != 0].count().sum())/movie_lens.size

movie_lens.to_csv("%s/%s.csv"%(diretorio_matriz_avaliacoes, nome_dataset), header=False, index=False)

## SUBAMOSTRA DATASET
num_amostras = 400
movie_lens_sampled = movie_lens.sample(n=num_amostras, axis=0).sample(n=num_amostras, axis=1)
esparsidade_dataset_sampled = float(movie_lens_sampled[movie_lens_sampled != 0].count().sum())/movie_lens_sampled.size

## REMOVE LINHAS COM TODOS OS VALORES ZERO
linhas_diferentes_de_zero = movie_lens_sampled.apply(sum, axis=1) >= 30
movie_lens_sampled = movie_lens_sampled.loc[linhas_diferentes_de_zero,:]
    
## REMOVE COLUNAS COM TODOS OS VALORES ZERO
colunas_diferentes_de_zero = movie_lens_sampled.apply(sum, axis=0) >= 25
movie_lens_sampled.shape
movie_lens_sampled = movie_lens_sampled.loc[:,colunas_diferentes_de_zero]

movie_lens_sampled.to_csv("%s/%s.csv"%(diretorio_matriz_avaliacoes, nome_dataset), header=False, index=False)

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
    subprocess.call(["rm","temp"])

def escreve_em_formato_csr_sem_cabecalho(nome_arquivo, df):
    saida = open(nome_arquivo, 'w')
    for linha in df.index:
        num_coluna = 1
        for coluna in df.columns:
            if df.ix[linha, coluna] != 0:
                saida.write("%d %f "%(num_coluna, df.ix[linha, coluna]))
            num_coluna = num_coluna + 1
        saida.write("\n")   
    saida.close()

def escreve_em_formato_csr_sem_cabecalho_binario(nome_arquivo, df):
    saida = open(nome_arquivo, 'w')
    for linha in df.index:
        num_coluna = 1
        for coluna in df.columns:
            if df.ix[linha, coluna] != 0:
                saida.write("%d %d "%(num_coluna, 1))
            num_coluna = num_coluna + 1
        saida.write("\n")   
    saida.close()
    
escreve_em_formato_csr_sem_cabecalho_binario('%s/%s.R.global.bin.csr'%(diretorio_matriz_ru_em_csr,nome_dataset), movie_lens_sampled)
escreve_em_formato_csr_com_cabecalho('%s/%s.R.csr'%(diretorio_matriz_cluto_em_csr,nome_dataset), movie_lens_sampled)