r"""
Cria e divide um conjunto de dados em subconjuntos de treino,
validação e teste para detecção de objetos de acordo com o padrão COCO.

- A criação é feita a partir de arquivos json no formato padrão do
LabelMe v4.5.5:
    {
       "shapes": [
           {
               "label": str,
               "points": [[float, float],
                          [float, float]],
               "group_id": null,
               "shape_type": "rectangle",
               "flags": {}
           }
       ],
       "imagePath": str,
       "imageData": null,
       "imageHeight": int,
       "imageWidth": int
    }
- A divisão é feita de forma aleatória. Para a estratificação é
necessário verificar e confirmar a divisão dos dados.

Referências:
LabelMe v4.5.5: https://github.com/wkentaro/labelme/releases/tag/v4.5.5
"""

import argparse
import copy
import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ID_INICIAL_IMAGEM = 1
ID_INICIAL_ANOTACAO = 1
ID_INICIAL_CATEGORIA = 1


def parse_args():
    args = argparse.ArgumentParser(
        description='Cria e divide um conjunto de dados de acordo com '
                    'o padrão COCO. A divisão é feita de forma '
                    'aleatória e estratificada.')
    args.add_argument('dir_anotacoes',
                      help='Diretório para as anotações.')
    args.add_argument('--percentuais',
                      default=[0.7, 0.15, 0.15], nargs='+',
                      help='Percentual das divisões dos conjuntos de '
                           'treino, validação e teste. Valores '
                           'padrões: 0.7, 0.15 e 0.15.')
    args.add_argument('--dir_saida',
                      default='./dataset',
                      help='Diretório para salvar os conjuntos '
                           'divididos. Se não especificado, '
                           'os arquivos serão salvos em ./dataset')
    args.add_argument('--seed',
                      default=None, type=int,
                      help='Semente para garantir que a divisão dos '
                           'dados sempre será igual. Se não '
                           'especificada, a seed irá variar de 0 até '
                           'n, com passo 1, até que a divisão ideal '
                           'seja encontrada.')
    args.add_argument('--mostrar_graficos', action='store_true',
                      help='Mostra os gráficos referentes às '
                           'estatísticas do conjunto de dados.')
    return args.parse_args()


def listar_jsons(dir_anotacoes: str) -> list:
    """
    Busca arquivos json em um diretório. A busca é feita apenas dentro
    do diretório informado e não em subpastas.

    :param dir_anotacoes: Diretório para buscar os arquivos json
    :return: Lista contendo o nome dos arquivos
    """

    arquivos = os.listdir(dir_anotacoes)
    jsons = [f for f in arquivos if f.endswith('.json')]
    return jsons


def carregar_categorias(dir_anotacoes: str,
                        id_inicial_categoria: int) -> list:
    """
    Carrega todas as categorias do conjunto de dados.

    :param dir_anotacoes: Diretório para buscar os arquivos json
    :param id_inicial_categoria: Número identificador para a
    primeira categoria
    :return: Lista contendo um dicionário com as informações de
    cada categoria
    """

    jsons = listar_jsons(dir_anotacoes)
    conjunto_categorias = set()
    for j in jsons:
        dir_json = os.path.join(dir_anotacoes, j)
        dados = carregar_json(dir_json)
        for shape in dados['shapes']:
            conjunto_categorias.add(shape['label'])
    categorias_ordenadas = sorted(list(conjunto_categorias))

    lista_categorias = list()
    for categoria in categorias_ordenadas:
        lista_categorias.append(dict(
            id=id_inicial_categoria,
            name=categoria))
        id_inicial_categoria += 1
    return lista_categorias


def carregar_json(dir_json: str) -> dict:
    """
    Carrega os dados de um arquivo json.

    :param dir_json: Diretório para o arquivo json a ser carregado
    :return: Dicionário com os dados do arquivo json
    """

    with open(dir_json, 'r') as arquivo:
        dados = json.load(arquivo)
    return dados


def buscar_identificador_categoria(nome: str, categorias: list) -> int:
    """
    Busca uma categoria por nome para saber o número do seu
    identificador.

    :param nome: Nome da categoria
    :param categorias: Lista de categorias
    :return: Número identificador da categoria
    """

    for categoria in categorias:
        if categoria['name'] == nome:
            return categoria['id']


def criar_dataset(dir_anotacoes: str) -> dict:
    """
    Cria um conjunto de dados no padrão COCO.

    :param dir_anotacoes: Diretório para buscar os arquivos json
    :return: Dicionário contendo as imagens, anotações e categorias
    do conjunto de dados no formato COCO
    """

    id_imagem = ID_INICIAL_IMAGEM
    id_anotacao = ID_INICIAL_ANOTACAO

    jsons = listar_jsons(dir_anotacoes)
    categorias = carregar_categorias(dir_anotacoes,
                                     ID_INICIAL_CATEGORIA)
    dataset = dict(images=list(), annotations=list(), categories=list())
    for j in jsons:
        json_path = os.path.join(dir_anotacoes, j)
        dados = carregar_json(json_path)

        imagem = formatar_dados_imagem(dados, id_imagem)
        dataset['images'].append(imagem)

        for shape in dados['shapes']:
            anotacao = formatar_dados_anotacao(imagem, id_anotacao,
                                               categorias, shape)
            dataset['annotations'].append(anotacao)
            id_anotacao += 1
        id_imagem += 1
    dataset['categories'] = categorias
    return dataset


def formatar_dados_anotacao(imagem: dict, id_anotacao: int,
                            categorias: list, shape: dict) -> dict:
    """
    Formata os dados de uma anotação para o padrão COCO.

    :param imagem: Imagem que contém a anotação
    :param id_anotacao: Número que será usado para identificar a
    anotação
    :param categorias: Lista de categorias
    :param shape: Dicionário com informações do nome da categoria e
    dos pontos da anotação
    :return: Dicionário contendo informações da anotação no padrão COCO
    """

    nome_categoria = shape['label']
    pontos = shape['points']
    xmin = int(min(pontos[0][0], pontos[1][0]))
    ymin = int(min(pontos[0][1], pontos[1][1]))
    xmax = int(max(pontos[0][0], pontos[1][0]))
    ymax = int(max(pontos[0][1], pontos[1][1]))

    # processamento para evitar anotações fora da imagem
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(xmax, imagem['width'])
    ymax = min(ymax, imagem['height'])

    assert xmin >= 0, 'Valor negativo para xmin'
    assert ymin >= 0, 'Valor negativo para ymin'
    assert xmax >= 0, 'Valor negativo para xmax'
    assert ymax >= 0, 'Valor negativo para ymax'

    anotacao = dict(
        id=id_anotacao,
        image_id=imagem['id'],
        category_id=buscar_identificador_categoria(nome_categoria,
                                                   categorias),
        bbox=[xmin, ymin, xmax, ymax],
        area=(xmax - xmin) * (ymax - ymin),
        iscrowd=0
    )
    return anotacao


def formatar_dados_imagem(dados: dict, id_imagem: int) -> dict:
    """
    Formata os dados de uma imagem para o padrão COCO.

    :param dados: Dados do arquivo json que contém as informações de
    uma imagem do conjunto de dados
    :param id_imagem: Número que será usado para identificar a imagem
    :return: Dicionário contendo informações da imagem no padrão COCO
    """

    imagem = dict(
        id=id_imagem,
        file_name=dados['imagePath'],
        height=dados['imageHeight'],
        width=dados['imageWidth'])
    return imagem


def dividir_dataset(dataset: dict, divisoes: list, seed: int) -> (dict,
                                                                  dict,
                                                                  dict):
    """
    Divide o conjunto de dados de acordo com as divisões informadas.

    :param dataset: Conjunto de dados no padrão COCO a ser dividido
    :param divisoes: Lista contendo os percentuais (números entre 0 e 1)
    de cada divisão
    :param seed: Semente para embaralhar os dados
    :return: Dicionário contendo os conjuntos de treino, validação e
    teste
    """

    random.Random(seed).shuffle(dataset['images'])

    assert len(divisoes) == 3, 'Por favor, informe valores de ' \
                               'divisão para os conjuntos de treino, ' \
                               'validação e teste'

    treino = dict(images=list(), annotations=list(),
                  categories=dataset['categories'])
    validacao = dict(images=list(), annotations=list(),
                     categories=dataset['categories'])
    teste = dict(images=list(), annotations=list(),
                 categories=dataset['categories'])

    quantidade_imagens = len(dataset['images'])
    quantidade_imagens_treino = int(quantidade_imagens * float(divisoes[
                                                                   0]))
    quantidade_imagens_validacao = int(quantidade_imagens * float(
        divisoes[1]))

    imagens = dataset['images']
    anotacoes = copy.deepcopy(dataset['annotations'])

    treino['images'] = imagens[:quantidade_imagens_treino]
    validacao['images'] = imagens[
                          quantidade_imagens_treino:quantidade_imagens_treino + quantidade_imagens_validacao]
    teste['images'] = imagens[
                      quantidade_imagens_treino + quantidade_imagens_validacao:]

    identificadores_imagens_treino = [imagem['id'] for imagem in treino[
        'images']]
    identificadores_imagens_validacao = [imagem['id'] for imagem in
                                         validacao['images']]

    while anotacoes:
        if anotacoes[0]['image_id'] in identificadores_imagens_treino:
            treino['annotations'].append(anotacoes.pop(0))
        elif anotacoes[0][
            'image_id'] in identificadores_imagens_validacao:
            validacao['annotations'].append(anotacoes.pop(0))
        else:
            teste['annotations'].append(anotacoes.pop(0))
    return treino, validacao, teste


def verificar_tamanho(tamanho: int) -> str:
    """
    Calcula a raiz quadrada do tamanho de uma anotação e retorna se é
    pequena, média ou grande, de acordo com o padrão COCO.

    :param tamanho: tamanho (altura * largura) de uma anotação
    :return: String representando o tamanho da anotação
    """

    raiz = tamanho ** 0.5
    if raiz < 32:
        return 'pequeno'
    elif 32 <= raiz < 96:
        return 'medio'
    return 'grande'


def mostrar_informacoes(treino: dict, validacao: dict, teste: dict):
    """
    Imprime no terminal as informações das divisões do conjunto de
    dados.

    :param treino: Conjunto de treino
    :param validacao: Conjunto de validacao
    :param teste: Conjunto de teste
    :return: Informações do conjunto de dados
    """

    quantidade_imagens_treino = len(treino['images'])
    quantidade_anotacoes_treino = len(treino['annotations'])
    quantidade_imagens_validacao = len(validacao['images'])
    quantidade_anotacoes_validacao = len(validacao['annotations'])
    quantidade_imagens_teste = len(teste['images'])
    quantidade_anotacoes_teste = len(teste['annotations'])

    total_imagens = quantidade_imagens_treino + \
                    quantidade_imagens_validacao + quantidade_imagens_teste
    df_imagens = pd.DataFrame(
        data=[[quantidade_imagens_treino, quantidade_imagens_validacao,
               quantidade_imagens_teste,
               total_imagens,
               round(quantidade_imagens_treino / total_imagens, 2),
               round(quantidade_imagens_validacao / total_imagens, 2),
               round(quantidade_imagens_teste / total_imagens, 2),
               round(total_imagens / total_imagens, 2)]],
        columns=['treino', 'validacao', 'teste', 'total', 'treino %',
                 'validacao %', 'teste %', 'total %'],
        index=['imagens']
    )
    print('O conjunto de dados possui:')
    print(total_imagens, 'imagens')
    print(df_imagens)
    print()

    categorias = treino['categories']
    print(len(categorias), 'categorias')
    print([categoria['name'] for categoria in categorias])
    print()

    dados = np.zeros((len(categorias), 8))
    informacoes_dataset = pd.DataFrame(
        data=dados,
        columns=['treino', 'validacao', 'teste', 'total', 'treino %',
                 'validacao %', 'teste %', 'total %'],
        index=[categoria['name'] for categoria in categorias]
    )

    tamanhos_geral = dict(pequeno=0, medio=0, grande=0)
    tamanhos_conjunto = dict(
        treino=dict(pequeno=0, medio=0, grande=0),
        validacao=dict(pequeno=0, medio=0, grande=0),
        teste=dict(pequeno=0, medio=0, grande=0)
    )

    # contagem das anotações do conjunto de dados.
    # o decremento ocorre para o índice da categoria ser igual ao
    # índice do dataframe de informações do dataset.
    # desta maneira o valor é incrementado de acordo com o conjunto e
    # a categoria
    for anotacao in treino['annotations']:
        id_categoria = anotacao['category_id'] - 1
        informacoes_dataset['treino'][id_categoria] += 1
        informacoes_dataset['total'][id_categoria] += 1
        tamanho = verificar_tamanho(anotacao['area'])
        tamanhos_geral[tamanho] += 1
        tamanhos_conjunto['treino'][tamanho] += 1
    for anotacao in validacao['annotations']:
        id_categoria = anotacao['category_id'] - 1
        informacoes_dataset['validacao'][id_categoria] += 1
        informacoes_dataset['total'][id_categoria] += 1
        tamanho = verificar_tamanho(anotacao['area'])
        tamanhos_geral[tamanho] += 1
        tamanhos_conjunto['validacao'][tamanho] += 1
    for anotacao in teste['annotations']:
        id_categoria = anotacao['category_id'] - 1
        informacoes_dataset['teste'][id_categoria] += 1
        informacoes_dataset['total'][id_categoria] += 1
        tamanho = verificar_tamanho(anotacao['area'])
        tamanhos_geral[tamanho] += 1
        tamanhos_conjunto['teste'][tamanho] += 1

    for indice in range(len(informacoes_dataset.index)):
        n_train = informacoes_dataset['treino'][indice]
        n_val = informacoes_dataset['validacao'][indice]
        n_test = informacoes_dataset['teste'][indice]
        n_total = informacoes_dataset['total'][indice]

        informacoes_dataset['treino %'][indice] = round(n_train /
                                                        n_total, 2)
        informacoes_dataset['validacao %'][indice] = round(n_val /
                                                           n_total, 2)
        informacoes_dataset['teste %'][indice] = round(n_test /
                                                       n_total, 2)
        informacoes_dataset['total %'][indice] = round(
            (n_train + n_val + n_test) / n_total, 2)

    # conversão dos dados tipo float para int
    informacoes_dataset[['treino', 'validacao', 'teste',
                         'total']] = informacoes_dataset[['treino',
                                                          'validacao',
                                                          'teste',
                                                          'total']].astype(
        int)

    total_anotacoes = quantidade_anotacoes_treino + \
                      quantidade_anotacoes_validacao + quantidade_anotacoes_teste
    anotacoes = pd.DataFrame(
        data=[
            [quantidade_anotacoes_treino,
             quantidade_anotacoes_validacao,
             quantidade_anotacoes_teste, total_anotacoes,
             round(quantidade_anotacoes_treino / total_anotacoes, 2),
             round(quantidade_anotacoes_validacao / total_anotacoes,
                   2),
             round(quantidade_anotacoes_teste / total_anotacoes, 2),
             round((quantidade_anotacoes_treino +
                    quantidade_anotacoes_validacao +
                    quantidade_anotacoes_teste) / total_anotacoes,
                   2)]],
        columns=['treino', 'validacao', 'teste', 'total', 'treino %',
                 'validacao %', 'teste %', 'total %'],
        index=['anotacoes'],

    )

    print('Anotações por conjunto e categoria')
    print(informacoes_dataset)
    print()
    print(total_anotacoes, 'anotações')
    print(anotacoes)
    print()
    print('Percentual de anotaçãoes por tamanho:')
    print('Pequeno:', round(tamanhos_geral['pequeno'] / total_anotacoes, 2))
    print('Médio:', round(tamanhos_geral['medio'] / total_anotacoes, 2))
    print('Grande:', round(tamanhos_geral['grande'] / total_anotacoes, 2))
    print()
    print('Tamanho por conjunto:')
    print('Treino - pequeno:', round(tamanhos_conjunto['treino'][
                                         'pequeno'] /
                                     len(
        treino['annotations']), 2))
    print('Treino - médio:', round(tamanhos_conjunto['treino']['medio'] / len(
        treino['annotations']), 2))
    print('Treino - grande:', round(tamanhos_conjunto['treino']['grande'] / len(
        treino['annotations']), 2))
    print('Validação - pequeno:', round(tamanhos_conjunto['validacao'][
                                            'pequeno'] /
                                        len(
        validacao['annotations']), 2))
    print('Validação - médio:', round(tamanhos_conjunto['validacao']['medio'] / len(
        validacao['annotations']), 2))
    print('Validação - grande:', round(tamanhos_conjunto['validacao']['grande'] / len(
        validacao['annotations']), 2))
    print('Teste - pequeno:', round(tamanhos_conjunto['teste']['pequeno'] /
                                    len(
        teste['annotations']), 2))
    print('Teste - médio:', round(tamanhos_conjunto['teste']['medio'] / len(
        teste['annotations']), 2))
    print('Teste - grande:', round(tamanhos_conjunto['teste']['grande'] / len(
        teste['annotations']), 2))
    print()

    return informacoes_dataset


def verificar_divisoes(divisoes: tuple) -> bool:
    """
    Pergunta ao usuário se as informações do conjunto de dados estão
    de acordo com as divisões solicitadas.

    :param divisoes: Tupla com as divisões do conjunto de dados
    :return: bool
    """

    informacoes_dataset = mostrar_informacoes(divisoes[0], divisoes[1],
                                              divisoes[2])

    opcao = ''
    while opcao != 's':
        opcao = input('Você concorda com as divisões realizadas? '
                      'Digite s para confirmar ou n para realizar '
                      'uma nova divisão e tecle ENTER\n')

        if opcao == 's':
            return True, informacoes_dataset
        elif opcao == 'n':
            return False, None
        else:
            print('Opção inválida. Tente novamente.')


def salvar_dataset(divisoes: tuple, dir_saida: str):
    """
    Salva as divisões de treino, validação e teste em arquivos jsons
    no diretório de saída informado.

    :param divisoes: Arquivos de treino, validação e teste (na ordem)
    :param dir_saida: Diretório para salvar os arquivos
    """

    dir_treino = os.path.join(dir_saida, 'treino.json')
    dir_validacao = os.path.join(dir_saida, 'validacao.json')
    dir_teste = os.path.join(dir_saida, 'teste.json')

    with open(dir_treino, 'w') as arquivo_treino, open(dir_validacao,
                                                       'w') as \
            arquivo_validacao, open(dir_teste, 'w') as arquivo_teste:
        json.dump(divisoes[0], arquivo_treino)
        json.dump(divisoes[1], arquivo_validacao)
        json.dump(divisoes[2], arquivo_teste)
        print('Divisões do conjunto de dados salvas em', dir_saida)


def anotacoes_por_categoria(informacoes_dataset):
    """
    Exibe um gráfico de barras contendo o nome das categorias no eixo x
    e a quantidade de categorias no eixo y.

    :param informacoes_dataset: Dataframe com as informações do
    conjunto de dados
    """

    categorias = list(informacoes_dataset.index)
    contador_treino = np.zeros(len(categorias))
    contador_validacao = np.zeros(len(categorias))
    contador_teste = np.zeros(len(categorias))
    for categoria in categorias:
        id_categoria = categorias.index(categoria)
        contador_treino[id_categoria] = informacoes_dataset[
            'treino'][categoria]
        contador_validacao[id_categoria] = informacoes_dataset[
            'validacao'][categoria]
        contador_teste[id_categoria] = informacoes_dataset[
            'teste'][categoria]

    eixo_x = range(len(categorias))
    plt.bar(eixo_x, contador_treino)
    plt.bar(eixo_x, contador_validacao, bottom=contador_treino)
    plt.bar(eixo_x, contador_teste,
            bottom=contador_treino + contador_validacao)
    # plt.xticks(eixo_x, categorias, rotation=45)
    plt.xticks(eixo_x, range(1, len(categorias) + 1))
    plt.xlabel('Id. Categoria')
    plt.ylabel('Quantidade de anotações')
    plt.legend(['Treino', 'Validação', 'Teste'])
    plt.grid(linestyle='-', axis='y', alpha=0.7, linewidth=3)
    plt.show()


def categorias_por_imagem(divisoes):
    """
    Exibe um gráfico de linhas contendo a quantidade de categorias no
    eixo x e o percentual de imagens no eixo y.

    :param divisoes: Divisões de treino, validação e teste do
    conjunto de dados
    """

    treino = divisoes[0]
    validacao = divisoes[1]
    teste = divisoes[2]
    categorias = treino['categories']

    imagens_treino = dict()
    for anotacao in treino['annotations']:
        id_imagem = anotacao['image_id']
        id_categoria = anotacao['category_id']

        if id_imagem not in imagens_treino.keys():
            imagens_treino[id_imagem] = set()
        imagens_treino[id_imagem].add(id_categoria)

    imagens_validacao = dict()
    for anotacao in validacao['annotations']:
        id_imagem = anotacao['image_id']
        id_categoria = anotacao['category_id']

        if id_imagem not in imagens_validacao.keys():
            imagens_validacao[id_imagem] = set()
        imagens_validacao[id_imagem].add(id_categoria)

    imagens_teste = dict()
    for anotacao in teste['annotations']:
        id_imagem = anotacao['image_id']
        id_categoria = anotacao['category_id']

        if id_imagem not in imagens_teste.keys():
            imagens_teste[id_imagem] = set()
        imagens_teste[id_imagem].add(id_categoria)

    contador_treino = np.zeros(len(categorias))
    contador_validacao = np.zeros(len(categorias))
    contador_teste = np.zeros(len(categorias))

    for imagem in imagens_treino.keys():
        contador_treino[len(imagens_treino[imagem]) - 1] += 1
    for imagem in imagens_validacao.keys():
        contador_validacao[len(imagens_validacao[imagem]) - 1] += 1
    for imagem in imagens_teste.keys():
        contador_teste[len(imagens_teste[imagem]) - 1] += 1

    quantidade_categorias = list(range(1, len(categorias) + 1))
    plt.plot(quantidade_categorias, contador_treino / len(
        imagens_treino.keys()), linewidth=3,
             marker='.', markersize=15)
    plt.plot(quantidade_categorias, contador_validacao / len(
        imagens_validacao.keys()), linestyle='--', linewidth=3,
             marker='.', markersize=15)
    plt.plot(quantidade_categorias, contador_teste / len(
        imagens_teste.keys()), linestyle=':', linewidth=3,
             marker='.', markersize=15)

    plt.grid(linestyle='-', axis='y', alpha=0.7, linewidth=3)
    plt.xlabel('Quantidade de categorias')
    plt.ylabel('Percentual de imagens')
    plt.legend(['Treino', 'Validação', 'Teste'])
    plt.show()


def anotacoes_por_imagem(divisoes):
    """
    Exibe um gráfico de linhas contendo a quantidade de anotações no
    eixo x e o percentual de imagens no eixo y.

    :param divisoes: Divisões de treino, validação e teste do
    conjunto de dados
    """

    treino = divisoes[0]
    validacao = divisoes[1]
    teste = divisoes[2]

    imagens_treino = dict()
    for anotacao in treino['annotations']:
        id_imagem = anotacao['image_id']
        if id_imagem not in imagens_treino.keys():
            imagens_treino[id_imagem] = 0
        imagens_treino[id_imagem] += 1

    imagens_validacao = dict()
    for anotacao in validacao['annotations']:
        id_imagem = anotacao['image_id']
        if id_imagem not in imagens_validacao.keys():
            imagens_validacao[id_imagem] = 0
        imagens_validacao[id_imagem] += 1

    imagens_teste = dict()
    for anotacao in teste['annotations']:
        id_imagem = anotacao['image_id']
        if id_imagem not in imagens_teste.keys():
            imagens_teste[id_imagem] = 0
        imagens_teste[id_imagem] += 1

    eixo_x = list(range(1, max(max(imagens_treino.values()),
                               max(imagens_validacao.values()),
                               max(imagens_teste.values())) + 1))

    contador_treino = np.zeros(len(eixo_x))
    contador_validacao = np.zeros(len(eixo_x))
    contador_teste = np.zeros(len(eixo_x))

    for imagem in imagens_treino.keys():
        contador_treino[imagens_treino[imagem] - 1] += 1
    for imagem in imagens_validacao.keys():
        contador_validacao[imagens_validacao[imagem] - 1] += 1
    for imagem in imagens_teste.keys():
        contador_teste[imagens_teste[imagem] - 1] += 1

    plt.plot(eixo_x, contador_treino / len(
        imagens_treino.keys()), linewidth=3,
             marker='.', markersize=15)
    plt.plot(eixo_x, contador_validacao / len(
        imagens_validacao.keys()), linestyle='--', linewidth=3,
             marker='.', markersize=15)
    plt.plot(eixo_x, contador_teste / len(
        imagens_teste.keys()), linestyle=':', linewidth=3,
             marker='.', markersize=15)

    plt.grid(linestyle='-', axis='y', alpha=0.7, linewidth=3)
    plt.xlabel('Quantidade de anotações')
    plt.ylabel('Percentual de imagens')
    plt.xticks(eixo_x)
    plt.legend(['Treino', 'Validação', 'Teste'])
    plt.show()


def box_plot(divisoes):
    """
    Exibe um gráfico de caixas contendo as informações de altura,
    largura e área para cada classe.

    :param divisoes: Divisões de treino, validação e teste do
    conjunto de dados
    """

    categorias = divisoes[0]['categories']
    anotacoes = dict()
    for divisao in divisoes:
        for anotacao in divisao['annotations']:
            id_anotacao = anotacao['id']
            id_categoria = anotacao['category_id']
            xmin, ymin, xmax, ymax = anotacao['bbox']
            altura = ymax - ymin
            largura = xmax - xmin
            area = (altura * largura) ** 0.5

            anotacoes[id_anotacao] = dict(
                categoria=id_categoria,
                altura=altura,
                largura=largura,
                area=area
            )

    df_anotacoes = pd.DataFrame().from_dict(anotacoes,
                                            orient='index', columns=[
            'categoria', 'altura', 'largura', 'area'])

    print()
    print('Informações gerais')
    print(df_anotacoes[['altura', 'largura', 'area']].describe())

    fig, axs = plt.subplots(nrows=1, ncols=3)
    titulos = ['altura', 'largura', 'area']
    for ax, titulo in zip(axs, titulos):
        anotacoes_categorias = list()
        nomes_categorias = list()
        for categoria in categorias:
            id_categoria = categoria['id']
            nomes_categorias.append(categoria['name'])
            anotacoes_categorias.append(df_anotacoes[df_anotacoes[
                                                         'categoria'] == id_categoria][
                                            titulo])
            print()
            print('Informações:', titulo)
            print('Categoria:', nomes_categorias[-1])
            print(anotacoes_categorias[-1].describe())
        ax.boxplot(anotacoes_categorias)
        ax.grid(linestyle='-', axis='y', alpha=0.7, linewidth=3)
        ax.set_xticks(list(range(1, len(nomes_categorias) + 1)))
        # ax.set_xticklabels(nomes_categorias, rotation=45)
        ax.set_xticklabels(range(1, len(nomes_categorias) + 1))
        ax.set_title(titulo)
    fig.text(0.5, 0.01, 'Id. Categoria', ha='center')
    fig.text(0.01, 0.5, 'Pixels', va='center', rotation='vertical')
    plt.show()


def main():
    args = parse_args()

    if not os.path.exists(args.dir_saida):
        os.makedirs(args.dir_saida)

    dataset = criar_dataset(args.dir_anotacoes)
    with open(os.path.join(args.dir_saida, 'dataset.json'), 'w') as \
            json_file:
        json.dump(dataset, json_file)

    seed = args.seed
    if seed is None:
        seed = 0

    resposta = False
    while not resposta:
        print('seed:', seed)
        divisoes = dividir_dataset(copy.deepcopy(dataset),
                                   args.percentuais, seed)
        resposta, informacoes_dataset = verificar_divisoes(divisoes)
        if not resposta:
            seed += 1
    salvar_dataset(divisoes, args.dir_saida)

    print('Seed de seleção dos dados:', seed)

    if args.mostrar_graficos:
        plt.rcParams.update({'font.size': 22})
        plt.rc('axes', axisbelow=True)
        anotacoes_por_categoria(informacoes_dataset)
        categorias_por_imagem(divisoes)
        anotacoes_por_imagem(divisoes)
        box_plot(divisoes)


if __name__ == '__main__':
    main()