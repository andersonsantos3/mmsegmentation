"""
A etapa de divisão é responsável por criar subimagens das imagens de
um conjunto de dados no padrão COCO, garantindo que as anotações
divididas não sejam perdidas.

Código para calcular intersecção extraído de:
https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
12/04/2021
"""


import argparse
import json
import os
from copy import deepcopy


def arg_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'dir_dataset',
        help='Diretório em que os arquivos treino.json, validacao.json e teste.json estão armazenados'
    )
    parser.add_argument(
        'n_subimagens',
        help='Divide as imagens em n_subimagens*n_subimagens subimagens',
        type=int,
        default=1
    )
    parser.add_argument(
        '--diretorio_subimagens',
        help='Diretório para salvar as subimagens. Serão salvas apenas se o diretório for informado',
        type=str,
        default=''
    )

    args = parser.parse_args()
    return args


def carregar_json(diretorio: str) -> dict:
    """
    Carrega um arquivo json e retorna seus dados

    :param diretorio: Diretório do arquivo json a ser carregado
    :return: Dicionário com os dados do arquivo json
    """

    with open(diretorio, 'r') as arquivo_json:
        dados = json.load(arquivo_json)
    return dados


def calcular_intervalos_subimagens(altura_imagem: int, largura_imagem: int, n_subimagens: int) -> list:
    """
    Calcula os intervalos das divisões para realizar os recortes na imagem e gerar as subimagens

    :param altura_imagem: Altura (em pixels) da imagem a ser dividida
    :param largura_imagem: Largura (em pixels) da imagem a ser dividida
    :param n_subimagens: Número de divisões, para altura e largura, da imagem a ser dividida
    :return: Lista com as coordenadas das subimagens criadas
    """

    # o limite do range é configurado com + 1 para incluir o valor limite (último pixel)
    intervalos_altura = list(range(0, altura_imagem + 1, altura_imagem // n_subimagens))
    intervalos_largura = list(range(0, largura_imagem + 1, largura_imagem // n_subimagens))

    # caso o último valor da lista de altura ou largura for diferente da altura ou largura da imagem, este valor é
    # substituído pela altura ou largura da imagem
    if intervalos_largura[-1] != largura_imagem:
        intervalos_largura.pop(-1)
        intervalos_largura.append(largura_imagem)
    if intervalos_altura[-1] != altura_imagem:
        intervalos_altura.pop(-1)
        intervalos_altura.append(altura_imagem)

    # verifica se a quantidade de subimagens na altura e na largura estão de acordo com o que foi solicitado
    assert len(intervalos_altura) - 1 == n_subimagens, \
        'Esta quantidade de subimagens para a altura não pode ser processada = ' + str(n_subimagens)
    assert len(intervalos_largura) - 1 == n_subimagens, \
        'Esta quantidade de subimagens para a largura não pode ser processada = ' + str(n_subimagens)

    subimagens = list()
    for i in range(len(intervalos_altura) - 1):
        for j in range(len(intervalos_largura) - 1):
            xmin = intervalos_largura[j]
            ymin = intervalos_altura[i]
            xmax = intervalos_largura[j + 1]
            ymax = intervalos_altura[i + 1]
            subimagens.append((xmin, ymin, xmax, ymax))
    return subimagens


def agrupar_anotacoes_por_id_imagem(dataset: dict) -> dict:
    """
    Agrupa as anotações pelo id da imagem

    :param dataset: Dataset com imagens e anotações
    :return: Dicionário cujas chaves são os identificadores das imagens e os valores são suas anotações
    """

    anotacoes = dict()
    for anotacao in dataset['annotations']:
        if 'segmentation' in anotacao:
            del (anotacao['segmentation'])
        id_imagem = anotacao['image_id']
        if id_imagem not in anotacoes.keys():
            anotacoes[id_imagem] = list()
        anotacoes[id_imagem].append(anotacao)
    return anotacoes


def atualizar_imagem(novo_id_imagem: int, novo_nome_imagem: str, nova_altura: int, nova_largura: int) -> dict:
    """
    Atribui um novo id para a imagem, bem como um novo nome, altura e largura

    :param novo_id_imagem: Novo identificador para imagem
    :param novo_nome_imagem: Novo nome para a imagem
    :param nova_altura: Nova altura para a imagem
    :param nova_largura: Nova largura para a imagem
    :return: Dicionário contendo os dados da nova imagem
    """

    nova_imagem = dict(
        id=novo_id_imagem,
        file_name=novo_nome_imagem,
        height=nova_altura,
        width=nova_largura
    )
    return nova_imagem


def atualizar_anotacao(anotacao: dict, novo_id_anotacao: int, novo_id_imagem: int) -> dict:
    """
    Atribui um novo id para a anotação e atualiza o id da imagem ou subimagem em que a anotação está contida

    :param anotacao: Dicionário contendo os dados da anotação
    :param novo_id_anotacao: Novo identificador para a anotação
    :param novo_id_imagem: Novo identificador para a imagem que contém a anotação
    :return: Dicionário com os novos identificadores da anotação
    """

    anotacao['id'] = novo_id_anotacao
    anotacao['image_id'] = novo_id_imagem
    return anotacao


def dividir_anotacoes(imagem: dict, anotacoes_imagem: dict, n_subimagens: int) -> dict:
    """
    Cria novas anotações e as divide, caso necessário

    :param imagem: Imagem que possui anotações
    :param anotacoes_imagem: Anotações da imagem
    :param n_subimagens: Quantidade de divisões, na altura e na largura, para dividir uma imagem
    :return: Dicionário cujas chaves são as coordenadas das subimagens e os valores são as coordenadas das anotações de
    cada divisão
    """

    subimagens = calcular_intervalos_subimagens(imagem['height'], imagem['width'], n_subimagens)

    novas_anotacoes = dict()
    for subimagem in subimagens:
        for anotacao in anotacoes_imagem:
            bbox = anotacao['bbox']

            # coordenadas do retângulo de intersecção
            xmin = max(subimagem[0], bbox[0])
            ymin = max(subimagem[1], bbox[1])
            xmax = min(subimagem[2], bbox[2])
            ymax = min(subimagem[3], bbox[3])

            area_interseccao = max(0, xmax - xmin) * max(0, ymax - ymin)
            if area_interseccao > 0:
                nova_anotacao = deepcopy(anotacao)
                # coordenadas relativas à nova subimagem
                nova_bbox = [xmin - subimagem[0], ymin - subimagem[1], xmax - subimagem[0], ymax - subimagem[1]]

                if subimagem not in novas_anotacoes.keys():
                    novas_anotacoes[subimagem] = list()
                nova_area = (nova_bbox[2] - nova_bbox[0]) * (nova_bbox[3] - nova_bbox[1])
                nova_anotacao['bbox'] = nova_bbox
                nova_anotacao['area'] = nova_area
                novas_anotacoes[subimagem].append(nova_anotacao)
    return novas_anotacoes


def salvar_json(
        conjunto: str,
        novas_imagens: list,
        novas_anotacoes: list,
        categorias: list,
        dir_dataset_subimagens: str
):
    """
    Salva os dados dos novos arquivos de treino, validação e teste no formato .json

    :param conjunto: Nome do conjunto de treino, validação ou teste a ser salvo
    :param novas_imagens: Lista com as novas imagens
    :param novas_anotacoes: Lista com as novas anotações
    :param categorias: Lista com as categorias
    :param dir_dataset_subimagens: Diretório para salvar os novos arquivos de treino, validação e teste
    """

    if not os.path.exists(dir_dataset_subimagens):
        os.makedirs(dir_dataset_subimagens)

    dataset = dict(
        images=novas_imagens,
        annotations=novas_anotacoes,
        categories=categorias
    )
    json_path = os.path.join(dir_dataset_subimagens, conjunto + '.json')
    with open(json_path, 'w') as json_file:
        json.dump(dataset, json_file, indent=4)


def criar_subimagens(dataset: dict, n_subimagens: int, dir_dataset_subimagens: str):
    """
    Para todas as imagens do dataset, cria uma grade com NxN subimagens, em que N = n_subimagens. Uma subimagem só é
    incluída ao dataset caso tenha pelo menos uma anotação ou parte de uma anotação. Novas anotações serão criadas caso
    uma anotação seja dividida entre duas ou mais subimagens. Todas as coordenadas das novas anotações serão ajustadas
    de acordo com a subimagem, assim como as coordenadas das anotações que não são divididas. Novos arquivos de treino,
    validação e teste são gerados.

    :param dataset: Dicionário com os dados dos arquivos de treino, validação e teste
    :param n_subimagens: (n_subimagens * n_subimagens) subimagens verticais e horizontais para uma imagem
    :param dir_dataset_subimagens: Diretório para salvar os novos arquivos de treino, validação e teste
    """

    novo_id_imagem = 1
    novo_id_anotacao = 1

    # para cada conjunto de treino, validação e teste
    for conjunto in dataset:
        novas_imagens = list()
        novas_anotacoes = list()
        dados = dataset[conjunto]

        anotacoes = agrupar_anotacoes_por_id_imagem(dados)
        imagens = [imagem for imagem in dados['images'] if imagem['id'] in anotacoes]

        for imagem in imagens:
            anotacoes_imagem = anotacoes[imagem['id']]
            anotacoes_divididas = dividir_anotacoes(imagem, anotacoes_imagem, n_subimagens)

            for subimagem in anotacoes_divididas.keys():
                for anotacao in anotacoes_divididas[subimagem]:
                    nova_anotacao = atualizar_anotacao(anotacao, novo_id_anotacao, novo_id_imagem)
                    novas_anotacoes.append(nova_anotacao)
                    novo_id_anotacao += 1

                nova_imagem = atualizar_imagem(novo_id_imagem,
                                               imagem['file_name'],
                                               # o nome não muda :)
                                               subimagem[3] - subimagem[1],
                                               subimagem[2] - subimagem[0])
                nova_imagem['subimagem'] = subimagem
                novas_imagens.append(nova_imagem)
                novo_id_imagem += 1

        categorias = dataset[conjunto]['categories']
        salvar_json(conjunto, novas_imagens, novas_anotacoes, categorias, dir_dataset_subimagens)


def main():
    args = arg_parse()

    dir_subimagens = os.path.join(args.dir_dataset, 'subimagens')
    dir_dataset_subimagens = os.path.join(dir_subimagens, str(args.n_subimagens) + 'x' + str(args.n_subimagens))
    if not os.path.exists(dir_dataset_subimagens):
        os.makedirs(dir_dataset_subimagens)

    dataset = dict()
    lista_arquivos = ['treino', 'validacao', 'teste']
    for arquivo in lista_arquivos:
        dir_arquivo = os.path.join(args.dir_dataset, arquivo)
        dataset[arquivo] = carregar_json(dir_arquivo + '.json')

    criar_subimagens(dataset, args.n_subimagens, dir_dataset_subimagens)


if __name__ == '__main__':
    main()
