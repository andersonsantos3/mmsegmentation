from argparse import ArgumentParser
from collections import defaultdict
from copy import deepcopy
from json import load
from typing import Optional, Union

from numpy import mean, sqrt


CATEGORIAS: tuple[int]
TIPO_NAO_IMPLEMENTADO = NotImplementedError('Tipo não implementado')
tipo: str


def parse_args():
    parser = ArgumentParser(description='Argumentos para a métrica de distância')

    parser.add_argument(
        'caminho_anotacoes_imagens',
        help='Caminho para o arquivo json com as bbox de anotações nas imagens'
    )
    parser.add_argument(
        'caminho_anotacoes_subimagens',
        help='Caminho para o arquivo json com as bbox de anotações nas subimagens'
    )
    parser.add_argument(
        'caminho_predicoes_deteccao',
        help='Caminho para o arquivo json com as bbox de predições de detecções nas subimagens'
    )
    parser.add_argument(
        'caminho_predicoes_segmentacao',
        help='Caminho para o arquivo json com as bbox de predições de segmentação nas subimagens'
    )
    parser.add_argument('limiar_score', help='Score mínimo para validar uma predição', type=float)
    parser.add_argument('limiar_distancia', help='Distância máxima (em pixels) para validar uma predição', type=float)
    parser.add_argument('quantidade_de_categorias', help='Quantidade de categorias', type=int)

    args = parser.parse_args()
    return args


def agrupar_anotacoes_por_nome_da_imagem(anotacoes: dict) -> dict:
    anotacoes_por_nome_da_imagem = defaultdict(list)

    images_por_id = {image['id']: image for image in anotacoes['images']}
    for anotacao in anotacoes['annotations']:
        image_id = anotacao['image_id']

        if tipo == 'det':
            file_name = images_por_id[image_id]['file_name'].removesuffix('.jpg')
            subimagem = '_'.join(map(str, images_por_id[image_id]['subimagem']))
            nome_imagem = file_name + '_' + subimagem + '.jpg'
        elif tipo == 'seg':
            nome_imagem = image_id
        else:
            raise TIPO_NAO_IMPLEMENTADO

        anotacoes_por_nome_da_imagem[nome_imagem].append(anotacao)
    return anotacoes_por_nome_da_imagem


def agrupar_anotacoes_por_nome_da_imagem_e_por_categoria(anotacoes: dict) -> dict[str, dict[str, list]]:
    anotacoes_agrupadas = agrupar_anotacoes_por_nome_da_imagem(anotacoes)

    anotacoes_agrupadas_por_categoria = defaultdict(dict)
    for nome_da_subimagem, anotacoes_na_subimagem in anotacoes_agrupadas.items():
        for anotacao in anotacoes_na_subimagem:
            categoria = anotacao['category_id']
            if not anotacoes_agrupadas_por_categoria[nome_da_subimagem].get(categoria):
                anotacoes_agrupadas_por_categoria[nome_da_subimagem][categoria] = list()
            anotacoes_agrupadas_por_categoria[nome_da_subimagem][categoria].append(anotacao)
    return anotacoes_agrupadas_por_categoria

def agrupar_predicoes_por_nome_da_imagem_e_por_categoria(predicoes: dict) -> dict[str, dict[str, list]]:
    predicoes_agrupadas = ajustar_predicoes(predicoes)

    predicoes_agrupadas_por_categoria = defaultdict(dict)
    for nome_da_subimagem, predicoes_na_subimagem in predicoes_agrupadas.items():
        for predicao in predicoes_na_subimagem:
            if tipo == 'det':
                categoria = predicao[-1]
            elif tipo == 'seg':
                categoria = predicao['category_id']
            else:
                raise TIPO_NAO_IMPLEMENTADO

            if not predicoes_agrupadas_por_categoria[nome_da_subimagem].get(categoria):
                predicoes_agrupadas_por_categoria[nome_da_subimagem][categoria] = list()
            predicoes_agrupadas_por_categoria[nome_da_subimagem][categoria].append(predicao)
    return predicoes_agrupadas_por_categoria


def ajustar_predicoes(predicoes: dict) -> dict:
    predicoes_por_nome_da_imagem = defaultdict(list)

    if tipo == 'det':
        for nome_subimagem, predicoes_categorias in predicoes.items():
            for id_categoria, predicoes_categoria in enumerate(predicoes_categorias):
                for predicao_categoria in predicoes_categoria:
                    predicao_categoria.append(id_categoria + 1)
                    predicoes_por_nome_da_imagem[nome_subimagem].append(predicao_categoria)
    elif tipo == 'seg':
        for predicao in predicoes:
            image_id = predicao['image_id']

            bbox = predicao['bbox']
            bbox[2] += bbox[0]
            bbox[3] += bbox[1]

            predicoes_por_nome_da_imagem[image_id].append(predicao)
    else:
        raise TIPO_NAO_IMPLEMENTADO
    return predicoes_por_nome_da_imagem


def calcular_centro(
        xmin: Union[float, int],
        ymin: Union[float, int],
        xmax: Union[float, int],
        ymax: Union[float, int]
) -> tuple[int, int]:
    centro_x = round((xmin + xmax) / 2)
    centro_y = round((ymin + ymax) / 2)
    return centro_x, centro_y


def calcular_distancia(ponto_1: tuple[int, int], ponto_2: tuple[int: int]) -> float:
    return sqrt((ponto_2[0] - ponto_1[0]) ** 2 + (ponto_2[1] - ponto_1[1]) ** 2)


def calcular_percentual_de_acerto(anotacoes: list[dict], predicoes: list[list], limiar_distancia: float) -> float:
    matriz_de_distancias = obter_matriz_de_distancias(anotacoes, predicoes)
    distancias_minimas = obter_distancias_minimas(matriz_de_distancias)

    # acertos = [1 if distancia <= limiar_distancia else 0 for distancia in distancias_minimas]
    # menor = min(len(anotacoes), len(predicoes))
    # maior = max(len(anotacoes), len(predicoes))
    # pontuacao_maxima = 1 / maior * menor
    # return mean(acertos).item() * pontuacao_maxima

    # operação equivalente
    acertos = [distancia for distancia in distancias_minimas if distancia <= limiar_distancia]
    maior = max(len(anotacoes), len(predicoes))
    return len(acertos) / maior


def calcular_taxa_de_acerto_geral(anotacoes: dict, predicoes: dict, limiar_distancia: float ) -> float:
    """Considera apenas a detecção e ignora a categoria"""

    anotacoes_por_nome_da_imagem = agrupar_anotacoes_por_nome_da_imagem(anotacoes)
    predicoes_por_nome_da_imagem = ajustar_predicoes(predicoes)

    acertos_por_subimagem = list()
    for nome_da_subimagem, anotacoes in anotacoes_por_nome_da_imagem.items():
        predicoes = predicoes_por_nome_da_imagem.get(nome_da_subimagem)
        if not anotacoes and not predicoes:
            acertos_por_subimagem.append(1)
        elif not anotacoes and predicoes:
            acertos_por_subimagem.append(0)
        elif anotacoes and not predicoes:
            acertos_por_subimagem.append(0)
        else:
            percentual = calcular_percentual_de_acerto(anotacoes, predicoes, limiar_distancia)
            acertos_por_subimagem.append(percentual)
    return mean(acertos_por_subimagem).item()


def calcular_taxa_de_acerto_por_categoria(
        anotacoes: dict,
        predicoes: dict,
        limiar_distancia: float
) -> dict[int, float]:
    anotacoes_agrupadas = agrupar_anotacoes_por_nome_da_imagem_e_por_categoria(anotacoes)
    predicoes_agrupadas = agrupar_predicoes_por_nome_da_imagem_e_por_categoria(predicoes)

    acertos_por_categoria = {categoria: list() for categoria in CATEGORIAS}
    for nome_da_subimagem, anotacoes_na_subimagem in anotacoes_agrupadas.items():
        predicoes_na_subimagem = predicoes_agrupadas.get(nome_da_subimagem, dict())
        for categoria in CATEGORIAS:
            if not anotacoes_na_subimagem.get(categoria, dict()) and not predicoes_na_subimagem.get(categoria, dict()):
                acertos_por_categoria[categoria].append(1)
            elif not anotacoes_na_subimagem.get(categoria, dict()) and predicoes_na_subimagem.get(categoria, dict()):
                acertos_por_categoria[categoria].append(0)
            elif anotacoes_na_subimagem.get(categoria, dict()) and not predicoes_na_subimagem.get(categoria, dict()):
                acertos_por_categoria[categoria].append(0)
            else:
                percentual = calcular_percentual_de_acerto(
                    anotacoes_na_subimagem.get(categoria, dict()),
                    predicoes_na_subimagem.get(categoria, dict()),
                    limiar_distancia
                )
                acertos_por_categoria[categoria].append(percentual)
    for categoria in CATEGORIAS:
        acertos_por_categoria[categoria] = round(mean(acertos_por_categoria[categoria]), 3)
    return acertos_por_categoria



def carregar_json(caminho_json: str) -> Union[dict, list]:
    with open(caminho_json, 'r', encoding='utf_8') as arquivo:
        dados = load(arquivo)
    return dados


def filtrar_predicoes_por_score(predicoes: Union[dict, list], limiar_score: float) -> Optional[list]:
    if tipo == 'det':
        for _, predicoes_na_subimagem in predicoes.items():
            for i, predicoes_por_categoria in enumerate(predicoes_na_subimagem):
                predicoes_na_subimagem[i] = [
                    predicao
                    for predicao in predicoes_por_categoria
                    if predicao[-1] >= limiar_score
                ]
    elif tipo == 'seg':
        predicoes = [predicao for predicao in predicoes if predicao['score'] >= limiar_score]
        return predicoes
    else:
        raise TIPO_NAO_IMPLEMENTADO


def obter_distancias_minimas(matriz_de_distancias: list[list]) -> list[float]:
    """O tamanho da lista de distâncias mínimas é sempre igual ao menor entre o tamanho de anotações e o tamanho de
    predições"""

    distancias = list()
    for i, linha in enumerate(matriz_de_distancias):
        for j, distancia in enumerate(matriz_de_distancias[i]):
            distancias.append((distancia, i, j))
    distancias.sort(key=lambda x: x[0])

    distancias_minimas = list()
    while distancias:
        distancia, i, j = distancias[0]
        distancias_minimas.append(distancia)
        distancias = [d for d in distancias if d[1] != i and d[2] != j]
    return distancias_minimas


def obter_matriz_de_distancias(anotacoes: list[dict], predicoes: list[list]) -> list[list]:
    """O tamanho da matriz de distâncias sempre será igual ao tamanho das anotações"""

    matriz_de_distancias = list()
    for anotacao in anotacoes:
        distancias = list()

        bbox_anotacao = anotacao['bbox']
        centro_da_anotacao = calcular_centro(*bbox_anotacao)
        for predicao in predicoes:
            if tipo == 'det':
                bbox_predicao = predicao[:4]
            elif tipo == 'seg':
                bbox_predicao = predicao['bbox']
            else:
                raise TIPO_NAO_IMPLEMENTADO

            centro_da_predicao = calcular_centro(*bbox_predicao)
            distancia = calcular_distancia(centro_da_anotacao, centro_da_predicao)
            distancias.append(distancia)
        matriz_de_distancias.append(distancias)
    return matriz_de_distancias


def main():
    global CATEGORIAS
    global tipo

    args = parse_args()

    CATEGORIAS = tuple(range(1, args.quantidade_de_categorias + 1))

    tipo = 'det'
    anotacoes_subimagens = carregar_json(args.caminho_anotacoes_subimagens)
    predicoes_deteccao = carregar_json(args.caminho_predicoes_deteccao)
    filtrar_predicoes_por_score(predicoes_deteccao, args.limiar_score)

    taxa_de_acerto_geral = calcular_taxa_de_acerto_geral(
        anotacoes_subimagens,
        deepcopy(predicoes_deteccao),
        args.limiar_distancia
    )
    taxa_de_acerto_por_categoria = calcular_taxa_de_acerto_por_categoria(
        anotacoes_subimagens,
        deepcopy(predicoes_deteccao),
        args.limiar_distancia
    )

    print('Resultados com detecção')
    print(f'Limiar de distância: {args.limiar_distancia}')
    print(f'Taxa de acerto geral: {round(taxa_de_acerto_geral, 3)}')
    print(f'Taxa de acerto por categoria: {taxa_de_acerto_por_categoria}')
    print()

    tipo = 'seg'
    anotacoes_imagens = carregar_json(args.caminho_anotacoes_imagens)
    predicoes_segmentacao = carregar_json(args.caminho_predicoes_segmentacao)
    predicoes_segmentacao = filtrar_predicoes_por_score(predicoes_segmentacao, args.limiar_score)

    taxa_de_acerto_geral = calcular_taxa_de_acerto_geral(
        anotacoes_imagens,
        deepcopy(predicoes_segmentacao),
        args.limiar_distancia
    )
    taxa_de_acerto_por_categoria = calcular_taxa_de_acerto_por_categoria(
        anotacoes_imagens,
        deepcopy(predicoes_segmentacao),
        args.limiar_distancia
    )

    print('Resultados com segmentação')
    print(f'Limiar de distância: {args.limiar_distancia}')
    print(f'Taxa de acerto geral: {round(taxa_de_acerto_geral, 3)}')
    print(f'Taxa de acerto por categoria: {taxa_de_acerto_por_categoria}')


if __name__ == '__main__':
    main()
