from argparse import ArgumentParser
from collections import defaultdict
from copy import deepcopy
from json import load
from typing import Optional, Union

from numpy import mean, sqrt
from pandas import DataFrame, concat
from torch import float32, int64, tensor
from torchvision.ops import batched_nms


CATEGORIAS: tuple[int]
TIPO_NAO_IMPLEMENTADO = NotImplementedError('Tipo não implementado')
imagem_inteira: bool
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
    parser.add_argument('quantidade_de_categorias', help='Quantidade de categorias', type=int)
    parser.add_argument('limiar_score', help='Score mínimo para validar uma predição', type=float)
    parser.add_argument('limiar_distancia', help='Distância máxima (em pixels) para validar uma predição', type=float)
    parser.add_argument('distancia_area_de_uniao', help='Distância que representa o tamanho da área de união', type=int)
    parser.add_argument(
        'distancia_deteccoes',
        help='Distância mínima entre dois pontos para unir duas detecções',
        type=int
    )
    parser.add_argument(
        '--imagem_inteira',
        help='Informe esta flag para avaliar predições unidas, montado a imagem inteira',
        action='store_true'
    )

    args = parser.parse_args()
    return args


def agrupar_anotacoes_por_nome_da_imagem(anotacoes: dict) -> dict:
    anotacoes_por_nome_da_imagem = defaultdict(list)

    images_por_id = {image['id']: image for image in anotacoes['images']}
    for anotacao in anotacoes['annotations']:
        image_id = anotacao['image_id']

        if tipo == 'det':
            file_name = images_por_id[image_id]['file_name'].removesuffix('.jpg')
            nome_imagem = file_name
            if not imagem_inteira:
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


def unir_predicoes(predicoes: dict, distancia_area_de_uniao: int, distancia_deteccoes: int) -> dict:
    """Une as detecções de cada imagem, após aplicação de batched_nms. Batched_nms é usado para fazer a supressão de
    não máximos para detecções que estão na mesma subimagem e que são da mesma classe, considerando um limiar de IoU de
    0.3. As detecções são unidas de acordo com algumas regras:
    1. Qualquer um dos pontos da detecção deve estar dentro de uma área de união, ou seja, a distância entre um dos
    pontos da detecção para uma das bordas da subimagem deve ser menor ou igual a distancia_uniao. Detecções que não
    estão em área de união não serão unidas;
    2. Duas detecções serão unidas se a distância entre dois de quaisquer de seus pontos médios for menor ou igual a
    distancia_pixels. A união de duas detecções cria uma nova detecção e as duas detecções anteriores são excluídas;
    3. As detecções podem ser divididas entre duas ou mais subimagens. Portanto, ao unir duas detecções, duas
    subimagens se tornam uma nova subimagem, e as subimagens anteriores são excluídas;
    4. Duas detecções só podem ser unidas quando suas subimagens não possuem intersecção, mesmo após a junção de
    subimagens."""

    deteccoes_unidas = defaultdict(DataFrame)
    deteccoes_unidas_ = dict()
    if tipo == 'det':
        imagens = defaultdict(list)
        for subimagem, deteccoes in predicoes.items():
            # armazena detecções da imagem após aplicar batched_nms em cada subimagem
            boxes = list()
            labels = list()
            scores = list()
            for label, deteccoes_classe in enumerate(deteccoes):
                if deteccoes_classe:
                    for deteccao in deteccoes_classe:
                        boxes.append(deteccao[:4])
                        labels.append(label)
                        scores.append(deteccao[-1])

            # aplica batched_nms em cada subimagem e retorna os índices a serem mantidas
            keep_idx = batched_nms(
                tensor(boxes, dtype=float32),
                tensor(scores, dtype=float32),
                tensor(labels, dtype=int64),
                0.3
            ).numpy().tolist()
            boxes = [boxes[idx] for idx in keep_idx]
            labels = [labels[idx] for idx in keep_idx]
            scores = [scores[idx] for idx in keep_idx]

            df_subimagem = DataFrame(dict(
                subimagens=[subimagem for i in range(len(boxes))],
                boxes=boxes,
                labels=labels,
                scores=scores
            ))

            nome_imagem = subimagem.split('_')[0]
            imagens[nome_imagem].append(df_subimagem)

        # cria um dataframe para cada imagem contendo as informações das subimagens
        for imagem, values in imagens.items():
            df_imagem = concat(values, ignore_index=True)

            # verifica se as detecções da imagem estão na área de união
            df_uniao = DataFrame()
            for box in df_imagem.iloc:
                xmin, ymin, xmax, ymax = box['boxes']
                label = box['labels']
                score = box['scores']
                subimagem = list(map(int, box['subimagens'].removesuffix('.jpg').split('_')[1:]))

                xmin += subimagem[0]
                ymin += subimagem[1]
                xmax += subimagem[0]
                ymax += subimagem[1]

                # áreas de união de uma subimagem
                area1 = [subimagem[0], subimagem[1], subimagem[2], subimagem[1] + distancia_area_de_uniao]
                area2 = [subimagem[0], subimagem[1], subimagem[0] + distancia_area_de_uniao, subimagem[3]]
                area3 = [subimagem[2] - distancia_area_de_uniao, subimagem[1], subimagem[2], subimagem[3]]
                area4 = [subimagem[0], subimagem[3] - distancia_area_de_uniao, subimagem[2], subimagem[3]]

                # verifica se a detecção possui intersecção com uma área de união
                box_ = [xmin, ymin, xmax, ymax]
                tem_interseccao = any([
                    verificar_interseccao(box_, area1),
                    verificar_interseccao(box_, area2),
                    verificar_interseccao(box_, area3),
                    verificar_interseccao(box_, area4)
                ])

                # se a detecção estiver em área de união, então ela será adicionada no dataframe de união para
                # processamento posterior. Caso contrário, a detecção não será unida com nenhuma outra, então será
                # considerada como já unida e será adicionada no dataframe de detecções unidas
                if tem_interseccao:
                    # df_uniao guarda uma detecção, a ser unida, por linha
                    df_uniao = concat([
                        df_uniao,
                        DataFrame(
                            data=[[subimagem, [xmin, ymin, xmax, ymax], int(label), score]],
                            columns=['subimagens', 'boxes', 'labels', 'scores']
                        )
                    ], ignore_index=True)
                else:
                    deteccoes_unidas[imagem] = concat([
                        deteccoes_unidas[imagem],
                        DataFrame(
                            data=[[[xmin, ymin, xmax, ymax], int(label), score]],
                            columns=['boxes', 'labels', 'scores']
                        )
                    ], ignore_index=True)

            # as uniões são realizadas de duas maneiras diferentes: na vertical e na horizontal
            df_deteccoes_unidas_horizontalmente = unir_deteccoes_horizontalmente(
                deepcopy(df_uniao),
                distancia_area_de_uniao,
                distancia_deteccoes
            )
            df_deteccoes_unidas = unir_deteccoes_verticalmente(
                deepcopy(df_deteccoes_unidas_horizontalmente),
                distancia_area_de_uniao,
                distancia_deteccoes
            )
            deteccoes_unidas[imagem] = concat([deteccoes_unidas[imagem], df_deteccoes_unidas], ignore_index=True)
        for nome_imagem, df in deteccoes_unidas.items():
            for deteccao in df.iloc:
                if nome_imagem not in deteccoes_unidas_:
                    deteccoes_unidas_[nome_imagem] = [[] for _ in CATEGORIAS]
                deteccoes_unidas_[nome_imagem][deteccao['labels']].append(deteccao['boxes'] + [deteccao['scores']])
    else:
        raise TIPO_NAO_IMPLEMENTADO
    return deteccoes_unidas_


def unir_deteccoes_horizontalmente(df_uniao: DataFrame, distancia_uniao: int, distancia_deteccoes: int) -> DataFrame:
    """
    Une detecções próximas horizontalmente

    :param df_uniao: DataFrame com detecções
    :param distancia_uniao: Distância de um dos pontos da anotação até uma das bordas da imagem
    :param distancia_deteccoes: Distância máxima entre os pontos médios das detecções

    :return: DataFrame com as detecções horizontalmente unidas
    """

    df_deteccoes_unidas_horizontalmente = DataFrame()
    while not df_uniao.empty:
        # os dados sempre serão ordenados
        df_uniao.sort_values(by=['boxes'], inplace=True, ignore_index=True)

        # se houver apenas uma detecção para ser unida, então ela simplesmente é passada para frente e este laço é
        # rompido, pois não existem outras detecções para realizar união
        if len(df_uniao) == 1:
            box1 = df_uniao.iloc[0]
            subimagem = box1['subimagens']
            xmin, ymin, xmax, ymax = box1['boxes']
            box1_pontuacao = box1['scores']
            df_deteccoes_unidas_horizontalmente = concat([
                df_deteccoes_unidas_horizontalmente,
                DataFrame(
                    data=[[subimagem, [xmin, ymin, xmax, ymax], int(box1['labels']), box1_pontuacao]],
                    columns=['subimagens', 'boxes', 'labels', 'scores']
                )
            ], ignore_index=True)

            df_uniao.drop(labels=[0], axis=0, inplace=True)
            df_uniao.reset_index(inplace=True, drop=True)
            break

        for i in range(len(df_uniao) - 1):
            box1 = df_uniao.iloc[i]
            xmin, ymin, xmax, ymax = box1['boxes']
            box1_pontuacao = box1['scores']
            box1_subimagem = box1['subimagens']

            box1_pontos_medios = [(xmin, (ymin + ymax) / 2), (xmax, (ymin + ymax) / 2)]

            parar = False  # utilizado quando acontece união, então é necessário interromper os laços
            for j in range(i + 1, len(df_uniao)):
                box2 = df_uniao.iloc[j]
                box2_subimagem = box2['subimagens']

                # quando uma detecção é unida, suas subimagens também são. A variável interseccao guarda o valor lógico
                # de interseccao entre uma subimagem e outra. se houver interseccao, a união das detecções não pode ser
                # realizada, para evitar que uniões na mesma subimagem sejam feitas
                interseccao = verificar_interseccao(box1_subimagem, box2_subimagem)

                # para uma união ocorrer, os labels das anotações devem ser iguais e a interseccao entre duas
                # subimagens deve ser falsa
                if box1['labels'] != box2['labels'] or interseccao:
                    continue

                xmin_, ymin_, xmax_, ymax_ = box2['boxes']
                box2_pontuacao = box2['scores']
                box2_pontos_medios = [(xmin_, (ymin_ + ymax_) / 2), (xmax_, (ymin_ + ymax_) / 2)]

                # a distância é calculada a partir dos pontos médios das box1 e box2
                dist_p2_p3 = calcular_distancia(box1_pontos_medios[0], box2_pontos_medios[1])
                dist_p3_p2 = calcular_distancia(box1_pontos_medios[1], box2_pontos_medios[0])

                if dist_p2_p3 <= distancia_deteccoes or dist_p3_p2 <= distancia_deteccoes:
                    # une as duas subimagens e cria uma nova
                    nova_subimagem_xmin = min(box1_subimagem[0], box2_subimagem[0])
                    nova_subimagem_ymin = min(box1_subimagem[1], box2_subimagem[1])
                    nova_subimagem_xmax = max(box1_subimagem[2], box2_subimagem[2])
                    nova_subimagem_ymax = max(box1_subimagem[3], box2_subimagem[3])
                    nova_subimagem = [nova_subimagem_xmin, nova_subimagem_ymin,
                                      nova_subimagem_xmax, nova_subimagem_ymax]

                    # une as duas detecções e cria uma nova
                    novo_xmin = min(xmin, xmin_)
                    novo_ymin = min(ymin, ymin_)
                    novo_xmax = max(xmax, xmax_)
                    novo_ymax = max(ymax, ymax_)

                    # verifica se a nova detecção está em área de união com a nova subimagem
                    area1 = [nova_subimagem[0], nova_subimagem[1],
                             nova_subimagem[2], nova_subimagem[1] + distancia_uniao]
                    area2 = [nova_subimagem[0], nova_subimagem[1],
                             nova_subimagem[0] + distancia_uniao, nova_subimagem[3]]
                    area3 = [nova_subimagem[2] - distancia_uniao, nova_subimagem[1],
                             nova_subimagem[2], nova_subimagem[3]]
                    area4 = [nova_subimagem[0], nova_subimagem[3] - distancia_uniao,
                             nova_subimagem[2], nova_subimagem[3]]

                    box_ = [novo_xmin, novo_ymin, novo_xmax, novo_ymax]
                    tem_interseccao = verificar_interseccao(box_, area1) or verificar_interseccao(box_, area2) or \
                                      verificar_interseccao(box_, area3) or verificar_interseccao(box_, area4)
                    nova_pontuacao = max(box1_pontuacao, box2_pontuacao)

                    # se não houver intersecção entre a nova subimagem e a nova detecção unida, então é necessário
                    # adicionar a nova detecção ao df_deteccoes_unidas_horizontalmente e remover as duas detecções
                    # antigas do df_uniao
                    if not tem_interseccao:
                        df_deteccoes_unidas_horizontalmente = concat([
                            df_deteccoes_unidas_horizontalmente,
                            DataFrame(
                                data=[[nova_subimagem, box_, int(box1['labels']), nova_pontuacao]],
                                columns=['subimagens', 'boxes', 'labels', 'scores']
                            )
                        ], ignore_index=True)

                        df_uniao.drop(labels=[i, j], axis=0, inplace=True)
                        df_uniao.reset_index(inplace=True, drop=True)
                        parar = True
                        break

                    # caso houver intersecção entre a nova subimagem e a nova detecção unida, a nova detecção deve
                    # permanecer em processo de união
                    df_uniao = concat([
                        df_uniao,
                        DataFrame(
                            data=[[nova_subimagem, box_, int(box1['labels']), nova_pontuacao]],
                            columns=['subimagens', 'boxes', 'labels', 'scores']
                        )
                    ], ignore_index=True)

                    df_uniao.drop(labels=[i, j], axis=0, inplace=True)
                    df_uniao.reset_index(inplace=True, drop=True)
                    parar = True
                    break

            # se houve união, é necessário interromper o laço for e voltar ao while
            if parar:
                break
            # caso contrário, a detecção i não poderá ser unida com nenhuma outra
            df_deteccoes_unidas_horizontalmente = concat([
                df_deteccoes_unidas_horizontalmente,
                DataFrame(
                    data=[[box1_subimagem, [xmin, ymin, xmax, ymax], int(box1['labels']), box1_pontuacao]],
                    columns=['subimagens', 'boxes', 'labels', 'scores']
                )
            ], ignore_index=True)

            df_uniao.drop(labels=[i], axis=0, inplace=True)
            df_uniao.reset_index(inplace=True, drop=True)
            break
    return df_deteccoes_unidas_horizontalmente


def unir_deteccoes_verticalmente(df_uniao: DataFrame, distancia_uniao: int, distancia_deteccoes: int) -> DataFrame:
    """
    Une detecções próximas verticalmente

    :param df_uniao: DataFrame com detecções
    :param distancia_uniao: Distância de um dos pontos da anotação até uma das bordas da imagem
    :param distancia_deteccoes: Distância máxima entre os pontos médios das detecções

    :return: DataFrame com as detecções verticalmente unidas
    """

    df_deteccoes_unidas = DataFrame()
    # enquanto o daframe de união não for vazio, ou seja, enquanto houverem detecções a serem unidas
    while not df_uniao.empty:
        # os dados sempre serão ordenados
        df_uniao.sort_values(by=['boxes'], inplace=True, ignore_index=True)

        # se houver apenas uma detecção para ser unida, então ela simplesmente é passada para frente e este laço é
        # rompido, pois não existem outras detecções para realizar união
        if len(df_uniao) == 1:
            box1 = df_uniao.iloc[0]
            xmin, ymin, xmax, ymax = box1['boxes']
            box1_pontuacao = box1['scores']
            df_deteccoes_unidas = concat([
                df_deteccoes_unidas,
                DataFrame(
                    data=[[[xmin, ymin, xmax, ymax], int(box1['labels']), box1_pontuacao]],
                    columns=['boxes', 'labels', 'scores']
                )
            ], ignore_index=True)

            df_uniao.drop(labels=[0], axis=0, inplace=True)
            df_uniao.reset_index(inplace=True, drop=True)
            break

        for i in range(len(df_uniao) - 1):
            box1 = df_uniao.iloc[i]
            xmin, ymin, xmax, ymax = box1['boxes']
            box1_pontuacao = box1['scores']
            box1_subimagem = box1['subimagens']

            box1_pontos_medios = [((xmin + xmax) / 2, ymin), ((xmin + xmax) / 2, ymax)]

            parar = False  # utilizado quando acontece união, então é necessário interromper os laços
            for j in range(i + 1, len(df_uniao)):
                box2 = df_uniao.iloc[j]
                box2_subimagem = box2['subimagens']

                # quando uma detecção é unida, suas subimagens também são. A variável interseccao guarda o valor lógico
                # de interseccao entre uma subimagem e outra. se houver interseccao, a união das detecções não pode ser
                # realizada, para evitar que uniões na mesma subimagem sejam feitas
                interseccao = verificar_interseccao(box1_subimagem, box2_subimagem)

                # para uma união ocorrer, os labels das anotações devem ser iguais e a interseccao entre duas subimagens
                # deve ser falsa
                if box1['labels'] != box2['labels'] or interseccao:
                    continue

                xmin_, ymin_, xmax_, ymax_ = box2['boxes']
                box2_pontuacao = box2['scores']
                box2_pontos_medios = [((xmin_ + xmax_) / 2, ymin_), ((xmin_ + xmax_) / 2, ymax_)]

                # a distância é calculada a partir dos pontos médios das box1 e box2
                dist_p1_p4 = calcular_distancia(box1_pontos_medios[0], box2_pontos_medios[1])
                dist_p4_p1 = calcular_distancia(box1_pontos_medios[1], box2_pontos_medios[0])

                if dist_p1_p4 <= distancia_deteccoes or dist_p4_p1 <= distancia_deteccoes:
                    # une as duas subimagens e cria uma nova
                    nova_subimagem_xmin = min(box1_subimagem[0], box2_subimagem[0])
                    nova_subimagem_ymin = min(box1_subimagem[1], box2_subimagem[1])
                    nova_subimagem_xmax = max(box1_subimagem[2], box2_subimagem[2])
                    nova_subimagem_ymax = max(box1_subimagem[3], box2_subimagem[3])
                    nova_subimagem = [nova_subimagem_xmin, nova_subimagem_ymin, nova_subimagem_xmax,
                                      nova_subimagem_ymax]

                    # une as duas detecções e cria uma nova
                    novo_xmin = min(xmin, xmin_)
                    novo_ymin = min(ymin, ymin_)
                    novo_xmax = max(xmax, xmax_)
                    novo_ymax = max(ymax, ymax_)

                    # verifica se a nova detecção está em área de união com a nova subimagem
                    area1 = [nova_subimagem[0], nova_subimagem[1],
                             nova_subimagem[2], nova_subimagem[1] + distancia_uniao]
                    area2 = [nova_subimagem[0], nova_subimagem[1],
                             nova_subimagem[0] + distancia_uniao, nova_subimagem[3]]
                    area3 = [nova_subimagem[2] - distancia_uniao,
                             nova_subimagem[1], nova_subimagem[2], nova_subimagem[3]]
                    area4 = [nova_subimagem[0], nova_subimagem[3] - distancia_uniao,
                             nova_subimagem[2], nova_subimagem[3]]

                    box_ = [novo_xmin, novo_ymin, novo_xmax, novo_ymax]
                    tem_interseccao = verificar_interseccao(box_, area1) or verificar_interseccao(box_, area2) or \
                                      verificar_interseccao(box_, area3) or verificar_interseccao(box_, area4)
                    nova_pontuacao = max(box1_pontuacao, box2_pontuacao)
                    if not tem_interseccao:
                        df_deteccoes_unidas = concat([
                            df_deteccoes_unidas,
                            DataFrame(
                                data=[[box_, int(box1['labels']), nova_pontuacao]],
                                columns=['boxes', 'labels', 'scores']
                            )
                        ], ignore_index=True)

                        df_uniao.drop(labels=[i, j], axis=0, inplace=True)
                        df_uniao.reset_index(inplace=True, drop=True)
                        parar = True
                        break

                    df_uniao = concat([
                        df_uniao,
                        DataFrame(
                            data=[[nova_subimagem, box_, int(box1['labels']), nova_pontuacao]],
                            columns=['subimagens', 'boxes', 'labels', 'scores']
                        )
                    ], ignore_index=True)

                    df_uniao.drop(labels=[i, j], axis=0, inplace=True)
                    df_uniao.reset_index(inplace=True, drop=True)
                    parar = True
                    break
            if parar:
                break

            df_deteccoes_unidas = concat([
                df_deteccoes_unidas,
                DataFrame(
                    data=[[[xmin, ymin, xmax, ymax], int(box1['labels']), box1_pontuacao]],
                    columns=['boxes', 'labels', 'scores']
                )
            ], ignore_index=True)

            df_uniao.drop(labels=[i], axis=0, inplace=True)
            df_uniao.reset_index(inplace=True, drop=True)
            break
    return df_deteccoes_unidas


def verificar_interseccao(retangulo1: list, retangulo2: list) -> bool:
    """Verifica se há interseccção entre dois retângulos"""

    xmin = max(retangulo1[0], retangulo2[0])
    ymin = max(retangulo1[1], retangulo2[1])
    xmax = min(retangulo1[2], retangulo2[2])
    ymax = min(retangulo1[3], retangulo2[3])
    area_interseccao = max(0, xmax - xmin) * max(0, ymax - ymin)
    return area_interseccao > 0


def main():
    global CATEGORIAS
    global imagem_inteira
    global tipo

    args = parse_args()
    imagem_inteira = args.imagem_inteira

    CATEGORIAS = tuple(range(1, args.quantidade_de_categorias + 1))

    tipo = 'det'
    anotacoes_subimagens = carregar_json(args.caminho_anotacoes_subimagens)
    predicoes_deteccao = carregar_json(args.caminho_predicoes_deteccao)
    filtrar_predicoes_por_score(predicoes_deteccao, args.limiar_score)

    if imagem_inteira:
        predicoes_deteccao = unir_predicoes(predicoes_deteccao, args.distancia_area_de_uniao, args.distancia_deteccoes)

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
