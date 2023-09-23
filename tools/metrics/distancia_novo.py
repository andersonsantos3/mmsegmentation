from collections import defaultdict
from copy import deepcopy
from json import load
from os import environ
from typing import Optional, Union

from numpy import array, mean, sqrt, zeros
from pandas import DataFrame, concat
from scipy.optimize import linear_sum_assignment
from torch import float32, int64, tensor
from torchvision.ops import batched_nms


CATEGORIAS = tuple(range(1, int(environ.get('QUANTIDADE_DE_CATEGORIAS')) + 1))
DISTANCIA_DA_AREA_DE_UNIAO = float(environ.get('DISTANCIA_DA_AREA_DE_UNIAO'))
DISTANCIA_EM_PIXELS_ENTRE_PONTOS_MEDIOS = float(environ.get('DISTANCIA_EM_PIXELS_ENTRE_PONTOS_MEDIOS'))


def aplicar_batched_nms(predicoes: dict) -> None:
    for chave, valor in predicoes.items():
        boxes = list()
        labels = list()
        scores = list()
        for categoria in range(len(valor)):
            if valor[categoria]:
                boxes += [box[:-1] for box in valor[categoria]]
                labels += [categoria] * len(valor[categoria])
                scores += [box[-1] for box in valor[categoria]]
        keep_idx = batched_nms(
            tensor(boxes, dtype=float32),
            tensor(scores, dtype=float32),
            tensor(labels, dtype=int64),
            0.3
        ).numpy().tolist()
        boxes = [boxes[idx] for idx in keep_idx]
        labels = [labels[idx] for idx in keep_idx]
        scores = [scores[idx] for idx in keep_idx]

        predicoes_ = [[] for _ in range(len(valor))]
        for box, label, score in zip(boxes, labels, scores):
            box.append(score)
            predicoes_[label].append(box)
        predicoes[chave] = predicoes_


def calcular_centro(box: list[Union[int, float]]) -> tuple[Union[int, float], Union[int, float]]:
    x = (box[0] + box[2]) / 2
    y = (box[1] + box[3]) / 2
    return x, y


def calcular_distancia(
        ponto_a: tuple[Union[int, float], Union[int, float]],
        ponto_b: tuple[Union[int, float], Union[int, float]]
) -> float:
    return sqrt((ponto_b[0] - ponto_a[0]) ** 2 + (ponto_b[1] - ponto_a[1]) ** 2)


def calcular_metrica_nas_imagens_com_categoria(anotacoes_imagens: dict, deteccoes_imagens: dict) -> float:
    return calcular_metrica_nas_subimagens_com_categoria(anotacoes_imagens, deteccoes_imagens)


def calcular_metrica_nas_imagens_sem_categoria(anotacoes_imagens: dict, deteccoes_imagens: dict) -> float:
    return calcular_metrica_nas_subimagens_sem_categoria(anotacoes_imagens, deteccoes_imagens)


def calcular_metrica_nas_subimagens_com_categoria(anotacoes_subimagens: dict, deteccoes_subimagens: dict) -> float:
    anotacoes_convertidas = converter_anotacoes_para_o_padrao_de_deteccoes(anotacoes_subimagens)
    percentual_de_acerto_por_subimagem = list()
    for nome_imagem, anotacoes in anotacoes_convertidas.items():
        predicoes = deteccoes_subimagens.get(nome_imagem)
        if existe_bbox(anotacoes) and existe_bbox(predicoes):
            percentual_de_acerto = calcular_percentual_de_acerto_por_subimagem_com_categoria(anotacoes, predicoes)
            percentual_de_acerto_por_subimagem.append(percentual_de_acerto)
        elif existe_bbox(anotacoes) and not existe_bbox(predicoes):
            percentual_de_acerto_por_subimagem.append(0)
        elif not existe_bbox(anotacoes) and not existe_bbox(predicoes):
            percentual_de_acerto_por_subimagem.append(1)
        elif not existe_bbox(anotacoes) and existe_bbox(predicoes):
            percentual_de_acerto_por_subimagem.append(0)
    return round(mean(percentual_de_acerto_por_subimagem).item(), 3)


def calcular_metrica_nas_subimagens_sem_categoria(anotacoes_subimagens: dict, deteccoes_subimagens: dict) -> float:
    anotacoes_convertidas = converter_anotacoes_para_o_padrao_de_deteccoes(anotacoes_subimagens)
    percentual_de_acerto_por_subimagem = list()
    for nome_imagem, anotacoes in anotacoes_convertidas.items():
        predicoes = deteccoes_subimagens.get(nome_imagem)
        if existe_bbox(anotacoes) and existe_bbox(predicoes):
            percentual_de_acerto = calcular_percentual_de_acerto_por_subimagem_sem_categoria(anotacoes, predicoes)
            percentual_de_acerto_por_subimagem.append(percentual_de_acerto)
        elif existe_bbox(anotacoes) and not existe_bbox(predicoes):
            percentual_de_acerto_por_subimagem.append(0)
        elif not existe_bbox(anotacoes) and not existe_bbox(predicoes):
            percentual_de_acerto_por_subimagem.append(1)
        elif not existe_bbox(anotacoes) and existe_bbox(predicoes):
            percentual_de_acerto_por_subimagem.append(0)
    return round(mean(percentual_de_acerto_por_subimagem).item(), 3)


def calcular_percentual_de_acerto_por_subimagem_com_categoria(anotacoes: list, predicoes: list) -> float:
    centros_das_anotacoes = [[] for _ in range(len(anotacoes))]
    centros_das_predicoes = [[] for _ in range(len(predicoes))]
    for i, boxes in enumerate(anotacoes):
        for box in boxes:
            centros_das_anotacoes[i].append(calcular_centro(box))
    for i, boxes in enumerate(predicoes):
        for box in boxes:
            centros_das_predicoes[i].append(calcular_centro(box))

    percentuais_de_acertos_por_categoria = list()
    for categoria_anotacao, categoria_predicao in zip(centros_das_anotacoes, centros_das_predicoes):
        if categoria_anotacao and categoria_predicao:
            matriz_de_distancias = zeros((len(categoria_anotacao), len(categoria_predicao)))
            for i, centro_i in enumerate(categoria_anotacao):
                for j, centro_j in enumerate(categoria_predicao):
                    distancia = calcular_distancia(centro_i, centro_j)
                    matriz_de_distancias[i][j] = distancia
            row_ind, col_ind = linear_sum_assignment(matriz_de_distancias)
            acertos = list()
            for row, col in zip(row_ind, col_ind):
                if matriz_de_distancias[row][col] <= float(environ.get('DISTANCIA_MINIMA_CENTROS')):
                    acertos.append(1)
            maior = max(len(categoria_anotacao), len(categoria_predicao))
            percentuais_de_acertos_por_categoria.append(len(acertos) / maior)
        elif categoria_anotacao and not categoria_predicao:
            percentuais_de_acertos_por_categoria.append(0)
        elif not categoria_anotacao and not categoria_predicao:
            percentuais_de_acertos_por_categoria.append(1)
        elif not categoria_anotacao and categoria_predicao:
            percentuais_de_acertos_por_categoria.append(0)
    return mean(percentuais_de_acertos_por_categoria)


def calcular_percentual_de_acerto_por_subimagem_sem_categoria(anotacoes: list, predicoes: list) -> float:
    centros_anotacoes = list()
    centros_predicoes = list()
    for categoria in anotacoes:
        for box in categoria:
            centros_anotacoes.append(calcular_centro(box))
    for categoria in predicoes:
        for box in categoria:
            centros_predicoes.append(calcular_centro(box))
    matriz_de_distancias = criar_matriz_de_distancias(centros_anotacoes, centros_predicoes)
    row_ind, col_ind = linear_sum_assignment(matriz_de_distancias)
    acertos = list()
    for row, col in zip(row_ind, col_ind):
        if matriz_de_distancias[row][col] <= float(environ.get('DISTANCIA_MINIMA_CENTROS')):
            acertos.append(1)
    maior = max(len(centros_anotacoes), len(centros_predicoes))
    return len(acertos) / maior


def carregar_json(caminho_arquivo: str) -> Union[dict, list]:
    with open(caminho_arquivo, 'r', encoding='utf_8') as arquivo:
        dados = load(arquivo)
    return dados


def converter_anotacoes_para_o_padrao_de_deteccoes(anotacoes_subimagens: dict) -> dict:
    anotacoes_padronizadas = dict()
    anotacoes_padronizadas_por_id_da_imagem = dict()
    for imagem in anotacoes_subimagens['images']:
        nome_da_imagem = ''.join(imagem['file_name'].split('.')[:-1])
        extensao_da_imagem = imagem['file_name'].split('.')[-1]
        chave = f'{nome_da_imagem}.{extensao_da_imagem}'
        if 'subimagem' in imagem:
            subimagem = '_'.join(map(str, imagem['subimagem']))
            chave = f'{nome_da_imagem}_{subimagem}.{extensao_da_imagem}'
        anotacoes_padronizadas[chave] = [[] for _ in range(len(anotacoes_subimagens['categories']))]
        anotacoes_padronizadas_por_id_da_imagem[imagem['id']] = chave

    for anotacao in anotacoes_subimagens['annotations']:
        id_da_imagem = anotacao['image_id']
        categoria_da_anotacao = anotacao['category_id']
        bbox = anotacao['bbox']
        subimagem = anotacoes_padronizadas_por_id_da_imagem[id_da_imagem]
        anotacoes_padronizadas[subimagem][categoria_da_anotacao - 1].append(bbox)
    return anotacoes_padronizadas


def criar_matriz_de_distancias(
        centros_anotacoes: list[tuple[Union[int, float], Union[int, float]]],
        centros_predicoes: list[tuple[Union[int, float], Union[int, float]]]
) -> array:
    matriz_de_distancias = zeros((len(centros_anotacoes), len(centros_predicoes)))
    for i, centro_i in enumerate(centros_anotacoes):
        for j, centro_j in enumerate(centros_predicoes):
            distancia = calcular_distancia(centro_i, centro_j)
            matriz_de_distancias[i][j] = distancia
    return array(matriz_de_distancias)


def existe_bbox(lista: list[list[Union[int, float]]]) -> bool:
    for elemento in lista:
        if elemento:
            return True
    return False


def remover_predicoes_com_a_mesma_localizacao(predicoes: dict) -> None:
    for chave, valor in predicoes.items():
        boxes = list()  # armazena boxes de todas as categorias em um único lugar para facilitar comparação
        for categoria in range(len(valor)):
            boxes += [box + [categoria] for box in valor[categoria]]

        boxes_para_remover = list()
        for i in range(len(boxes) - 1):
            xmin_i, ymin_i, xmax_i, ymax_i, score_i, _ = boxes[i]
            for j in range(i + 1, len(boxes)):
                xmin_j, ymin_j, xmax_j, ymax_j, score_j, _ = boxes[j]
                if xmin_i == xmin_j and ymin_i == ymin_j and xmax_i == xmax_j and ymax_i == ymax_j:
                    if score_i > score_j:
                        boxes_para_remover.append(j)
                    elif score_j > score_i:
                        boxes_para_remover.append(i)

        predicoes[chave] = [[] for _ in range(len(valor))]
        for i, box in enumerate(boxes):
            categoria = box[-1]
            if i not in boxes_para_remover:
                predicoes[chave][categoria].append(box[:-1])


def remover_predicoes_com_score_baixo(predicoes: Union[dict, list]) -> Optional[list]:
    score_minimo = float(environ.get('SCORE_MINIMO'))

    if isinstance(predicoes, dict):
        for key, predicoes_ in predicoes.items():
            for i, boxes in enumerate(predicoes_):
                predicoes_[i] = [box for box in boxes if box[-1] >= score_minimo]
            predicoes[key] = predicoes_
    elif isinstance(predicoes, list):
        predicoes = [predicao for predicao in predicoes if predicao['score'] >= score_minimo]
        return predicoes


def unir_deteccoes_das_subimagens(deteccoes_subimagens: dict) -> dict:
    """Une as detecções de cada subimagem de acordo com as seguintes regras:
    1. Qualquer um dos pontos da detecção deve estar dentro de uma área de união, ou seja, a distância entre um dos
    pontos da detecção para uma das bordas da subimagem deve ser menor ou igual a DISTANCIA_DA_AREA_DE_UNIAO.
    Detecções que não estão em área de união não serão unidas;
    2. Duas detecções serão unidas se a distância entre dois de quaisquer de seus pontos médios for menor ou igual a
    DISTANCIA_EM_PIXELS_ENTRE_PONTOS_MEDIOS. A união de duas detecções cria uma nova detecção e as duas detecções
    anteriores são excluídas;
    3. As detecções podem ser divididas entre duas ou mais subimagens. Portanto, ao unir duas detecções, duas
    subimagens se tornam uma nova subimagem, e as subimagens anteriores são excluídas;
    4. Duas detecções só podem ser unidas quando suas subimagens não possuem intersecção, mesmo após a junção de
    subimagens."""

    deteccoes_unidas = defaultdict(DataFrame)
    deteccoes_unidas_ = dict()
    imagens = defaultdict(list)
    for subimagem, deteccoes in deteccoes_subimagens.items():
        boxes = list()
        labels = list()
        scores = list()
        for label, deteccoes_classe in enumerate(deteccoes):
            if deteccoes_classe:
                for deteccao in deteccoes_classe:
                    boxes.append(deteccao[:4])
                    labels.append(label)
                    scores.append(deteccao[-1])
        df_subimagem = DataFrame(dict(
            subimagens=[subimagem for _ in range(len(boxes))],
            boxes=boxes,
            labels=labels,
            scores=scores
        ))
        nome_imagem = subimagem.split('_')[0]
        extensao = subimagem.split('.')[-1]
        imagens[f'{nome_imagem}.{extensao}'].append(df_subimagem)

    # cria um dataframe para cada imagem contendo as informações das subimagens
    for imagem, values in imagens.items():
        df_imagem = concat(values, ignore_index=True)

        # verifica as detecções da imagem estão na área de união
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
            area1 = [subimagem[0], subimagem[1], subimagem[2], subimagem[1] + DISTANCIA_DA_AREA_DE_UNIAO]
            area2 = [subimagem[0], subimagem[1], subimagem[0] + DISTANCIA_DA_AREA_DE_UNIAO, subimagem[3]]
            area3 = [subimagem[2] - DISTANCIA_DA_AREA_DE_UNIAO, subimagem[1], subimagem[2], subimagem[3]]
            area4 = [subimagem[0], subimagem[3] - DISTANCIA_DA_AREA_DE_UNIAO, subimagem[2], subimagem[3]]

            # verifica se a detecção possui intersecção com uma área de união
            box_ = [xmin, ymin, xmax, ymax]
            tem_interseccao = any([
                verificar_interseccao(box_, area1),
                verificar_interseccao(box_, area2),
                verificar_interseccao(box_, area3),
                verificar_interseccao(box_, area4)
            ])

            # se a detecção estiver em área de união, então ela será adicionada no dataframe de união para
            # processamento posterior. Caso contrário, a detecção não será unida com nenhuma outra e será adicionada no
            # dataframe de detecções unidas
            if tem_interseccao:
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

        # as uniões serão realizadas na horizontal e depois na vertical
        df_deteccoes_unidas_horizontalmente = unir_deteccoes_horizontalmente(deepcopy(df_uniao))
        df_deteccoes_unidas = unir_deteccoes_verticalmente(deepcopy(df_deteccoes_unidas_horizontalmente))
        deteccoes_unidas[imagem] = concat([deteccoes_unidas[imagem], df_deteccoes_unidas], ignore_index=True)
    for nome_imagem, df in deteccoes_unidas.items():
        for deteccao in df.iloc:
            if nome_imagem not in deteccoes_unidas_:
                deteccoes_unidas_[nome_imagem] = [[] for _ in CATEGORIAS]
            deteccoes_unidas_[nome_imagem][deteccao['labels']].append(deteccao['boxes'] + [deteccao['scores']])
    return deteccoes_unidas_


def unir_deteccoes_horizontalmente(df_uniao: DataFrame) -> DataFrame:
    """Une detecções horizontalmente próximas"""

    df_deteccoes_unidas_horizontalmente = DataFrame()
    while not df_uniao.empty:
        # os dados sempre serão ordenados
        df_uniao.sort_values(by=['boxes'], inplace=True, ignore_index=True)

        # se houver apenas uma detecção para ser unida, então ela é adicionada ao df_deteccoes_unidas_horizontalmente e
        # o laço é rompido
        if len(df_uniao) == 1:
            box_1 = df_uniao.iloc[0]
            subimagem = box_1['subimagens']
            xmin, ymin, xmax, ymax = box_1['boxes']
            box_1_pontuacao = box_1['scores']
            df_deteccoes_unidas_horizontalmente = concat([
                df_deteccoes_unidas_horizontalmente,
                DataFrame(
                    data=[[subimagem, [xmin, ymin, xmax, ymax], int(box_1['labels']), box_1_pontuacao]],
                    columns=['subimagens', 'boxes', 'labels', 'scores']
                )
            ], ignore_index=True)

            df_uniao.drop(labels=[0], axis=0, inplace=True)
            df_uniao.reset_index(inplace=True, drop=True)
            break

        for i in range(len(df_uniao) - 1):
            box_1 = df_uniao.iloc[i]
            xmin, ymin, xmax, ymax = box_1['boxes']
            box_1_pontuacao = box_1['scores']
            box_1_subimagem = box_1['subimagens']

            box_1_pontos_medios = [(xmin, (ymin + ymax) / 2), (xmax, (ymin + ymax) / 2)]

            parar = False  # utilizado quando acontece união, então é necessário interromper os laços
            for j in range(i + 1, len(df_uniao)):
                box_2 = df_uniao.iloc[j]
                box_2_subimagem = box_2['subimagens']

                # quando uma detecção é unida, duas subimagens também são. A variável interseccao guarda o valor lógico
                # de interseccao entre uma subimagem e outra. Se houver intersecção, a união das detecções não pode ser
                # realizada. Isso evita que detecções em uma mesma subimagem sejam unidas
                interseccao = verificar_interseccao(box_1_subimagem, box_2_subimagem)

                # quando labels são diferentes ou há intersecção, então estas detecções não devem ser unidas e o
                # restante do código é ignorado
                if box_1['labels'] != box_2['labels'] or interseccao:
                    continue

                xmin_, ymin_, xmax_, ymax_ = box_2['boxes']
                box_2_pontuacao = box_2['scores']
                box_2_pontos_medios = [(xmin_, (ymin_ + ymax_) / 2), (xmax_, (ymin_ + ymax_) / 2)]

                # a distância é calculada a partir dos pontos médios das box_1 e box_2
                dist_p2_p3 = calcular_distancia(box_1_pontos_medios[0], box_2_pontos_medios[1])
                dist_p3_p2 = calcular_distancia(box_1_pontos_medios[1], box_2_pontos_medios[0])

                if (dist_p2_p3 <= DISTANCIA_EM_PIXELS_ENTRE_PONTOS_MEDIOS
                        or dist_p3_p2 <= DISTANCIA_EM_PIXELS_ENTRE_PONTOS_MEDIOS):
                    # une as subimagens e cria uma nova
                    nova_subimagem_xmin = min(box_1_subimagem[0], box_2_subimagem[0])
                    nova_subimagem_ymin = min(box_1_subimagem[1], box_2_subimagem[1])
                    nova_subimagem_xmax = max(box_1_subimagem[2], box_2_subimagem[2])
                    nova_subimagem_ymax = max(box_1_subimagem[3], box_2_subimagem[3])
                    nova_subimagem = [
                        nova_subimagem_xmin,
                        nova_subimagem_ymin,
                        nova_subimagem_xmax,
                        nova_subimagem_ymax
                    ]

                    # une as detecções e cria uma nova
                    novo_xmin = min(xmin, xmin_)
                    novo_ymin = min(ymin, ymin_)
                    novo_xmax = max(xmax, xmax_)
                    novo_ymax = max(ymax, ymax_)

                    # verifica se a nova detecção está em área de união com a nova subimagem
                    area_1 = [
                        nova_subimagem[0],
                        nova_subimagem[1],
                        nova_subimagem[2],
                        nova_subimagem[1] + DISTANCIA_DA_AREA_DE_UNIAO
                    ]
                    area_2 = [
                        nova_subimagem[0],
                        nova_subimagem[1],
                        nova_subimagem[0] + DISTANCIA_DA_AREA_DE_UNIAO,
                        nova_subimagem[3]
                    ]
                    area_3 = [
                        nova_subimagem[2] - DISTANCIA_DA_AREA_DE_UNIAO,
                        nova_subimagem[1],
                        nova_subimagem[2],
                        nova_subimagem[3]
                    ]
                    area_4 = [
                        nova_subimagem[0],
                        nova_subimagem[3] - DISTANCIA_DA_AREA_DE_UNIAO,
                        nova_subimagem[2],
                        nova_subimagem[3]
                    ]

                    box_ = [novo_xmin, novo_ymin, novo_xmax, novo_ymax]
                    tem_interseccao = any([
                        verificar_interseccao(box_, area_1),
                        verificar_interseccao(box_, area_2),
                        verificar_interseccao(box_, area_3),
                        verificar_interseccao(box_, area_4)
                    ])
                    nova_pontuacao = max(box_1_pontuacao, box_2_pontuacao)

                    # se não houver intersecção entre a nova subimagem e a nova detecção unida, então é necessário
                    # adicionar a nova detecção ao df_deteccoes_unidas_horizontalmente e remover as duas detecções
                    # antigas do df_uniao
                    if not tem_interseccao:
                        df_deteccoes_unidas_horizontalmente = concat([
                            df_deteccoes_unidas_horizontalmente,
                            DataFrame(
                                data=[[nova_subimagem, box_, int(box_1['labels']), nova_pontuacao]],
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
                            data=[[nova_subimagem, box_, int(box_1['labels']), nova_pontuacao]],
                            columns=['subimagens', 'boxes', 'labels', 'scores']
                        )
                    ], ignore_index=True)

                    df_uniao.drop(labels=[i, j], axis=0, inplace=True)
                    df_uniao.reset_index(inplace=True, drop=True)
                    parar = True
                    break

            # se houve união, então é necessário interromper o laço for e voltar ao while
            if parar:
                break
            # caso contrário, a detecção i não poderá ser unida com nenhuma outra
            df_deteccoes_unidas_horizontalmente = concat([
                df_deteccoes_unidas_horizontalmente,
                DataFrame(
                    data=[[box_1_subimagem, [xmin, ymin, xmax, ymax], int(box_1['labels']), box_1_pontuacao]],
                    columns=['subimagens', 'boxes', 'labels', 'scores']
                )
            ], ignore_index=True)

            df_uniao.drop(labels=[i], axis=0, inplace=True)
            df_uniao.reset_index(inplace=True, drop=True)
            break
    return df_deteccoes_unidas_horizontalmente


def unir_deteccoes_verticalmente(df_uniao: DataFrame) -> DataFrame:
    """Une detecções verticalmente próximas"""

    df_deteccoes_unidas = DataFrame()
    while not df_uniao.empty:
        # os dados sempre serão ordenados
        df_uniao.sort_values(by=['boxes'], inplace=True, ignore_index=True)

        # se houver apenas uma detecção para ser unida, então ela é adicionada ao df_deteccoes_unidas e o laço é rompido
        if len(df_uniao) == 1:
            box_1 = df_uniao.iloc[0]
            xmin, ymin, xmax, ymax = box_1['boxes']
            box_1_pontuacao = box_1['scores']
            df_deteccoes_unidas = concat([
                df_deteccoes_unidas,
                DataFrame(
                    data=[[[xmin, ymin, xmax, ymax], int(box_1['labels']), box_1_pontuacao]],
                    columns=['boxes', 'labels', 'scores']
                )
            ], ignore_index=True)

            df_uniao.drop(labels=[0], axis=0, inplace=True)
            df_uniao.reset_index(inplace=True, drop=True)
            break

        for i in range(len(df_uniao) - 1):
            box_1 = df_uniao.iloc[i]
            xmin, ymin, xmax, ymax = box_1['boxes']
            box_1_pontuacao = box_1['scores']
            box_1_subimagem = box_1['subimagens']

            box_1_pontos_medios = [((xmin + xmax) / 2, ymin), ((xmin + xmax) / 2, ymax)]

            parar = False  # utilizado quando acontece união, então é necessário interromper os laços
            for j in range(i + i, len(df_uniao)):
                box_2 = df_uniao.iloc[j]
                box_2_subimagem = box_2['subimagens']

                # quando uma detecção é unida, suas subimagens também são. A variável interseccao guarda o valor lógico
                # de intersecção entre uma subimagem e outra. Se houver intersecção, a união das detecções não pode ser
                # realizada, para evitar que uniões na mesma subimagem sejam feitas
                interseccao = verificar_interseccao(box_1_subimagem, box_2_subimagem)

                # quando labels são diferentes ou há intersecção, então estas detecções não devem ser unidas e o
                # restante do código é ignorado
                if box_1['labels'] != box_2['labels'] or interseccao:
                    continue

                xmin_, ymin_, xmax_, ymax_ = box_2['boxes']
                box_2_pontuacao = box_2['scores']
                box_2_pontos_medios = [((xmin_ + xmax_) / 2, ymin_), ((xmin_ + xmax_) / 2, ymax_)]

                # a distância é calculada a partir dos pontos médios das box_1 e box_2
                dist_p1_p4 = calcular_distancia(box_1_pontos_medios[0], box_2_pontos_medios[1])
                dist_p4_p1 = calcular_distancia(box_1_pontos_medios[1], box_2_pontos_medios[0])

                if (dist_p1_p4 <= DISTANCIA_EM_PIXELS_ENTRE_PONTOS_MEDIOS
                        or dist_p4_p1 <= DISTANCIA_EM_PIXELS_ENTRE_PONTOS_MEDIOS):
                    # une as duas subimagens e cria uma nova
                    nova_subimagem_xmin = min(box_1_subimagem[0], box_2_subimagem[0])
                    nova_subimagem_ymin = min(box_1_subimagem[1], box_2_subimagem[1])
                    nova_subimagem_xmax = max(box_1_subimagem[2], box_2_subimagem[2])
                    nova_subimagem_ymax = max(box_1_subimagem[3], box_2_subimagem[3])
                    nova_subimagem = [
                        nova_subimagem_xmin,
                        nova_subimagem_ymin,
                        nova_subimagem_xmax,
                        nova_subimagem_ymax
                    ]

                    # une as duas detecções e cria uma nova
                    novo_xmin = min(xmin, xmin_)
                    novo_ymin = min(ymin, ymin_)
                    novo_xmax = max(xmax, xmax_)
                    novo_ymax = max(ymax, ymax_)

                    # verifica se a nova detecção está em área de união com a nova subimagem
                    area1 = [
                        nova_subimagem[0],
                        nova_subimagem[1],
                        nova_subimagem[2],
                        nova_subimagem[1] + DISTANCIA_DA_AREA_DE_UNIAO
                    ]
                    area2 = [
                        nova_subimagem[0],
                        nova_subimagem[1],
                        nova_subimagem[0] + DISTANCIA_DA_AREA_DE_UNIAO,
                        nova_subimagem[3]
                    ]
                    area3 = [
                        nova_subimagem[2] - DISTANCIA_DA_AREA_DE_UNIAO,
                        nova_subimagem[1],
                        nova_subimagem[2],
                        nova_subimagem[3]
                    ]
                    area4 = [
                        nova_subimagem[0],
                        nova_subimagem[3] - DISTANCIA_DA_AREA_DE_UNIAO,
                        nova_subimagem[2],
                        nova_subimagem[3]
                    ]

                    box_ = [novo_xmin, novo_ymin, novo_xmax, novo_ymax]
                    tem_interseccao = any([
                        verificar_interseccao(box_, area1),
                        verificar_interseccao(box_, area2),
                        verificar_interseccao(box_, area3),
                        verificar_interseccao(box_, area4)
                    ])
                    nova_pontuacao = max(box_1_pontuacao, box_2_pontuacao)
                    if not tem_interseccao:
                        df_deteccoes_unidas = concat([
                            df_deteccoes_unidas,
                            DataFrame(
                                data=[[box_, int(box_1['labels']), nova_pontuacao]],
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
                            data=[[nova_subimagem, box_, int(box_1['labels']), nova_pontuacao]],
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
                    data=[[[xmin, ymin, xmax, ymax], int(box_1['labels']), box_1_pontuacao]],
                    columns=['boxes', 'labels', 'scores']
                )
            ], ignore_index=True)

            df_uniao.drop(labels=[i], axis=0, inplace=True)
            df_uniao.reset_index(inplace=True, drop=True)
            break
    return df_deteccoes_unidas


def verificar_interseccao(retangulo_1: list, retangulo_2: list) -> bool:
    """Verifica se há intersecção entre dois retângulos"""

    xmin = max(retangulo_1[0], retangulo_2[0])
    ymin = max(retangulo_1[1], retangulo_2[1])
    xmax = min(retangulo_1[2], retangulo_2[2])
    ymax = min(retangulo_1[3], retangulo_2[3])
    area_interseccao = max(0, xmax - xmin) * max(0, ymax - ymin)
    return area_interseccao > 0


def main():
    anotacoes_imagens = carregar_json(environ.get('ARQUIVO_ANOTACOES_IMAGENS'))
    anotacoes_subimagens = carregar_json(environ.get('ARQUIVO_ANOTACOES_SUBIMAGENS'))
    deteccoes_subimagens = carregar_json(environ.get('ARQUIVO_DETECCOES_SUBIMAGENS'))
    predicoes_segmentacao = carregar_json(environ.get('ARQUIVO_PREDICOES_SEGMENTACAO'))

    remover_predicoes_com_score_baixo(deteccoes_subimagens)
    predicoes_segmentacao = remover_predicoes_com_score_baixo(predicoes_segmentacao)
    remover_predicoes_com_a_mesma_localizacao(deteccoes_subimagens)
    aplicar_batched_nms(deteccoes_subimagens)

    deteccoes_imagens = unir_deteccoes_das_subimagens(deteccoes_subimagens)

    print('Pontuação nas imagens considerando apenas a localização e ignorando as categorias')
    print('Detecção: {}\t Segmentação: {}\n'.format(
        calcular_metrica_nas_imagens_sem_categoria(anotacoes_imagens, deteccoes_imagens),
        'Implementando...'
    ))

    print('Pontuação nas imagens considerando localização e categorias')
    print('Detecção: {}\t Segmentação: {}\n'.format(
        calcular_metrica_nas_imagens_com_categoria(anotacoes_imagens, deteccoes_imagens),
        'Implementando...'
    ))

    print('Pontuação nas subimagens considerando apenas a localização e ignorando as categorias')
    print('Detecção: {}\t Segmentação: {}\n'.format(
        calcular_metrica_nas_subimagens_sem_categoria(anotacoes_subimagens, deteccoes_subimagens),
        'Implementando...'
    ))

    print('Pontuação nas subimagens considerando localização e categorias')
    print('Detecção: {}\t Segmentação: {}\n'.format(
        calcular_metrica_nas_subimagens_com_categoria(anotacoes_subimagens, deteccoes_subimagens),
        'Implementando...'
    ))


if __name__ == '__main__':
    main()
