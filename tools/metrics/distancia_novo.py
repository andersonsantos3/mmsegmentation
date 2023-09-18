from json import load
from os import environ
from typing import Union

from numpy import array, mean, sqrt, zeros
from scipy.optimize import linear_sum_assignment
from torch import float32, int64, tensor
from torchvision.ops import batched_nms


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


def calcular_metrica_nas_subimagens_com_categoria(anotacoes_subimagens: dict, deteccoes_subimagens: dict) -> None:
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
    print('Pontuação nas subimagens considerando localização e categorias')
    print(mean(percentual_de_acerto_por_subimagem).item())
    print()


def calcular_metrica_nas_subimagens_sem_categoria(anotacoes_subimagens: dict, deteccoes_subimagens: dict) -> None:
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
    print('Pontuação nas subimagens considerando apenas a localização e ignorando as categorias')
    print(mean(percentual_de_acerto_por_subimagem).item())
    print()


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


def remover_predicoes_com_score_baixo(predicoes: dict) -> None:
    score_minimo = float(environ.get('SCORE_MINIMO'))
    for key, predicoes_ in predicoes.items():
        for i, boxes in enumerate(predicoes_):
            predicoes_[i] = [box for box in boxes if box[-1] >= score_minimo]
        predicoes[key] = predicoes_


def main():
    anotacoes_subimagens = carregar_json(environ.get('ARQUIVO_ANOTACOES_SUBIMAGENS'))
    deteccoes_subimagens = carregar_json(environ.get('ARQUIVO_DETECCOES_SUBIMAGENS'))

    remover_predicoes_com_score_baixo(deteccoes_subimagens)
    remover_predicoes_com_a_mesma_localizacao(deteccoes_subimagens)
    aplicar_batched_nms(deteccoes_subimagens)

    calcular_metrica_nas_subimagens_sem_categoria(anotacoes_subimagens, deteccoes_subimagens)
    calcular_metrica_nas_subimagens_com_categoria(anotacoes_subimagens, deteccoes_subimagens)


if __name__ == '__main__':
    main()
