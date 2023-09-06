from json import load
from os import environ
from typing import Union

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
            predicoes_[label] += [[box] + [score]]
        predicoes[chave] = predicoes_


def carregar_json(caminho_arquivo: str) -> Union[dict, list]:
    with open(caminho_arquivo, 'r', encoding='utf_8') as arquivo:
        dados = load(arquivo)
    return dados


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


if __name__ == '__main__':
    main()
