from collections import defaultdict
from json import load
from os import environ
from typing import Union


def carregar_json(caminho_arquivo: str) -> Union[dict, list]:
    with open(caminho_arquivo, 'r', encoding='utf_8') as arquivo:
        dados = load(arquivo)
    return dados


def excluir_predicoes_iguais(predicoes: dict) -> dict:
    for imagem_id, boxes in predicoes.items():
        ids_para_excluir = list()
        for i in range(0, len(boxes) - 1):
            box_i = boxes[i]['bbox']
            score_i = boxes[i]['score']
            for j in range(i + 1, len(boxes)):
                box_j = boxes[j]['bbox']
                score_j = boxes[j]['score']
                if box_i[0] == box_j[0] and box_i[1] == box_j[1] and box_i[2] == box_j[2] and box_i[3] == box_j[3]:
                    if score_i > score_j:
                        ids_para_excluir.append(j)
                    elif score_j > score_i:
                        ids_para_excluir.append(i)
        predicoes[imagem_id] = [predicao for id_, predicao in enumerate(boxes) if id_ not in ids_para_excluir]
        print(len(boxes), len(predicoes[imagem_id]))
    return predicoes


def filtrar_predicoes_por_score(predicoes: list, score: float) -> list:
    predicoes = [predicao for predicao in predicoes if predicao['score'] >= score]
    return predicoes


def obter_anotacoes_por_imagem_id(anotacoes: dict) -> dict:
    annotations = defaultdict(list)
    for annotation in anotacoes['annotations']:
        annotations[annotation['image_id']].append(annotation)
    return annotations


def obter_predicoes_por_imagem_id(predicoes: dict) -> dict:
    predictions = defaultdict(list)
    for pred in predicoes:
        predictions[pred['image_id']].append(pred)
    return predictions


def main():
    anotacoes_imagens = carregar_json(environ.get('CAMINHO_ANOTACOES_IMAGENS'))
    predicoes_deteccoes = carregar_json(environ.get('CAMINHO_PREDICOES_DETECCOES'))
    score_minimo = float(environ.get('SCORE_MINIMO'))
    predicoes_deteccoes = filtrar_predicoes_por_score(predicoes_deteccoes, score_minimo)

    categorias = anotacoes_imagens['categories']
    anotacoes_por_imagem_id = obter_anotacoes_por_imagem_id(anotacoes_imagens)
    predicoes_por_imagem_id = obter_predicoes_por_imagem_id(predicoes_deteccoes)
    predicoes_por_imagem_id = excluir_predicoes_iguais(predicoes_por_imagem_id)


if __name__ == '__main__':
    main()
