from collections import defaultdict
from json import load
from os import environ
from typing import Union


def carregar_json(caminho_arquivo: str) -> Union[dict, list]:
    with open(caminho_arquivo, 'r', encoding='utf_8') as arquivo:
        dados = load(arquivo)
    return dados


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

    categorias = anotacoes_imagens['categories']
    anotacoes_por_imagem_id = obter_anotacoes_por_imagem_id(anotacoes_imagens)
    predicoes_por_imagem_id = obter_predicoes_por_imagem_id(predicoes_deteccoes)


if __name__ == '__main__':
    main()
