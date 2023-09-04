from json import load
from os import environ
from typing import Union


def carregar_json(caminho_arquivo: str) -> Union[dict, list]:
    with open(caminho_arquivo, 'r', encoding='utf_8') as arquivo:
        dados = load(arquivo)
    return dados


def remover_predicoes_com_score_baixo(predicoes: dict):
    score_minimo = float(environ.get('SCORE_MINIMO'))
    for key, predicoes_ in predicoes.items():
        for i, boxes in enumerate(predicoes_):
            predicoes_[i] = [box for box in boxes if box[-1] >= score_minimo]
        predicoes[key] = predicoes_


def main():
    anotacoes_subimagens = carregar_json(environ.get('ARQUIVO_ANOTACOES_SUBIMAGENS'))
    deteccoes_subimagens = carregar_json(environ.get('ARQUIVO_DETECCOES_SUBIMAGENS'))

    remover_predicoes_com_score_baixo(deteccoes_subimagens)


if __name__ == '__main__':
    main()
