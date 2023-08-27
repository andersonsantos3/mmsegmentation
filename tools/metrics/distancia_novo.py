from json import load
from os import environ
from typing import Union


def carregar_json(caminho_arquivo: str) -> Union[dict, list]:
    with open(caminho_arquivo, 'r', encoding='utf_8') as arquivo:
        dados = load(arquivo)
    return dados


def main():
    anotacoes_imagens = carregar_json(environ.get('CAMINHO_ANOTACOES_IMAGENS'))
    predicoes_deteccoes = carregar_json(environ.get('CAMINHO_PREDICOES_DETECCOES'))


if __name__ == '__main__':
    main()
