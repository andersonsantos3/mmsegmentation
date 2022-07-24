from argparse import ArgumentParser
from collections import defaultdict
from json import load
from os import makedirs
from os.path import exists, join
from PIL.Image import Image, fromarray, open as open_image
from typing import DefaultDict, Dict, List, NamedTuple, Union

from tifffile import imwrite
from numpy import arange, flip, ndarray, outer, zeros, float32


class Anotacao(NamedTuple):
    xmin: int
    ymin: int
    xmax: int
    ymax: int
    categoria: int


class Imagem(NamedTuple):
    nome: str
    extensao: str
    subimagem: Union[List[int], None]
    altura: int
    largura: int
    anotacoes: List[Anotacao]


def parse_args():
    parser = ArgumentParser(
        description='Converte anotações no padrão COCO para máscaras de segmentação no formato de gaussianas.'
    )

    parser.add_argument('diretorio_jsons', help='Diretório que contém os arquivos jsons')
    parser.add_argument('arquivos_json', help='Nomes dos arquivos json. Exemplo: treino.json validacao.json teste.json')
    parser.add_argument('diretorio_imagens', help='Diretório que contém as imagens')
    parser.add_argument(
        'diretorio_saida',
        help='Diretório para salvar os dados de saída. Caso já existir, os dados podem ser sobreescritos'
    )
    parser.add_argument('--int', help='Salvar a máscara com números inteiros', action='store_true')

    args = parser.parse_args()
    return args


def abrir_imagem(diretorio_imagem: str, imagem: Imagem) -> Image:
    img = open_image(join(diretorio_imagem, imagem.nome + imagem.extensao))
    if imagem.subimagem:
        img = img.crop(tuple(imagem.subimagem))
    return img


def criar_diretorios_saida(diretorio_saida: str) -> None:
    if not exists(diretorio_saida):
        makedirs(diretorio_saida, exist_ok=True)
    makedirs(join(diretorio_saida, 'annotations', 'treino'), exist_ok=True)
    makedirs(join(diretorio_saida, 'annotations', 'validacao'), exist_ok=True)
    makedirs(join(diretorio_saida, 'annotations', 'teste'), exist_ok=True)
    makedirs(join(diretorio_saida, 'images', 'treino'), exist_ok=True)
    makedirs(join(diretorio_saida, 'images', 'validacao'), exist_ok=True)
    makedirs(join(diretorio_saida, 'images', 'teste'), exist_ok=True)
    makedirs(join(diretorio_saida, 'imagesLists'), exist_ok=True)


def criar_images_lists(imagens: List[Imagem], diretorio_saida: str, arquivo_saida: str) -> None:
    linhas = '\n'.join([criar_nome_imagem(imagem) for imagem in imagens])
    with open(join(diretorio_saida, arquivo_saida), 'w', encoding='utf_8') as arquivo, \
            open(join(diretorio_saida, 'all.txt'), 'a', encoding='utf_8') as arquivo_all:
        arquivo.writelines(linhas)
        arquivo_all.writelines(linhas + '\n')


def criar_mascara(imagem: Imagem) -> ndarray:
    mascara = zeros(shape=(6, imagem.altura, imagem.largura), dtype=float32)
    for anotacao in imagem.anotacoes:
        largura_anotacao = anotacao.xmax - anotacao.xmin
        altura_anotacao = anotacao.ymax - anotacao.ymin

        meio_x = largura_anotacao // 2
        meio_y = altura_anotacao // 2

        x = zeros(largura_anotacao)
        y = zeros(altura_anotacao)
        x[:meio_x] = (arange(meio_x + 1) / (meio_x + 1))[1:]
        y[:meio_y] = (arange(meio_y + 1) / (meio_y + 1))[1:]
        x[meio_x:] = flip(arange(len(x[meio_x:]) + 1) / len(x[meio_x:]))[:-1]
        y[meio_y:] = flip(arange(len(y[meio_y:]) + 1) / len(y[meio_y:]))[:-1]

        mascara[anotacao.categoria - 1, anotacao.ymin:anotacao.ymax, anotacao.xmin:anotacao.xmax] = outer(y, x)
        if anotacao.categoria != 1:
            mascara[anotacao.categoria - 1, anotacao.ymin:anotacao.ymax, anotacao.xmin:anotacao.xmax] += anotacao.categoria
    return mascara


def criar_mascara_int(imagem: Imagem) -> ndarray:
    mascara = zeros(shape=(imagem.altura, imagem.largura), dtype=float)
    for anotacao in imagem.anotacoes:
        largura_anotacao = anotacao.xmax - anotacao.xmin
        altura_anotacao = anotacao.ymax - anotacao.ymin

        mascara_anotacao = zeros(shape=(altura_anotacao, largura_anotacao)) + anotacao.categoria
        mascara[anotacao.ymin:anotacao.ymax, anotacao.xmin:anotacao.xmax] = mascara_anotacao
    return mascara


def criar_nome_imagem(imagem: Imagem) -> str:
    nome_imagem = imagem.nome
    if imagem.subimagem:
        nome_imagem = '_'.join(map(str, [nome_imagem] + imagem.subimagem))
    return nome_imagem


def listar_anotacoes_por_id_imagem(anotacoes: List[Dict[str, Union[int, str]]]) -> DefaultDict[str, List[Anotacao]]:
    anotacoes_por_id_imagem = defaultdict(list)
    for anotacao in anotacoes:
        anotacoes_por_id_imagem[anotacao['image_id']].append(
            Anotacao(
                xmin=anotacao['bbox'][0],
                ymin=anotacao['bbox'][1],
                xmax=anotacao['bbox'][2],
                ymax=anotacao['bbox'][3],
                categoria=anotacao['category_id']
            )
        )
    return anotacoes_por_id_imagem


def listar_imagens(dados: Dict[str, List[Dict[str, Union[int, str]]]]) -> List[Imagem]:
    anotacoes_por_id_imagem = listar_anotacoes_por_id_imagem(dados['annotations'])
    imagens = list()
    for imagem in dados['images']:
        imagens.append(
            Imagem(
                nome=imagem['file_name'][:-4],
                extensao=imagem['file_name'][-4:],
                subimagem=imagem.get('subimagem'),
                altura=imagem['height'],
                largura=imagem['width'],
                anotacoes=anotacoes_por_id_imagem[imagem['id']]
            )
        )
    return imagens


def salvar_mascara(mascara: ndarray, caminho_destino: str) -> None:
    imwrite(caminho_destino, mascara)


def salvar_mascara_int(mascara: ndarray, caminho_destino: str) -> None:
    img = fromarray(mascara).convert('L')
    img.save(caminho_destino)


def main():
    args = parse_args()

    criar_diretorios_saida(args.diretorio_saida)
    for arquivo_json in args.arquivos_json.split():
        with open(join(args.diretorio_jsons, arquivo_json), 'r', encoding='utf_8') as arquivo:
            print(join(args.diretorio_jsons, arquivo_json))
            dados = load(arquivo)

        imagens = listar_imagens(dados)
        for imagem in imagens:
            img = abrir_imagem(args.diretorio_imagens, imagem)
            img.save(join(args.diretorio_saida, 'images', arquivo_json[:-5], criar_nome_imagem(imagem) + imagem.extensao))

            if args.int:
                mascara = criar_mascara_int(imagem)
                salvar_mascara_int(
                    mascara=mascara,
                    caminho_destino=join(
                        args.diretorio_saida,
                        'annotations',
                        arquivo_json[:-5],
                        criar_nome_imagem(imagem) + '.png'
                    )
                )
            else:
                mascara = criar_mascara(imagem)
                salvar_mascara(
                    mascara=mascara,
                    caminho_destino=join(
                        args.diretorio_saida,
                        'annotations',
                        arquivo_json[:-5],
                        criar_nome_imagem(imagem) + '.tif'
                    )
                )
        criar_images_lists(imagens, join(args.diretorio_saida, 'imagesLists'), arquivo_json[:-5] + '.txt')


if __name__ == '__main__':
    main()
