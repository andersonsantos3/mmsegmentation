from argparse import ArgumentParser
from collections import defaultdict
from json import load as load_json
from os import listdir
from os.path import basename, join
from pickle import load as load_pickle
from typing import Union

from mmcv import imread, imshow, imwrite
import cv2


diretorio_imagens: str = None


def parse_args():
    parser = ArgumentParser(description='Argumentos para exibir anotações e predições')

    parser.add_argument('diretorio_imagens', help='Diretório até as imagens inteiras')
    parser.add_argument('caminho_anotacoes', help='Caminho até o arquivo json com as anotações inteiras')
    parser.add_argument(
        'caminho_anotacoes_subimagens',
        help='Caminho até o arquivo json com as anotações em subimagens'
    )
    parser.add_argument(
        'caminho_predicoes_segmentacao',
        help='Caminho até o arquivo json com as predições da segmentação transformadas em bbox'
    )
    parser.add_argument('caminho_predicoes_deteccao', help='Caminho até o arquivo json com as predições de detecção')

    args = parser.parse_args()
    return args


def exibir(
        imagens: list[str],
        anotacoes_imagens: dict,
        anotacoes_subimagens: dict,
        predicoes_segmentacao: dict,
        predicoes_deteccao: list
) -> None:
    id_file_name_imagem = {
        imagem['id']: imagem['file_name'].removesuffix('.jpg')
        for imagem in anotacoes_imagens['images']
    }

    imagens_por_nome = defaultdict(dict)
    for imagem in imagens:
        nome_imagem = basename(imagem).removesuffix('.jpg')
        imagens_por_nome[nome_imagem] = dict(anotacoes=list(), predicoes_segmentacao=list(), predicoes_deteccao=list())

    for anotacao in anotacoes_imagens['annotations']:
        image_id = anotacao['image_id']
        bbox = anotacao['bbox']
        category_id = anotacao['category_id']
        nome_da_imagem = id_file_name_imagem[image_id]
        imagens_por_nome[nome_da_imagem]['anotacoes'].append(dict(bbox=bbox, category_id=category_id))

    for predicao in predicoes_segmentacao:
        image_id = predicao['image_id']
        bbox = predicao['bbox']
        category_id = predicao['category_id']
        nome_da_imagem = id_file_name_imagem[image_id]
        imagens_por_nome[nome_da_imagem]['predicoes_segmentacao'].append(dict(bbox=bbox, category_id=category_id))

    assert len(anotacoes_subimagens['images']) == len(predicoes_deteccao)
    alinhar_predicoes_deteccao(imagens_por_nome, predicoes_deteccao)

    for nome_imagem in imagens_por_nome:
        if imagens_por_nome[nome_imagem]['anotacoes'] \
                or imagens_por_nome[nome_imagem]['predicoes_segmentacao'] \
                or imagens_por_nome[nome_imagem]['predicoes_deteccao']:
            caminho_imagem = join(diretorio_imagens, nome_imagem + '.jpg')
            im = imread(caminho_imagem)

            for anotacao in imagens_por_nome[nome_imagem]['anotacoes']:
                bbox = anotacao['bbox']
                category_id = anotacao['category_id']
                im = cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

            for predicao in imagens_por_nome[nome_imagem]['predicoes_segmentacao']:
                bbox = predicao['bbox']
                bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
                category_id = anotacao['category_id']
                im = cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)

            for predicao in imagens_por_nome[nome_imagem]['predicoes_deteccao']:
                xmin, ymin, xmax, ymax = predicao['bbox']
                category_id = anotacao['category_id']
                im = cv2.rectangle(im, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)

            # largura = int(im.shape[1] * 0.4)
            # altura = int(im.shape[0] * 0.4)
            # dim = (largura, altura)
            # im = cv2.resize(im, dim, interpolation=cv2.INTER_AREA)

            # imshow(im, nome_imagem)
            base = '/home/anderson/PycharmProjects/mmsegmentation/work_dirs/v3.0.0/b3/ann_seg_bbox'
            imwrite(im, join(base, nome_imagem + '.jpg'))
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()


def alinhar_predicoes_deteccao(imagens_por_nome, predicoes_deteccao):
    for nome_imagem, predicoes in predicoes_deteccao.items():
        nome_imagem = nome_imagem.removesuffix('.jpg')
        subimagem = list(map(int, nome_imagem.split('_')[1:]))
        nome_imagem = nome_imagem.split('_')[0]
        for category_id, predicoes_da_classe in enumerate(predicoes, start=1):
            for xmin, ymin, xmax, ymax, score in predicoes_da_classe:
                if score >= 0.7:
                    bbox = list(map(
                        int,
                        [xmin + subimagem[0], ymin + subimagem[1], xmax + subimagem[0], ymax + subimagem[1]]
                    ))
                    imagens_por_nome[nome_imagem]['predicoes_deteccao'].append(dict(
                        bbox=bbox, category_id=category_id
                    ))


def ler_json(caminho_arquivo: str) -> Union[dict, list]:
    with open(caminho_arquivo, 'r', encoding='utf_8') as arquivo:
        dados = load_json(arquivo)
    return dados


def ler_pickle(caminho_arquivo: str) -> list:
    with open(caminho_arquivo, 'rb') as arquivo:
        dados = load_pickle(arquivo)
    return dados


def main():
    global diretorio_imagens

    args = parse_args()
    diretorio_imagens = args.diretorio_imagens

    imagens = [join(diretorio_imagens, imagem) for imagem in listdir(diretorio_imagens) if imagem.endswith('.jpg')]
    anotacoes = ler_json(args.caminho_anotacoes)
    anotacoes_subimagens = ler_json(args.caminho_anotacoes_subimagens)
    predicoes_segmentacao = ler_json(args.caminho_predicoes_segmentacao)
    predicoes_deteccao = ler_json(args.caminho_predicoes_deteccao)

    exibir(imagens, anotacoes, anotacoes_subimagens, predicoes_segmentacao, predicoes_deteccao)


if __name__ == '__main__':
    main()
