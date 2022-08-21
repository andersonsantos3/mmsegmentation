# Copyright (c) OpenMMLab. All rights reserved.
from json import load
from os import environ
from os.path import join
from typing import List, NamedTuple
import os.path as osp

import mmcv
import numpy as np
from numpy import arange, flip, ndarray, outer, zeros, float32

from ..builder import PIPELINES


class Anotacao(NamedTuple):
    xmin: int
    ymin: int
    xmax: int
    ymax: int
    categoria: int


class Imagem(NamedTuple):
    altura: int
    largura: int
    anotacoes: List[Anotacao]


@PIPELINES.register_module()
class LoadImageFromFile(object):
    """Load an image from file.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'cv2'
    """

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='cv2'):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('img_prefix') is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']
        img_bytes = self.file_client.get(filename)
        img = mmcv.imfrombytes(
            img_bytes, flag=self.color_type, backend=self.imdecode_backend)
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32},'
        repr_str += f"color_type='{self.color_type}',"
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str


@PIPELINES.register_module()
class LoadAnnotations(object):
    """Load annotations for semantic segmentation.

    Args:
        reduce_zero_label (bool): Whether reduce all label value by 1.
            Usually used for datasets where 0 is background label.
            Default: False.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'pillow'
    """

    def __init__(self,
                 reduce_zero_label=False,
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='pillow'):
        self.reduce_zero_label = reduce_zero_label
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('seg_prefix', None) is not None:
            filename = osp.join(results['seg_prefix'],
                                results['ann_info']['seg_map'])
        else:
            filename = results['ann_info']['seg_map']
        img_bytes = self.file_client.get(filename)
        gt_semantic_seg = mmcv.imfrombytes(
            img_bytes, flag='unchanged',
            backend=self.imdecode_backend).squeeze().astype(np.uint8)
        # modify if custom classes
        if results.get('label_map', None) is not None:
            # Add deep copy to solve bug of repeatedly
            # replace `gt_semantic_seg`, which is reported in
            # https://github.com/open-mmlab/mmsegmentation/pull/1445/
            gt_semantic_seg_copy = gt_semantic_seg.copy()
            for old_id, new_id in results['label_map'].items():
                gt_semantic_seg[gt_semantic_seg_copy == old_id] = new_id
        # reduce zero_label
        if self.reduce_zero_label:
            # avoid using underflow conversion
            gt_semantic_seg[gt_semantic_seg == 0] = 255
            gt_semantic_seg = gt_semantic_seg - 1
            gt_semantic_seg[gt_semantic_seg == 254] = 255
        results['gt_semantic_seg'] = gt_semantic_seg
        results['seg_fields'].append('gt_semantic_seg')
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_zero_label={self.reduce_zero_label},'
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str


@PIPELINES.register_module()
class LoadGaussianAnnotations(object):
    """Load annotations for semantic segmentation.

    Args:
        reduce_zero_label (bool): Whether reduce all label value by 1.
            Usually used for datasets where 0 is background label.
            Default: False.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'pillow'
    """

    def __init__(self,
                 reduce_zero_label=False,
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='cv2'):
        self.reduce_zero_label = reduce_zero_label
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        conjunto = results['seg_prefix'].split('/')[-1] + '.json'

        nome_imagem = results['img_info']['filename']
        extensao_imagem = nome_imagem.split('.')[-1]
        infos_imagem = nome_imagem.split('.')[0].split('_')
        nome_imagem = '.'.join([infos_imagem[0], extensao_imagem])
        subimagem = list(map(int, infos_imagem[1:]))
        altura, largura, _ = results['img_shape']

        with open(join(environ['ARQUIVOS'], conjunto), 'r', encoding='utf_8') as arquivo:
            dados = load(arquivo)

        id_imagem = [
            imagem['id']
            for imagem in dados['images']
            if imagem['file_name'] == nome_imagem and imagem['subimagem'] == subimagem
        ]

        assert len(id_imagem) == 1, '{} ids encontrados, apenas um é permitido'.format(len(id_imagem))
        id_imagem = id_imagem[0]

        anotacoes = [Anotacao(
            xmin=anotacao['bbox'][0],
            ymin=anotacao['bbox'][1],
            xmax=anotacao['bbox'][2],
            ymax=anotacao['bbox'][3],
            categoria=anotacao['category_id']
        ) for anotacao in dados['annotations'] if anotacao['image_id'] == id_imagem]

        imagem = Imagem(
            altura=altura,
            largura=largura,
            anotacoes=anotacoes
        )
        gt_semantic_seg = criar_mascara(imagem)

        # modify if custom classes
        if results.get('label_map', None) is not None:
            # Add deep copy to solve bug of repeatedly
            # replace `gt_semantic_seg`, which is reported in
            # https://github.com/open-mmlab/mmsegmentation/pull/1445/
            gt_semantic_seg_copy = gt_semantic_seg.copy()
            for old_id, new_id in results['label_map'].items():
                gt_semantic_seg[gt_semantic_seg_copy == old_id] = new_id
        # reduce zero_label
        if self.reduce_zero_label:
            # avoid using underflow conversion
            gt_semantic_seg[gt_semantic_seg == 0] = 255
            gt_semantic_seg = gt_semantic_seg - 1
            gt_semantic_seg[gt_semantic_seg == 254] = 255
        results['gt_semantic_seg'] = gt_semantic_seg
        results['seg_fields'].append('gt_semantic_seg')
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_zero_label={self.reduce_zero_label},'
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str


def criar_mascara(imagem: Imagem) -> ndarray:
    # mascara = zeros(shape=(imagem.altura, imagem.largura, 6), dtype=float32)
    mascara = zeros(shape=(imagem.altura, imagem.largura), dtype=float32)
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

        # mascara[anotacao.ymin:anotacao.ymax, anotacao.xmin:anotacao.xmax, anotacao.categoria - 1] = outer(y, x)
        # if anotacao.categoria != 1:
        #     mascara[anotacao.ymin:anotacao.ymax, anotacao.xmin:anotacao.xmax, anotacao.categoria - 1] += \
        #         anotacao.categoria

        mascara[anotacao.ymin:anotacao.ymax, anotacao.xmin:anotacao.xmax] = outer(y, x)
        if anotacao.categoria != 1:
            mascara[anotacao.ymin:anotacao.ymax, anotacao.xmin:anotacao.xmax] += \
                anotacao.categoria
    return mascara
