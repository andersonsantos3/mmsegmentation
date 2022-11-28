# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='MixVisionTransformer',
        in_channels=3,
        embed_dims=32,
        num_stages=4,
        num_layers=[2, 2, 2, 2],
        num_heads=[1, 2, 5, 8],
        patch_sizes=[7, 3, 3, 3],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3),
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1),
    decode_head=dict(
        type='SegformerHead',
        ignore_index=255,
        in_channels=[32, 64, 160, 256],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            # class_weight=[  # pesos com todo o dataset
            #     1.44263754e-01,
            #     5.17139259e+02,
            #     2.19240992e+01,
            #     1.82440613e+02,
            #     2.23583035e+02,
            #     1.40073904e+02,
            #     2.76751305e+02
            # ],
            # class_weight=[  # pesos do conjunto de treino
            #     1.44268376e-01,
            #     4.95711274e+02,
            #     2.20576611e+01,
            #     1.66404423e+02,
            #     2.37547322e+02,
            #     1.38480993e+02,
            #     2.71692673e+02
            # ],
            use_sigmoid=False,
            loss_weight=1.0,
            avg_non_ignore=True
        )
    ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
