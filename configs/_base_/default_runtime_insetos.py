# yapf:disable
log_config = dict(
    interval=1,
    by_epoch=True,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=True),
        # dict(type='TensorboardLoggerHook', by_epoch=True)
        # dict(type='PaviLoggerHook') # for internal services
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1), ('val', 1)]
cudnn_benchmark = True
