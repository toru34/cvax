config = {
    'Stage1': {
        'Block1': (
            {'kernel_shape': (64, 64, 1, 1)},
            {'kernel_shape': (64, 64, 3, 3)},
            {'kernel_shape': (256, 256, 1, 1)},
        ),
        'Block2': (
            {'kernel_shape': (64, 256, 1, 1)},
            {'kernel_shape': (64, 64, 3, 3)},
            {'kernel_shape': (256, 256, 1, 1)},
        ),
        'Block3': (
            {'kernel_shape': (64, 256, 1, 1)},
            {'kernel_shape': (64, 64, 3, 3)},
            {'kernel_shape': (256, 256, 1, 1)},
        )
    },
    'Stage2': {
        'Block1': (
            {'kernel_shape': (128, 256, 1, 1)},
            {'kernel_shape': (128, 128, 3, 3)},
            {'kernel_shape': (512, 128, 1, 1)},
        ),
        'Block2': (
            {'kernel_shape': (128, 512, 1, 1)},
            {'kernel_shape': (128, 128, 3, 3)},
            {'kernel_shape': (512, 128, 1, 1)},
        ),
        'Block3': (
            {'kernel_shape': (128, 512, 1, 1)},
            {'kernel_shape': (128, 128, 3, 3)},
            {'kernel_shape': (512, 128, 1, 1)},
        ),
        'Block4': (
            {'kernel_shape': (128, 512, 1, 1)},
            {'kernel_shape': (128, 128, 3, 3)},
            {'kernel_shape': (512, 128, 1, 1)},
        )

    }
}