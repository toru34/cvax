from typing import Union, Literal
from pathlib import Path

import numpy as np
from PIL import Image

from cvax import config

class Imagenette:

    def __init__(self,
        split: Literal['train', 'val', 'test'],
        image_size: Union[int, tuple[int, int]],
        ):
        """
        """

        self.image_paths = []
        self.labels = []

        self.image_size = (image_size, image_size) if type(image_size) == int else image_size

        # TODO: Download dataset

        data_dir = config.DATASET_DIR / f"imagenette2/{split}"

        self._class_ids = sorted([p.stem for p in data_dir.glob('n0*')])
        
        for _class_id in self._class_ids:
            _class_dir = data_dir / _class_id

            _image_paths = sorted(_class_dir.glob("*.JPEG"))

            self.image_paths += _image_paths
            self.labels += len(_image_paths) * [_class_id]


    def __getitem__(self, i):
        image = Image.open(self.image_paths[i])

        image = image.resize(self.image_size)
        image = np.array(image)
        label = self.labels[i]
        
        return image, label

    def __len__(self):
        return len(self.image_paths)

    @staticmethod
    def _download():
        pass