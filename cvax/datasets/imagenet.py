from typing import Union, Literal
from pathlib import Path

import numpy as np
from PIL import Image

from cvax.utils import config

class ImageNet:
    def __init__(self,
        split: Literal["train", "val", "test"],
        image_size: Union[int, tuple[int, int]] = (224, 224)
        ):
        """
        """
        self.image_paths = []
        self.labels = []

        self.image_size = (image_size, image_size) if type(image_size) == int else image_size

        # TODO: Download dataset

        data_dir = config.DATASET_DIR / f"imagenette2/{split}"

        self.names = sorted([p.stem for p in data_dir.glob("n0*")])
        self.ids = [id_ for id_ in range(len(self.names))]
        self.id2name = {id_: name for id_, name in enumerate(self.names)}
        
        for id_, name in sorted(self.id2name.items(), key=lambda x: x[0]):
            image_dir = data_dir / name

            image_paths = sorted(image_dir.glob("*.JPEG"))

            self.image_paths += image_paths
            self.labels += len(image_paths) * [id_]


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
        raise NotImplementedError