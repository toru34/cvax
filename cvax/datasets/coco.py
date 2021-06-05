import json
from typing import Union, Literal

from cvax import config

class COCO:
    def __init__(self,
        split: Literal["train", "val", "test"],
        image_size: Union[int, tuple[int, int]] = (224, 224),
        year: Literal["2014", "2017"] = 2017,
        ):
        """
        """

        self.image_paths = []
        self.labels = []

        self.image_size = (image_size, image_size) if type(image_size) == int else image_size

        # TODO: Download dataset

        image_dir = config.DATASET_DIR / f"coco/{split}{year}"
        label_path = config.DATASET_DIR / f"coco/annotations/instances_{split}{year}.json" # TODO: Add test set format

        dataset_info = json.load(open(label_path, 'r'))

        self.category2id = {c['name']: i for i, c in enumerate(dataset_info['categories'])}
        self.id2category = {i: category for category, i in self.category2id.items()}
