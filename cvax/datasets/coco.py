import json
from typing import Literal
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw

from cvax.utils import config


class COCODataset:
    def __init__(
        self,
        split: Literal['train', 'val', 'test'],
        year: Literal['2014', '2017'],
        task: Literal['bbox', 'keypoints'],
    ):

        self.images = list()
        self.annotations = list()

        imageid2imageinfo = dict()
        imageid2annotations = defaultdict(list)

        # TODO: Download dataset

        image_dir = config.DATASET_DIR / f'coco/{split}{year}'

        if task == 'bbox':
            annotation_path = config.DATASET_DIR / f"coco/annotations/instances_{split}{year}.json"
        elif task == 'keypoints':
            annotation_path = config.DATASET_DIR / f"coco/annotations/person_keypoints_{split}{year}.json"
        else:
            raise NotImplementedError

        dataset_info = json.load(open(annotation_path, 'r'))

        _id2category = {c['id']: c['name'] for c in dataset_info['categories']}
        self.category2id = {c['name']: i for i, c in enumerate(dataset_info['categories'])}
        self.id2category = {i: category for category, i in self.category2id.items()}

        for _image_info in dataset_info['images']:

            image_id = _image_info['id']

            image_info = {
                'image_path': image_dir / _image_info['file_name'],
                'height': _image_info['height'],
                'width': _image_info['width'],
            }

            imageid2imageinfo[image_id] = image_info
        
        for _annotation in dataset_info['annotations']:

            if _annotation['iscrowd']:
                continue # TODO
            
            image_id = _annotation['image_id']
            category = _id2category[_annotation['category_id']]

            annotation = {
                'category': category,
                'category_id': self.category2id[category],
                'polygons': _annotation['segmentation'],
                'bbox': {
                    'xmin': _annotation['bbox'][0],
                    'ymin': _annotation['bbox'][1],
                    'xmax': _annotation['bbox'][0] + _annotation['bbox'][2],
                    'ymax': _annotation['bbox'][1] + _annotation['bbox'][3],
                },
                'keypoints': np.array(_annotation['keypoints']).reshape(-1, 3),
            }

            imageid2annotations[image_id].append(annotation)
        
        for image_id in imageid2imageinfo.keys():
            
            self.images.append(imageid2imageinfo[image_id])
            self.annotations.append(imageid2annotations[image_id])
    

    def __getitem__(self, i):
        
        image_info = self.images[i]
        annotations = self.annotations[i]

        image = np.array(Image.open(image_info['image_path']))

        for j in range(len(annotations)):

            mask = Image.fromarray(np.zeros((image_info['height'], image_info['width']), dtype='uint8'))
            draw = ImageDraw.Draw(mask)

            for polygon in annotations[j]['polygons']:
                draw.polygon(polygon, fill=1, outline=1)
            
            annotations[j]['mask'] = np.array(mask, dtype='uint8')

        return image, annotations
    

    def __len__(self):
        return len(self.images)
    

    @staticmethod
    def _download_dataset():
        # TODO
        raise NotImplementedError