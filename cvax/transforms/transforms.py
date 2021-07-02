import numpy as np
from PIL import Image


class Resize:
    def __init__(
        self,
        size: tuple[int, int],
        keep_aspect_ratio: bool = True,
        pad_value: int = 0
    ):

        self.height = size[0]
        self.width = size[1]

        self.keep_aspect_ratio = keep_aspect_ratio
        self.pad_value = pad_value
    

    def __call__(self, image):

        image = Image.fromarray(image)

        if self.keep_aspect_ratio:
            aspect_ratio = image.size[0] / image.size[1] # width / height

            pad_top, pad_bottom, pad_left, pad_right = 0, 0, 0, 0

            if aspect_ratio < self.width / self.height:
                height = self.height
                width = round(height * aspect_ratio)

                pad_left = round((self.width - width) / 2)
                pad_right = self.width - width - pad_left

                image = image.resize((width, height))

            else:
                width = self.width
                height = round(width / aspect_ratio)

                pad_top = round((self.height - height) / 2)
                pad_bottom = self.height - height - pad_top

                image = image.resize((width, height))

            image = np.pad(
                array=np.array(image),
                pad_width=((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                mode='constant',
                constant_values=self.pad_value,
            )
        
        else:
            image = np.array(image.resize((self.width, self.height)))

        return image