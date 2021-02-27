import numpy as np

class DataLoader:
    def __init__(self, dataset, batch_size, shuffle):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.pointer = 0
    
    def __iter__(self):
        self.pointer = 0
        # TODO: Shuffle dataset
        return self
    
    def __next__(self):
        if self.pointer < len(self.dataset):

            images, labels = zip(*[self.dataset[i] for i in range(self.pointer, self.pointer+self.batch_size)])
            
            images = np.array(images)
            labels = np.array(labels)

            self.pointer += self.batch_size

            return images, labels
        else:
            raise StopIteration