import numpy as np

class DataLoader:
    def __init__(self, dataset, batch_size, shuffle):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.pointer = 0
    
    def __iter__(self):
        self.pointer = 0
        return self
    
    def __next__(self):
        if self.pointer < len(self.dataset):

            batch_images, batch_labels = zip(*[self.dataset[i] for i in range(self.pointer, self.pointer+self.batch_size)])
            
            batch_images = np.array(batch_images)
            batch_labels = np.array(batch_labels)

            self.pointer += self.batch_size

            return batch_images, batch_labels
        else:
            raise StopIteration