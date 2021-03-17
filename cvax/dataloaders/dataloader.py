import numpy as np

class DataLoader:
    def __init__(self, dataset, batch_size, shuffle=True, seed=34):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.n_samples = len(self.dataset)
        self.rng = np.random.default_rng(seed=seed)
    
    def __iter__(self):
        self.pointer = 0
        if self.shuffle:
            self.indices = self.rng.permutation(self.n_samples)
        else:
            self.indices = np.arange(self.n_samples)

        return self
    
    def __next__(self):
        if self.pointer < self.n_samples:

            images, labels = zip(*[self.dataset[i] for i in self.indices[self.pointer:self.pointer+self.batch_size]])
            
            images = np.asarray(images)
            labels = np.asarray(labels)

            self.pointer += self.batch_size

            return images, labels
        else:
            raise StopIteration