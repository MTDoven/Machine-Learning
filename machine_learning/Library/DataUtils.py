from decimal import Decimal
import numpy as np
import random


class DataLoader:
    """
    DataLoader:
        __init__(self, dataset:callable, divide:tuple=(1.0,0.0), shuffle:bool=True, seed:int=None):
            :param divide: To divide the data to train/evaluate/test.
            :param shuffle: To shuffle the dataset, default:True
        __call__(self, batchsize:int=0, mode:str="train"): It will return an iterator.
            :param batchsize: just batchsize in this iterator
            :param mode: [train]/evaluate/test, which part the data comes from
    demo:
        dataset = MyDataset(...)
        dataloader = DataLoader(dataset, divide:tuple=(0.7, 0.2, 0.1), shuffle=True)
        for i in dataloader(batchsize=10, mode="train"):
            print(i)
        for i in dataloader(batchsize=10, mode="eval"):
            print(i)
        for i in dataloader(batchsize=10, mode="test"):
            print(i)
    attention:
        If you try: for i in dataloader: print(i) // dataloader[0] // ...
        it will do the same as what you do to the Dataset.
    """
    def __init__(self, dataset:callable, divide:tuple=(1.0,0.0), shuffle:bool=True, seed:int=None):
        self.dataset = dataset
        self.datalength = len(dataset)
        self._divide_dataset(divide,shuffle,seed)
    def __call__(self, batchsize:int=0, mode:str="train"):
        if batchsize==0:
            if mode == "train": batchsize = len(self.train_list)
            elif mode == "eval": batchsize = len(self.eval_list)
            elif mode == "test": batchsize = len(self.test_list)
            else: raise ValueError("self.mode only can be selected from 'train'/'eval'/'test'.")
        assert batchsize!=0, """You cannot generate data from a empty dataset, check the param:'divide' please.
            If you still have this problem check your Dataset to make sure it works right"""
        if mode == "train": batchsize = len(self.train_list) if batchsize>len(self.train_list) else batchsize
        elif mode == "eval": batchsize = len(self.eval_list) if batchsize>len(self.eval_list) else batchsize
        elif mode == "test": batchsize = len(self.test_list) if batchsize>len(self.test_list) else batchsize
        else: raise ValueError("self.mode only can be selected from 'train'/'eval'/'test'.")
        # Set Iterator class
        class Iterator:
            def __init__(self, dataset, datalist, batchsize):
                self.datalist = datalist
                self.dataset = dataset
                self.batchsize = batchsize
                self.count = -1
            def __iter__(self):
                return self
            def __next__(self):
                temp_list = []
                for _ in range(self.batchsize):
                    self.count += 1
                    try:  # To handle StopIteration
                        temp_index = self.datalist[self.count]
                    except IndexError:
                        raise StopIteration
                    temp_list.append(self.dataset[temp_index])
                return np.array(temp_list)
        if mode == "train":
            return Iterator(self.dataset, self.train_list, batchsize)
        elif mode == "eval":
            return Iterator(self.dataset, self.eval_list, batchsize)
        elif mode == "test":
            return Iterator(self.dataset, self.test_list, batchsize)
        else: # The mode is not train/test/eval
            raise ValueError("self.mode only can be selected from 'train'/'eval'/'test'.")
    def _divide_dataset(self, divide, shuffle, seed):
        if seed is not None:
            random.seed(seed)
        index = [i for i in range(self.datalength)]
        if shuffle:
            index_shuffled = index.copy()
            random.shuffle(index_shuffled)
            number2number_dict = dict(zip(index, index_shuffled))
        else: # Not shuffle
            number2number_dict = dict(zip(index, index))
        if len(divide)==2:
            assert Decimal(str(divide[0]))+Decimal(str(divide[1]))==Decimal("1.0"),\
                "The sum of all the division is not 1, check your param:'divide' please."
            index = int(self.datalength * divide[0])
            self.train_list = [i for i in range(index)]
            self.eval_list = []
            self.test_list = list(set(range(self.datalength))-set(self.train_list))
        elif len(divide)==3:
            assert Decimal(str(divide[0])) + Decimal(str(divide[1])) + Decimal(str(divide[2])) == Decimal("1.0"), \
                "The sum of all the division is not 1, check your param:'divide' please."
            index = int(self.datalength * divide[0])
            self.train_list = [i for i in range(index)]
            index = int(self.datalength * divide[0] + self.datalength * divide[1])
            self.eval_list = list(set([i for i in range(index)])-set(self.train_list))
            self.test_list = list(set(range(self.datalength))-set(self.eval_list)-set(self.train_list))
        else: raise ValueError("The length of 'divide' must be 2(train,test) or 3(train,eval,test).")
        self.train_list = [number2number_dict[i] for i in self.train_list]
        self.eval_list = [number2number_dict[i] for i in self.eval_list]
        self.test_list = [number2number_dict[i] for i in self.test_list]
    def __len__(self):
        return self.datalength
    def __getitem__(self, item):
        return self.dataset[item]
    def __iter__(self):
        return self.dataset.__iter__()
    def __next__(self):
        return self.dataset.__next__()
    def __str__(self):
        strs = self.dataset.__str__()+"\n"
        strs += self.__repr__()
        return strs


class Dataset:
    """
    Dataset:
        you should inherit this class and rewrite __init__, __getitem__, __len__
        __getitem__: just return one piece of data one time, never return a 2D or 3D array.
        __len__: the length of the dataset, or maximum number can be putted to dataset[number]
            If you don’t recover __len__, it will try to traverse your dataset to test its length
    demo:
        class MyDataset(Dataset):
            def __init__(self, path):
                self.X = load(path)
            def __getitem__(self, item):
                return self.X[item]
            def __len__(self):
                return len(self.X)
        # Rewrite __init__, __getitem__, __len__
        dataset = MyDataset("./dataset/pictures")
        print(len(dataset))
        print(dataset[0])
    """
    def __init__(self):
        """__init__ should be recovered"""
        self.data_test = np.random.random(size=(100,2))
    def __getitem__(self, item):
        """__getitem__ should be recovered"""
        return self.data_test[item]
    def __len__(self):
        """__getitem__ should be recovered, (not necessary)"""
        self.length = 0
        while True:
            try: self.__getitem__(self.length)
            except IndexError: return self.length
            self.length += 1
    def __iter__(self):
        self._count_index = -1
        return self
    def __next__(self):
        try: # for StopIteration
            self._count_index += 1
            return self[self._count_index]
        except IndexError:
            raise StopIteration
    def __str__(self):
        strs = f"Example: {repr(self.__getitem__(0))}\n"
        strs += f"Length of data is: {len(self)}\n"
        strs += self.__repr__()
        return strs


