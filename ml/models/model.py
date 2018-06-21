from abc import abstractmethod


class ModelBuilder(object):

    @staticmethod
    @abstractmethod
    def build(input_shape, num_classes=2, config={}):
        pass
