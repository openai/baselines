import tensorflow as tf
from models.model import Model


class NatureConvnet(Model):
    def __init__(self, name='nature_cnn'):
        super(NatureConvnet, self).__init__(name=name)

    def __call__(self, inputs):
        pass
