import typing

from dataclasses import dataclass


@dataclass
class ModelConfig:
    model_callable: typing.Callable
    optimizer: typing.Callable
    loss: typing.Callable
    dropout_rate1: int
    dropout_rate2: int
    freeze: bool
    batch_size: int  # Deprecated
    rotation_range: int  # Deprecated


# TODO: Integrate the generator
@dataclass
class GeneratorConfig:
    generator_callable: typing.Callable
    rotation_range: int = 30
    width_shift_range: float = 0.1
    height_shift_range: float = 0.1
    shear_range: float = 0
    zoom_range: int = 0.1
    horizontal_flip = True
    vertical_flip = False


@dataclass
class ParamGrid:
    data: typing.Tuple[str]
    generator: typing.Tuple[GeneratorConfig]
    model: typing.Tuple[ModelConfig]
    seed: typing.Tuple[int] = (0, 1)
    val_split: typing.Tuple[int] = (0.1, 0.2)

    # TODO: Batch size to paramgrid
    # batch_size: typing.List[int] = None

    def __init__(self, **kwargs):
        if not isinstance(kwargs['generator'], GeneratorConfig):
            generators = tuple(
                GeneratorConfig(**gen) for gen in kwargs['generator'])
            self.generator = generators
        if not isinstance(kwargs['model'], ModelConfig):
            models = tuple(ModelConfig(**mod) for mod in kwargs['model'])
            self.model = models

        self.data = kwargs['data']
        self.seed = kwargs['seed']
        self.val_split = kwargs['val_split']
