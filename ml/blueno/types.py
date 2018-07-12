"""Define new parameters here.

By being explict about parameters, it will be easier to share
information around the team.

All generator and model functions should take in **kwargs as an argument.
"""
import typing

from dataclasses import dataclass

__all__ = ['PreprocessConfig',
           'DataConfig',
           'ModelConfig',
           'GeneratorConfig',
           'ParamConfig',
           'ParamGrid']


# TODO(#65): Implement this config
@dataclass
class PreprocessConfig:
    pass


@dataclass
class DataConfig:
    data_dir: str
    labels_path: str
    index_col: str
    label_col: str


@dataclass
class ModelConfig:
    model_callable: typing.Callable
    optimizer: typing.Callable
    loss: typing.Callable

    dropout_rate1: int
    dropout_rate2: int
    freeze: bool


@dataclass
class GeneratorConfig:
    generator_callable: typing.Callable
    rotation_range: int = 30
    width_shift_range: float = 0.1
    height_shift_range: float = 0.1
    shear_range: float = 0
    zoom_range: int = 0.1
    horizontal_flip: bool = True
    vertical_flip: bool = False


# Keep the below two classes in sync

@dataclass
class ParamConfig:
    data: DataConfig
    generator: GeneratorConfig
    model: ModelConfig
    batch_size: int
    seed: int
    val_split: float

    job_fn: typing.Callable = None


@dataclass
class ParamGrid:
    data: typing.Tuple[DataConfig]
    generator: typing.Tuple[GeneratorConfig]
    model: typing.Tuple[ModelConfig]
    batch_size: typing.List[int]
    seed: typing.Tuple[int]
    val_split: typing.Tuple[int]

    job_fn: typing.Callable = None

    def __init__(self, **kwargs):
        # TODO(#65): Implement preprocessing config
        if not isinstance(kwargs['data'], DataConfig):
            data = tuple(
                DataConfig(**d) for d in kwargs['data'])
            self.data = data
        if not isinstance(kwargs['generator'], GeneratorConfig):
            generators = tuple(
                GeneratorConfig(**gen) for gen in kwargs['generator'])
            self.generator = generators
        if not isinstance(kwargs['model'], ModelConfig):
            models = tuple(ModelConfig(**mod) for mod in kwargs['model'])
            self.model = models

        self.seed = kwargs['seed']
        self.val_split = kwargs['val_split']
        self.batch_size = kwargs['batch_size']

        if 'job_fn' in kwargs:
            self.job_fn = kwargs['job_fn']
