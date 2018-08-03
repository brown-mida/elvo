"""Configuration type definitions for bluenot.py
"""
import typing
import warnings

import keras


class DataConfig:
    def __init__(self, data_dir: str, labels_path: str, index_col: str,
                 label_col: str, gcs_url: str):
        self.data_dir = data_dir
        self.labels_path = labels_path
        self.index_col = index_col
        self.label_col = label_col
        self.gcs_url = gcs_url


# This class is experimental and subject to a lot of change.
# When a clear solution for appending preprocessing to the
# config grid arises, this will likely be replaced.
class LukePipelineConfig(DataConfig):
    """
    Defines a configuration to preprocess raw numpy data.
    """

    def __init__(self,
                 pipeline_callable: typing.Callable,
                 height_offset: int,
                 mip_thickness: int,
                 pixel_value_range: typing.Sequence,
                 data_dir: str,
                 labels_path: str,
                 index_col: str,
                 label_col: str,
                 gcs_url: str):
        self.pipeline_callable = pipeline_callable
        self.height_offset = height_offset
        self.mip_thickness = mip_thickness
        self.pixel_value_range = pixel_value_range
        super().__init__(data_dir, labels_path, index_col, label_col, gcs_url)


class ModelConfig:
    def __init__(self,
                 model_callable: typing.Callable,
                 optimizer: keras.optimizers.Optimizer,
                 loss: typing.Callable,
                 dropout_rate1: float = 0.8,
                 dropout_rate2: float = 0.8,
                 freeze: bool = False):
        """

        :param model_callable: must take in **kwargs as an argument
        :param optimizer:
        :param loss:
        :param dropout_rate1:
        :param dropout_rate2:
        :param freeze:
        """
        self.model_callable = model_callable
        self.optimizer = optimizer
        self.loss = loss
        self.dropout_rate1 = dropout_rate1
        self.dropout_rate2 = dropout_rate2
        self.freeze = freeze


class GeneratorConfig:
    def __init__(self, generator_callable: typing.Callable,
                 rotation_range: int = 30,
                 width_shift_range: float = 0.1,
                 height_shift_range: float = 0.1,
                 shear_range: float = 0,
                 zoom_range: typing.Union[float,
                                          typing.Tuple[float, float]] = 0.1,
                 horizontal_flip: bool = True,
                 vertical_flip: bool = False):
        """

        :param generator_callable should return a tuple containing
          x_train, y_train, x_test, y_test and mst take in **kwargs
          as an argument
        :param rotation_range:
        :param width_shift_range:
        :param height_shift_range:
        :param shear_range:
        :param zoom_range:
        :param horizontal_flip:
        :param vertical_flip:
        """
        self.generator_callable = generator_callable
        self.rotation_range = rotation_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.shear_range = shear_range
        self.zoom_range = zoom_range
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip


class EvalConfig:
    """
    Config data structure required to evaluate a given model.
    """

    def __init__(self,
                 model: ModelConfig,
                 model_weights: str,
                 data: DataConfig,
                 val_split: float,
                 seed: int, ):
        self.model = model
        self.model_weights = model_weights
        self.data = data
        self.val_split = val_split
        self.seed = seed


# Keep the below two classes in sync
class ParamConfig:
    def __init__(self,
                 data: DataConfig,
                 generator: GeneratorConfig,
                 model: ModelConfig,
                 batch_size: int,
                 seed: int,
                 val_split: float,
                 max_epochs: int = 100,
                 early_stopping: bool = True,
                 reduce_lr: bool = False,
                 job_fn: typing.Callable = None,
                 job_name: str = None):
        self.data = data
        self.generator = generator
        self.model = model
        self.batch_size = batch_size
        self.seed = seed
        self.val_split = val_split
        self.max_epochs = max_epochs
        self.early_stopping = early_stopping
        self.reduce_lr = reduce_lr
        self.job_fn = job_fn
        self.job_name = job_name


class ParamGrid:
    def __init__(self, **kwargs):
        warnings.warn('This class is deprecated. Use ParamConfig instead.',
                      category=DeprecationWarning)

        # With a grid you can only define one of DataConfig and
        # PipelineConfig, use a list of ParamConfigs instead if you want
        # more configurability
        if not isinstance(kwargs['data'], DataConfig):
            if 'pipeline_callable' in kwargs:
                data = tuple(LukePipelineConfig(**d) for d in kwargs['data'])
                self.data = data
            else:
                data = tuple(DataConfig(**d) for d in kwargs['data'])
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
