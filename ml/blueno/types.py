import typing


class ModelConfig(typing.NamedTuple):
    model_callable: typing.Callable
    optimizer: typing.Callable
    loss: typing.Callable
    dropout_rate1: int
    dropout_rate2: int
    freeze: bool
    batch_size: int  # Deprecated
    rotation_range: int  # Deprecated


# TODO: Implement the generator
class GeneratorConfig(typing.NamedTuple):
    generator_callable: typing.Callable
    rotation_range: int
    width_shift_range: float = 0.1
    height_shift_range: float = 0.1
    shear_range: float = 0
    zoom_range: int = 0.1
    horizontal_flip = True
    vertical_flip = False


class ParamGrid(typing.NamedTuple):
    data: typing.List[str]
    generator: typing.List[GeneratorConfig]
    model: typing.List[ModelConfig]
    seed: typing.List[int] = [0, 42]
    val_split: typing.List[int] = [0.1, 0.2]
    # batch_size: typing.List[int] = None
