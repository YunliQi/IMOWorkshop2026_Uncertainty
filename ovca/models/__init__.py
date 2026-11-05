"""Model registry: add new models here."""
from .base import BaseModel
from .growth_death import GrowthDeath
from .poptk import PopTK
from .resistance import SensitiveResistant

MODELS = {m.name: m for m in (GrowthDeath, PopTK, SensitiveResistant)}


def get_model(name, **options) -> BaseModel:
    """A model instance; options go to its constructor (e.g. bev="brake" for resistance)."""
    try:
        cls = MODELS[name]
    except KeyError:
        raise ValueError(f"unknown model {name!r}; available: {sorted(MODELS)}") from None
    return cls(**options)
