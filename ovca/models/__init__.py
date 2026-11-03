"""Model registry: add new models here."""
from .base import BaseModel
from .growth_death import GrowthDeath

MODELS = {m.name: m for m in (GrowthDeath,)}


def get_model(name) -> BaseModel:
    try:
        return MODELS[name]()
    except KeyError:
        raise ValueError(f"unknown model {name!r}; available: {sorted(MODELS)}") from None
