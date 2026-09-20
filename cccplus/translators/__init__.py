from .base import (  # noqa: F401
    ActivationStore,
    Translator,
    TranslatorPair,
    collect_activations,
    split_documents,
)
from .procrustes import ProcrustesTranslator  # noqa: F401
from .crosscoder import CrosscoderTranslator, DFCTranslator  # noqa: F401
from .causal_baselines import LatentStitchTranslator, MASTranslator, StitchTranslator  # noqa: F401

REGISTRY = {
    "procrustes": ProcrustesTranslator,
    "crosscoder": CrosscoderTranslator,
    "dfc": DFCTranslator,
    "mas": MASTranslator,
    "latent_stitch": LatentStitchTranslator,
    "stitch": StitchTranslator,
}

UNSUPERVISED = ("procrustes", "crosscoder", "dfc")
BEHAVIOR_SUPERVISED = ("mas", "latent_stitch", "stitch")
