from .unets import U_Net
from .conditional_encoders import SimpleConditionalEncoder


CONDITIONAL_ENCODER_REGISTRY = {
    "simple_conditional_encoder": SimpleConditionalEncoder,
}
