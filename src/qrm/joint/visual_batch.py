"""Bounded frozen-vision batching; trainable VLA microbatch remains one."""
import torch

from .model import frozen_visual


@torch.no_grad()
def encode_visual_chunk(bridge, preps, visual_encoder=None):
    if not 1 <= len(preps) <= 2:
        raise ValueError('Only frozen vision batch sizes 1 and 2 are validated')
    if any(image.shape[0] != 1 for prep in preps for image in prep[0]):
        raise ValueError('Expected one observation per history time')
    encode = visual_encoder if visual_encoder is not None else lambda prep: frozen_visual(bridge, prep)
    if len(preps) == 1:
        return [encode(preps[0])]
    cameras = len(preps[0][0])
    if any(len(prep[0]) != cameras or len(prep[1]) != cameras for prep in preps):
        raise ValueError('Camera layout differs across observations')
    images = [torch.cat([prep[0][slot] for prep in preps], 0) for slot in range(cameras)]
    masks = [torch.cat([prep[1][slot] for prep in preps], 0) for slot in range(cameras)]
    values = encode((images, masks))
    # Independent storage prevents a saved frame retaining the whole batch.
    return [tuple(value[i:i+1].clone() for value in values) for i in range(len(preps))]
