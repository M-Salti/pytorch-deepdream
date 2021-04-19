from typing import List, Tuple, Union

import timm
import torch
import torchvision.transforms.functional as TF
from PIL.Image import Image, open
from timm.data import resolve_data_config
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

patch_typeguard()


@typechecked
def open_image(path: str) -> Image:
    return open(path).convert("RGB")


@typechecked
def transform(image: Image, mean: TensorType[3, 1, 1], std: TensorType[3, 1, 1]) -> TensorType[3, -1, -1]:
    return TF.to_tensor(image).to(mean.device).sub(mean).div(std)


@typechecked
def untransform(tensor: TensorType[3, -1, -1], mean: TensorType[3, 1, 1], std: TensorType[3, 1, 1]) -> Image:
    return TF.to_pil_image(tensor.mul(std).add(mean))


@typechecked
def create_model(
    name: str, device: Union[str, torch.device], pretrained=True
) -> Tuple[torch.nn.Module, TensorType[3, 1, 1], TensorType[3, 1, 1]]:
    model = timm.create_model(name, pretrained=pretrained, scriptable=True).to(device)
    model.eval()
    config = resolve_data_config({}, model=model)
    mean = torch.tensor(config["mean"], device=device).view(3, 1, 1)
    std = torch.tensor(config["std"], device=device).view(3, 1, 1)

    return model, mean, std


@typechecked
def random_roll(tensor: TensorType[3, -1, -1], max_roll: int) -> Tuple[List[int], TensorType]:
    shifts = torch.randint(-max_roll, max_roll, (2,)).tolist()
    rolled = torch.roll(tensor, shifts, [1, 2])
    return shifts, rolled
