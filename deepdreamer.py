from functools import partial
from typing import Dict, List, Union

import IPython.display as display
import torch
import torch.nn as nn
from PIL.Image import Image
from torch.utils.hooks import RemovableHandle
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

import utils

patch_typeguard()


@typechecked
class OutputsSaver:
    def __init__(self, model: nn.Module, modules_names: List[str]):
        self.handles: List[RemovableHandle] = []
        self.outputs: Dict[str, TensorType] = dict()

        def save_layer_output(module, input, output, name):
            self.outputs[name] = output

        for name in modules_names:
            module = getattr(model, name)
            handle = module.register_forward_hook(partial(save_layer_output, name=name))
            self.handles.append(handle)

    def clear(self):
        for handle in self.handles:
            handle.remove()
        self.outputs.clear()
        self.handles.clear()


@typechecked
class DeepDreamer(nn.Module):
    """Simple implementation of Deep Dream

        Args:
            model_name (str): Pretrained model name (inception_v3, vgg16, resnet50, etc.)
            layers_names (List[str]): Names of the layers to maximize their activations
            device (Union[str, torch.device]): cpu or cuda
        """
    def __init__(self, model_name: str, layers_names: List[str], device: Union[str, torch.device]):
        super().__init__()
        self.device = device
        self.model, mean, std = utils.create_model(model_name, self.device)
        assert self.model.training == False

        self.transform = partial(utils.transform, mean=mean, std=std)
        self.untransform = partial(utils.untransform, mean=mean, std=std)
        self.output_saver = OutputsSaver(self.model, layers_names)

    def set_layers(self, layers_names: List[str]):
        self.output_saver.clear()
        self.output_saver = OutputsSaver(self.model, layers_names)

    def forward(self, img, steps: int, step_size: TensorType[float]):
        for _ in range(steps):
            self.model.zero_grad()
            self.model.forward_features(img)
            loss = torch.tensor(0.0).to(self.device)
            for activation in self.output_saver.outputs.values():
                loss += activation.mean()

            gradients = torch.autograd.grad(loss, img, only_inputs=True)[0]
            gradients = gradients / (gradients.std() + 1e-8)

            img = img + gradients * step_size
            img = torch.clamp(img, -1, 1)

        return loss, img

    def run_deep_dream_simple(self, img: Image, steps=100, step_size=1e-2, verbose=False) -> Image:
        img = self.transform(img).unsqueeze(0).to(self.device)
        img.requires_grad_(True)

        step_size = torch.tensor(step_size).to(self.device)
        steps_remaining = steps
        step = 0
        while steps_remaining:
            if steps_remaining > 100:
                run_steps = 100
            else:
                run_steps = steps_remaining
            steps_remaining -= run_steps
            step += run_steps

            loss, img = self.forward(img, run_steps, step_size)

            if verbose:
                print(f"step: {step}, loss: {loss.item()}")
                display.display(self.untransform(img.detach().squeeze()))

        result = self.untransform(img.detach().squeeze())
        if verbose:
            print("final")
            display.display(result)

        return result

    def run_deep_dream_with_octaves(
        self, original_img: Image, steps_per_octave=100, step_size=1e-2, octaves=range(-2, 3), octave_scale=1.3
    ) -> Image:
        img = original_img
        base_shape = img.size
        base_shape = torch.tensor(base_shape).float()

        for n in octaves:
            new_shape = base_shape * (octave_scale ** n)
            new_shape = new_shape.int().tolist()
            img = img.resize(new_shape)
            img = self.run_deep_dream_simple(img, steps_per_octave, step_size)

        return img.resize(original_img.size)
