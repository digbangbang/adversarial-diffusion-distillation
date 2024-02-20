import torch
import torchvision
import numpy as np

from diffusers import UNet2DModel
from diffusers import DDPMScheduler

from PIL import Image

repo_id = "google/ddpm-cifar10-32"
model = UNet2DModel.from_pretrained(repo_id)

def show_images(x):
    """Given a batch of images x, make a grid and convert to PIL"""
    x = x * 0.5 + 0.5  # Map from (-1, 1) back to (0, 1)
    grid = torchvision.utils.make_grid(x)
    grid_im = grid.detach().cpu().permute(1, 2, 0).clip(0, 1) * 255
    grid_im = Image.fromarray(np.array(grid_im).astype(np.uint8))
    grid_im.save("ddpm_generated_image.png")
    return grid_im

def sample_images(model, noise_scheduler, device: str = "cuda", c: int = 0, bs: int = 16,
                  num_inference_steps: int = 1000):
    model.eval()
    model.to(device)

    x = torch.randn((bs, 3, 32, 32)).to(device)
    # x = torch.randn((bs, 3, 32, 32))

    noise_scheduler.set_timesteps(num_inference_steps=num_inference_steps)

    for t in noise_scheduler.timesteps:
        model_input = noise_scheduler.scale_model_input(x, t)

        t_batch = torch.full(
            size=(x.shape[0],),
            fill_value=t.item(),
            dtype=torch.long
        ).to(device)

        with torch.no_grad():
            noise_pred = model(
                model_input,
                t_batch,
                return_dict=False
            )[0]

        x = noise_scheduler.step(noise_pred, t, x).prev_sample

    return x

noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="linear")

sampled_images = sample_images(
    model=model,
    noise_scheduler=noise_scheduler,
    num_inference_steps=100,
)

show_images(sampled_images)

