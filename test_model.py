from diffusers import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
import torch
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
#################
# Setup
#################
output = './output'
#################
# Load Model
#################
device = torch.device('cuda')
pipeline = StableDiffusionPipeline.from_pretrained(output).to(device)
image = pipeline('a girl standing on the beach').images[0]
image.save('./test_image.png')
