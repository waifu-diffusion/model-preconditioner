from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    StableDiffusionPipeline,
    PNDMScheduler
)
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
import pandas as pd
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
#################
# Setup
#################
target_sd_model = 'stabilityai/stable-diffusion-2-1'
output = './output'
#################
# Load Booru Tags
#################
label_names = pd.read_csv("metadata/selected_tags.csv")
#################
# Load SD Models
#################
print('loading models...')
tokenizer = CLIPTokenizer.from_pretrained(
    target_sd_model, subfolder="tokenizer"
)
text_encoder = CLIPTextModel.from_pretrained(
    target_sd_model, subfolder="text_encoder"
)
vae = AutoencoderKL.from_pretrained(
    'cafeai/wd14-vae', subfolder="v2-512"
)
unet = UNet2DConditionModel.from_pretrained(
    target_sd_model, subfolder="unet"
)
#################
# Update Tokenizer
#################
print('updating tokenizer...')
vocab = tokenizer.get_vocab()
for _, y in enumerate(label_names['name'][0:4]):
    tokenizer.add_tokens(f"rating:{y}")
for _, y in enumerate(label_names['name'][4:]):
    name = y.replace('_', ' ')
    if name not in vocab:
        tokenizer.add_tokens(name)
print(tokenizer.get_added_vocab())
print(f'resizing text_encoder embedding matrix (-> {len(tokenizer)})')
text_encoder.resize_token_embeddings(len(tokenizer))
#################
# Create Pipeline
#################
print('creating pipeline...')
pipeline = StableDiffusionPipeline(
    text_encoder=text_encoder,
    vae=vae,
    unet=unet,
    tokenizer=tokenizer,
    scheduler=PNDMScheduler.from_pretrained(
        target_sd_model,
        subfolder="scheduler",
    ),
    safety_checker=StableDiffusionSafetyChecker.from_pretrained(
        "CompVis/stable-diffusion-safety-checker"
    ),
    feature_extractor=CLIPFeatureExtractor.from_pretrained(
        "openai/clip-vit-base-patch32"
    ),
)
#################
# Save Model
#################
print('saving model...')
pipeline.save_pretrained(output, safe_serialization=True)
