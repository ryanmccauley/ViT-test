import os
import torch

from transformers import ViTImageProcessor, ViTModel
from PIL import Image

model_name = "google/vit-base-patch16-224"
extractor = ViTImageProcessor.from_pretrained(model_name)
model = ViTModel.from_pretrained(model_name)

def extract_embeddings(model: torch.nn.Module):
    device = model.device

    def pp(image):
        inputs = extractor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0].cpu()  

        return embeddings

    return pp

batch_size = 24
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def get_embedding(image_path):
  image = Image.open(image_path).convert("RGB")
  embeddings = extract_embeddings(model)(image)

  return { "path": image_path, "embeddings": embeddings }

def compute_scores(emb_one, emb_two):
    scores = torch.nn.functional.cosine_similarity(emb_one["embeddings"], emb_two["embeddings"])
    score = scores.numpy().tolist()

    print(f"{emb_one['path']} <-> {emb_two['path']}: {score}")

image_paths = []

for file in os.listdir("."):
    if file.endswith(".png"):
        image_paths.append(file)

embeddings = list(map(get_embedding, image_paths))

for i in range(len(embeddings)):
    for j in range(len(embeddings)):
        if i != j and embeddings[i]["path"] == "query.png":
            compute_scores(embeddings[i], embeddings[j])