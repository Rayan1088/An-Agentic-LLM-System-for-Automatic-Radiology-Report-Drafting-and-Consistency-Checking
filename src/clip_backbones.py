import sys
import torch
import open_clip
import numpy as np
from PIL import Image
from typing import List
from src.logger import logging
from src.exception import CustomException
from src.utils import normalize_embeddings
from transformers import CLIPProcessor, CLIPModel

logger = logging.getLogger("clip_backbones") 
defalt_device="cuda" if torch.cuda.is_available() else "cpu"

### All CLIP model run on CPU and GPU(Single GPU but In multiple GPU MedCLIP can through GPU errors specifically on a multi-GPU box)
class OpenAICLIP:
    name= "OpenAICLIP"
    base_model_checkpoint="openai/clip-vit-base-patch32"
     ### Developer : OpenAI
     ### Image encoder : ViT-B/32 Vision Transformer
     ### Text encoder : Masked self-attention Transformer (~63M params, 12 layers, 512 hidden width, 8 attention heads)
     ### Image input size : 224 x 224
     ### Patch size : 32 x 32
     ### Embedding dim : 512
     ### Approx. parameters : ~151M total (ViT ~88M + text ~63M)
     ### Training data : WIT (WebImageText) — ~400M image-text pairs crawled from the public internet
     ### Training objective : InfoNCE contrastive loss
     ### Domain scope : General/natural images
     ### Released : 2021
    
    ### CLIP's text tower has a hard 77-token context limit regardless of what
    ### the tokenizer's own `model_max_length` metadata claims.
    max_text_tokens=77 
    
    ### NOTE (not a bugfix — just a defensive design choice): different `transformers` versions / output classes for CLIPModel.get_image_features()
    ### and get_text_features() have historically varied in whether they return a raw tensor or a wrapper object exposing `.image_embeds` / `.pooler_output`.
    ### This helper tries all known shapes so OpenAICLIP/PubMedCLIP keep working across transformers versions without needing another key-remap fix like
    ### MedCLIP needed. No actual failure was observed for OpenAICLIP/PubMedCLIP in this debugging session — this is just why the abstraction exists.
    @staticmethod
    def extract_embedding(output, prefer_attrs):

        if torch.is_tensor(output):
            return output
        for attr in prefer_attrs:
            val = getattr(output, attr, None)
            if val is not None:
                return val
        raise AttributeError(
            f"Could not extract an embedding tensor from output of type {type(output)}; "
            f"tried attributes: {prefer_attrs}"
        )
    
    def __init__(self,
                 model_checkpoint: str = None,
                 batch_size: int = 32,
                 device: str = None):
        
        self.model_checkpoint = model_checkpoint
        self.device = device if device is not None else defalt_device
        self.batch_size = batch_size
        self.normalize_embeddings= normalize_embeddings
        
        if model_checkpoint:
            try:
                ### fine-tune is saved as torch.save()
                self.model = CLIPModel.from_pretrained(self.base_model_checkpoint)
                self.processor = CLIPProcessor.from_pretrained(self.base_model_checkpoint)
                ### Image Size[224×224]--Resiz[Bicubic]--Crop[CenterCrop]--Normalize Mean[(0.48145466, 0.4578275, 0.40821073)]--Normalize Std[(0.26862954, 0.26130258, 0.27577711)]]--RGB[Yes]
                state_dict=torch.load(self.model_checkpoint, map_location=self.device)
                self.model.load_state_dict(state_dict)
                logger.info(f"OpenAICLIP loaded with fine-tuned weights and preprocessor from: [{self.model_checkpoint}] successfully")
            except Exception as e:
                logger.error(f"Error occurred during load OpenAICLIP fine-tuned weights and preprocessor from: {self.model_checkpoint}: {e}")
                raise CustomException(e, sys)   
        else:
            try:
                self.model = CLIPModel.from_pretrained(self.base_model_checkpoint)
                self.processor = CLIPProcessor.from_pretrained(self.base_model_checkpoint)
                ### Image Size[224×224]--Resiz[Bicubic]--Crop[CenterCrop]--Normalize Mean[(0.48145466, 0.4578275, 0.40821073)]--Normalize Std[(0.26862954, 0.26130258, 0.27577711)]]--RGB[Yes]
                logger.info(f"OpenAICLIP loaded with base pretrained weights and preprocessor from: [{self.base_model_checkpoint}] successfully")
            except Exception as e:
                logger.error(f"Error occurred during load OpenAICLIP base pretrained weights and preprocessor from: {self.base_model_checkpoint}: {e}")
                raise CustomException(e, sys)    
        
        self.model.to(self.device).eval()
    
    @torch.no_grad()
    def encode_images(self, 
                      images: List[Image.Image]):
        outputs = []
        try:
            for i in range(0, len(images), self.batch_size):
                batch = images[i:i+self.batch_size]
                inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
                raw_output = self.model.get_image_features(**inputs)
                image_embeddings= self.extract_embedding(raw_output, prefer_attrs=("image_embeds", "pooler_output"))
                outputs.append(self.normalize_embeddings(image_embeddings).cpu().numpy())
        except Exception as e:
            logger.error(f"Error occurred while encoding images for OpenAICLIP/PubMedCLIP: {e}")
            raise CustomException(e, sys)
        
        ### np.vstack(outputs) guarantees a 2D array (stacking 1D row vectors into a matrix)
        ### and .astype(np.float32) guarantees the dtype FAISS expects.
        return np.vstack(outputs).astype(np.float32) 

    @torch.no_grad()
    def encode_texts(self, 
                     texts: List[str]):
        outputs = []
        try:
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i+self.batch_size]
                inputs = self.processor(text=batch, return_tensors="pt", padding=True, truncation=True, max_length=self.max_text_tokens).to(self.device)
                raw_output = self.model.get_text_features(**inputs)
                text_embeddings=self.extract_embedding(raw_output, prefer_attrs=("text_embeds", "pooler_output"))
                outputs.append(self.normalize_embeddings(text_embeddings).cpu().numpy())
        except Exception as e:
            logger.error(f"Error occurred while encoding texts for OpenAICLIP/PubMedCLIP: {e}")
            raise CustomException(e, sys)
        
        ### np.vstack(outputs) guarantees a 2D array (stacking 1D row vectors into a matrix)
        ### and .astype(np.float32) guarantees the dtype FAISS expects.
        return np.vstack(outputs).astype(np.float32)

class PubMedCLIP(OpenAICLIP):
    name= "PubMedCLIP"
    base_model_checkpoint="flaviagiammarino/pubmed-clip-vit-base-patch32"
     ### Developer : Eslami et al. (fine-tunes OpenAI's CLIP on ROCO dataset)
     ### Base architecture : Same as OpenAI CLIP (also released as RN50/RN50x4 and variants; this repo/checkpoint uses ViT-32 only)
     ### Image encoder : ViT-B/32 (inherited, unchanged from base CLIP)
     ### Text encoder : Same CLIP text Transformer (unchanged)
     ### Image input size : 224 x 224
     ### Patch size : 32 x 32
     ### Embedding dim : 512
     ### Approx. parameters : ~151M (identical architecture to base CLIP)
     ### Training data : ROCO dataset — ~82K radiology image-caption pairs from PMC articles (X-ray, MRI, ultrasound, fluoroscopy, various body regions)
     ### Training details : 50 epochs, batch size 64, Adam, lr 1e-5
     ### Training objective : InfoNCE contrastive loss (fine-tuning)
     ### Domain scope : Radiology only
     ### Released : 2023
    
    def __init__(self, 
                 model_checkpoint: str = None,      
                 batch_size: int = 32,
                 device: str = None):
    
        ### Save fine-tuned model in torch.save()    
        super().__init__(model_checkpoint=model_checkpoint, batch_size=batch_size, device=device)
    
class BiomedCLIP:
    name= "BiomedCLIP"
    base_model_checkpoint="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
     ### Developer : Microsoft — "Large-Scale Domain-Specific Pretraining for Biomedical Vision-Language Processing" (arXiv 2303.00915)
     ### Image encoder : ViT-B/16 Vision Transformer
     ### Text encoder : PubMedBERT, domain-specific BERT pretrained on PubMed abstracts/articles; supports a 256-token context length
     ### Image input size : 224 x 224
     ### Patch size : 16 x 16
     ### Approx. parameters : ~196M (ViT-B/16 ~86M + PubMedBERT ~110M) — estimated, not an official published total
     ### Training data : PMC-15M figure caption pairs from PubMed Central Open-Access articles, covering diverse biomedical image types (microscopy, radiography, histology, etc.)
     ### Training objective : InfoNCE contrastive loss
     ### Domain scope : Broad biomedical (not limited to one modality, unlike MedCLIP/PubMedCLIP)
     ### Released : 2023
                 
    def __init__(self,
                 model_checkpoint: str = None,
                 batch_size: int = 32,
                 device: str = None):
        
        self.model_checkpoint=model_checkpoint
        self.device = device if device is not None else defalt_device
        self.batch_size = batch_size
        self.normalize_embeddings=normalize_embeddings

        if model_checkpoint:
            try:
                ### Model save uing torch.save()
                self.model, _, self.preprocessor = open_clip.create_model_and_transforms(self.base_model_checkpoint, pretrained=None)
                 ### Image Size[224×224]--Resiz[Bicubic]--Crop[CenterCrop]--Normalize Mean[(0.48145466, 0.4578275, 0.40821073)]--Normalize Std[(0.26862954, 0.26130258, 0.27577711)]]--RGB[Yes]
                state_dict=torch.load(self.model_checkpoint, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self.tokenizer = open_clip.get_tokenizer(self.base_model_checkpoint)
                logger.info(f"BiomedCLIP loaded with fine-tuned weights and tokenizer from: [{self.model_checkpoint}] successfully")
            except Exception as e:
                logger.error(f"Error occurred during load BiomedCLIP fine-tuned weights and tokenizer from: {self.model_checkpoint}: {e}")
                raise CustomException(e, sys)       
        else:
            try:
                self.model, _, self.preprocessor = open_clip.create_model_and_transforms(self.base_model_checkpoint)
                ### Image Size[224×224]--Resiz[Bicubic]--Crop[CenterCrop]--Normalize Mean[(0.48145466, 0.4578275, 0.40821073)]--Normalize Std[(0.26862954, 0.26130258, 0.27577711)]]--RGB[Yes]
                self.tokenizer = open_clip.get_tokenizer(self.base_model_checkpoint)
                logger.info(f"BiomedCLIP loaded with base pretrained weights and tokenizer from: [{self.base_model_checkpoint}] successfully")
            except Exception as e:
                logger.error(f"Error occurred during load BiomedCLIP base pretrained weights and tokenizer from: {self.base_model_checkpoint}: {e}")
                raise CustomException(e, sys)    
                
        self.model.to(self.device).eval()
                 
    @torch.no_grad()
    def encode_images(self, 
                      images: List[Image.Image]):
        outputs = []
        try:
            for i in range(0, len(images), self.batch_size):
                batch = images[i:i+self.batch_size]
                pixel_values = torch.stack([self.preprocessor(image) for image in batch]).to(self.device)
                image_embeddings = self.model.encode_image(pixel_values)
                outputs.append(self.normalize_embeddings(image_embeddings).cpu().numpy())
        except Exception as e:
            logger.error(f"Error occurred while encoding images for BiomedCLIP: {e}")
            raise CustomException(e, sys)
        return np.vstack(outputs).astype(np.float32)

    @torch.no_grad()
    def encode_texts(self, 
                     texts: List[str]):
        outputs = []
        try:
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i+self.batch_size]
                tokenized_texts = self.tokenizer(batch).to(self.device)
                text_embeddings = self.model.encode_text(tokenized_texts)
                outputs.append(self.normalize_embeddings(text_embeddings).cpu().numpy())
        except Exception as e:
            logger.error(f"Error occurred while encoding texts for BiomedCLIP: {e}")
            raise CustomException(e, sys)
        return np.vstack(outputs).astype(np.float32)