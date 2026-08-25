import os
import sys
import json
import faiss
import torch
import numpy as np
from PIL import Image
from src.logger import logging
from src.utils import load_config
from typing import List, Dict, Any
from src.exception import CustomException
from src.clip_backbones import OpenAICLIP, PubMedCLIP, BiomedCLIP

logger = logging.getLogger("retrieval_database_build") 
defalt_device="cuda" if torch.cuda.is_available() else "cpu"

class IUXRayDataset:
    def __init__(self, file_path: str):
        try:
            with open(file_path) as f:
                self.data: Dict[str, List[Dict[str, Any]]] = json.load(f)
            logger.info(f"Loaded annotation.json:"
                        + ", ".join(f"{split}={len(entries)}" for split, entries in self.data.items()))
        except Exception as e:
            logger.error(f"Error loading annotation.json file: {e}")
            raise CustomException(e, sys)
        
    def get_split(self, split: str):
        if split not in self.data:
            raise ValueError(f"Split '{split}' not found in the dataset. Available splits: {list(self.data.keys())}")
        return self.data[split]  
    
    def get_all_entries(self, split: str):
        try:
            entries=self.get_split(split)
            logger.info(f"Fetched {len(entries)} reports for split '{split}'")  
            return entries
        except Exception as e:
            logger.error(f"Error occurred while fetching reports for split '{split}': {e}")
            raise CustomException(e, sys)
    
    def get_all_reports(self, split: str):
        try:
            entries = self.get_split(split)
            reports = [entry['report'] for entry in entries]
            logger.info(f"Fetched {len(reports)} reports for split '{split}'")
            return reports
        except Exception as e:
            logger.error(f"Error occurred while fetching reports for split '{split}': {e}")
            raise CustomException(e, sys)
    
    def get_all_image_report_pairs(self, 
                                   split: str, 
                                   image_dir: str="",
                                   frontal_view_only: bool=False):
        try:
            entries = self.get_split(split)
            pairs=[]
            for entry in entries:
                img_paths=entry['image_path'][:1] if frontal_view_only else entry['image_path']
                for img_path in img_paths:
                    full_path = f"{image_dir.rstrip('/')}/{img_path}" if image_dir else img_path
                    pairs.append({"id": entry['id'], "image_path": full_path, "report": entry['report']})
            logger.info(f"Fetched {len(pairs)} image-report pairs for split '{split}' (frontal_view_only={frontal_view_only})")
            return pairs
        except Exception as e:
            logger.error(f"Error occurred while fetching image paths for split '{split}': {e}")
            raise CustomException(e, sys)   
    
    def __repr__(self):
        counts = ", ".join(f"{split}={len(entries)}" for split, entries in self.data.items())
        return f"IUXRayDataset({counts})"
    
class RetrievalDatabaseBuilder:
    clip_factory={
        "OpenAICLIP": (OpenAICLIP, None),
        "PubMedCLIP": (PubMedCLIP, None),
        "BiomedCLIP": (BiomedCLIP, None),
        # "fine-tuned_CLIP": (OpenAICLIP, "checkpoints/clip_finetuned.pt"),
        # "fine-tuned_PubMedCLIP": (PubMedCLIP, "checkpoints/pubmedclip_finetuned.pt"),
        # "fine-tuned_BiomedCLIP": (BiomedCLIP, "checkpoints/biomedclip_finetuned.pt"),  
    }

    def __init__(self, backbone_name: str, 
                       device: str=None,
                       image_batch_size: int=32):
        
        self.device=device if device is not None else defalt_device
        self.image_batch_size=image_batch_size
        self.clip_backbone = self.get_backbone(backbone_name)
        self.index = None
        self.reports: List[str] = []
        self.image_paths: List[str]=[]
        self.study_ids: List[str]=[]
        self.retrieval_mode: str="image" 
    
    def get_backbone(self, name: str):
        try:
            clip_backbone, checkpoint=self.clip_factory[name]
            clip_backbone =clip_backbone(model_checkpoint=checkpoint, device=self.device)
        except KeyError as e:
            logger.error(f"Unknown backbone {name}, Available backbone: {list(self.clip_factory)}")
            raise CustomException(e, sys)
        except Exception as e:
            logger.error(f"Failed to load backbone {name}: {e}")
            raise CustomException(e, sys)
        logger.info(f"Backbone {name} loaded successfully.")
        return clip_backbone
    
    def build_database_from_image(self, pairs: List[Dict[str, Any]]):
        try:
            if not pairs:
                raise ValueError("Pair list is empty. Can't build database")
            
            all_embeddings=[]
            kept_reports, kept_paths, kept_ids=[],[],[]
            skipped=0
            
            for start in range(0, len(pairs), self.image_batch_size):
                chunk=pairs[start:start+self.image_batch_size]
                images, chunk_reports, chunk_paths, chunk_ids=[],[],[],[]
                for pair in chunk:
                    try:
                        img=Image.open(pair["image_path"]).convert("RGB")
                    except Exception as e:
                        skipped+=1
                        logger.warning(f"Skipping unreadable image [{pair['image_path']}]: {e}")
                        continue
                    images.append(img)
                    chunk_reports.append(pair["report"])
                    chunk_paths.append(pair["image_path"])
                    chunk_ids.append(pair["id"])
                
                if not images:
                    continue
                
                embeddings=self.clip_backbone.encode_images(images)
                all_embeddings.append(embeddings)
                kept_reports.extend(chunk_reports)
                kept_paths.extend(chunk_paths)
                kept_ids.extend(chunk_ids)
                logger.info(f"[{self.clip_backbone.name}] Encoded {len(kept_reports)}/{len(pairs)} train images.....")
                
            if not all_embeddings:
                raise ValueError(f"No images are encoded and no database built.")
            
            ### L2 normalization fro create faiss index (Double check)
            embeddings=np.vstack(all_embeddings).astype(np.float32)
            faiss.normalize_L2(embeddings)
            index=faiss.IndexFlatIP(embeddings.shape[1])
            index.add(embeddings)
            
            self.index=index
            self.reports=kept_reports
            self.image_paths=kept_paths
            self.study_ids=kept_ids
            self.retrieval_mode="image"
    
            if skipped:
                logger.warning(f"[{self.clip_backbone.name}] {skipped} images were unreadable and excluded from the database.")
            logger.info(f"[{self.clip_backbone.name}] Image database build with [{index.ntotal}] vectors [dim={embeddings.shape[1]}].")      
            return self
        except CustomException:
            raise
        except Exception as e:
            logger.error(f"Error occurred during build image database.: {e}")
            raise CustomException(e, sys)
                       
    def build_database_from_text(self, reports: List[str]):
        try:
            if not reports:
                raise ValueError("Reports list is empty. cannot build database.")
            ### Using Text Encoder Only Of ALL CLIP Varient For Build The Vector Database
            embeddings = self.clip_backbone.encode_texts(reports).astype(np.float32) 
            faiss.normalize_L2(embeddings)
            index = faiss.IndexFlatIP(embeddings.shape[1])
            index.add(embeddings)
            self.index = index
            self.reports = list(reports)
            self.image_paths=[]
            self.study_ids=[]
            self.retrieval_mode="text"
            logger.info(f"[{self.clip_backbone.name}] Text database build with [{index.ntotal}] vectors [dim={embeddings.shape[1]}].")
            return self
        except CustomException:
            raise
        except Exception as e:
            logger.error(f"Error occurred during build text database: {e}")
            raise CustomException(e, sys)
    
    def save_database(self, reports_path: str):
        try:
            os.makedirs(reports_path, exist_ok=True)
            faiss.write_index(self.index, os.path.join(reports_path, "index.faiss"))
            
            with open(os.path.join(reports_path, "reports.json"), 'w') as f:
                json.dump(self.reports, f)
                
            if self.image_paths:
                with open(os.path.join(reports_path, "sources.json"), 'w') as f:
                   json.dump([{"study_id": sid, "image_path": path} for sid, path in zip(self.study_ids, self.image_paths)], f) 
                    
            with open(os.path.join(reports_path, "meta.json"), 'w') as f:
                json.dump({"clip_backbone": self.clip_backbone.name,
                           "retrieval_mode": self.retrieval_mode,
                           "num_vectors": int(self.index.ntotal),
                           "embedding_dim": int(self.index.d),
                           "metric": "inner_product_on_l2_normalised"}, f)
                
            logger.info(f"[{self.clip_backbone.name}] Database of {len(self.reports)} entries (mode={self.retrieval_mode}) saved to {reports_path}")
        except Exception as e:
            logger.error(f"Error saving database: {e}")
            raise CustomException(e, sys)

    def run_build(self,
                  output_base_dir: str,
                  pairs: List[Dict[str, Any]]=None,
                  reports: list[str]=None,
                  retrieval_mode: str="image"):
        try:
            if retrieval_mode=="image":
                if not pairs:
                    raise ValueError("retrieval_mode='image' requires the 'pairs' argument.")
                self.build_database_from_image(pairs)
                suffix=f"retrieval_database_image_{self.clip_backbone.name}"
            elif retrieval_mode=="text":
                if not reports:
                    raise ValueError("retrieval_mode='text' requires the 'reports' argument.")
                self.build_database_from_text(reports)
                suffix=f"retrieval_database_text_{self.clip_backbone.name}"
            else:
                raise ValueError(f"Unknon mode '{retrieval_mode}.")
                
            output_path=os.path.join(output_base_dir, suffix)
            self.save_database(output_path)
        finally:
            if hasattr(self, "clip_backbone"):
                del self.clip_backbone
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
if __name__ == "__main__":
    config_path="config.yaml"
    config = load_config(config_path)   
    
    device = defalt_device
    logger.info(f"Using device: {device}")
    
    ### Load IUXRayDataset and fetch reports for the 'train' split
    dataset= IUXRayDataset(config["IU_DATASET_ANNOTATION_JSON_PATH"])
    logger.info(f"Successfully loaded dataset: {dataset}")
    
    ### For image to image retrieval [index the train images each paired with its report]
    train_pair=dataset.get_all_image_report_pairs("train",
                                                  image_dir=config["IU_DATASET_IMAGES_PATH"],
                                                  frontal_view_only=True) 
    logger.info(f"Train image-reports pair fetched: {len(train_pair)}")
    ### return [{"id": "CXR1", "image_path": "datas/iu_xray/images/CXR1_IM-0001-3001.png", "report": "The heart is..."},
    ###         {"id": "CXR1", "image_path": "datas/iu_xray/images/CXR1_IM-0001-4001.png", "report": "The heart is..."},
    ###         {"id": "CXR2", "image_path": "datas/iu_xray/images/CXR2_IM-0002-1001.png", "report": "No acute..."},...]
        
    ### For test image to test reportst cross modal retrieval    
    train_reports = dataset.get_all_reports("train")
    logger.info(f"Train reports fetched: {len(train_reports)}")
   
    ### For CLIP, PubMedCLIP, and BiomedCLIP Zero-shot and fine-tune
    for backbone_name in ["OpenAICLIP", "PubMedCLIP", "BiomedCLIP", 
                          # "fine-tuned_CLIP", "fine-tuned_PubMedCLIP", "fine-tuned_BiomedCLIP"
                          ]:
        try:
            ### For image to image retrieval [index the train images each paired with its report]
            image_embeddings_builder = RetrievalDatabaseBuilder(backbone_name, device=device)
            image_embeddings_builder.run_build(output_base_dir=config["RET_DATABASE_OUTPUT_DIR"],
                                               pairs=train_pair,
                                               retrieval_mode="image")
            
            ### For test image to train report retrieval
            text_embeddings_builder = RetrievalDatabaseBuilder(backbone_name, device=device)
            text_embeddings_builder.run_build(output_base_dir=config["RET_DATABASE_OUTPUT_DIR"],
                                              reports=train_reports, 
                                              retrieval_mode="text")
        except Exception as e:
            logger.error(f"Skipping {backbone_name} due to error: {e}")
            continue
