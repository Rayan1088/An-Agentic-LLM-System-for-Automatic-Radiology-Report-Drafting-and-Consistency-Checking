import os
import sys
import json
import faiss
from PIL import Image
from typing import List, Union, Optional, Sequence, Any 
from src.logger import logging
from src.exception import CustomException

logger = logging.getLogger("retrieval_agent") 

class RetrievalAgent:
    name="retrieval_agent"

    def __init__(self, clip_backbone):
        self.clip_backbone = clip_backbone
        self.index: faiss.Index | None = None
        self.reports: List[str] = []

    def load_retrieval_database(self, reports_path: str):
        
        self.index = faiss.read_index(os.path.join(reports_path, "index.faiss"))
       
        with open(os.path.join(reports_path, "reports.json"), 'r') as f:
            self.reports = json.load(f)
       
        with open(os.path.join(reports_path, "meta.json"), 'r') as f:
            meta = json.load(f)
       
        if meta["clip_backbone"] != self.clip_backbone.name:
            raise ValueError(f"Clip backbone mismatch: \n"
                             f"Index was built with clip backbone :{meta['clip_backbone']}\n"
                             f"But passed clip backbone is :{self.clip_backbone.name}\n"
                             f"Embedding from different clip backbone are not comparable\n"
                             f"Expected clip backbone is :{self.clip_backbone.name}\n")
        
        if self.index.ntotal != len(self.reports):
            raise ValueError(f"Database inconsistency detected: index contains {self.index.ntotal} "
                             f"vectors but reports.json has {len(self.reports)} entries. "
                             f"These files must come from the same build of {reports_path}.")

        logger.info(f"Loaded retrieval database with {len(self.reports)} reports from {reports_path}.")
        
        return self
    
    def retrieve_reports(self, image: Union[Image.Image, List[Image.Image]], 
                               top_k: int = 5):
        all_results=[]
        
        if top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k}.")
        
        if self.index is None:
            raise RuntimeError("No database loaded.")
        
        single_img=isinstance(image, Image.Image)
        imgs=[image] if single_img else image
        if not imgs:
            raise ValueError("image must contain at least one image.")
        logger.info("Images loaded successfully from given input.")

        try:
            query_embeds=self.clip_backbone.encode_images(imgs)
            logger.info(f"Image embedding extraction successfully using {self.clip_backbone.name}.")
            scores, indices=self.index.search(query_embeds,top_k) 
            ### faiss.Index.search requires a 2D float32 numpy array.
            logger.info("successfully similarity score and their indices are extracted from retrieval database depending on Input image embeddings.")
        except Exception as e:
            logger.error("Error occurred during extract input image embeddings, similarity score and their indices")
            raise CustomException(e,sys)
        
        for row_scores, row_indices in zip(scores, indices):
            results=[]
            for idx, score in zip(row_indices, row_scores):
                if idx == -1:
                    continue
                if idx >= len(self.reports):
                    logger.error(f"Index {idx} returned by FAISS is out of range for reports list of size {len(self.reports)}.")
                    continue
                results.append((int(idx), self.reports[idx], float(score)))
           
            all_results.append(results)
            
        return all_results[0] if single_img else all_results

    def save_retrievals(self,
                        batch_retrieved: Sequence[Sequence[Any]],
                        output_path: str,
                        study_ids: Optional[Sequence[Any]]=None,
                        references: Optional[Sequence[Any]]=None,
                        image_path: Optional[Sequence[str]]=None,
                        extra_meta: Optional[dict]=None):
        try:
            out_dir=os.path.dirname(output_path)
            retrieval_dir=os.path.join(out_dir, "retrieval_agent") if out_dir else "retrieval_agent"
            os.makedirs(retrieval_dir, exist_ok=True)
            final_path=os.path.join(retrieval_dir, os.path.basename(output_path))            
            
            records=[]
            for i, retrieved in enumerate(batch_retrieved):
                record={"index": i, "retrieved":[{"db_index": int(idx), "report": str(report), "score": float(score)} for idx, report, score in retrieved],
                        "num_retrieved": len(retrieved)}
                if study_ids is not None and i<len(study_ids):
                    record["study_id"]=study_ids[i]
                if references is not None and i< len(references):
                    record["reference"]=references[i]     
                if image_path is not None and i<len(image_path):
                    record["image_path"]=image_path[i]
                records.append(record)

            payload={"agent": self.name,
                     "clip_backbone": self.clip_backbone.name,
                     "db_size": len(self.reports),
                     "num_queries": len(records),
                     "num_empty": sum(1 for r in batch_retrieved if not r),
                     "retrievals": records}   
            if extra_meta:
                payload.update(extra_meta)
            
            with open(final_path, "w") as f:
                json.dump(payload, f, indent=2)
            logger.info(f"[{self.name}] save {len(records)} retrievals to {final_path}")
            return final_path
        except CustomException:
            raise
        except Exception as e:
            logger.error(f"[{self.name}] error saving retrievals to {output_path}: {e}")
            raise CustomException(e, sys)

    @staticmethod
    def load_retrievals(output_path: str):
        try:
            with open(output_path, "r") as f:
                payload=json.load(f)
            batch_retrieved=[[(r["db_index"], r["report"], r["score"]) for r in record["retrieved"]]
                              for record in payload["retrievals"]]
            logger.info(f"Loaded {len(batch_retrieved)} retrievals from {output_path}.")
            return batch_retrieved
        except Exception as e:
            logger.error(f"Error occurred during load retrievals from {output_path}: {e}")
            raise CustomException(e, sys)    

    def __len__(self):
        return len(self.reports)
    
    def __repr__(self):
        return (f"RetrievalAgent(backbone={self.clip_backbone.name}, db_size={len(self.reports)})")
                        
                        