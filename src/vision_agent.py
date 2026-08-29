import os 
import sys
import json 
from PIL import Image
from src.logger import logging
from src.exception import CustomException
from typing import Any, List, Optional, Sequence

logger=logging.getLogger("vission_agent")

class VisionAgent:
    name="vision_agent"

    def __init__(self,
                 vlm: Any,
                 system_prompt: Optional[str]=None):
        
        if not system_prompt:
            raise CustomException("VisionAgent requires system_prompt.", sys)
        self.vlm=vlm
        self.system_prompt=system_prompt
    
    @staticmethod
    def clean_vlm_output(text: str):
        text=(text or "").strip()
        if "</think>" in text:
            text=text.split("</think>")[-1].strip()
        if text.startswith("```"):
            text="\n".join(ln for ln in text.splitlines() if not ln.strip().startswith("```")).strip()
        for prefix in ("visual description:", "description:", "findings:", "report:", "the chest x-ray shows:"):
            if text.lower().startswith(prefix):
                text=text[len(prefix):].strip()
                break
        text=" ".join(line.strip() for line in text.splitlines() if line.strip())
        return text

    def single_run(self, image: Image.Image):
        try:
            visual_description=self.vlm.run_inference(image, self.system_prompt)
            clean_output=self.clean_vlm_output(visual_description)
            return clean_output
        except CustomException:
            raise
        except Exception as e:
            logger.error(f"[{self.name}] error generating visual description.: {e}")
            raise CustomException(e, sys)
    
    def batch_run(self,
                  images: Sequence[Image.Image],
                  log_every: int=20,
                  skip_error: bool=True):

        visual_description: List[str]=[]
        fails=0
        for i, img in enumerate(images, start=1):
            try:
                visual_description.append(self.single_run(img))
            except Exception as e:
                fails+=1
                if not skip_error:
                    raise 
                logger.error(f"[{self.name}] description {i} failed, using empty string.: {e}")
                visual_description.append("")
            if log_every and i % log_every==0:
                logger.info(f"[{self.name}] described {i}/{len(images)} images...............")
        if fails:
           logger.warning(f"[{self.name}] {fails}/{len(images)} description failed, using empty string.") 
        
        logger.info(f"[{self.name}] completed {len(visual_description)} visual descriptions")
        if hasattr(self.vlm, "log_usage"):
            self.vlm.log_usage()
        return visual_description

    def save_visual_description(self,
                                visual_descriptions: Sequence[str],
                                output_path: str, 
                                study_ids: Optional[Sequence[Any]]=None,
                                image_paths: Optional[Sequence[str]]=None,
                                references: Optional[Sequence[Any]]=None,
                                extra_meta: Optional[dict]=None):
        try:
            out_dir=os.path.dirname(output_path)
            vision_dir=os.path.join(out_dir, "vision_agent") if out_dir else "vision_agent"
            os.makedirs(vision_dir, exist_ok=True)
            final_path=os.path.join(vision_dir, os.path.basename(output_path)) 

            records=[]
            for i, description in enumerate(visual_descriptions):
                record={"index": i, "visual_description": description, "description_len": len(str(description).split())}
                if study_ids is not None and i<len(study_ids):
                    record["study_id"]=study_ids[i]
                if image_paths is not None and i<len(image_paths):
                    record["image_path"]=image_paths[i]
                if references is not None and i< len(references):
                    record["reference"]=references[i]     
                records.append(record)   
            
            payload={"agent": self.name,
                     "system_prompt": self.system_prompt,
                     "num_descriptions": len(records),
                     "num_empty": sum(1 for d in visual_descriptions if not str(d).strip()),
                     "vlm": self.vlm.llava_provenance() if hasattr(self.vlm, "llava_provenance") else str(self.vlm),
                     "usage": self.vlm.usage_summary() if hasattr(self.vlm, "usage_summary") else {},
                     "visual_descriptions": records} 
            
            if extra_meta:
                payload.update(extra_meta)
                        
            with open(final_path, "w") as f:
                json.dump(payload, f, indent=2)
            logger.info(f"[{self.name}] save {len(records)} visual descriptions to {final_path}")
            return final_path
        except CustomException:
            raise
        except Exception as e:
            logger.error(f"[{self.name}] error saving visual descriptions to {output_path}: {e}")
            raise CustomException(e, sys)

    @staticmethod
    def load_visual_descriptions(output_path: str):
        try:
            with open(output_path, "r") as f:
                payload=json.load(f)
            visual_descriptions=[record["visual_description"] for record in payload["visual_descriptions"]] 
            logger.info(f"Loaded {len(visual_descriptions)} visual descriptions from {output_path}.")
            return visual_descriptions
        except Exception as e:
            logger.error(f"Error occurred during load visual descriptions from {output_path}: {e}")
            raise CustomException(e, sys)   
    
    def __repr__(self):
        return f"VisionAgent(vlm={self.vlm!r})"





