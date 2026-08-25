import os
import sys
import json
from src.logger import logging
from src.exception import CustomException
from typing import Any, List, Optional, Sequence, Callable, Tuple

logger=logging.getLogger("refiner_agent")

class RefinerAgent:
    name="refiner_agent"
    
    def __init__(self,
                 llm: Any,
                 system_prompt: Optional[str]=None,
                 top_k: Optional[int]=None, # for more acuracey use top-3 or top-2 or any from retrieved top-5
                 result_parser: Optional[Callable[[Any], Tuple[str, float]]]=None,
                 include_similarity: bool=False,
                 deduplicate: bool=True):
        
        self.llm=llm
        self.system_prompt=system_prompt
        self.top_k=top_k
        self.result_parser=result_parser
        self.include_similarity=include_similarity
        self.deduplicate=deduplicate
    
    def parse(self, item: Any):
        if self.result_parser is not None:
            return self.result_parser(item)
    
        if isinstance(item, (tuple, list)) and len(item)==3:
            return str(item[1]), float(item[2])
        
        if isinstance(item, (tuple, list)) and len(item)==2:
            return str(item[0]), float(item[1])      
        
        if isinstance(item, str):
            return item, 0.0
        raise ValueError(f"Could not parse retrieval result item: {item!r}. Pass result_parser to RefinerAgent.")
    
    @staticmethod
    def clean_llm_output(text: str):
        text=(text or "").strip()
        if "</think>" in text:
            text=text.split("</think>")[-1].strip()        
        
        if text.startswith("```"):
            text="\n".join(ln for ln in text.splitlines() if not ln.strip().startswith("```")).strip()
        
        for prefix in ("revised report:", "final report:", "report:", "findings:", "impression:", "here is the revised report:"):
            if text.lower().startswith(prefix):
                text=text[len(prefix):].strip()
                break
        return " ".join(line.strip() for line in text.splitlines() if line.strip())
        
    def build_user_prompt(self, 
                          draft: str,
                          retrieved: Sequence[Any]):
        blocks=[]
        reports=[self.parse(item) for item in retrieved]
        if self.deduplicate:
            seen, unique=set(), []
            for text, score in reports:
                key=text.strip()
                if key in seen:
                    continue
                seen.add(key)
                unique.append((text, score))
            reports=unique
        if self.top_k is not None:
            reports=reports[:self.top_k]
            
        for i, (text, score) in enumerate(reports, start=1):
            header=f"Prior report {i}"
            if self.include_similarity:
                header+=f" (similarity {score:.3f})"
            blocks.append(f"{header}:\n{text.strip()}")
        
        joined="\n\n".join(blocks)
        
        return (f"Preliminary report:\n{(draft or '').strip()}\n\n"
                f"Here are the {len(reports)} prior reports it was drafted from.\n\n"
                f"{joined}\n\n"
                f"Revise the preliminary report now.")
    
    def single_run(self, 
                   draft:str,
                   retrieved: Sequence[Any]):
        try:
            if not str(draft).strip():
                logger.info(f"[{self.name}] empty draft supplied, returning empty refinement.")
                return ""
            
            if not retrieved:
                logger.info(f"[{self.name}] no retrieved supplied, returning draft unchanged.")
                return self.clean_llm_output(draft)
            
            user_prompt=self.build_user_prompt(draft, retrieved)
            refined=self.llm.generate_response(system_prompt=self.system_prompt, user_prompt=user_prompt)                
            cleaned=self.clean_llm_output(refined)
            if not cleaned:
                logger.warning(f"[{self.name}] LLM returned an empty refinement, falling back to the original draft.")
                return self.clean_llm_output(draft)
            return cleaned
        except CustomException:
            raise
        except Exception as e:
            logger.error(f"[{self.name}] error generating refined report.: {e}")
            raise CustomException(e, sys)
            
    def batch_run(self,
                  drafts: Sequence[str],
                  batch_retrieved: Sequence[Sequence[Any]],
                  log_every: int=50,
                  skip_error: bool=True):
        
        refinements: List[str]=[]
        fails=0
        if len(drafts)!=len(batch_retrieved):
            raise CustomException(f"[{self.name}] drafts ({len(drafts)}) and batch_retrieved ({len(batch_retrieved)}) length mismatch.", sys) 
        
        for i, (draft, retrieved) in enumerate(zip(drafts, batch_retrieved), start=1):
            try:
                refinements.append(self.single_run(draft, retrieved))
            except Exception as e:
                fails+=1
                if not skip_error:
                    raise
                logger.error(f"[{self.name}] refinement [{i}] failed, falling back to unrefined draft.: {e} ")
                refinements.append(self.clean_llm_output(draft))
            
            if log_every and i % log_every==0:
                logger.info(f"[{self.name}] refined {i}/{len(drafts)} reports......")
        if fails:
            logger.warning(f"[{self.name}] {fails}/{len(drafts)} refinements failed and fall back to the unrefined draft.")
         
        logger.info(f"[{self.name}] completed {len(refinements)} refined reports.") 
        if hasattr(self.llm, "log_usage"):
            self.llm.log_usage()
        return refinements
    
    def save_refinements(self, 
                        refinements: Sequence[str],
                        output_path: str,
                        drafts: Optional[Sequence[str]]=None,
                        study_ids: Optional[Sequence[Any]]=None,
                        references: Optional[Sequence[Any]]=None,
                        retrieved: Optional[Sequence[Sequence[Any]]]=None,
                        extra_meta: Optional[dict]=None):
        
        try:
            out_dir=os.path.dirname(output_path)
            refiner_dir=os.path.join(out_dir, "refiner_agent") if out_dir else "refiner_agent"
            os.makedirs(refiner_dir, exist_ok=True)
            final_path=os.path.join(refiner_dir, os.path.basename(output_path))            
                    
            records=[]
            for i, refined in enumerate(refinements):
                record={"index": i, "refined": refined, "refined_len": len(str(refined).split())}
                if drafts is not None and i<len(drafts):
                    record["draft"]=drafts[i]
                    record["changed"]=self.clean_llm_output(drafts[i])!=str(refined).strip()
                if study_ids is not None and i<len(study_ids):
                    record["study_id"]=study_ids[i]
                if references is not None and i< len(references):
                    record["reference"]=references[i]     
                if retrieved is not None and i<len(retrieved):
                    record["retrieved_reports"]=[self.parse(item)[0] for item in retrieved[i]]
                records.append(record)    
        
            payload={"agent": self.name,
                     "system_prompts": self.system_prompt,
                     "top_k": self.top_k,
                     "deduplicate": self.deduplicate, 
                     "num_refinements": len(records),
                     "num_empty": sum(1 for d in refinements if not str(d).strip()),
                     "num_changed": sum(1 for r in records if r.get("changed")),
                     "llm": self.llm.llm_provenance() if hasattr(self.llm, "llm_provenance") else str(self.llm),
                     "usage": self.llm.usage_summary() if hasattr(self.llm, "usage_summary") else {},
                     "refinements": records} 
              
            if extra_meta:
                payload.update(extra_meta)
                        
            with open(final_path, "w") as f:
                json.dump(payload, f, indent=2)
            logger.info(f"[{self.name}] save {len(records)} refinements to {final_path}")
            return final_path
        except CustomException:
            raise
        except Exception as e:
            logger.error(f"[{self.name}] error saving refinements to {output_path}: {e}")
            raise CustomException(e, sys)
                
    @staticmethod
    def load_refinements(output_path: str):
        try:
            with open(output_path, "r") as f:
                payload=json.load(f)
            refinements=[record["refined"] for record in payload["refinements"]] 
            logger.info(f"Loaded {len(refinements)} refinements from {output_path}.")
            return refinements
        except Exception as e:
            logger.error(f"Error occurred during load refinements from {output_path}: {e}")
            raise CustomException(e, sys)               

    def __repr__(self):
        return f"RefinerAgent(llm={self.llm!r}, top_k={self.top_k}, deduplicate={self.deduplicate})"
        
            
            
            
            
            
            