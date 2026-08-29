import os
import sys
import json
from src.logger import logging
from src.exception import CustomException
from typing import Any, Callable, List, Optional, Sequence, Tuple

logger=logging.getLogger("draft_agent")

class DraftAgent:
    name="draft_agent"
    
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
        self.deduplicate=deduplicate ### For drop repeated report 
    
    ### parsing retrieval output
    def parse(self, item: Any):
        if self.result_parser is not None:
            return self.result_parser(item)
    
        if isinstance(item, (tuple, list)) and len(item)==3:
            return str(item[1]), float(item[2])
        
        if isinstance(item, (tuple, list)) and len(item)==2:
            return str(item[0]), float(item[1])      
        
        if isinstance(item, str):
            return item, 0.0
        raise ValueError(f"Could not parse retrieval result item: {item!r}. Pass result_parser to DraftAgent.")
    
    @staticmethod
    def clean_llm_output(text: str):
        text=(text or "").strip()
        if "</think>" in text:
            text=text.split("</think>")[-1].strip()
            
        if text.startswith("```"):
            text="\n".join(ln for ln in text.splitlines() if not ln.strip().startswith("```")).strip()

        for prefix in ("preliminary report:", "final report:", "report:", "findings:", "impression:", "here is the report:"):
            if text.lower().startswith(prefix):
                text=text[len(prefix):].strip()
                break
        text=" ".join(line.strip() for line in text.splitlines() if line.strip())
        return text
        
    def build_user_prompt(self, retrieved: Sequence[Any]):
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

        blocks=[]
        for i, (text, score) in enumerate(reports, start=1):
            header=f"Prior report {i}"
            if self.include_similarity:
                header+=f" (similarity {score:.3f})"
            blocks.append(f"{header}:\n{text.strip()}") 
        
        joined="\n\n".join(blocks)
        return (f"Here are {len(reports)} reports from similar prior chest X-ray cases.\n\n"
                f"{joined}\n\n"
                f"Compose the preliminary report now.")     
        
    def single_run(self, retrieved: Sequence[Any]):
        try:
            if not retrieved:
                logger.info(f"[{self.name}] No retrieved reports supplied, returning empty draft.")    
                return ""
            user_prompt=self.build_user_prompt(retrieved)
            draft=self.llm.generate_response(system_prompt=self.system_prompt, user_prompt=user_prompt)      
            return self.clean_llm_output(draft)
        except CustomException:
            raise
        except Exception as e:
            logger.error(f"[{self.name}] error generating draft report.: {e}")
            raise CustomException(e, sys)
        
    def batch_run(self,
                  batch_retrieved: Sequence[Sequence[Any]],
                  log_every: int=50,
                  skip_error: bool=True):
        drafts: List[str]=[]
        fails=0
        for i, retrieved in enumerate(batch_retrieved, start=1):
            try:
                drafts.append(self.single_run(retrieved))
            except Exception as e:
                fails+=1
                if not skip_error:
                    raise
                logger.error(f"[{self.name}] draft {i} failed, using empty string.: {e}")      
                drafts.append("")
            
            if log_every and i % log_every==0:
                logger.info(f"[{self.name}] drafted {i}/{len(batch_retrieved)} reports......")
        if fails:
            logger.warning(f"[{self.name}] {fails}/{len(batch_retrieved)} draft failed and were replaced with empty strings.") 
        
        logger.info(f"[{self.name}] completed {len(drafts)} draft reports.") 
        if hasattr(self.llm, "log_usage"):
            self.llm.log_usage()
        return drafts
    
    def save_drafts(self, 
                    drafts: Sequence[str],
                    output_path: str,
                    study_ids: Optional[Sequence[Any]]=None,
                    references: Optional[Sequence[Any]]=None,
                    retrieved: Optional[Sequence[Sequence[Any]]]=None,
                    extra_meta: Optional[dict]=None):
        try:
            out_dir=os.path.dirname(output_path)
            draft_dir=os.path.join(out_dir, "draft_agent") if out_dir else "draft_agent"
            os.makedirs(draft_dir, exist_ok=True)
            final_path=os.path.join(draft_dir, os.path.basename(output_path))            
            
            records=[]
            for i, draft in enumerate(drafts):
                record={"index": i, "draft": draft, "draft_len": len(str(draft).split())}
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
                     "num_drafts": len(records),
                     "num_empty": sum(1 for d in drafts if not str(d).strip()),
                     "llm": self.llm.llm_provenance() if hasattr(self.llm, "llm_provenance") else str(self.llm),
                     "usage": self.llm.usage_summary() if hasattr(self.llm, "usage_summary") else {},
                     "drafts": records}   
            if extra_meta:
                payload.update(extra_meta)
            
            with open(final_path, "w") as f:
                json.dump(payload, f, indent=2)
            logger.info(f"[{self.name}] save {len(records)} drafts to {final_path}")
            return final_path
        except CustomException:
            raise
        except Exception as e:
            logger.error(f"[{self.name}] error saving drafts to {output_path}: {e}")
            raise CustomException(e, sys)
    
    @staticmethod
    def load_drafts(output_path: str):
        try:
            with open(output_path, "r") as f:
                payload=json.load(f)
            drafts=[record["draft"] for record in payload["drafts"]] 
            logger.info(f"Loaded {len(drafts)} drafts from {output_path}.")
            return drafts
        except Exception as e:
            logger.error(f"Error occurred during load drafts from {output_path}: {e}")
            raise CustomException(e, sys)               

    def __repr__(self):
        return f"DraftAgent(llm={self.llm!r}, top_k={self.top_k}, deduplicate={self.deduplicate})"
