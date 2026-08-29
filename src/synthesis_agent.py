import os
import sys
import json
from typing import List, Union, Callable, Optional, Tuple, Sequence, Any 
from src.logger import logging
from src.exception import CustomException

logger = logging.getLogger("synthesis_agent") 

class SynthesisAgent:
    name="synthesis_agent"

    def __init__(self,
                 llm: Any,
                 system_prompt: Optional[str]=None):

        self.llm=llm
        self.system_prompt=system_prompt

    def build_user_prompt(self,
                          draft_report: str,
                          refined_report: str,
                          visual_description: str):
        sections=[]
        sections.append("PRELIMINARY REPORT (drafted from similar prior cases):\n"
                        f"{str(draft_report).strip() or '(none available)'}")
        sections.append("KEY FINDINGS (distilled from the preliminary report and prior cases):\n"
                        f"{str(refined_report).strip() or '(none available)'}")
        sections.append("FINDINGS OBSERVED IN THIS PATIENT'S X-RAY:\n"
                        f"{str(visual_description).strip() or '(no visual description available)'}")
        sections="\n\n".join(sections)+"\n\nWrite the final radiology report for this patient."
        return sections
    
    @staticmethod
    def clean_llm_output(text: str):
        text=(text or "").strip()
        if "</think>" in text:
            text=text.split("</think>")[-1].strip()        
        
        if text.startswith("```"):
            text="\n".join(ln for ln in text.splitlines() if not ln.strip().startswith("```")).strip()
        
        for prefix in ("final radiology report:", "final report:", "radiology report:",
                       "report:", "findings:", "impression:", "here is the final report:"):
            if text.lower().startswith(prefix):
                text=text[len(prefix):].strip()
                break
        text=" ".join(line.strip() for line in text.splitlines() if line.strip())
        return text
    
    def single_run(self,
                   draft_report: str,
                   refined_report: str,
                   visual_description: str):
        
        try:
            if not any(str(x).strip() for x in (draft_report, refined_report, visual_description)):
                logger.warning(f"[{self.name}] all inputs are empty, generate empty final report.")
                return ""
            
            user_prompt=self.build_user_prompt(draft_report, refined_report, visual_description)
            final_report=self.llm.generate_response(system_prompt=self.system_prompt, user_prompt=user_prompt)
            cleaned=self.clean_llm_output(final_report) 

            if not cleaned:
                fallback=str(refined_report).strip() or str(draft_report).strip()
                logger.warning(f"[{self.name}] empty synthesis output and falling back to earlier stage.")     
                return fallback

            return cleaned
        except CustomException:
            raise
        except Exception as e:
            logger.error(f"[{self.name}] error generating final report.: {e}")
            raise CustomException(e, sys)

    def batch_run(self,
                   draft_reports: Sequence[str],
                   refined_reports: Sequence[str],
                   visual_descriptions: Sequence[str],
                   log_every: int=20,
                   skip_error: bool=True):
        
        final_reports: List[str]=[]
        fails=0
        if not (len(draft_reports))==(len(refined_reports))==(len(visual_descriptions)):
            raise ValueError(f"[{self.name}] draft_reports={len(draft_reports)}, refined_reports={len(refined_reports)} and visual_descriptions={len(visual_descriptions)}"
                             f"Len. of all agent report must match 1:1")
        
        for i, (draft, refined, visual) in enumerate(zip(draft_reports, refined_reports, visual_descriptions), start=1):
            try:
                final_reports.append(self.single_run(draft, refined, visual))
            except Exception as e:
                fails+=1
                if not skip_error:
                    raise
                fallback=str(refined).strip() or str(draft).strip()
                logger.error(f"[{self.name}] synthesis {i} failed and falling back to earlier stag.: {e}")
                final_reports.append(fallback)
        
            if log_every and i%log_every==0:
                logger.info(f"[{self.name}] synthesised {i}/{len(draft_reports)} report.................")

        if fails:
            logger.warning(f"[{self.name}] {fails}/{len(draft_reports)} syntheses failed and fallback to earlier stage.")
        
        chnaged_refined=sum(1 for f, r in zip(final_reports, refined_reports) if str(f).strip() != str(r).strip())
        chnaged_draft=sum(1 for f, d in zip(final_reports, draft_reports) if str(f).strip() != str(d).strip())
        logger.info(f"[{self.name}] completed {len(final_reports)} final reports."
                    f"{chnaged_refined} differ from the refined report and {chnaged_draft} differ from the draft.")
       
        if hasattr(self.llm, "log_usage"):
            self.llm.log_usage()
        return final_reports        

    def save_final_reports(self,
                           final_reports: Sequence[str],
                           output_path: str,
                           study_ids: Optional[Sequence[Any]]=None,
                           references: Optional[Sequence[Any]]=None,
                           draft_reports: Optional[Sequence[Any]]=None,
                           refined_reports: Optional[Sequence[Any]]=None,
                           visual_descriptions: Optional[Sequence[Any]]=None,
                           extra_meta: Optional[dict]=None):
        try:
            out_dir=os.path.dirname(output_path)
            final_dir=os.path.join(out_dir, "synthesis_agent") if out_dir else "synthesis_agent"
            os.makedirs(final_dir, exist_ok=True)
            final_path=os.path.join(final_dir, os.path.basename(output_path))     

            records=[]
            for i, report in enumerate(final_reports):
                record={"index": i, "final_report": report, "final_report_len": len(str(report).split())}
                if study_ids is not None and i<len(study_ids):
                    record["study_id"]=study_ids[i]
                if references is not None and i< len(references):
                    record["reference"]=references[i]     
                if draft_reports is not None and i<len(draft_reports):
                    record["draft_report"]=draft_reports[i]
                    record["changed_from_draft"]=(str(report).strip()!=str(draft_reports[i]).strip())
                if refined_reports is not None and i<len(refined_reports):
                    record["refined_report"]=refined_reports[i]
                    record["changed_from_refined"]=(str(report).strip()!=str(refined_reports[i]).strip())
                if visual_descriptions is not None and i< len(visual_descriptions):
                    record["visual_description"]=visual_descriptions[i]     
                records.append(record)    

            payload={"agent": self.name,
                     "system_prompts": self.system_prompt,
                     "num_final_reports": len(records),
                     "num_empty": sum(1 for r in final_reports if not str(r).strip()),
                     "num_changed_from_draft": sum(1 for d in records if d.get("changed_from_draft")),
                     "num_changed_from_refined": sum(1 for r in records if r.get("changed_from_refined")),
                     "llm": self.llm.llm_provenance() if hasattr(self.llm, "llm_provenance") else str(self.llm),
                     "usage": self.llm.usage_summary() if hasattr(self.llm, "usage_summary") else {},                      
                     "final_reports": records} 

            if extra_meta:
                payload.update(extra_meta) 

            with open(final_path, "w") as f:
                json.dump(payload, f, indent=2) 
            logger.info(f"[{self.name}] save {len(records)} final reports to {final_path}")
            return final_path
        except CustomException:
            raise
        except Exception as e:
            logger.error(f"[{self.name}] error saving final reports to {output_path}: {e}")
            raise CustomException(e, sys)
        
    @staticmethod
    def load_final_reports(output_path: str):
        try:
            with open(output_path, "r") as f:
                payload=json.load(f)
            reports=[record["final_report"] for record in payload["final_reports"]] 
            logger.info(f"Loaded {len(reports)} final reports from {output_path}.")
            return reports
        except Exception as e:
            logger.error(f"Error occurred during load final reports from {output_path}: {e}")
            raise CustomException(e, sys)               

    def __repr__(self):
        return f"SynthesisAgent(llm={self.llm!r})"
        
              
                   


