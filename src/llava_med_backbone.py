import os
import sys
import time
import torch
from PIL import Image
from typing import Any, Dict, Optional
from transformers import BitsAndBytesConfig
from transformers import AutoProcessor, LlavaForConditionalGeneration
from src.logger import logging
from src.exception import CustomException

logger=logging.getLogger("LLaVA_Med_backbone")
defalt_device="cuda" if torch.cuda.is_available() else "cpu"

class LLaVAMedVLM:
    name="LLaVA-Med"
    hf_repo="chaoyinshe/llava-med-v1.5-mistral-7b-hf"
    
    def __init__(self,
                 model_id: Optional[str]=None,
                 revision: str="",
                 device: Optional[str]=None,
                 load_in_4bit: bool=False,
                 max_new_tokens: int=160,
                 temperature: float=0.0,
                 lazy: bool=False):
        
        self.model_id=model_id
        self.revision=revision
        self.device=device if device is not None else defalt_device
        self.load_in_4bit=load_in_4bit
        self.max_new_tokens=max_new_tokens
        self.temperature=temperature
        self.lazy=lazy
        self.model=None
        self.processor=None
        self.total_calls=0
        self.total_latency=0.0
        
        if not self.lazy:
            self.load_model()
        
    def load_model(self):
        
        if self.model is not None:
            return self
        
        if self.model_id is None:
            self.model_id=self.hf_repo
        
        try:
            kwargs: Dict[str, Any]={"device_map": "auto"}
            if self.revision:
                kwargs["revision"]=self.revision
            
            if self.load_in_4bit and self.device=="cuda":
                kwargs["quantization_config"]=BitsAndBytesConfig(
                        load_in_4bit=True, 
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True)
            else:
                kwargs["torch_dtype"]=(torch.float16 if self.device=="cuda" else torch.float32)
            
            logger.info(f"[{self.name}] loading {self.model_id} in device={self.device} and 4bit={self.load_in_4bit}....")

            self.processor=AutoProcessor.from_pretrained(self.model_id, **({"revision": self.revision} if self.revision else{}))
            self.model=LlavaForConditionalGeneration.from_pretrained(self.model_id, **kwargs) 
            logger.info(f"[{self.name}] model loaded successfully.")
            self.model.eval()
            return self
        except CustomException:
            raise
        except Exception as e:
            logger.error(f"[{self.name}] failed to load {self.model_id}: {e}")
            raise CustomException(e, sys)     

    def unload_model(self):
        self.model=None
        self.processor=None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(f"[{self.name}] model unloaded from {self.device}.")

    def run_inference(self, image: Image.Image, prompt: str):
        
        if self.model is None:
            self.load_model()
        
        try:
            ### LLaVA v1.5 template. The literal <image> token is required
            messages=f"USER: <image>\n{prompt} ASSISTANT:"
            inputs= self.processor(images=image, text=messages, return_tensors='pt')
            inputs=inputs.to(self.model.device)

            start=time.time()
            with torch.inference_mode():
                output_ids=self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=self.temperature>0, 
                                               temperature=self.temperature if self.temperature>0 else None)    
            self.total_latency+=time.time()-start
            self.total_calls+=1

            ### keep only model generated
            generated_ids=output_ids[0, inputs["input_ids"].shape[-1]:]
            decode_text=self.processor.decode(generated_ids, skip_special_tokens=True).strip()
            
            if not decode_text:
                logger.info(f"(input {inputs['input_ids'].shape[-1]} tokens, output {output_ids.shape[-1]} tokens)."
                            f"Check that '<image>' is present in the prompt template.")

            return decode_text
        except CustomException:
            raise
        except Exception as e:
            logger.error(f"[{self.name}] error occurred during generate descriptions: {e}")
            raise CustomException(e, sys)
    
    def llava_provenance(self):
        return {"backbone": self.name,
                "model_id": self.model_id,
                "hf_repo": self.hf_repo,
                "revision": self.revision,
                "device": self.device,
                "load_in_4bit": self.load_in_4bit,
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
                "prompt_template": "USER: <image>\\n{prompt} ASSISTANT:"}

    def usage_summary(self):
        return {"model_id": self.model_id, "total_calls": self.total_calls, 
                "avg_latency_sec": (self.total_latency/self.total_calls) if self.total_calls else 0.0}
    
    def log_usage(self):
        usage=self.usage_summary()
        logger.info(f"[{self.name}] total calls: {usage['total_calls']}, avg latency: {usage['avg_latency_sec']:.2f}s")
    
    def __repr__(self):
        return f"LLaVAMedVLM(model_id={self.model_id}, device={self.device}, 4bit={self.load_in_4bit}, max_new_tokens={self.max_new_tokens})"
            
        
        
        