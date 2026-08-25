import os
import sys
import json
import time
import hashlib
from pathlib import Path
from openai import OpenAI
from src.logger import logging
from typing import Optional, Any, Dict, List
from src.exception import CustomException

logger=logging.getLogger("llm_backbones")

### LLMs are API-based, not local. All three (Llama, Qwen, DeepSeek) go through the HuggingFace router via the OpenAI client 
### the actual model inference happens on HuggingFace's/the provider's servers. There's no torch, no weights loaded locally, no device concept at all.
class BaseLLM:
    name: str="base_llm"
    model_id: str=""
    hf_repo: str="" ### Hugging face repo
    revision: str="" ### for SHA weights if commit
    
    def __init__(self,
                 seed: Optional[int]=42,
                 temperature: float=0.0,
                 max_tokens: int=200,
                 max_retries: int=4,
                 retry_backoff: float=2.0,
                 cache_dir: Optional[str]=None):
        
        self.seed=seed
        self.temperature=temperature
        self.max_tokens=max_tokens
        self.max_retries=max_retries
        self.retry_backoff=retry_backoff
        self.cache_dir=Path(cache_dir) if cache_dir else None
        if cache_dir:
            self.cache_dir=self.cache_dir/"llm_cache"/self.name.lower()
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.total_input_tokens=0
        self.total_output_tokens=0
        self.cached_input_tokens=0
        self.cached_output_tokens=0
        self.total_calls=0
        self.total_latency=0.0
        self.cache_hits=0
        self.served_models=set()
        
    def cache_keys(self, system_prompt: str, user_prompt: str):
        payload=json.dumps({"model": self.model_id,
                      "temperature": self.temperature,
                      "max_tokens": self.max_tokens,
                      "seed": self.seed,
                      "extra_body": getattr(self, "extra_body", {}),
                      "system_prompt": system_prompt,
                      "user_prompt": user_prompt}, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
    
    def read_cache(self, key: str):
        if not self.cache_dir:
            return None
        path=self.cache_dir/f"{key}.json"
        if not path.exists():
            return None
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"[{self.name}] can not read cache file from [{path}]: {e}")
            return None
    
    def write_cache(self, key: str,
                    system_prompt: str,
                    user_prompt: str, 
                    response: str,
                    usage: Optional[Dict[str, int]]=None,
                    served_model: Optional[str]=None):
        
        if not self.cache_dir:
            return
        try:
            with open(self.cache_dir/f"{key}.json", "w") as f:
                json.dump({"model": self.model_id,
                           "hf_repo": self.hf_repo, 
                           "temperature": self.temperature,
                           "max_tokens": self.max_tokens,
                           "seed": self.seed,
                           "extra_body": getattr(self, "extra_body", {}),
                           "system_prompt": system_prompt,
                           "user_prompt": user_prompt,
                           "response": response,
                           "usage": usage or {},
                           "served_model": served_model}, f)
        except Exception as e:
            logger.warning(f"[{self.name}] can not write cache file: {e}")
            
    def generate_response(self, system_prompt: str, user_prompt: str):
        last_error=None
        key=self.cache_keys(system_prompt, user_prompt)
        cached=self.read_cache(key)
        if cached is not None:
            self.cache_hits+=1
            usage=cached.get("usage") or {}
            self.cached_input_tokens+=usage.get("input_tokens", 0)
            self.cached_output_tokens+=usage.get("output_tokens", 0)
            served_model=cached.get("served_model")
            if served_model:
                self.served_models.add(served_model)
            return cached["response"]
        
        for attempt in range(1, self.max_retries+1):
            try:
                start=time.time()
                response, usage, served_model=self.call_api(system_prompt, user_prompt)
                self.total_latency+=time.time()-start
                self.total_calls+=1 
                self.total_input_tokens+=usage.get("input_tokens", 0)
                self.total_output_tokens+=usage.get("output_tokens", 0)
                if served_model:
                    self.served_models.add(served_model)
                self.write_cache(key, system_prompt, user_prompt, response, usage, served_model)
                return response
            except Exception as e:
                last_error=e
                if attempt==self.max_retries:
                    logger.warning(f"[{self.name}] llm call failed [attempt {attempt}/{self.max_retries}]: {e}")
                    break
                wait=self.retry_backoff**attempt
                logger.warning(f"[{self.name}] llm call failed [attempt {attempt}/{self.max_retries}]: {e} "
                               f"Retrying to hit the api again.............................................") 
                time.sleep(wait)
        
        logger.error(f"[{self.name}] all [{self.max_retries}] attempts are failed.")       
        raise CustomException(last_error, sys)
    
    def llm_provenance(self):
        return {"backbone": self.name,
                "model_id": self.model_id,
                "hf_repo": self.hf_repo, 
                "revision": self.revision,
                "provider_base_url": getattr(self, "base_url", None),
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "seed": self.seed,
                "served_models": sorted(self.served_models)}
    
    @staticmethod
    def save_llm_provenance(llms: List["BaseLLM"], path: str):
        try:
            out_dir=os.path.dirname(path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            with open(path, "w") as f:
                json.dump([{**llm.llm_provenance(), **llm.usage_summary()} for llm in llms], f, indent=2)
            logger.info(f"Model provenance saved to {path}")
        except Exception as e:
            logger.error(f"Error saving provenance: {e}")
            raise CustomException(e, sys)
    
    def usage_summary(self):
        return {"model_id": self.model_id,
                "total_calls": self.total_calls,
                "total_cache_hits": self.cache_hits,
                "input_tokens": self.total_input_tokens,
                "output_tokens": self.total_output_tokens,
                "cached_input_tokens": self.cached_input_tokens,     
                "cached_output_tokens": self.cached_output_tokens,  
                "avg_latency_sec": (self.total_latency/self.total_calls) if self.total_calls else 0.0}
    
    def log_usage(self):
        usage=self.usage_summary()
        logger.info(f"[{self.name}] calls: {usage['total_calls']}, cache hits: {usage['total_cache_hits']},"
                    f"input_tokens: {usage['input_tokens']}, output tokens: {usage['output_tokens']},"
                    f"avg latency: {usage['avg_latency_sec']:.2f}s")
    
    def __repr__(self):
        return f"{self.__class__.__name__}(model_id={self.model_id}, temperature={self.temperature})"
 
class LLMRequests(BaseLLM):
    api_providers={"huggingface":{"base_url": "https://router.huggingface.co/v1", "env_key": "HF_TOKEN"}} ### One token, one base_url, for any hosted model.
    ### I use hugging face for model apis providers.  
    ### router in front of many backend providers (Together, Fireworks, Novita, Baseten, Ollama ect)
    
    def __init__(self,
                 model_id: Optional[str]=None,
                 api_provider: str="huggingface",
                 base_url: Optional[str]=None,
                 api_key: Optional[str]=None,
                 hf_repo: Optional[str]=None,
                 revision: str="",
                 extra_body: Optional[Dict[str, Any]]=None,
                 **kwargs):
        super().__init__(**kwargs)
        
        if api_provider not in self.api_providers and not base_url:
            raise CustomException(f"Unknown api provider [{api_provider}].", sys)
        preset=self.api_providers.get(api_provider, {})
        self.api_provider=api_provider
        self.base_url=base_url or preset.get("base_url")
        if model_id:
            self.model_id=model_id
        if hf_repo:
            self.hf_repo=hf_repo
        self.revision=revision
        self.extra_body=extra_body or {}
        
        if not self.model_id:
            raise CustomException("model_id is not define.", sys)
        
        get_api_key=api_key or os.environ.get(preset.get("env_key", "HF_TOKEN")) or "not_needed"
        ### Hugging face appi push in .env file
        
        try:
            self.client=OpenAI(api_key=get_api_key, base_url=self.base_url)
            logger.info(f"[{self.name}] initialised, model=[{self.model_id}], api provider=[{self.api_provider}], temp=[{self.temperature}], max tokens=[{self.max_tokens}]")
        except Exception as e:
            logger.error(f"[{self.name}] failed to initialise client: {e}")
            raise CustomException(e, sys)
    
    def call_api(self, system_prompt: str, user_prompt: str):
        request={"model": self.model_id,
                 "temperature": self.temperature,
                 "max_tokens": self.max_tokens,
                 "messages": [{"role": "system", "content": system_prompt},
                              {"role": "user", "content": user_prompt}]}
        
        if self.seed is not None:
            request["seed"]=self.seed
        if self.extra_body:
            request["extra_body"]=self.extra_body
        
        response=self.client.chat.completions.create(**request)
        msg=response.choices[0].message
        finish_reason=getattr(response.choices[0], "finish_reason", None)
        text=msg.content or ""
        if not text:
            reasoning_content=getattr(msg, "reasoning_content", None)
            if reasoning_content and finish_reason!="length":
                logger.warning(f"[{self.name}] 'content' is empty,recovered text from 'reasoning_content' instead.")
                text=reasoning_content.strip()
            else:
                logger.warning(f"[{self.name}] API returned empty content. finish_reason={getattr(response.choices[0], 'finish_reason', None)}, "
                               f"raw_message={msg!r}")
        
        usage_obj=getattr(response, "usage", None)
        usage={"input_tokens": getattr(usage_obj, "prompt_tokens", 0) or 0,
               "output_tokens": getattr(usage_obj, "completion_tokens", 0) or 0,} if usage_obj is not None else {}
        
        served_model=getattr(response, "model", None)

        return text, usage, served_model

### Non-Reasoning model             
class LlamaLLM(LLMRequests):
    name="Llama"
    model_id="meta-llama/Llama-3.3-70B-Instruct"
    hf_repo="meta-llama/Llama-3.3-70B-Instruct"
    
### Non-Reasoning model    
class QwenLLM(LLMRequests):
    name="Qwen"
    model_id="Qwen/Qwen3-235B-A22B-Instruct-2507"
    hf_repo="Qwen/Qwen3-235B-A22B-Instruct-2507"

### Reasoning model
class DeepSeekLLM(LLMRequests):
    name="DeepSeek"
    model_id="deepseek-ai/DeepSeek-V4-Flash-0731"
    hf_repo="deepseek-ai/DeepSeek-V4-Flash-0731"

    def __init__(self,
                thinking: bool=False, ### Reasoning/thinking modes produce longer, hedged prose. IU X-Ray references average ~28 tokens, so thinking is disabled by default.
                **kwargs):
        extra=kwargs.pop("extra_body", {}) or {}
        if not thinking:
            extra.setdefault("chat_template_kwargs", {"enable_thinking": False})
        super().__init__(extra_body=extra,
                         **kwargs)
        
    