import sys
import torch
import random
import statistics
import pandas as pd
from src.logger import logging
from rouge_score import rouge_scorer
from dataclasses import dataclass, field
from src.exception import CustomException
from typing import List, Optional, Sequence
from bert_score import score as bert_score_fn
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction

logger = logging.getLogger("evaluation_metrics")
defalt_device="cuda" if torch.cuda.is_available() else "cpu"

### Text generation evaluation for any agent in the pipeline.
### Input: list of ground truth texts of test data, list of generated texts, and a list of evaluation metrics to compute.
###   For list of generated texts, from retrieval agent, it will be the top-1 retrieved reports for each test image.
###   For LLM agent[like draft agent, refiner agent and final synthesis agent], it will be the generated report for each test image.
### Output: BLEU[1,2,3,4], ROUGE[1,2,L], METEOR, BERTScore and stacked into a single ablation table[pandas DataFrame].
@dataclass
class AgentEvalMetrics:
    
    name: str="system"
    use_bert_score: bool=False ### Is set by default to False, cuz its takes a lot of time to compute BERTScore for large test datasets.
    bert_score_model: str="distilbert-base-uncased"
    bert_score_batch_size: int=32
    bert_score_device: Optional[str]=None
    rescale_bert_score_with_baseline: bool=False 
    use_stemmer: bool=True
    lowercase: bool=True
    bootstrap_samples: int=0
    random_seed: int=42
    per_sample: List[dict]=field(default_factory=list, init=False, repr=False)
    n_gram: tuple=("bleu_1", "bleu_2", "bleu_3", "bleu_4", "rouge_1", "rouge_2", "rouge_L", "meteor")
    bert_metrics: tuple=("bert_score_precision", "bert_score_recall", "bert_score_f1")
    
    def __post_init__(self):
        self.rouge=rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=self.use_stemmer)
        self.smoother=SmoothingFunction().method1
        self.bert_score_device=self.bert_score_device if self.bert_score_device is not None else defalt_device

    def metric_names(self):
        names=list(self.n_gram)
        if self.use_bert_score:
            names +=list(self.bert_metrics)
        return names
    
    def aggregate(self):
        if not self.per_sample:
            raise ValueError(f"[{self.name}] No samples to aggregate. Run evaluate() first.")
        try:
            results={key: statistics.mean(entry.get(key, 0.0) for entry in self.per_sample) for key in self.metric_names()}
            pred_lens=[entry["prediction_len"] for entry in self.per_sample]
            ref_lens=[entry["reference_len"] for entry in self.per_sample]
            mean_pred=statistics.mean(pred_lens)
            mean_ref=statistics.mean(ref_lens)
            results["avg_prediction_len"]=mean_pred
            results["avg_reference_len"]=mean_ref
            results["len_ratio"]=mean_pred/mean_ref if mean_ref else float("nan")
            results["num_samples"]=len(self.per_sample)
            results["name"]=self.name
            return results
        except CustomException:
            raise
        except Exception as e:
            logger.error(f"[{self.name}] Error during aggregation: {e}")
            raise CustomException(e, sys)
    
    ### Resample per sample scores to get 95% CIs shows whether gaps are real
    def bootstrap_ci(self, confidence: float=0.95):
       try:
            if self.bootstrap_samples<100:
               logger.warning(f"[{self.name}] bootstrap_samples={self.bootstrap_samples} is very low; "
                              f"confidence intervals will be unreliable. Use 1000 or more.")
               
            rng=random.Random(self.random_seed)     
            len_samples=len(self.per_sample)
            intervals={}
            for name in self.metric_names():
                values=[entry.get(name, 0.0) for entry in self.per_sample]
                means=[]
                for _ in range(self.bootstrap_samples):
                    resample=[values[rng.randrange(len_samples)] for _ in range(len_samples)]
                    means.append(statistics.mean(resample))
                means.sort()
                lower_idx=int((1-confidence)/2*self.bootstrap_samples)
                upper_idx=int((1+confidence)/2*self.bootstrap_samples)-1
                upper_idx=max(lower_idx, min(upper_idx, self.bootstrap_samples-1))
                intervals[name]=(means[lower_idx], means[upper_idx])
            return intervals
       except Exception as e:
           logger.error(f"[{self.name}] Error during bootstrap: {e}")
           raise CustomException(e, sys)
        
    def tokenize(self, text: str):
        text=str(text)
        if self.lowercase:
            text=text.lower()
        return text.split()
    
    ### BLEU-1,2,3,4, ROUGE-1/2/L and METEOR for one prediction/reference pair.
    def compute_blue_rouge_metero_per_pair(self, prediction: str, reference: str):
        try:
            pred_tokens=self.tokenize(prediction)
            ref_tokens=self.tokenize(reference)
            
            all_scores={}
            for n_gram in range(1, 5):
                all_scores[f"bleu_{n_gram}"]=sentence_bleu([ref_tokens], pred_tokens, weights=tuple((1.0/n_gram,)*n_gram), smoothing_function=self.smoother)
            
            rouge=self.rouge.score(str(reference), str(prediction))
            all_scores["rouge_1"]=rouge["rouge1"].fmeasure
            all_scores["rouge_2"]=rouge["rouge2"].fmeasure
            all_scores["rouge_L"]=rouge["rougeL"].fmeasure
            
            all_scores["meteor"]=meteor_score([ref_tokens], pred_tokens)
            return all_scores
        except Exception as e:
            logger.error(f"[{self.name}] Error scoring in evaluating per pair metrics: {e}")
            raise CustomException(e, sys)
        
    def compute_bleu_for_corpus(self, predictions: Sequence[str], references: Sequence[str]):
        try:
            list_of_refs=[[self.tokenize(ref)] for ref in references]
            list_of_preds=[self.tokenize(pred) for pred in predictions]
            corpus_bleu_score=corpus_bleu(list_of_refs, list_of_preds, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=self.smoother)
            return corpus_bleu_score
        except Exception as e:
            logger.error(f"[{self.name}] Error scoring in evaluating corpus BLEU metrics: {e}")
            raise CustomException(e, sys)
    
    def compute_bert_score(self, predictions: Sequence[str], references: Sequence[str]):
        try:
            logger.info(f"[{self.name}] Computing BERTScore with {self.bert_score_model} on {self.bert_score_device} "
                        f"(rescale_with_baseline={self.rescale_bert_score_with_baseline}) "
                        f"for {len(predictions)} samples...............................")
            precision, recall, f1=bert_score_fn([str(pred) for pred in predictions], [str(ref) for ref in references],
                                                model_type=self.bert_score_model, lang="en", batch_size=self.bert_score_batch_size,
                                                rescale_with_baseline=self.rescale_bert_score_with_baseline, verbose=False,
                                                device=self.bert_score_device)
            
            for entry, p, r, f in zip(self.per_sample, precision.tolist(), recall.tolist(), f1.tolist()):
                entry["bert_score_precision"]=float(p)
                entry["bert_score_recall"]=float(r)
                entry["bert_score_f1"]=float(f)
            logger.info(f"[{self.name}] BERTScore finished.")
        except CustomException:
            raise
        except Exception as e:
            logger.error(f"[{self.name}] Error during BERTScore computation: {e}")    
            raise CustomException(e, sys)
    
    def run_evaluation(self, predictions: Sequence[str], references: Sequence[str]):
        
        if len(predictions)!=len(references):
            raise ValueError(f"[{self.name}] predictions has {len(predictions)} entries but references has {len(references)}; they must match 1:1.")
    
        if not predictions:
            raise ValueError(f"[{self.name}] predictions is empty; nothing to evaluate.")
        
        empty=sum(1 for p in predictions if not str(p).strip())
        if empty:
            logger.warning(f"[{self.name}] {empty}/{len(predictions)} predictions are empty, they will score ~0 on every metric.")
        
        self.per_sample=[]
        for prediction, reference in zip(predictions, references):
            entry=self.compute_blue_rouge_metero_per_pair(prediction, reference)
            entry["prediction"]=str(prediction)
            entry["reference"]=str(reference)
            entry["prediction_len"]=len(self.tokenize(prediction)) 
            entry["reference_len"]=len(self.tokenize(reference))
            self.per_sample.append(entry)
        
        if self.use_bert_score:
            self.compute_bert_score(predictions, references)
        results=self.aggregate()
        
        #######Corpus level BLEU-4 (what MT/RRG papers usually mean by "BLEU")#######
        results["corpus_bleu_4"]=self.compute_bleu_for_corpus(predictions, references)     
        
        if self.bootstrap_samples>0:
            results["confidence_intervals"]=self.bootstrap_ci()
        
        return results    
    
    ### For print and save report
    def print_report(self, results: dict):
        cis=results.get("confidence_intervals", {})
        logger.info("+"*80)
        logger.info(f"NLG Evaluation [{results.get('name', self.name)}]  "
                    f"N = {results['num_samples']}") 
        logger.info("+"*80)
        for name in self.metric_names():
            line=f" {name.upper():<22}: {results[name]:.4f}" 
            if name in cis:
                line+=f" 95% CI [{cis[name][0]:.4f}, {cis[name][1]:.4f}]"
            logger.info(line)
        logger.info(f" {'CORPUS_BLEU_4':<22}: {results['corpus_bleu_4']:.4f}")    
        logger.info("+"*80)
        logger.info(f" {'AVG PRED LENGTH':<22}: {results['avg_prediction_len']:.1f} tokens\n"
                    f" {'AVG REF LENGTH':<22}: {results['avg_reference_len']:.1f} tokens\n"
                    f" {'LENGTH RATIO':<22}: {results['len_ratio']:.2f}")
        logger.info("+"*80)
    
    '''def to_row(self, results: dict, label: Optional[str]=None):
        row ={"system": label or results.get("name", self.name)}
        for name in self.metric_names():
            row[name.upper()]=round(results[name], 4)
        row["CORPUS_BLEU_4"]=round(results["corpus_bleu_4"], 4)
        row["LENGTH_RATIO"]=round(results["len_ratio"], 2)
        row["NUMBER_OF_SAMPLES"]=results["num_samples"]
        return row'''
    
    @staticmethod
    def coparison_table_pd(all_results: Sequence[dict]):
        
        rows=[]
        for results in all_results:
            row ={"system":results.get("name", "system")}
            for key, value in results.items():
                if key in ("name", "num_samples", "confidence_intervals", "avg_prediction_len", "avg_reference_len", "len_ratio"):
                    continue
                if isinstance(value, (int, float)):
                    row[key.upper()]=round(value, 4)
            row["LENGTH_RATIO"]=round(results["len_ratio"], 2)
            row["NUMBER_OF_SAMPLES"]=results.get("num_samples")
            rows.append(row)
        return pd.DataFrame(rows)
    
    
    
    
    
    
    
    