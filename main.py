from dotenv import load_dotenv
load_dotenv() ### .env file in project root
import os
import torch
import faiss
from src.utils import load_config
from collections import Counter
from src.logger import logging
from src.llm_backbones import DeepSeekLLM, QwenLLM, LlamaLLM, BaseLLM
from src.img_loader import ImageLoader
from src.evaluation_metrics import AgentEvalMetrics
from src.exception import CustomException
from src.clip_backbones import OpenAICLIP, PubMedCLIP, BiomedCLIP
from src.retrieval_agent import RetrievalAgent
from src.vision_agent import VisionAgent
from src.llava_med_backbone import LLaVAMedVLM
from src.draft_agent import DraftAgent
from src.refiner_agent import RefinerAgent
from src.prompts import (sys_prompts_and_versions_for_draft_agent, sys_prompts_and_versions_for_refiner_agent, sys_prompts_and_versions_for_vision_agent,
                         draft_prompt_version, refiner_prompt_version, vlm_prompt_version)
from src.retrieval_database_build import IUXRayDataset

logger = logging.getLogger("main.py")
defalt_device="cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    config_path="config.yaml"
    config = load_config(config_path)

    device=defalt_device
    logger.info(f"Using device: {device}")

    ##############################################################################################
    ########################### Load Test Data And Reeferences ###################################
    ##############################################################################################
    iu_dataset= IUXRayDataset(config["IU_DATASET_ANNOTATION_JSON_PATH"])
    ### Load test images and their Groun Truth reports
    test_image_report_pairs = iu_dataset.get_all_image_report_pairs("test",
                                                                    image_dir=config["IU_DATASET_IMAGES_PATH"],
                                                                    frontal_view_only=True)
    logger.info(f"Test image-report pairs fetched: {len(test_image_report_pairs)}")
    test_image_paths = [pair["image_path"] for pair in test_image_report_pairs]
    report_lookup={pair["image_path"]: pair["report"] for pair in test_image_report_pairs}
    study_id_lookup={pair["image_path"]: pair["id"] for pair in test_image_report_pairs}

    image_loader=ImageLoader(image_paths=test_image_paths)
    filenames, images = image_loader.load_raw_images()
    logger.info(f"Loaded {len(images)} test images for retrieval agent")

    ### Ground truth report text for each test image
    ### This is the REFERENCE for every stage of the pipeline (retrieval baseline, draft, refiner, final synthesis).
    gt_report=[report_lookup[fname] for fname in filenames]
    study_ids=[study_id_lookup[fname] for fname in filenames]
    logger.info(f"Unique ground truth reports: {len(set(gt_report))} / {len(gt_report)}")

    ##############################################################################################
    ################################# For Retrieval Agent ########################################
    ##############################################################################################
    retrieved_per_clip_backbone_results={}
    all_clip_backbone_eval_results=[]
    clip_eval_by_backbone={}
    models_and_retrive_image_database={
        "OpenAICLIP (zero-shot)": (OpenAICLIP(device=device), config["RET_DATABSE_IMAGE_FOR_BASE_OPENAICLIP"]),
        "PubMedCLIP (zero-shot)": (PubMedCLIP(device=device), config["RET_DATABSE_IMAGE_FOR_BASE_PUBMEDCLIP"]),
        "BiomedCLIP (zero-shot)": (BiomedCLIP(device=device), config["RET_DATABSE_IMAGE_FOR_BASE_BIOMEDCLIP"]),
        # "OpenAICLIP (fine-tuned)": (OpenAICLIP(config["FINE_TUNED_CLIP_CHECKPOINT"], device=device), config["RET_DATABSE_IMAGE_FOR_FINE_TUNED_OPENAICLIP"]),
        # "PubMedCLIP (fine-tuned)": (PubMedCLIP(config["FINE_TUNED_PUBMEDCLIP_CHECKPOINT"], device=device), config["RET_DATABSE_IMAGE_FOR_FINE_TUNED_PUBMEDCLIP"]),
        # "BiomedCLIP (fine-tuned)": (BiomedCLIP(config["FINE_TUNED_BIOMEDCLIP"], device=device), config["RET_DATABSE_IMAGE_FOR_FINE_TUNED_BIOMEDCLIP"])
        }

    '''models_and_retrive_text_database={
        "OpenAICLIP (zero-shot)": (OpenAICLIP(device=device), config["RET_DATABSE_TEXT_FOR_BASE_OPENAICLIP"]),
        "PubMedCLIP (zero-shot)": (PubMedCLIP(device=device), config["RET_DATABSE_TEXT_FOR_BASE_PUBMEDCLIP"]),
        "BiomedCLIP (zero-shot)": (BiomedCLIP(device=device), config["RET_DATABSE_TEXT_FOR_BASE_BIOMEDCLIP"]),
        # "OpenAICLIP (fine-tuned)": (OpenAICLIP(config["FINE_TUNED_CLIP_CHECKPOINT"], device=device), config["RET_DATABSE_TEXT_FOR_FINE_TUNED_OPENAICLIP"]),
        # "PubMedCLIP (fine-tuned)": (PubMedCLIP(config["FINE_TUNED_PUBMEDCLIP_CHECKPOINT"], device=device), config["RET_DATABSE_TEXT_FOR_FINE_TUNED_PUBMEDCLIP"]),
        # "BiomedCLIP (fine-tuned)": (BiomedCLIP(config["FINE_TUNED_BIOMEDCLIP"], device=device), config["RET_DATABSE_TEXT_FOR_FINE_TUNED_BIOMEDCLIP"])
        }'''

    for name, (model, retrival_database_path) in models_and_retrive_image_database.items():
        ret_agent=None
        logger.info(f"Evaluating retrieval agent for backbone: {name}..........................")
        try:
            ret_agent=RetrievalAgent(model)
            ret_agent.load_retrieval_database(retrival_database_path)
            ret_agent_batch_raw_results=ret_agent.retrieve_reports(images, config["TOP_K_RETRIEVAL"])
            retrieved_per_clip_backbone_results[name]=ret_agent_batch_raw_results
            ret_agent.save_retrievals(batch_retrieved=ret_agent_batch_raw_results,
                                      output_path=os.path.join(config["SAVE_RETRIEVED_REPORTS"], f"retrievals_{model.name.lower()}.json"),
                                      study_ids=study_ids,
                                      references=gt_report,
                                      image_path=filenames,
                                      extra_meta={"top_k": config["TOP_K_RETRIEVAL"]})
            logger.info(f"Retrieved top-{config['TOP_K_RETRIEVAL']} reports for {len(images)} test images.")

            top1_reports=[]
            top1_simi=[]
            for ret_results in ret_agent_batch_raw_results:
                if not ret_results:
                    logger.warning(f"[{name}] empty retrieval results fot one query, using empty text.")
                    top1_reports.append("")
                    continue
                top1_reports.append(ret_results[0][1])
                top1_simi.append(ret_results[0][2])

            top1_counts=Counter(top1_reports)
            most_text, most_count=top1_counts.most_common(1)[0]
            logger.info(f"[{name}] unique top-1 reports: {len(top1_counts)}/{len(top1_reports)} | most frequent returned {most_count} times ({100*most_count/len(top1_reports):.1f}%)")
            logger.info(f"[{name}] top hub text: {most_text[:120]}")

            ### Pick the backbone with the highest ROUGE_L / BERT_SCORE_F1, not the highest cosine similarity
            logger.info(f"Evaluating retrieval results for [{name}]............................")
            retrieval_agent_evaluator=AgentEvalMetrics(
                name=f"Retrieval only Top1 [{name}]",
                use_bert_score=config.get("USE_BERT_SCORE", False),
                rescale_bert_score_with_baseline=True,
                bootstrap_samples=config.get("BOOTSTRAP_SAMPLES", 0))
            retrieval_agent_evl_results=retrieval_agent_evaluator.run_evaluation(predictions=top1_reports, references=gt_report)
            retrieval_agent_evaluator.print_report(retrieval_agent_evl_results)
            if top1_simi:
                avg_simi=sum(top1_simi)/len(top1_simi)
                logger.info(f"[{name}] Average top-1 similarity: {avg_simi:.4f} [not comparable across backbones]")
            all_clip_backbone_eval_results.append(retrieval_agent_evl_results)
            clip_eval_by_backbone[name]=retrieval_agent_evl_results

        except Exception as e:
            logger.info(f"Skipping [{name}] due to error: {e}")
            continue
        finally:
            if ret_agent is not None:
                del ret_agent
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if not retrieved_per_clip_backbone_results:
        logger.error("No retrieval backbone succeeded, con't run llm agents.")
        raise SystemExit(1)

    ##############################################################################################
    ############################# Select best CLIP for all agents ################################
    ##############################################################################################
    selection_clip_metric=config.get("CLIP_SELECTION_METRIC", "rouge_L")
    forced_select_clip=config.get("SELECTED_CLIP_BACKBONE")

    if forced_select_clip and forced_select_clip in retrieved_per_clip_backbone_results:
        selected_clip_backbone=forced_select_clip
        logger.info(f"Retrieval clip backbone fixed by config [{selected_clip_backbone}]")
    else:
        if forced_select_clip:
            logger.warning(f"[{forced_select_clip}] from config is not available. auto selecting.....")

        clip_scored={name: res for name, res in clip_eval_by_backbone.items() if selection_clip_metric in res}
        
        if clip_scored:
            selected_clip_backbone=max(clip_scored, key=lambda n: clip_scored[n][selection_clip_metric])
            ranked=sorted(clip_scored.items(), key=lambda kv: kv[1][selection_clip_metric], reverse=True)
            ranking=", ".join(f"{n}={res[selection_clip_metric]:.4f}" for n, res in ranked)
            logger.info(f"Auto selected retrieval clip backbone by {selection_clip_metric}: {selected_clip_backbone}\n"
                        f"CLIP Backbone ranking [{selection_clip_metric}]: {ranking}")
            
            values=[res[selection_clip_metric] for res in clip_scored.values()]
            spread=max(values)-min(values)
            if spread<0.01:
                logger.info(f"CLIP backbone {selection_clip_metric} spread is only {spread:.4f}. so all clip backbones are nearly same.")
        else:
            selected_clip_backbone=next(iter(retrieved_per_clip_backbone_results))
            logger.warning(f"No backbone had metric [{selection_clip_metric}]. Falling back to first clip backbone [{selected_clip_backbone}].")

    batch_retrieved=retrieved_per_clip_backbone_results[selected_clip_backbone]
    logger.info(f"Using retrieval output from [{selected_clip_backbone}] for thr LLM agents.")

    ##############################################################################################
    ################################# Initialize 3 LLM ###########################################
    ##############################################################################################
    llms={"Llama": LlamaLLM(api_provider=config["LLM_PROVIDER"], model_id=config.get("LLAMA_MODEL_ID") or None, cache_dir=config["CACHE_DIR_LLMS"], temperature=config["LLM_TEMPERATURE"], max_tokens=config["LLAMA_MAX_TOKENS"]),
          "Qwen": QwenLLM(api_provider=config["LLM_PROVIDER"], model_id=config.get("QWEN_MODEL_ID") or None , cache_dir=config["CACHE_DIR_LLMS"], temperature=config["LLM_TEMPERATURE"], max_tokens=config["QWEN_MAX_TOKENS"]),
          "Deepseek": DeepSeekLLM(api_provider=config["LLM_PROVIDER"], model_id=config.get("DEEPSEEK_MODEL_ID") or None, cache_dir=config["CACHE_DIR_LLMS"], temperature=config["LLM_TEMPERATURE"], max_tokens=config["DEEPSEEK_MAX_TOKENS"])}
    
    for llm_name, llm in llms.items():
        logger.info(f"LLM ready: {llm_name} - {llm}")

    ##############################################################################################
    ##################################### For Draft Agent ########################################
    ##############################################################################################
    all_llm_backbone_eval_results_for_draft_agent=[]
    retrieved_subset=batch_retrieved[:config["TEST_IMAGE_LIMIT"]] if config["TEST_IMAGE_LIMIT"] else batch_retrieved
    gt_subset=gt_report[:config["TEST_IMAGE_LIMIT"]] if config["TEST_IMAGE_LIMIT"] else gt_report
    ids_subset= study_ids[:config["TEST_IMAGE_LIMIT"]] if config["TEST_IMAGE_LIMIT"] else study_ids
    logger.info(f"Running the draft agent on [{len(retrieved_subset)}] studies (limit={config['TEST_IMAGE_LIMIT']}).")
    drafts_per_llm={}

    for llm_name, llm in llms.items():
        logger.info(f"Draft agrnt with LLM: {llm_name}..............................................................")
        try:
            draft_agent=DraftAgent(llm=llm, system_prompt=sys_prompts_and_versions_for_draft_agent[draft_prompt_version])
            logger.info(f"[{llm_name}] example user prompt:\n{draft_agent.build_user_prompt(retrieved_subset[0])}")
            drafts=draft_agent.batch_run(retrieved_subset)
            drafts_per_llm[llm_name]=drafts

            draft_agent.save_drafts(drafts=drafts,
                                    output_path=os.path.join(config["SAVE_DRAFT_REPORTS"], f"drafts_{llm_name.lower()}_{draft_prompt_version}.json"),
                                    study_ids=ids_subset,
                                    references=gt_subset,
                                    retrieved=retrieved_subset,
                                    extra_meta={"prompt_version": draft_prompt_version, "retrieval_backbone": selected_clip_backbone})

            logger.info(f"Evaluating draft results for [{llm_name}]............................")
            draft_agent_evaluator= AgentEvalMetrics(name=f"Draft Agent [{llm_name}]",
                                                    use_bert_score=config.get("USE_BERT_SCORE", False),
                                                    rescale_bert_score_with_baseline=True,
                                                    bootstrap_samples=config.get("BOOTSTRAP_SAMPLES", 0))
            draft_agent_evl_results=draft_agent_evaluator.run_evaluation(predictions=drafts, references=gt_subset)
            draft_agent_evaluator.print_report(draft_agent_evl_results)
            all_llm_backbone_eval_results_for_draft_agent.append(draft_agent_evl_results)

        except Exception as e:
            logger.error(f"Skipping draft agent for [{llm_name}] due to error: {e}")
            continue

    ##############################################################################################
    ##################################### For Refiner Agent ######################################
    ##############################################################################################
    all_llm_backbone_eval_results_for_refiner_agent=[]
    refinements_per_llm={}

    for llm_name, llm in llms.items():
        logger.info(f"Refiner agrnt with LLM: {llm_name}..............................................................")
        try:
            if llm_name not in drafts_per_llm:
                logger.warning(f"[{llm_name}] no drafts in memory (draft stage failed), skipping refiner.")
                continue
            current_drafts=drafts_per_llm[llm_name]
            refiner_agent=RefinerAgent(llm=llm, system_prompt=sys_prompts_and_versions_for_refiner_agent[refiner_prompt_version])
            logger.info(f"[{llm_name}] example refiner user prompt:\n{refiner_agent.build_user_prompt(current_drafts[0],retrieved_subset[0])}")
            refinements=refiner_agent.batch_run(current_drafts, retrieved_subset)
            refinements_per_llm[llm_name]=refinements

            refiner_agent.save_refinements(refinements=refinements,
                                           output_path=os.path.join(config["SAVE_REFINED_REPORTS"], f"refinements_{llm_name.lower()}_{refiner_prompt_version}.json"),
                                           drafts=current_drafts,
                                           study_ids=ids_subset,
                                           references=gt_subset,
                                           retrieved=retrieved_subset,
                                           extra_meta={"prompt_version": refiner_prompt_version, "draft_prompt_version": draft_prompt_version, "retrieval_backbone": selected_clip_backbone})

            logger.info(f"Evaluating refine results for [{llm_name}]............................")
            refiner_agent_evaluator=AgentEvalMetrics(name=f"Refiner Agent [{llm_name}]",
                                                     use_bert_score=config.get("USE_BERT_SCORE", False),
                                                     rescale_bert_score_with_baseline=True,
                                                     bootstrap_samples=config.get("BOOTSTRAP_SAMPLES", 0))
            refiner_agent_evl_results=refiner_agent_evaluator.run_evaluation(predictions=refinements, references=gt_subset)
            refiner_agent_evaluator.print_report(refiner_agent_evl_results)
            all_llm_backbone_eval_results_for_refiner_agent.append(refiner_agent_evl_results)
        except Exception as e:
            logger.error(f"Skipping refiner agent for [{llm_name}] due to error: {e}")
            continue

    ##############################################################################################
    ##################################### For Vision Agent ######################################
    ##############################################################################################
    vision_agent_eval_results=[]
    visual_descriptions=None
    images_subset=images[:config["TEST_IMAGE_LIMIT"]] if config["TEST_IMAGE_LIMIT"] else images
    paths_subset=filenames[:config["TEST_IMAGE_LIMIT"]] if config["TEST_IMAGE_LIMIT"] else filenames
    visual_descriptions_saved_path=os.path.join(config["SAVE_VISUAL_DESCRIPTIONS"], f"visual_descriptions_{vlm_prompt_version}.json")

    visual_descriptions_cache_path=os.path.join(os.path.dirname(visual_descriptions_saved_path), "vision_agent", os.path.basename(visual_descriptions_saved_path))
    ### Check visual_descriptions saved file exist
    if os.path.exists(visual_descriptions_cache_path):
        try:### then load cached visual_descriptions from save path for vision agent
            cached=VisionAgent.load_visual_descriptions(visual_descriptions_cache_path)
            if len(cached)!=len(ids_subset):
                logger.warning(f"[vision_agent] cached count {len(cached)} != current subset {len(ids_subset)}; regenerating.")
                visual_descriptions=None
            else:
                visual_descriptions=cached
                logger.info(f"[vision_agent] reusing {len(visual_descriptions)} saved descriptions from [{visual_descriptions_cache_path}]")
        except Exception as e:
            logger.error(f"[vision_agent] could not read cache at [{visual_descriptions_cache_path}]: {e}. Regenerating.")

    if config.get("FORCE_REGENERATE_VISUAL_DESCRIPTIONS", False):
        logger.info("[vision_agent] force regenerate set. ignoring any cache")
        visual_descriptions=None

    if visual_descriptions is None:
        vlm=LLaVAMedVLM(device=device, 
                        load_in_4bit=True)
        vision_agent=VisionAgent(vlm=vlm, vlm_prompt=sys_prompts_and_versions_for_vision_agent[vlm_prompt_version])
        visual_descriptions=vision_agent.batch_run(images_subset)
        vision_agent.save_visual_description(visual_descriptions=visual_descriptions,
                                             output_path=visual_descriptions_saved_path,
                                             study_ids=ids_subset,
                                             image_paths=paths_subset,
                                             references=gt_subset,
                                             extra_meta={"vlm_prompt_version": vlm_prompt_version})
        vlm.unload_model()
        del vlm, vision_agent
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        logger.info(f"[vision_agent] reusing {len(visual_descriptions)} saved descriptions from {visual_descriptions_cache_path}")

    try:
        logger.info("Evaluating vision agent results................................")
        vision_agent_evaluator=AgentEvalMetrics(name=f"Vision Agent [LLaVA-Med {vlm_prompt_version}]",
                                                use_bert_score=config.get("USE_BERT_SCORE", False),
                                                rescale_bert_score_with_baseline=True,
                                                bootstrap_samples=config.get("BOOTSTRAP_SAMPLES", 0))
        vision_agent_evl_results=vision_agent_evaluator.run_evaluation(predictions=visual_descriptions, references=gt_subset)
        vision_agent_evaluator.print_report(vision_agent_evl_results)
        vision_agent_eval_results.append(vision_agent_evl_results)
    except Exception as e:
        logger.error(f"Skipping vision agent evaluation due to error: {e}")

    ##############################################################################################
    ############ Save llm_provenance & eval results comparison table for all agent ###############
    ##############################################################################################
    BaseLLM.save_llm_provenance(list(llms.values()), os.path.join(config["SAVE_LLM_PROVENANCE"], "llm_provenance.json"))

    ### Save comparison table For All CLIP Backbones for retrieval agent to local
    if all_clip_backbone_eval_results:
        comparison_df_ret_agent=AgentEvalMetrics.coparison_table_pd(all_clip_backbone_eval_results)
        logger.info(f"Comparison of all clip backbone models:\n\n{comparison_df_ret_agent.to_string(index=False)}\n")
        output_path_ret_agent=os.path.join(config["EVAL_RESULTS_DIR"], "retrieval_agent" ,"image_retrieval_agent_backbone_comparison_results.csv")
        os.makedirs(os.path.dirname(output_path_ret_agent), exist_ok=True)
        comparison_df_ret_agent.to_csv(output_path_ret_agent, index=False)
        logger.info(f"Comparison results saved to {output_path_ret_agent}")
    else:
        logger.info("No backbone was evaluated successfully; nothing to compare.")

    ### Save comparison table For All llm Backbones for draft agent to local
    if all_llm_backbone_eval_results_for_draft_agent:
        comparison_df_draft_agent=AgentEvalMetrics.coparison_table_pd(all_llm_backbone_eval_results_for_draft_agent)
        logger.info(f"Comparison of all LLM backbone models (draft):\n\n{comparison_df_draft_agent.to_string(index=False)}\n")
        output_path_draft_agent=os.path.join(config["EVAL_RESULTS_DIR"], "draft_agent" ,f"draft_agent_llm_comparison_results_{draft_prompt_version}.csv")
        os.makedirs(os.path.dirname(output_path_draft_agent), exist_ok=True)
        comparison_df_draft_agent.to_csv(output_path_draft_agent, index=False)
        logger.info(f"Comparison results saved to {output_path_draft_agent}")
    else:
        logger.info("No drafts results was evaluated successfully, nothing to compare.")

    ### Save comparison table For All llm Backbones for refiner agent to local
    if all_llm_backbone_eval_results_for_refiner_agent:
        comparison_df_refiner_agent=AgentEvalMetrics.coparison_table_pd(all_llm_backbone_eval_results_for_refiner_agent)
        logger.info(f"Comparison of all LLM backbone models (refiner):\n\n{comparison_df_refiner_agent.to_string(index=False)}\n")
        output_path_refiner_agent=os.path.join(config["EVAL_RESULTS_DIR"], "refiner_agent" ,f"refiner_agent_llm_comparison_results_{refiner_prompt_version}.csv")
        os.makedirs(os.path.dirname(output_path_refiner_agent), exist_ok=True)
        comparison_df_refiner_agent.to_csv(output_path_refiner_agent, index=False)
        logger.info(f"Comparison results saved to {output_path_refiner_agent}")
    else:
        logger.info("No refiner results was evaluated successfully; nothing to compare.")

    ### Save comparison table for vision agent to local
    if vision_agent_eval_results:
        comparison_df_vision_agent=AgentEvalMetrics.coparison_table_pd(vision_agent_eval_results)
        logger.info(f"Comparison vision agent results:\n\n{comparison_df_vision_agent.to_string(index=False)}\n")
        output_path_vision_agent=os.path.join(config["EVAL_RESULTS_DIR"], "vision_agent" ,f"vision_agent_results_{vlm_prompt_version}.csv")
        os.makedirs(os.path.dirname(output_path_vision_agent), exist_ok=True)
        comparison_df_vision_agent.to_csv(output_path_vision_agent, index=False)
        logger.info(f"Vision agent results saved to {output_path_vision_agent}")
    else:
        logger.info("Vision agent results was not evaluated successfully; nothing to compare.")