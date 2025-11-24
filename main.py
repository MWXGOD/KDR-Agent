import argparse
from arguments import Arguments
from tool import *
import json_repair
from jsonchecker import JsonChecker
import os
import logging

# init log
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# openai.api_key = "sk-*********************************"
openai.api_key = os.getenv("OPENAI_API_KEY")

# schema
answer_schema = {"sentence": str, "entities": list}
answer_inner_schema = {"name": str, "type": str}
answer_checker = JsonChecker(answer_schema)
answer_inner_checker = JsonChecker(answer_inner_schema)


parser = argparse.ArgumentParser()
parser.add_argument("--args_file", type=str, default="./config/Bio_BC5CDR.json")
args = parser.parse_args()
logger.info(f"Loaded config from: {args.args_file}")
args = Arguments(args.args_file)

logger.info("Loading test data...")
test_data = get_test_data(args.test_file_path)
logger.info(f"Loaded {len(test_data)} test samples.")

for data in test_data:
    sentence = data["sentence"]
    logger.info(f"Processing sentence: {sentence}")

    # =====================================================
    # Stage 1: Planner
    # =====================================================
    logger.info("Stage 1: Building planner prompt...")
    palnner_prompt = get_palnner_prompt(args.dataset, sentence)
    palnner_answer, _ = QA_GPT(palnner_prompt, args.model_name)
    logger.info("Planner answer received.")

    unfamiliar_concepts, ambiguous_concepts = get_wiki_and_disambiguation_object(palnner_answer)

    # ===== Wikipedia Query =====
    if 'none' not in str(unfamiliar_concepts).lower():
        logger.info(f"Detected unfamiliar concepts: {unfamiliar_concepts}")

        unfamiliar_concepts_answers = []
        for uc in unfamiliar_concepts:
            try:
                wiki_res = get_wiki_summary(uc)
                unfamiliar_concepts_answers.append(wiki_res)
                logger.info(f"Wikipedia summary for {uc}: OK")
            except Exception as e:
                logger.warning(f"Wikipedia failed for {uc}: {e}")
                unfamiliar_concepts_answers.append(f"[NO RESULT] {uc}")

        unfamiliar_concepts_answers = "\n".join(unfamiliar_concepts_answers)
    else:
        unfamiliar_concepts_answers = ''
        logger.info("No unfamiliar concepts detected.")

    # ===== Disambiguation =====
    if 'none' not in str(ambiguous_concepts).lower():
        logger.info(f"Detected ambiguous concepts: {ambiguous_concepts}")

        disambiguation_prompt = get_disambiguation_prompt(sentence, ambiguous_concepts)
        ambiguous_concepts_answers, _ = QA_GPT(disambiguation_prompt, args.model_name)
    else:
        ambiguous_concepts_answers = ''
        logger.info("No ambiguous concepts detected.")

    # =====================================================
    # Stage 2: First NER
    # =====================================================
    logger.info("Stage 2: Running first NER...")

    for i in range(args.max_loop):
        logger.info(f"First NER attempt #{i+1}")
        first_NER_prompt = get_first_NER_prompt(args.dataset, sentence, unfamiliar_concepts_answers, ambiguous_concepts_answers)
        first_NER_answer, _ = QA_GPT(first_NER_prompt, args.model_name)

        try:
            first_NER_answer_json = json_repair.loads(first_NER_answer)
        except Exception:
            logger.warning("JSON parse failed for first NER.")
            continue

        if answer_checker.check([first_NER_answer_json]) and \
           answer_inner_checker.check(first_NER_answer_json["entities"]):
            logger.info("First NER JSON valid.")
            break

        if i == args.max_loop - 1:
            logger.error("First NER failed after max attempts, fallback to empty.")
            first_NER_answer_json = {"sentence": sentence, "entities": []}

    # =====================================================
    # Stage 3: Reflection
    # =====================================================
    logger.info("Stage 3: Reflection correction...")

    for i in range(args.max_loop):
        logger.info(f"Reflection attempt #{i+1}")
        reflection_prompt = get_reflection_prompt(args.dataset, first_NER_answer_json)
        reflection_answer, _ = QA_GPT(reflection_prompt, args.model_name)

        try:
            reflection_answer_json = json_repair.loads(reflection_answer)
        except Exception:
            logger.warning("JSON parse failed for reflection.")
            continue

        if answer_checker.check([reflection_answer_json]) and \
           answer_inner_checker.check(reflection_answer_json["entities"]):
            logger.info("Reflection JSON valid.")
            break

        if i == args.max_loop - 1:
            logger.error("Reflection failed after max attempts, fallback to empty.")
            reflection_answer_json = {"sentence": sentence, "entities": []}

    # save result
    data["predicts"] = reflection_answer_json["entities"]


logger.info("Saving final predictions...")
save_json_file(args.save_file_path, test_data)
logger.info(f"Saved to: {args.save_file_path}")

# =====================================================
# Compute PRF
# =====================================================
logger.info("Calculating PRF metrics...")
P, R, F1 = get_PRF(test_data)
logger.info(f"PRF = {P:.4f}, {R:.4f}, {F1:.4f}")

print(f"\nPrecision: {P}, Recall: {R}, F1: {F1}")
