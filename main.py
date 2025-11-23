import argparse
from arguments import Arguments
from tool import *
import json_repair
from jsonchecker import JsonChecker

openai.api_key = "sk-*********************************"


# schema
answer_schema = {"sentence": str, "entities": list}
answer_inner_schema = {"name": str, "type": str}
answer_checker = JsonChecker(answer_schema)
answer_inner_checker = JsonChecker(answer_inner_schema)


parser = argparse.ArgumentParser()
parser.add_argument("--args_file", type=str, default="./config/Bio_BC5CDR.json")
args = parser.parse_args()
args = Arguments(args.args_file)


test_data = get_test_data(args.test_file_path)


for data in test_data:
    sentence = data["sentence"] # -> str
    entities = data["entities"] # -> list

    # stage1
    # planner
    palnner_prompt = get_palnner_prompt(args.dataset, sentence)
    palnner_answer,  _= QA_GPT(palnner_prompt, args.model_name)
    unfamiliar_concepts, ambiguous_concepts = get_wiki_and_disambiguation_object(palnner_answer)

    # unfamiliar_concepts
    if 'none' not in str(unfamiliar_concepts).lower():
        unfamiliar_concepts_answers = []
        for uc in unfamiliar_concepts:
            unfamiliar_concepts_answer = get_wiki_summary(uc)
            unfamiliar_concepts_answers.append(unfamiliar_concepts_answer)
        unfamiliar_concepts_answers = '\n'.join(unfamiliar_concepts_answers)
    else:
        unfamiliar_concepts_answers = ''

    # ambiguous_concepts
    if 'none' not in str(ambiguous_concepts).lower():
        disambiguation_prompt = get_disambiguation_prompt(sentence, ambiguous_concepts)
        ambiguous_concepts_answers, _ = QA_GPT(disambiguation_prompt, args.model_name)
    else:
        ambiguous_concepts_answers = ''

    # stage2
    # first_NER
    for i in range(args.max_loop):
        first_NER_prompt = get_first_NER_prompt(args.dataset, sentence, unfamiliar_concepts_answers, ambiguous_concepts_answers)
        first_NER_answer, _ = QA_GPT(first_NER_prompt, args.model_name)
        first_NER_answer_json = json_repair.loads(first_NER_answer)
        if answer_checker.check([first_NER_answer_json]):
            if answer_inner_checker.check(first_NER_answer_json["entities"]):
                break
        if i == 9:
            first_NER_answer_json = {
                "sentence": sentence,
                "entities": []
            }

    # reflection
    for i in range(args.max_loop):
        reflection_prompt = get_reflection_prompt(args.dataset, first_NER_answer_json)
        reflection_answer, _ = QA_GPT(reflection_prompt, args.model_name)
        reflection_answer_json = json_repair.loads(reflection_answer)
        if answer_checker.check([reflection_answer_json]):
            if answer_inner_checker.check(reflection_answer_json["entities"]):
                break
        if i == 9:
            reflection_answer_json = {
                "sentence": sentence,
                "entities": []
            }
    
    data["predicts"] = reflection_answer_json["entities"]
    break

save_json_file(args.save_file_path, test_data)

P, R, F1 = get_PRF(test_data)

print(P, R, F1)





