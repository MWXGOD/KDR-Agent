import json
import time
from datetime import datetime
import wikipedia
import requests
import openai

def get_test_data(test_file_path):
    return get_json_file(test_file_path)


def get_palnner_prompt(dataset, input_sentence):
    # answer_example: "**Truly Unfamiliar Concepts**: Blepharoconjunctivitis#photodermatitis **Ambiguous Concepts**: None" ->str
    # answer_example: "**Truly Unfamiliar Concepts**: None **Ambiguous Concepts**: grafting#infarction#emergency room ->str
    with open(f'data/{dataset}/define.json', 'r', encoding='utf-8') as f:
        type_list = json.load(f)
    type_str = ''
    for tl in type_list:
        type_str+=f'[{tl}], '
    
    prompt = f'''Please read the following text:
> [{input_sentence}]
    
Your task is to identify entities of the following types: {type_str}
    
Before that, identify up to 5 nouns or proper nouns that may require further interpretation, divided into:
    
Truly Unfamiliar Concepts: Nouns whose meaning is **not transparent even to a language model**, and require external knowledge sources (e.g., Wikipedia, PubChem, UMLS) to resolve. These are typically rare chemicals, technical terms, or domain-specific jargon.
    
Ambiguous Concepts: Familiar nouns with multiple possible meanings in the real world, requiring contextual disambiguation (e.g., “Jaguar” as car vs. animal). Do not include obscure terms here.
    
If there are no concepts in a category, write “None”. Return your answer in the following format (use # to separate items):
    
**Truly Unfamiliar Concepts**: concept1#concept2#...  
**Ambiguous Concepts**: concept3#concept4#...'''

    return prompt


def get_disambiguation_prompt(input_sentence, ambiguous_concepts):
   
    prompt = f'''
Given the following sentence:
    
> {input_sentence}
    
And a list of **Ambiguous Concepts** — these are commonly used nouns or terms that may have multiple interpretations in different contexts (e.g., “Jaguar” could refer to either an animal or a car).
    
{ambiguous_concepts}
    
Your task:
For each concept, briefly infer its most likely meaning **in the context of the sentence**. Disambiguate each term based on how it is used, and provide a short explanation for your interpretation.
    
Use the following output format:
    
Concept: <concept>
Interpretation: <a short explanation that includes both the most likely meaning and the reasoning behind it>
'''

    return prompt


def get_wiki_summary(unfamiliar_concepts, sentences=2, max_try_candidates=5):
    """
    Robust Wikipedia lookup for a single concept:
    1) Try direct summary first (with auto_suggest=True)
    2) If DisambiguationError occurs, try the top candidate pages one by one
    3) If PageError occurs, use wikipedia.search for a fuzzy search fallback
    4) If still failing, raise the exception
    """
    try:
        # First attempt: let Wikipedia auto-correct / fuzzy match the query
        return wikipedia.summary(unfamiliar_concepts, sentences=sentences, auto_suggest=True)
    except wikipedia.exceptions.DisambiguationError as e:
        # Disambiguation page: multiple entries share the same name
        for cand in e.options[:max_try_candidates]:
            try:
                return wikipedia.summary(cand, sentences=sentences, auto_suggest=False)
            except (wikipedia.exceptions.DisambiguationError,
                    wikipedia.exceptions.PageError,
                    requests.exceptions.RequestException):
                # Continue to try the next candidate
                continue
        # If all candidates fail, re-raise the original exception
        raise
    except wikipedia.exceptions.PageError:
        # Page not found: fall back to fuzzy search
        search_results = wikipedia.search(unfamiliar_concepts)
        if not search_results:
            raise

        for cand in search_results[:max_try_candidates]:
            try:
                return wikipedia.summary(cand, sentences=sentences, auto_suggest=False)
            except (wikipedia.exceptions.DisambiguationError,
                    wikipedia.exceptions.PageError,
                    requests.exceptions.RequestException):
                continue
        # All search candidates failed
        raise


def get_first_NER_prompt(dataset, sentence, unfamiliar_concepts_answers, ambiguous_concepts_answers):
    # with open('wiki_answer/All_wiki_dict.json', 'r', encoding='utf-8') as f:
    #     all_wiki_dict = json.load(f)
    with open(f'data/{dataset}/labels.json', 'r', encoding='utf-8') as f:
        labels = json.load(f)
    with open(f'data/{dataset}/define.json', 'r', encoding='utf-8') as f:
        type_define = json.load(f)
    with open(f'data/{dataset}/define_sample_min.json', 'r', encoding='utf-8') as f:
        few_shot_samples = json.load(f)
    

    base_prompt = f'''Please read the following text: 
>[{sentence}]

Your task is to infer the entities in the text with the entity category labels: {labels}.

Please infer all the entities in this sentence according to the following entity category definitions, and return the result in the following format:
{{"sentence": "", "entities": [{{"name": "...", "type": "..."}}, ...]}}

'''
    

    type_define_prompt = f'''Please read the following relevant knowledge:

Entity type and their natural-language type definitions:
{type_define}

'''
    
    
    few_shot_prompt = f'''Auxiliary reasoning few-shot examples:
{few_shot_samples}

'''

    # unfamiliar_concepts_answer -> dict
    unfamiliar_concepts_prompt = ''
    if unfamiliar_concepts_answers:
        unfamiliar_concepts_prompt = 'Unfamiliar concepts retrieved by the Wiki agent (orchestrated by LLMs):\n' + unfamiliar_concepts_answers
    # if "None" not in str(unfamiliar_concepts_answer) and unfamiliar_concepts_answer:
    #     unfamiliar_concepts_prompt = 'Unfamiliar concepts retrieved by the Wiki agent (orchestrated by LLMs):\n'
    #     for i, uc in enumerate(unfamiliar_concepts_answer):
    #         unfamiliar_concepts_prompt += f'''Unfamiliar Concept {i+1}:{uc}:{unfamiliar_concepts_answer[uc]}\n'''
    #     unfamiliar_concepts_prompt += '\n'
    
    # ambiguous_concepts_answer -> dict
    ambiguous_concepts_prompt = ""
    if ambiguous_concepts_answers:
        ambiguous_concepts_prompt = 'Disambiguated concepts derived through discussion by the LLM agent (orchestrated by LLMs):\n'+ambiguous_concepts_answers


    # if "None" not in str(ambiguous_concepts_answer) and ambiguous_concepts_answer:
    #     ambiguous_concepts_prompt = 'Disambiguated concepts derived through discussion by the LLM agent (orchestrated by LLMs):\n'
    #     for i, ac in enumerate(ambiguous_concepts_answer):
    #         ambiguous_concepts_prompt += f'''Disambiguated Concept {i+1}:{ac}:{ambiguous_concepts_answer[ac]}\n'''
    #     ambiguous_concepts_prompt += '\n'

    return base_prompt+type_define_prompt+few_shot_prompt+unfamiliar_concepts_prompt+ambiguous_concepts_prompt


def get_reflection_prompt(dataset, data_item):
    with open(f'data/{dataset}/define.json', 'r', encoding='utf-8') as f:
        type_define = json.load(f)
    prompt = f'''You are an expert in Named Entity Recognition (NER). Your task is to verify and correct the NER results generated by a model. Follow the instructions step by step.

Input Text:
>[{data_item['sentence']}]

Predicted Entities:
Entities: {data_item['entities']}

Entity type and their definitions:
{type_define}

Checking Criteria:
Check the predicted NER results according to the following four criteria:
1. Reflect on whether the entity boundaries are accurate.
2. Reflect on whether the entity types are accurate.
3. Reflect on whether both the entity boundaries and types are accurate.
4. Reflect on whether any entities have been missed.

Instructions:
1. Examine the predicted entities based on the above four criteria, and explicitly explain which entity spans satisfy or violate each rule.
2. Identify and explain any errors in prediction, including missing, spurious, or incorrectly labeled entities.
3. Revise the entity predictions according to your analysis, ensuring compliance with all four criteria.

Output the final corrected NER result in the following fixed format:
{{"sentence": "", "entities": [{{"name": "...", "type": "..."}}, ...]}}'''
    return prompt


def QA_GPT(prompt, model_name):
    message = [{"role": "user", "content": prompt}]
    response = openai.chat.completions.create(
        model=model_name,
        messages=message
    )
    answer =  response.choices[0].message.content
    message.append({"role": "assistant", "content": answer})
    return answer, message


def get_json_file(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)
    

def save_json_file(json_file, data):
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=8)


def get_wiki_and_disambiguation_object(palnner_answer):
    lines = palnner_answer.split("\n")
    unfamiliar_concepts = []
    ambiguous_concepts = []

    for line in lines:
        if "Truly Unfamiliar Concepts" in line:
            unfamiliar_concepts = [unfamiliar_concept for unfamiliar_concept in line.split(":")[1].strip().split("#") if unfamiliar_concept]
        if "Ambiguous Concepts" in line:
            ambiguous_concepts = [ambiguous_concept for ambiguous_concept in line.split(":")[1].strip().split("#") if ambiguous_concept]
    return unfamiliar_concepts, ambiguous_concepts


def get_PRF(test_data):
    P, R, C = 0, 0, 0
    for td in test_data:
        gold_lables = set([e_item["name"]+"-"+e_item["type"] for e_item in td["entities"]])
        pred_lables = set([pe_item["name"]+"-"+pe_item["type"] for pe_item in td["predicts"]])

        R += len(gold_lables)
        P += len(pred_lables)
        C += len(gold_lables & pred_lables)
    P = C / P if P > 0 else 0.0
    R = C / R if R > 0 else 0.0
    F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0.0


    return P, R, F1
