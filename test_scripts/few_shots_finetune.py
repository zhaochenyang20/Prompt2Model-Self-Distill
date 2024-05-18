import os
# TODO change card number
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WANDB_MODE"] = "offline"

import gc
import re
import wandb
from pathlib import Path
import json
import datasets
import torch
import shutil
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer
import numpy as np

from prompt2model.output_annotator import construct_meta_prompt
from prompt2model.prompt_parser import MockPromptSpec, TaskType
from prompt2model.utils.path import ROOT, STORE_ROOT, MODEL_PATH
from prompt2model.utils import count_tokens_from_string
from utils.inference import vllm_inference

import string
import re
from collections import Counter
from datasets import Dataset
import pandas as pd


def store_evaluation_content(
    content_store_path, tuned_model_generated_outputs, prompts, GROUND_TRUTH
):
    print(f"Genrated contents are stored in {content_store_path}")
    datasets.Dataset.from_dict(
        dict(
            model_output=tuned_model_generated_outputs,
            model_input=prompts,
            groud_truth=GROUND_TRUTH,
        )
    ).save_to_disk(content_store_path)

def find_last_occurrence(model_output: str, labels: list[str]) -> str:
    pattern = '|'.join(re.escape(label) for label in labels)
    regex = re.compile(pattern)
    matches = list(regex.finditer(model_output))
    return matches[-1].group() if matches else None

# cited from https://github.com/allenai/natural-instructions/blob/55a365637381ce7f3748fa2eac7aef1a113bbb82/eval/automatic/evaluation.py#L24
def normalize_answer(s):
    """Lower text and remove punctuation, and extra whitespace."""

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))

def exact_match(prediction, ground_truth, xlingual=False):
    # small changed based on our current code
    if prediction is None:
        return 0
    return (normalize_answer(prediction) == normalize_answer(ground_truth))

def exact_match_score(
    GROUND_TRUTH,
    tuned_model_generated_outputs,
):
    labels = list(Counter(GROUND_TRUTH).keys())
    index = 0
    n = len(GROUND_TRUTH)
    for i in range(n):
        index += exact_match(find_last_occurrence(tuned_model_generated_outputs[i], labels), GROUND_TRUTH[i])
    score = index / len(GROUND_TRUTH)
    return score

def lcs_length_dp(x, y):
    """Compute the length of the longest common subsequence between two strings using dynamic programming."""
    m, n = len(x), len(y)
    dp_table = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                dp_table[i][j] = 0
            elif x[i - 1] == y[j - 1]:
                dp_table[i][j] = dp_table[i - 1][j - 1] + 1
            else:
                dp_table[i][j] = max(dp_table[i - 1][j], dp_table[i][j - 1])

    return dp_table[m][n]

def rouge_l_score(GROUND_TRUTH, tuned_model_generated_outputs):
    scores = []
    for gt, gen in zip(GROUND_TRUTH, tuned_model_generated_outputs):
        lcs = lcs_length_dp(gt, gen)
        if lcs == 0:
            scores.append(0)
            continue
        precision = lcs / len(gen)
        recall = lcs / len(gt)
        f_measure = (2 * precision * recall) / (precision + recall)
        scores.append(f_measure)
    return sum(scores) / len(scores)

def extract_number(s):
    return int(re.search(r"\d+", s).group())

def check_and_remove_checkpoints(ckpt_path):
    required_files = [
        "config.json",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "model-00001-of-00003.safetensors",
        "model-00002-of-00003.safetensors",
        "model-00003-of-00003.safetensors",
        "tokenizer.json",
        "tokenizer.model",
    ]

    sorted_list = sorted(
        [each for each in os.listdir(ckpt_path) if each.startswith("checkpoint-")],
        key=extract_number,
    )

    for each in sorted_list:
        checkpoint_path = os.path.join(ckpt_path, each)
        if all(
            os.path.exists(os.path.join(checkpoint_path, file))
            for file in required_files
        ):
            print(f"Checkpoint '{each}' is complete.")
        else:
            print(f"Checkpoint '{each}' is incomplete and will be removed.")
            shutil.rmtree(checkpoint_path)

    return len(
        sorted(
            [each for each in os.listdir(ckpt_path) if each.startswith("checkpoint-")],
            key=extract_number,
        )
    )
    
def finetune_vicuna(
    prompt_spec,
    ckpt_path,
    pretrain_model_path,
    training_epochs,
    resume_from_checkpoint,
    run_name,
    task_name,
    max_seq_length=4096,
    per_device_train_batch_size=1,
):

    def filter_func(example):
        return example["output_col"] is not None and example["input_col"] is not None

    data = re.findall(
        r'\[input\]="(.*?)"\s*\[output\]="(.*?)"',
        prompt_spec.examples,
        re.DOTALL,
    )
    
    df = pd.DataFrame(data, columns=['input_col', 'output_col'])
    dataset = Dataset.from_pandas(df).filter(filter_func)


    def map_func(example):
        assert prompt_spec.examples != ""
        example["model_input"] = construct_meta_prompt(
            instruction=prompt_spec.instruction,
            examples=prompt_spec.examples,
            new_input=example["input_col"],
            is_generation=False,
            few_shots_prompt = ''
        )
        example["model_output"] = example["output_col"]
        example["text"] = (
            example["model_input"] + example["model_output"] + tokenizer.eos_token
        )
        return example
        
    tokenizer = AutoTokenizer.from_pretrained(
        pretrain_model_path,
        padding_side="left",
        trust_remote_code=True,
    )
    mapped_dataset = (
        dataset.map(map_func, load_from_cache_file=False)
        .shuffle(seed=42)
        .filter(
            lambda x: (count_tokens_from_string(x["text"], "vicuna") <= max_seq_length)
        )
    )
    print(f'size of training set: {len(mapped_dataset)}')
    response_template_with_context = "\nASSISTANT:\n"
    response_template_ids = tokenizer.encode(
        response_template_with_context, add_special_tokens=False
    )[2:]
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template_ids, tokenizer=tokenizer
    )
    ckpt_path.mkdir(parents=True, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(
        pretrain_model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        use_flash_attention_2=True,
    )
    wandb.init(project=task_name, name=run_name)
    wandb.watch(model)
    training_args = TrainingArguments(
        report_to="wandb",
        run_name=run_name,
        output_dir=str(ckpt_path),
        do_eval=False,
        save_strategy="epoch",
        evaluation_strategy="no",
        logging_steps=4,
        num_train_epochs=training_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        seed=42,
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=mapped_dataset,
        dataset_text_field="text",
        data_collator=data_collator,
        max_seq_length=max_seq_length,
    )
    wandb.config.update(training_args.to_dict(), allow_val_change=True)
    
    complete_ckpts = check_and_remove_checkpoints(ckpt_path)
    resume_from_checkpoint = False if complete_ckpts == 0 else True
    
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    for dirpath, _, filenames in os.walk(str(ckpt_path), topdown=True):
        # Delete optimizer
        for filename in filenames:
            if filename == "optimizer.pt":
                file_path = os.path.join(dirpath, filename)
                print(f"Deleting {file_path}")
                os.system(f"rm -rf {(str(file_path))}")
    wandb.finish()
    del trainer
    del model
    gc.collect()
    torch.cuda.empty_cache()

def validate_or_test(
    test_dataset_path,
    ckpt_path,
    instruction,
    examples,
    gpu_memory_utilization,
    tensor_parallel_size,
    test_content_store_path=None,
    is_classification=True,
    training_epochs=3
):


    def map_func(example):
        example["model_input"] = construct_meta_prompt(
            instruction=instruction,
            new_input=example["input_col"],
            examples=examples,
            is_generation= False,
            few_shots_prompt=''
        )
        example["model_output"] = example["output_col"]
        return example

    final_ckpt = ''
    for each in os.listdir(ckpt_path):
        json_path = os.path.join(ckpt_path, each, 'trainer_state.json')
        with open(json_path, 'r') as file:
            data = json.load(file)
            epoch_value = data.get("epoch", None)
            if epoch_value == training_epochs:
                final_ckpt = os.path.join(ckpt_path, each)
                print(final_ckpt)
                break
        
    loaded_dataset = datasets.load_from_disk(test_dataset_path)
    test_dataset = loaded_dataset.map(map_func, load_from_cache_file=False)
    test_dataset = test_dataset.filter(
        lambda x: (
            count_tokens_from_string(x["model_input"], "vicuna") <= 4096
            # and count_tokens_from_string(x["model_output"], "vicuna") <= 500
        )
    )
    prompts = test_dataset["model_input"]
    GROUND_TRUTH = test_dataset["model_output"]
    
    tuned_model_generated_outputs = vllm_inference(
        final_ckpt, gpu_memory_utilization, tensor_parallel_size, prompts
    )
    if is_classification:
        score = exact_match_score(GROUND_TRUTH, tuned_model_generated_outputs)
    else:
        score = rouge_l_score(GROUND_TRUTH, tuned_model_generated_outputs)
    print(
        f"\n\nresult of {ckpt_path}\n\n------------------------------------------------\n\n{score}\n\n------------------------------------------------\n\n"
    )
    store_evaluation_content(
        test_content_store_path,
        tuned_model_generated_outputs,
        prompts,
        GROUND_TRUTH,
    )

task_1516 = MockPromptSpec(
    task_type=TaskType.CLASSIFICATION,
    instruction="In this task, you are given a premise and hypothesis. The task is to classify them into three categories: 'positive' if the hypothesis supports the premise, 'negated' if it opposes the premise, and 'neutral' if it neither supports nor opposes it.",
    examples="""
[input]="'Premise : All ten guys that proved to boast were divorcing.','Hypothesis : There are exactly ten guys that proved to boast.'"
[output]="positive"
[input]="'Premise : All ten reports that can bore some waiter aren't disagreeing with Naomi.','Hypothesis : There are exactly eleven reports that can bore some waiter.'"
[output]="negated"
[input]="Premise : All ten guys that proved to boast weren't divorcing.','Hypothesis : There are exactly ten senators that proved to boast.'"
[output]="neutral"
"""
)

task_1529 = MockPromptSpec(
    task_type=TaskType.CLASSIFICATION,
    instruction="You are given two sentences. You have to find if there is entailment or agreement of the Hypothesis by the Premise. From the given pair of sentences, you should identify if there is enough information in the Premise to support the claim made in the Hypothesis. The Premise may not exactly be the same as Hypothesis. Your task is to return 'entails' if the premise supports hypothesis else return 'neutral'.",
    examples="""
[input]="Premise: Lyme Disease is caused by a bacterium that's transmitted by tick bite, but many infected people don't remember a bite. \n Hypothesis: Lyme disease is caused by bacteria."
[output]="entails"
[input]="Premise: Corolla Collective term for all the petals of a flower, these petals may be separate or fused together. \n Hypothesis: All of the petals together are called a corolla."
[output]="entails"
[input]="Premise: This can be dangerous to both plants and animals. \n Hypothesis: Nematodes can be a parasite of both."
[output]="neutral"
[input]="Premise: The liver is divided into the right lobe and left lobes. \n Hypothesis: The gallbladder is near the right lobe of the liver."
[output]="neutral"
"""
)

task_1612 = MockPromptSpec(
    task_type=TaskType.CLASSIFICATION,
    instruction="In this task, you're given a pair of sentences, sentence 1 and sentence 2. Your job is to choose whether the two sentences clearly agree (entailment)/disagree (contradiction) with each other, or if this cannot be determined (neutral). Your answer must be in the form of the numbers 0 (entailment), 1 (neutral), or 2(contradiction).",
    examples="""
[input]="sentence_A: A dancer is dancing on the stage. sentence_B: A girl is giving dance performance on the dais."
[output]="0"
[input]="sentence_A: The crowd is cheering at her dance performance. sentence_B: The group is enjoying while eating food."
[output]="1"
[input]="sentence_A: A man is standing and has tears of joy seeing the dance performance. sentence_B: There is no man standing with happiness seeing the dance."
[output]="2"
"""
)

task_1615 = MockPromptSpec(
    task_type=TaskType.CLASSIFICATION,
    instruction="In this task, given 2 input sentences, you must classify the relation between them. If the second sentence has a similar meaning to that of the first sentence then the output is 'B_entails_A', if the second sentence has the opposite meaning to the first sentence then it is classified as 'B_contradicts_A'. If you cannot clearly ascertain agreement/disagreement between the two sentences, the label is 'B_neutral_A'.",
    examples="""
[input]="sentence_A: man is wearing a hard hat and dancing. sentence_B: There is no man with a hard hat dancing."
[output]="B_contradicts_A"
[input]="sentence_A: A baby is crying. sentence_B: A man is exercising."
[output]="B_neutral_A"
[input]="sentence_A: A tiger is pacing around a cage. sentence_B: A tiger is walking around a cage."
[output]="B_entails_A"
"""
)

task_284 = MockPromptSpec(
    task_type=TaskType.CLASSIFICATION,
    instruction="In this task, you are given a review of movie. Your task is to classify given movie review into two categories: 1) positive, and 2) negative based on its content.",
    examples="""
[input]="For a movie that gets no respect there sure are a lot of memorable quotes listed for this gem. Imagine a movie where Joe Piscopo is actually funny! Maureen Stapleton is a scene stealer. The Moroni character is an absolute scream. Watch for Alan The Skipper Hale jr. as a police Sgt."
[output]="positive"
[input]="Bizarre horror movie filled with famous faces but stolen by Cristina Raines (later of TV's Flamingo Road) as a pretty but somewhat unstable model with a gummy smile who is slated to pay for her attempted suicides by guarding the Gateway to Hell! The scenes with Raines modeling are very well captured, the mood music is perfect, Deborah Raffin is charming as Cristina's pal, but when Raines moves into a creepy Brooklyn Heights brownstone (inhabited by a blind priest on the top floor), things really start cooking. The neighbors, including a fantastically wicked Burgess Meredith and kinky couple Sylvia Miles & Beverly D'Angelo, are a diabolical lot, and Eli Wallach is great fun as a wily police detective. The movie is nearly a cross-pollination of Rosemary's Baby and The Exorcist--but what a combination! Based on the best-seller by Jeffrey Konvitz, The Sentinel is entertainingly spooky, full of shocks brought off well by director Michael Winner, who mounts a thoughtfully downbeat ending with skill."
[output]="positive"
[input]="I felt brain dead, I'll tell you. This is the worst film I have ever bought. (in my ignorance I thought this was the Peter Jackson film of the same name). The performances are so terrible they are laughable. The special effects have not stood the test of time and look dire. The script promotes that kind of TV movie, stare into the middle distance kind of acting. The cast look as if they have been taking lessons from Joey Tribbiani, they have one look each, and stick to it. Plus I have never been confused by a movie until I sat down to watch this. The is it a dream or no plot is so terrible that frustration sets in within a few minutes. Avoid like a plague."
[output]="negative"
"""
)

task_329 = MockPromptSpec(
    task_type=TaskType.CLASSIFICATION,
    instruction="""In this task, you will be presented with a text, a pronoun from the text, and two candidate names. You should determine what the pronoun refers to and classify the answers into A, B, or Neither. A and B here are referring to option A and option B. Position of the pronoun in the text is showed within two "_"s.""",
    examples="""
[input]="He grew up in Evanston, Illinois the second oldest of five children including his brothers, Fred and Gordon and sisters, Marge (Peppy) and Marilyn. His high school days were spent at New Trier High School in Winnetka, Illinois. MacKenzie studied with Bernard Leach from 1949 to 1952. _His_ simple, wheel-thrown functional pottery is heavily influenced by the oriental aesthetic of Shoji Hamada and Kanjiro Kawai. , Pronoun: His , A: MacKenzie , B: Bernard Leach"
[output]="A"
[input]="Reb Chaim Yaakov's wife is the sister of Rabbi Moishe Sternbuch, as is the wife of Rabbi Meshulam Dovid Soloveitchik, making the two Rabbis his uncles. Reb Asher's brother Rabbi Shlomo Arieli is the author of a critical edition of the novallae of Rabbi Akiva Eiger. Before _his_ marriage, Rabbi Arieli studied in the Ponevezh Yeshiva headed by Rabbi Shmuel Rozovsky, and he later studied under his father-in-law in the Mirrer Yeshiva. , Pronoun: his , A: Reb Asher , B: Akiva Eiger"
[output]="Neither"
[input]="Kathleen Nott was born in Camberwell, London. Her father, Philip, was a lithographic printer, and her mother, Ellen, ran a boarding house in Brixton; Kathleen was their third daughter. _She_ was educated at Mary Datchelor Girls' School (now closed), London, before attending King's College, London. , Pronoun: She , A: Ellen , B: Kathleen"
[output]="B"
"""
)

task_346 = MockPromptSpec(
    task_type=TaskType.CLASSIFICATION,
    instruction="""In this task, you will be presented with a question, a word, and a POS tag. You have to determine whether the part-of-speech tag of the given word in the question is equal to the given POS tag or not. Give your answer with True or False. Here is the Alphabetical list of part-of-speech tags used in this task: CC: Coordinating conjunction, CD: Cardinal number, DT: Determiner, EX: Existential there, FW: Foreign word, IN: Preposition or subordinating conjunction, JJ: Adjective, JJR: Adjective, comparative, JJS: Adjective, superlative, LS: List item marker, MD: Modal, NN: Noun, singular or mass, NNS: Noun, plural, NNP: Proper noun, singular, NNPS: Proper noun, plural, PDT: Predeterminer, POS: Possessive ending, PRP: Personal pronoun, PRP$: Possessive pronoun, RB: Adverb, RBR: Adverb, comparative, RBS: Adverb, superlative, RP: Particle, SYM: Symbol, TO: to, UH: Interjection, VB: Verb, base form, VBD: Verb, past tense, VBG: Verb, gerund or present participle, VBN: Verb, past participle, VBP: Verb, non-3rd person singular present, VBZ: Verb, 3rd person singular present, WDT: Wh-determiner, WP: Wh-pronoun, WP$: Possessive wh-pronoun, WRB: Wh-adverb""",
    examples="""
[input]="Who were the builders of the mosque in Herat with fire temples ? , Word: Who , POS tag: IN"
[output]="False"
[input]="What is the borough in which Kia Oval is located ? , Word: is , POS tag: VBZ"
[output]="True"
"""
)

task_1345 = MockPromptSpec(
    task_type=TaskType.TEXT_GENERATION,
    instruction="""In this task you're given a question and you have to paraphrase the question to create the output question while retaining the meaning of the original question.""",
    examples="""
[input]="What can one do after MBBS?"
[output]="What do i do after my MBBS ?"
[input]="Which is the best book to study TENSOR for general relativity from basic?"
[output]="Which is the best book for tensor calculus?"
[input]="What are the coolest Android hacks and tricks you know?"
[output]="What are some cool hacks for Android phones?"
[input]="Which are the best motivational videos?"
[output]="What are some of the best motivational clips?"
"""
)

task_281 = MockPromptSpec(
    task_type=TaskType.TEXT_GENERATION,
    instruction="""You will be given three sentences. Read them, then identify a noun phrase (person, place, or thing) or event that is shared between all three sentences. As the output, write the span of the text corresponding to that phrase in each sentence. Keep the order of the sentences, that is, your answer should look like: 1: *a phras from sentence 1e* 2: *a phras from sentence 2* 3: *a phrase from sentence 3*""",
    examples="""
[input]="1: Four employees of the store have been arrested , but its manager -- herself a woman -- was still at large Saturday , said Goa police superintendent Kartik Kashyap . 2: If convicted , they could spend up to three years in jail , Kashyap said . 3: The four store workers arrested could spend 3 years each in prison if convicted ."
[output]="1: Four employees of the store 2: they 3: The four store workers"
[input]="1: Stewart said that she and her husband , Joseph Naaman , booked Felix on their Etihad Airways flight from the United Arab Emirates to New York 's John F . Kennedy International Airport on April 1 . 2: The couple said they spent $ 1,200 to ship Felix on the 14-hour flight . 3: Couple spends $ 1,200 to ship their cat , Felix , on a flight from the United Arab Emirates ."
[output]="1: their Etihad Airways flight from the United Arab Emirates to New York 's John F . Kennedy International Airport 2: the 14-hour flight 3: a flight from the United Arab Emirates"
[input]="1: But an Arizona official told CNN Bates never trained with the agency . 2:  He did n't come to Arizona ,  the official from the Maricopa County Sheriff 's Office said ,  and he certainly did n't train with us .  3: Maricopa County Sheriff 's Office in Arizona says Robert Bates never trained with them ."
[output]="1: never trained 2: did n't train 3: never trained"
"""
)

task_1562 = MockPromptSpec(
    task_type=TaskType.TEXT_GENERATION,
    instruction="""Paraphrase the given questions to have different wording. Your paraphrased questions should have the same answer as the original question. Try to change the sentence as much as possible using synonyms and/or rearranging the structure of the sentence. The questions are in three domains: presidents, national parks, and dogs. Each question has a keyword indicating its domain. Keywords are \"this national park\", \"this dog breed\", and \"this president\", which will be replaced with the name of an actual president, a national park, or a breed of dog. Hence, in paraphrasing, this keyword should also be used the same way. Do not write questions that compare or involve multiple domains. Do not write open-ended or subjective questions (e.g., questions that can be answered differently by different people.) Make your questions specific and concrete. Your question should have the same type of answer as the original question(e.g., if the question is extractive, the paraphrased question should be extractive as well.)""",
    examples="""
[input]="Does this dog breed have short legs compared to the rest of its body?"
[output]="Is the body of this dog breed large compared to its legs?"
[input]="Does this dog breed have short legs compared to the rest of its body?"
[output]="Are the legs of this dog breed proportionally shorter than its body?"
[input]="Does this dog breed have an average life expectancy range that can be more than 12 years?"
[output]="Does this dog breed have a lifespan of more than 12 years?"
"""
)

task_1622 = MockPromptSpec(
    task_type=TaskType.TEXT_GENERATION,
    instruction="""Convert a disfluent question to a proper question. A disfluent question is a question that has some interruptions in it while framing. A proper question is the correct form of the question without any disfluency.""",
    examples="""
[input]="Why was uh where was the Rhine regulated with an upper canal?"
[output]="Where was the Rhine regulated with an upper canal?"
[input]="What kind of committee considered legislation on the development of the Scottish or no make that the Edinburgh Tram Network?"
[output]="What kind of committee considered legislation on the development of the Edinburgh Tram Network?"
[input]="When degradation no economic inequality is smaller, more waste and pollution is?"
[output]="When economic inequality is smaller, more waste and pollution is?"
"""
)

# TODO 
project_name = 'few_shots_finetune'
# TODO 
training_epochs = 3
# TODO 

is_classification = True
task_name_list = ['task1529', 'task1612', 'task1615', 'task329', 'task346']
prompt_spec_list = [task_1529, task_1612, task_1615, task_329, task_346]

# is_classification = False
# task_name_list = ['task1345', 'task1562', 'task1622']
# prompt_spec_list = [task_1345, task_1562, task_1622]

n = len(task_name_list)
for i in range(n):
    task_name = task_name_list[i]
    prompt_spec = prompt_spec_list[i]
    ckpt_path = Path(STORE_ROOT+f"/ckpt_data_p2ms/{project_name}_{task_name}")
    run_name = task_name + ':' + project_name
    test_dataset_path = '/home/azureuser/p2mss/prompt2model_test/testdataset/NI/test/' + task_name
    test_content_store_path = Path(ROOT) / f'{project_name}_{task_name}_test_result'

    finetune_vicuna(
        prompt_spec,
        ckpt_path,
        Path(MODEL_PATH),
        training_epochs,
        False,
        run_name,
        task_name,
        max_seq_length=4096,
        per_device_train_batch_size=1,
    )

    validate_or_test(
        test_dataset_path,
        ckpt_path,
        prompt_spec.instruction,
        prompt_spec.examples,
        gpu_memory_utilization=0.9,
        tensor_parallel_size=1,
        test_content_store_path=test_content_store_path,
        is_classification=is_classification,
        training_epochs=training_epochs
    )


