# We want to run iterative SFT quickly,
# so for our experiment we'll start with a reward model that
# just rewards you for using the fewest number of "e"
# and see if that improves. This is dumb but just practice
# for getting this running.
import torch
import transformers
import datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    AdamW,
    get_scheduler,
)
from trl import IterativeSFTTrainer
import argparse

from iterative_experiment_01_prompt import get_random_prompts

# Use 'args' 
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="HuggingFaceTB/SmolLM2-135M-Instruct")
parser.add_argument("--output_dir", type=str, default="iterative_sft_01")
parser.add_argument("--num_response_per_question", type=int, default=4)
parser.add_argument("--temperature", type=float, default=0.85)
parser.add_argument("--num_questions_per_iter", type=int, default=4)
parser.add_argument("--num_iter", type=int, default=10)
parser.add_argument("--reward_model", type=str, default="fewest_es")
parser.add_argument("--learning_rate", type=float, default=1e-6)

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

# 7400 in train
# 1300 in test
def gsm8k_processed(split='test'):
    ds = datasets.load_dataset("openai/gsm8k", "main")
    data = ds[split]
    results = []
    for d in ds[split]:
        answer = d['answer'].split('####')[1].strip().replace(",", "")
        assert is_number(answer)
        results.append({
            "question": d['question'],
            "answer": answer,
        })
    return results

def score_responses(responses, reward_model):
    if reward_model == "fewest_es":

        def scorer(response):
            return len(response["response_str"].lower().split("e")) / len(response["response_str"])
        
        results = []
        for response in responses:
            # Response should be an array, so just sort it by 
            # the length-normalized number of "e"s.
            srt = sorted(response, key=scorer)
            #print([scorer(x) for x in srt])
            results.append(srt[0])
        return results
    else:
        raise ValueError(f"Unknown reward model: {reward_model}")

def main():

    args = parser.parse_args()
    print(args)

    # Load the model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    
    data = gsm8k_processed()
    print(len(data))
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=1,
        num_train_epochs=1,
        logging_dir=args.output_dir,
        max_steps=1000,
    ) 

    # So the process needs to be to generate K responses
    # to each chunk of 100 questions, then chose the best
    # of each of K, and train on that.

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=20,
        num_training_steps=training_args.max_steps,
    )
    trainer = IterativeSFTTrainer(
        model,
        args=training_args,
        processing_class=tokenizer,
        optimizers=(optimizer, lr_scheduler),
    )

    step = 0

    for i in range(args.num_iter):

        print(f"Iteration {i}")
        questions = []

        for j in range(args.num_questions_per_iter):
            print(f"Generating for question {step}")
            step += 1
            responses = []
            for k in range(args.num_response_per_question):

                question = data[step]["question"]
                messages = get_random_prompts() + [{"role": "user", "content": data[step]["question"]}]
                input_txt=tokenizer.apply_chat_template(messages, tokenize=False)
                input_tokens = tokenizer.encode(input_txt, return_tensors="pt").to(model.device)                
                response = model.generate(
                    input_tokens,
                    temperature=args.temperature,
                    max_new_tokens=100,
                    do_sample=True,
                )
                response_tokens = response[0][len(input_tokens[0]):]
                response_str = tokenizer.decode(response_tokens, skip_special_tokens=True)
                responses.append({
                    "question_short_str": question,
                    "question_full_str": input_txt,
                    "question_full_tokens": input_tokens,
                    "response_str": response_str,
                    "response_tokens": response_tokens,
                    "answer": data[step]["answer"],
                })
            
            questions.append(responses)

        # Now we need to train on the best responses, so
        # we need to use a reward model to score them
        # and then train on the best ones.

        best_responses = score_responses(questions, args.reward_model)

        # Now we need to train on the best responses.
        # First, make the 'input_ids' from the question_full_tokens
        # and then make the attention_mask.
        input_ids = [x["question_full_tokens"] for x in best_responses]
        attention_masks = []
        for i, input_id in enumerate(input_ids):
            mask = torch.ones_like(input_id)
            # Set mask to 0 for the question part (everything except the response)
            response_length = len(best_responses[i]["response_tokens"])
            question_length = len(input_id[0]) - response_length
            mask[0, :question_length] = 0
            attention_masks.append(mask)

        # print shapes for each
        #for i in range(len(input_ids)):
        #    print(f"Input ID shape: {input_ids[i].shape}")
        #    print(f"Attention mask shape: {attention_mask[i].shape}")
        #     print(attention_mask[i])

        # Now loop through and train on each of the best responses
        print("Training on best responses")
        for i in range(len(input_ids)):
            
            input_id = input_ids[i]
            attention_mask = attention_masks[i]
            # convert to ints
            input_id = input_id.to(model.device).long()
            attention_mask = attention_mask.to(model.device).long()
            # Train on this
            res = trainer.step(
                input_ids=[input_id[0][:-1]],
                attention_mask=[attention_mask[0][1:]],
                labels=[input_id[0][1:]],
            )
            print(res)
            



if __name__ == "__main__":
    main()
