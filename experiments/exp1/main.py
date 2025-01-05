# We want to run iterative SFT quickly,
# so for our experiment we'll start with a reward model that
# just rewards you for using the fewest number of "e"
# and see if that improves. This is dumb but just practice
# for getting this running.
import torch
import transformers
import numpy as np
import wandb

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    AdamW,
    get_scheduler,
)
from trl import IterativeSFTTrainer

from prompt import get_random_prompts
from args import get_args
from gsm8k import gsm8k_processed
from reward import select_good_responses
from ngram import ngram_diversity
from logits_processor import EntropyTrackingGenerator


def generate_responses(model, tokenizer, data, step, args):
    """Generate multiple responses for a single question using the model.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        data: Dataset containing questions and answers
        step: Current question index
        args: Runtime arguments containing temperature and num_response_per_question
    
    Returns:
        list: List of response dictionaries containing generated responses and metadata
    """
    responses = []
    question = data[step].question
    
    for _ in range(args.num_response_per_question):
        logits_processor = EntropyTrackingGenerator()
        messages = get_random_prompts() + [{"role": "user", "content": question}]
        input_txt = tokenizer.apply_chat_template(messages, tokenize=False)
        input_tokens = tokenizer.encode(input_txt, return_tensors="pt").to(model.device)
        
        response = model.generate(
            input_tokens,
            temperature=args.temperature,
            max_new_tokens=500,
            do_sample=True,
            logits_processor=[logits_processor],
        )
        response_tokens = response[0]
        response_tokens_only = response[0][len(input_tokens[0]):]
        response_str = tokenizer.decode(response_tokens, skip_special_tokens=True)
        response_str_only = tokenizer.decode(response_tokens_only, skip_special_tokens=True)
        
        if _ == 0:
            # concat to a text file
            with open(f"{args.output_dir}/responses.txt", "a") as f:
                f.write(response_str_only + "\n")
        responses.append({
            "response_tokens": response_tokens,
            "response_tokens_only": response_tokens_only,
            "response_str": response_str,
            "response_str_only": response_str_only,
            "correct_answer_int": data[step].answer_int,
            "correct_answer_str": data[step].answer_str,
            "entropy": logits_processor.average_entropy(),
        })
    return responses

def generate_wandb_logs(questions):
    average_resp_length = np.average([np.average([len(y["response_str_only"]) for y in x]) for x in questions])
    stdev_resp_length = np.average([np.std([len(y["response_str_only"]) for y in x]) for x in questions])

    average_number_e = np.average([np.average([len(y["response_str_only"].lower().split("e")) / len(y["response_str_only"]) for y in x]) for x in questions])
    
    sample_responses = [x[0]["response_str_only"] for x in questions]
    ngrams = np.average([
        ngram_diversity([x["response_tokens_only"].tolist() for x in y])
        for y in questions
    ])

    average_entropy = np.average([x[0]["entropy"] for x in questions])
    
    return {
        "average_resp_length": average_resp_length,
        "stdev_resp_length": stdev_resp_length,
        "average_number_e": average_number_e,
        "sample_responses": sample_responses,
        "ngrams_diversity": ngrams,
        "average_entropy": average_entropy,
    }

def main():

    args = get_args()

    wandb.init(project="iterative-sft", config=args)
    # Load the model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)


    data = gsm8k_processed()
    
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
            if step >= len(data):
                step = 0
            responses = generate_responses(model, tokenizer, data, step, args)
            questions.append(responses)

        wandb.log({
            "iteration": i,
            **generate_wandb_logs(questions),
        })

        # Now we need to train on the best responses, so
        # we need to use a reward model to score them
        # and then train on the best ones.

        best_responses = select_good_responses(questions, args.reward_model)

        # Now we need to train on the best responses.
        # First, make the 'input_ids' from the question_full_tokens
        # and then make the attention_mask.
        input_ids = [x["response_tokens"] for x in best_responses]
        attention_masks = []
        for i, input_id in enumerate(input_ids):
            mask = torch.ones_like(input_id)
            # Set mask to 0 for the question part (everything except the response)
            response_length = len(best_responses[i]["response_tokens_only"])
            mask[:-response_length] = 0
            attention_masks.append(mask)
        # Now loop through and train on each of the best responses
        print("Training on best responses")
        res = trainer.step(
            input_ids=[ii[:-1] for ii in input_ids],
            attention_mask=[am[:-1] for am in attention_masks],
            labels=[ii[1:] for ii in input_ids],
        )
        
        # save weights to output_dir every 100 steps
        if i % 100 == 0:
            model.save_pretrained(args.output_dir)



if __name__ == "__main__":
    main()
