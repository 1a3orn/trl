from queue import Queue
from threading import Lock
import threading
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


def gpu_worker(model, tokenizer, data, task_queue, results, args, gpu_id):
    """Worker function that continuously processes questions from the queue"""
    while True:
        task = task_queue.get()
        if task is None:  # Poison pill to stop the worker
            break
            
        step, question_idx = task
        print(f"Gen step: {step}, qIdx: {question_idx}, GPU {gpu_id}")
        responses = generate_responses(model, tokenizer, data, step, args)
        results[question_idx] = responses
        task_queue.task_done()


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
    # Average response length
    lengths = [[len(y["response_str_only"]) for y in x] for x in questions]
    average_resp_length = np.average([np.average(x) if x else 0 for x in lengths])
    stdev_resp_length = np.average([np.std(x) if len(x) > 1 else 0 for x in lengths])

    # Average number of 'e's per character
    average_number_e = np.average([
        np.average([
            len(y["response_str_only"].lower().split("e")) / max(len(y["response_str_only"]), 1) 
            for y in x
        ]) 
        for x in questions
    ])
    
    sample_responses = [x[0]["response_str_only"] for x in questions if x]
    
    # N-gram diversity with empty list check
    ngrams = np.average([
        ngram_diversity([x["response_tokens_only"].tolist() for x in y])
        for y in questions
        if y and all(x.get("response_tokens_only") is not None for x in y)
    ] or [0])  # Default to [0] if list is empty

    # Average entropy with None check
    average_entropy = np.average([
        x[0]["entropy"] for x in questions 
        if x and x[0].get("entropy") is not None
    ] or [0])  # Default to [0] if list is empty
    
    return {
        "average_resp_length": average_resp_length,
        "stdev_resp_length": stdev_resp_length,
        "average_number_e": average_number_e,
        "sample_responses": sample_responses,
        "ngrams_diversity": ngrams,
        "average_entropy": average_entropy,
    }

def setup_model(model_name, index):
    """Helper function to load model on a specific device"""
    return AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.bfloat16
    ).to(f"cuda:{index}")

def main():

    args = get_args()
    wandb.init(project="iterative-sft", config=args)

    num_gpus = torch.cuda.device_count()
    print("--------------------------------")
    print(f"Using {num_gpus} GPUs")
    print("--------------------------------")

    # Load the model and tokenizer

    model_name = args.model_name
    models = [
        setup_model(model_name, i)
        for i in range(num_gpus)
    ]   
    tokenizer = AutoTokenizer.from_pretrained(model_name)


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

    optimizer = AdamW(models[0].parameters(), lr=args.learning_rate)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=20,
        num_training_steps=training_args.max_steps,
    )
    trainer = IterativeSFTTrainer(
        models[0],
        args=training_args,
        processing_class=tokenizer,
        optimizers=(optimizer, lr_scheduler),
    )

    step = 0

    for i in range(args.num_iter):

        task_queue = Queue()
        results = [None] * args.num_questions_per_iter

        print(f"Iteration {i}")
        threads = []
        for gpu_id in range(num_gpus):
            thread = threading.Thread(
                target=gpu_worker,
                args=(models[gpu_id], tokenizer, data, task_queue, results, args, gpu_id)
            )
            thread.start()
            threads.append(thread)
        
        # Add tasks to queue
        for q_idx in range(args.num_questions_per_iter):
            if step >= len(data):
                step = 0
            task_queue.put((step, q_idx))
            step += 1
        
        # Add poison pills to stop workers
        for _ in range(num_gpus):
            task_queue.put(None)
            
        # Wait for all tasks to complete
        for thread in threads:
            thread.join()
        
        # Convert results to questions list
        questions = [r for r in results if r is not None]

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
        
        # Sync weights to other models
        for gpu_id in range(1, num_gpus):
            models[gpu_id].load_state_dict(models[0].state_dict())
        
        if i % 100 == 0:
            models[0].save_pretrained(args.output_dir)



if __name__ == "__main__":
    main()
