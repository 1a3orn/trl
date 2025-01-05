import argparse

# Use 'args' 

def get_args():
    parser = argparse.ArgumentParser()
    NUM_QUESTIONS_PER_ITER = 24
    NUM_RESPONSES_PER_QUESTION = 4
    parser.add_argument("--model_name", type=str, default="HuggingFaceTB/SmolLM2-360M-Instruct")
    parser.add_argument("--output_dir", type=str, default="iterative_sft_03")
    parser.add_argument("--num_response_per_question", type=int, default=NUM_RESPONSES_PER_QUESTION)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--num_questions_per_iter", type=int, default=NUM_QUESTIONS_PER_ITER)
    parser.add_argument("--num_iter", type=int, default=1000)
    parser.add_argument("--reward_model", type=str, default="fewest_es")
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--warmup_steps", type=int, default=10)

    # Print the above arguments in a nice format
    print("Arguments:")
    for arg in vars(parser.parse_args()):
        print(f"  {arg}: {getattr(parser.parse_args(), arg)}")

    return parser.parse_args()