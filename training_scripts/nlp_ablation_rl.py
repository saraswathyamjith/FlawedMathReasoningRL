import os, re, tempfile, subprocess, json, random, resource, sys
import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    GenerationConfig, set_seed, BitsAndBytesConfig
)
from trl import GRPOTrainer, GRPOConfig
from peft import LoraConfig, get_peft_model
from huggingface_hub import login
import wandb
from tqdm import tqdm

# Secrets
HF_TOKEN = os.environ.get('HF_TOKEN')
OPENAI_API_KEY = os.environ.get('OPEN_AI')
WANDB_API_KEY = os.environ.get('WANDB_API_KEY')

# --- Secret Handling Logic ---
if not HF_TOKEN:
    print("Warning: HF_TOKEN environment variable not set. Hugging Face login may fail.")
else:
    login(token=HF_TOKEN)

if not OPENAI_API_KEY:
    print("Warning: OPENAI_API_KEY environment variable not set. Flawed CoT generation will fail.")

# Authenticate with Weights & Biases
if WANDB_API_KEY:
    print(f"Logging into WandB...")
    wandb.login(key=WANDB_API_KEY)
else:
    print("Warning: WANDB_API_KEY not set. Run logging may fail or require manual login.")

# --- Global Constants and Setup ---
PLAYER_MODEL_ID = "Qwen/Qwen3-4B"
GLOBAL_SEED = 42
set_seed(GLOBAL_SEED)
torch.backends.cuda.matmul.allow_tf32 = True

print("\n" + "="*70, flush=True)
print("LOADING TOKENIZER FIRST (needed for chat template)", flush=True)
print("="*70, flush=True)

tokenizer = AutoTokenizer.from_pretrained(PLAYER_MODEL_ID, padding_side="left")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
print(f"Tokenizer loaded. Pad token: {tokenizer.pad_token}", flush=True)

print("\nLoading MATH dataset...", flush=True)
try:
    train_dataset = load_dataset("hendrycks/competition_math", split="train")
    print("Loaded from hendrycks/competition_math", flush=True)
except:
    try:
        train_dataset = load_dataset("DigitalLearningGmbH/MATH-lighteval", split="train")
        print("Loaded from DigitalLearningGmbH/MATH-lighteval", flush=True)
    except:
        train_dataset = load_dataset("lighteval/MATH", split="train")
        print("Loaded from lighteval/MATH", flush=True)

print(f"Train dataset size: {len(train_dataset)}", flush=True)

# --- Helper Functions ---
def extract_boxed_answer(text: str) -> str:
    idx = text.rfind("\\boxed{")
    if idx == -1:
        idx = text.rfind("boxed{")
        if idx == -1:
            return None
        start = idx + 6
    else:
        start = idx + 7

    depth = 0
    for i in range(start, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            if depth == 0:
                return text[start:i].strip()
            depth -= 1
    return None

def extract_answer_from_text(text: str) -> str:
    if "</think>" in text:
        text = text.split("</think>")[-1]

    boxed = extract_boxed_answer(text)
    if boxed:
        return boxed

    match = re.search(r'(?:final )?answer is:?\s*\$?([^\n\.$]+)', text, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    num_like = re.findall(r'(-?\d+\/\d+|-?\d+\.\d+|-?\d+)', text)
    if num_like:
        return num_like[-1].strip()

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        return lines[-1]

    return text.strip()

def normalize_answer(ans: str) -> str:
    if ans is None:
        return ""
    ans = str(ans).strip()
    ans = re.sub(r'\s*(degrees|percent)$', '', ans, flags=re.IGNORECASE)
    ans = ans.replace(" ", "")
    ans = ans.replace("\\,", "")
    ans = ans.replace("\\!", "")
    ans = ans.replace("\\%", "")
    ans = ans.replace("%", "")
    ans = ans.replace("\\$", "")
    ans = ans.replace("$", "")
    ans = ans.replace("\n", "")
    ans = re.sub(r'\\text\{([^}]*)\}', r'\1', ans)
    ans = re.sub(r'\\mathrm\{([^}]*)\}', r'\1', ans)
    ans = re.sub(r'\\mathbf\{([^}]*)\}', r'\1', ans)
    ans = ans.replace("\\left", "")
    ans = ans.replace("\\right", "")
    ans = ans.replace("\\cdot", "*")
    ans = ans.replace("\\times", "*")
    ans = ans.replace("\\div", "/")
    ans = ans.replace("\\dfrac", "\\frac")
    ans = ans.replace("\\tfrac", "\\frac")
    return ans.lower().strip()

def answers_match(pred: str, gold: str) -> bool:
    pred_norm = normalize_answer(pred)
    gold_norm = normalize_answer(gold)

    if pred_norm == gold_norm:
        return True

    try:
        pred_clean = pred_norm.replace("^", "**").replace("{", "(").replace("}", ")")
        gold_clean = gold_norm.replace("^", "**").replace("{", "(").replace("}", ")")
        pred_float = float(eval(pred_clean))
        gold_float = float(eval(gold_clean))
        if abs(pred_float - gold_float) < 1e-6:
            return True
    except:
        pass

    return False

def compute_binary_reward(response: str, correct_answer: str) -> float:
    pred_raw = extract_answer_from_text(response)

    if correct_answer is None or correct_answer == "":
        return -1.0

    if answers_match(pred_raw, correct_answer):
        return 1.0
    return -1.0

def process_math_item(item):
    problem = item.get("problem", "")
    solution = item.get("solution", "")
    answer = extract_boxed_answer(solution)
    if not answer:
        answer = item.get("answer", "")

    return {
        "problem": problem.strip(),
        "correct_answer": answer.strip() if answer else "",
    }

# --- Math Problem Processing ---
math_problems = [process_math_item(item) for item in train_dataset]
math_problems = [item for item in math_problems if item["problem"] and item["correct_answer"]]
print(f"Processed {len(math_problems)} valid math problems", flush=True)
random.shuffle(math_problems)

print(f"\nSample problem:", flush=True)
print(f"Problem: {math_problems[0]['problem'][:200]}...", flush=True)
print(f"Answer: {math_problems[0]['correct_answer']}", flush=True)

def build_prompt(problem: str) -> str:
    messages = [
        {
            "role": "system",
            "content": "You are a math expert. Solve the problem step by step. Always provide your final answer in the format: \\boxed{answer}"
        },
        {
            "role": "user",
            "content": f"Solve this problem step by step and provide your final answer in \\boxed{{answer}} format:\n\n{problem}"
        }
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

print("\nBuilding dataset with pre-applied chat template...", flush=True)
dataset_items = [{
    "prompt": build_prompt(item["problem"]),
    "correct_answer": item["correct_answer"],
} for item in math_problems]

raw_ds = Dataset.from_list(dataset_items)
print(f"Created dataset with {len(raw_ds)} training examples", flush=True)

print("\nSample formatted prompt:", flush=True)
print(raw_ds[0]['prompt'][:500], flush=True)
print("...", flush=True)

# --- Reward Function Setup ---
global_step = 0
correct_count = 0
total_count = 0

# Initialize all_rewards.csv to be empty at the start of training
with open("all_rewards_0.csv", "w") as f:
    f.write("reward\n") # Write header

def math_reward_function(prompts, completions, correct_answer, **kwargs):
    global global_step, correct_count, total_count
    rewards = []

    print(f"\n{'='*70}", flush=True)
    print(f"REWARD FUNCTION CALLED - Batch of {len(completions)} completions", flush=True)
    print(f"{'='*70}", flush=True)
    sys.stdout.flush()

    for idx, resp in enumerate(completions):
        if idx < len(correct_answer):
            gold = correct_answer[idx]
        else:
            print(f"WARNING: No correct answer at index {idx}", flush=True)
            rewards.append(-1.0)
            global_step += 1
            total_count += 1
            continue

        reward = compute_binary_reward(resp, gold)
        rewards.append(reward)

        extracted = extract_answer_from_text(resp)
        pred_norm = normalize_answer(extracted)
        gold_norm = normalize_answer(gold)

        total_count += 1
        if reward > 0:
            correct_count += 1

        print(f"\n--- Step {global_step} ---", flush=True)
        print(f"Response (first 200 chars): {resp[:200]}...", flush=True)
        print(f"Extracted: {extracted!r} -> {pred_norm!r}", flush=True)
        print(f"Expected:  {gold!r} -> {gold_norm!r}", flush=True)
        print(f"{'✓ CORRECT' if reward > 0 else '✗ INCORRECT'} | Reward: {reward}", flush=True)
        print(f"Running accuracy: {correct_count}/{total_count} = {100*correct_count/total_count:.1f}%", flush=True)

        try:
            wandb.log({
                "reward/step": reward,
                "reward/running_accuracy": correct_count/total_count,
                "train/global_step": global_step
            })
        except Exception as e:
            print(f"wandb log error: {e}", flush=True)

        with open("all_rewards_0.csv","a") as f:
           f.write(f"{reward}\n")
        global_step += 1

    print(f"\nBatch rewards: {rewards}", flush=True)
    sys.stdout.flush()
    return rewards

# --- Weights & Biases Initialization ---
wandb.init(
    project="math-grpo-training",
    config={
        "algo": "GRPO",
        "model": PLAYER_MODEL_ID,
        "dataset": "MATH",
        "dataset_size": len(math_problems),
        "seed": GLOBAL_SEED
    }
)
print("WandB initialized", flush=True)

# --- LoRA Configuration ---
lora_cfg = LoraConfig(
    r=128,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

print("\nLoading base model...", flush=True)
base_model = AutoModelForCausalLM.from_pretrained(
    PLAYER_MODEL_ID,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.float16,
)
print("Base model loaded", flush=True)

print("Applying LoRA...", flush=True)
policy_model = get_peft_model(base_model, lora_cfg)
policy_model.generation_config = GenerationConfig(
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    max_new_tokens=1024,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
)
print(f"Trainable params: {policy_model.num_parameters(only_trainable=True):,}", flush=True)

# --- GRPO Configuration ---
grpo_cfg = GRPOConfig(
    output_dir="qwen3-4b-recap-binary",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    num_generations=2,
    learning_rate=1e-5,
    seed=GLOBAL_SEED,
    logging_steps=1,
    save_steps=50,
    report_to="wandb",
    max_steps=150,
    beta=0.0,
    max_completion_length=1024,
    max_prompt_length=2048,
    temperature=0.7,
    top_p=0.9,
    log_completions=True,
)

policy_model.train()

print("\nCreating GRPOTrainer...", flush=True)
trainer = GRPOTrainer(
    model=policy_model,
    args=grpo_cfg,
    train_dataset=raw_ds,
    reward_funcs=math_reward_function,
    processing_class=tokenizer,
)
print("GRPOTrainer created", flush=True)

print("\n" + "="*70, flush=True)
print("Starting GRPO training...", flush=True)
print("="*70, flush=True)
sys.stdout.flush()

try:
    trainer.train()
except KeyboardInterrupt:
    print("\n\nTraining interrupted by user", flush=True)
except Exception as e:
    print(f"\n\nTRAINING ERROR: {e}", flush=True)
    import traceback
    traceback.print_exc()
    raise

print("\n" + "="*70, flush=True)
print(f"Training complete!", flush=True)
print(f"Final training accuracy: {correct_count}/{total_count} = {100*correct_count/total_count:.1f}%", flush=True)
print("="*70, flush=True)

print("\nSaving model...", flush=True)
trainer.save_model("./qwen3_4b_math_grpo_final")
tokenizer.save_pretrained("./qwen3_4b_math_grpo_final")
print("Done!", flush=True)

# --- Post-Training Evaluation on perturbed100.json ---
print("\n" + "="*70, flush=True)
print("Starting evaluation on perturbed100.json...", flush=True)
print("="*70, flush=True)

# Ensure the model is in evaluation mode
policy_model.eval()

# Load perturbed100.json dataset
print("Loading MATH dataset from JSON...")
try:
    with open("perturbed100.json", "r") as f:
        eval_dataset = json.load(f)
    print(f"Loaded {len(eval_dataset)} examples from perturbed100.json")
except FileNotFoundError:
    print("Error: perturbed100.json not found in the current directory.")
    eval_dataset = []

correct_eval = 0
total_eval = 0
limit_eval = 100 # Evaluate on up to 100 examples, or fewer if dataset is smaller

if eval_dataset:
    for i, example in enumerate(tqdm(eval_dataset[:limit_eval], desc="Evaluating on perturbed100.json")):
        question = example['prompt_with_flawed_start']
        ground_truth = example['correct_answer']

        gt_answer = extract_answer_from_text(ground_truth) # Reusing the training helper functions

        messages = [
            {"role": "system", "content": "You are a math expert. Always provide your final answer in the format: \\boxed{answer}"},
            {"role": "user", "content": f"Solve this problem step by step and provide your final answer in \\boxed{{answer}} format:\n\n{question}"}
        ]

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(policy_model.device)

        with torch.no_grad():
            outputs = policy_model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.3,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)

        pred_answer = extract_answer_from_text(response)

        is_correct = answers_match(pred_answer, gt_answer) # Reusing the training helper functions
        if is_correct:
            correct_eval += 1

        total_eval += 1

        if i < 5: # Print first 5 examples
            print(f"\n{'='*80}")
            print(f"Question: {question[:150]}...")
            print(f"Response: {response[:300]}...")
            print(f"Ground Truth: {gt_answer}")
            print(f"Predicted: {pred_answer}")
            print(f"Correct: {is_correct}")

        if (i + 1) % 50 == 0:
            print(f"\nProgress: {i+1}/{total_eval} - Accuracy: {correct_eval/total_eval:.2%}")

# Final evaluation results
print(f"\n{'='*80}")
print(f"Final Evaluation Results on perturbed100.json:")
print(f"Correct: {correct_eval}/{total_eval}")
if total_eval > 0:
    print(f"Accuracy: {correct_eval/total_eval:.2%}")
else:
    print("No examples evaluated.")
print("="*80)

print("\nScript finished.")
