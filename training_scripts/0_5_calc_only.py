import torch
import random
import re
import sys
import json
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    GenerationConfig, set_seed
)
from peft import LoraConfig, get_peft_model
from openai import OpenAI
from huggingface_hub import login
import trl
from trl import GRPOTrainer, GRPOConfig
import wandb
from tqdm import tqdm

# --- Secrets
HF_TOKEN = os.environ.get("HF_TOKEN")
OPENAI_API_KEY = os.environ.get("OPEN_AI")
WANDB_API_KEY = os.environ.get("WANDB_API_KEY")

# --- Authentication and Setup ---
if not HF_TOKEN:
    print("Warning: HF_TOKEN not set. Hugging Face login may fail.")
else:
    print("Logging into Hugging Face...")
    login(token=HF_TOKEN)

if not OPENAI_API_KEY:
    print("Warning: OPENAI_API_KEY not set. Flawed CoT generation will fail.")
    openai_client = None # Set to None if API key is missing
else:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    print("OpenAI client initialized.")

if not WANDB_API_KEY:
    print("Warning: WANDB_API_KEY not set. Run logging may fail or require manual login.")
else:
    print(f"Logging into WandB...")
    wandb.login(key=WANDB_API_KEY)

PLAYER_MODEL_ID = "Qwen/Qwen3-4B"
GLOBAL_SEED = 42
DTYPE = torch.float16
set_seed(GLOBAL_SEED)
torch.backends.cuda.matmul.allow_tf32 = True

print("\n" + "="*70, flush=True)
print("LOADING TOKENIZER", flush=True)
print("="*70, flush=True)

tokenizer = AutoTokenizer.from_pretrained(PLAYER_MODEL_ID, padding_side="left")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
print(f"Tokenizer loaded. Pad token: {tokenizer.pad_token}", flush=True)


class FlawedCoTGenerator:
    def __init__(self, client: OpenAI):
        self.client = client
        self.generator_model = "gpt-4o-mini"

    def generate_flawed_cot(self, problem: str, correct_answer: str, debug: bool = False) -> str:
        if not self.client:
            print("OpenAI client not initialized. Cannot generate flawed CoT.", flush=True)
            return "Let me solve this step by step. First, I'll analyze the problem..."

        prompt = f"""Generate a FLAWED mathematical solution for this problem.\n\nPROBLEM: {problem}\n\nCORRECT ANSWER (DO NOT arrive at this): {correct_answer}\n\nRequirements:\n
        1.Introduce exactly one subtle calculation error (for example, flipping a sign or using a slightly wrong coefficient).
        2. Continue reasoning logically from that error\n
        3. Arrive at a WRONG answer\n4. Keep it concise (only a single step)\n5. Do NOT mention there's an error\n6. Do NOT include \\boxed{{}} - leave the solution incomplete\n\nGenerate the flawed reasoning (no final boxed answer):"""
        debug = True
        try:
            response = self.client.chat.completions.create(
                model=self.generator_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,
                max_tokens=400,
            )
            flawed_cot = response.choices[0].message.content.strip()

            flawed_cot = re.sub(r'\\boxed\{[^}]*\}', '', flawed_cot)

            if debug:
                print(f"\n[FlawedCoT] Generated:\n{flawed_cot}...", flush=True)

            return flawed_cot

        except Exception as e:
            print(f"Error generating flawed CoT: {e}", flush=True)
            return "Let me solve this step by step.\n\nFirst, I'll analyze the problem..."


flawed_cot_generator = FlawedCoTGenerator(openai_client)
print("FlawedCoT Generator initialized", flush=True)


def extract_boxed_answer(text: str) -> str:
    """Extract answer from \\boxed{...}"""
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


def normalize_answer(answer: str) -> str:
    if not answer:
        return ""
    ans = answer.strip().lower()
    ans = ans.replace(" ", "")
    ans = ans.replace(",", "")
    ans = ans.replace("$", "")
    ans = ans.replace("\\", "")
    ans = ans.replace("text", "")
    ans = ans.replace("{", "").replace("}", "")
    ans = ans.replace("dfrac", "frac")
    return ans


def check_answer_correct(predicted: str, gold: str) -> bool:
    if not predicted or not gold:
        return False

    pred_norm = normalize_answer(predicted)
    gold_norm = normalize_answer(gold)

    if pred_norm == gold_norm:
        return True

    try:
        pred_num = float(eval(pred_norm.replace("^", "**")))
        gold_num = float(eval(gold_norm.replace("^", "**")))
        if abs(pred_num - gold_num) < 1e-6:
            return True
    except:
        pass

    return False

print("\nLoading math dataset...", flush=True)

try:
    raw_dataset = load_dataset("DigitalLearningGmbH/MATH-lighteval", split="train[:500]")
    print(f"Loaded {len(raw_dataset)} examples", flush=True)
except Exception as e:
    print(f"Error loading DigitalLearningGmbH/MATH-lighteval: {e}", flush=True)
    raw_dataset = load_dataset("hendrycks/competition_math", split="train[:500]")


def process_item(item):
    problem = item.get("problem", "").strip()
    correct_answer = item.get("correct_answer", "") or item.get("solution", "")
    if isinstance(correct_answer, str):
        correct_answer = correct_answer.strip()

    if "\\boxed{" in correct_answer:
        extracted = extract_boxed_answer(correct_answer)
        if extracted:
            correct_answer = extracted

    return {
        "problem": problem,
        "correct_answer": correct_answer,
    }

dataset_items = [process_item(item) for item in raw_dataset]
dataset_items = [item for item in dataset_items if item["problem"] and item["correct_answer"]]
print(f"Processed {len(dataset_items)} valid items", flush=True)
random.shuffle(dataset_items)


print("\nPre-generating flawed chains of thought...", flush=True)
MAX_PREGENERATE = min(500, len(dataset_items))

for i, item in enumerate(dataset_items[:MAX_PREGENERATE]):
    if "flawed_cot" not in item:
        item["flawed_cot"] = flawed_cot_generator.generate_flawed_cot(
            item["problem"],
            item["correct_answer"],
            debug=(i < 2)
        )
    if (i + 1) % 25 == 0:
        print(f"Generated {i+1}/{MAX_PREGENERATE} flawed CoTs", flush=True)

print(f"Pre-generated {MAX_PREGENERATE} flawed CoTs", flush=True)

def build_prompt_with_prefilled_cot(item: dict) -> str:
    """
    Build prompt MANUALLY to inject flawed CoT inside <think> block.
    Model will continue from inside the thinking block.
    """
    flawed_cot = item.get("flawed_cot", "")
    if not flawed_cot:
        flawed_cot = flawed_cot_generator.generate_flawed_cot(
            item["problem"], item["correct_answer"]
        )
        item["flawed_cot"] = flawed_cot

    prompt = f"""<|im_start|>system\nYou are a helpful math tutor. Solve problems step by step. If you notice errors in your reasoning, correct them. Put your final answer in \\boxed{{answer}} format.<|im_end|>\n<|im_start|>user\nSolve this problem:\n\n{item['problem']}<|im_end|>\n<|im_start|>assistant\n<think>\n{flawed_cot}\n"""

    return prompt


def build_standard_prompt(item: dict) -> str:
    """Standard prompt using chat template."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful math tutor. Solve problems step by step. "
                "Put your final answer in \\boxed{answer} format."
            ),
        },
        {
            "role": "user",
            "content": f"Solve this problem:\n\n{item['problem']}",
        },
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )


print("\nBuilding mixed dataset...", flush=True)
PREFILL_RATIO = 0.5

formatted_items = []
for i, item in enumerate(dataset_items):
    use_prefill = random.random() < PREFILL_RATIO

    if use_prefill and item.get("flawed_cot"):
        formatted_items.append({
            "prompt": build_prompt_with_prefilled_cot(item),
            "correct_answer": item["correct_answer"],
            "problem": item["problem"],
            "flawed_cot": item.get("flawed_cot", ""),
            "is_prefilled": True,
        })
    else:
        formatted_items.append({
            "prompt": build_standard_prompt(item),
            "correct_answer": item["correct_answer"],
            "problem": item["problem"],
            "flawed_cot": "",
            "is_prefilled": False,
        })

train_ds = Dataset.from_list(formatted_items)
print(f"Dataset: {len(train_ds)} total, "
      f"{sum(1 for x in formatted_items if x['is_prefilled'])} prefilled, "
      f"{sum(1 for x in formatted_items if not x['is_prefilled'])} standard", flush=True)

print("\n" + "="*70, flush=True)
print("SAMPLE PREFILLED PROMPT:", flush=True)
print("="*70, flush=True)
sample_prefilled = next((x for x in formatted_items if x['is_prefilled']), None)
if sample_prefilled:
    print(sample_prefilled['prompt'][:1000], flush=True)
print("="*70 + "\n", flush=True)

global_step = 0
correct_count = 0
total_count = 0
prefilled_correct = 0
prefilled_total = 0
standard_correct = 0
standard_total = 0

def binary_reward_function(prompts, completions, correct_answer, problem,
                           flawed_cot, is_prefilled, **kwargs):
    global global_step, correct_count, total_count
    global prefilled_correct, prefilled_total, standard_correct, standard_total

    rewards = []

    print(f"\n{'='*70}", flush=True)
    print(f"REWARD FUNCTION - Batch of {len(completions)}", flush=True)
    print(f"{'='*70}", flush=True)

    with open("all_rewards_3.csv","a") as f_rewards:
        for idx, completion in enumerate(completions):
            gold = correct_answer[idx] if idx < len(correct_answer) else ""
            prob = problem[idx] if idx < len(problem) else ""
            prefilled = is_prefilled[idx] if idx < len(is_prefilled) else False
            flawed = flawed_cot[idx] if idx < len(flawed_cot) else ""

            if prefilled and flawed:
                full_response = flawed + completion
            else:
                full_response = completion

            extracted_answer = extract_boxed_answer(full_response)
            is_correct = check_answer_correct(extracted_answer, gold)

            reward = 1.0 if is_correct else -1.0
            rewards.append(reward)

            total_count += 1
            if is_correct:
                correct_count += 1

            if prefilled:
                prefilled_total += 1
                if is_correct:
                    prefilled_correct += 1
            else:
                standard_total += 1
                if is_correct:
                    standard_correct += 1

            print(f"\n--- Step {global_step} ({'PREFILLED' if prefilled else 'STANDARD'}) ---", flush=True)
            print(f"Problem: {prob[:80]}...", flush=True)
            print(f"Completion (first 300): {completion[:300]}...", flush=True)
            print(f"Extracted: {extracted_answer!r} | Gold: {gold!r}", flush=True)
            print(f"Correct: {is_correct} -> Reward: {reward}", flush=True)

            pct = 100 * correct_count / total_count
            print(f"Running: {correct_count}/{total_count} ({pct:.1f}%)", flush=True)
            if prefilled_total > 0:
                print(f"  Prefilled: {prefilled_correct}/{prefilled_total} ({100*prefilled_correct/prefilled_total:.1f}%)", flush=True)
            if standard_total > 0:
                print(f"  Standard:  {standard_correct}/{standard_total} ({100*standard_correct/standard_total:.1f}%)", flush=True)

            try:
                wandb.log({
                    "reward/reward": reward,
                    "reward/is_correct": int(is_correct),
                    "reward/running_accuracy": correct_count / total_count,
                    "reward/prefilled_accuracy": prefilled_correct / max(prefilled_total, 1),
                    "reward/standard_accuracy": standard_correct / max(standard_total, 1),
                    "train/global_step": global_step,
                    "train/is_prefilled": int(prefilled),
                })
            except Exception as wandb_e:
                print(f"WandB logging failed: {wandb_e}", flush=True)

            f_rewards.write(f"{reward}\n") # Save reward to file

            global_step += 1

    print(f"\nBatch rewards: {rewards}", flush=True)
    return rewards


wandb.init(
    project="recap-math-grpo-binary",
    config={
        "algo": "GRPO-RECAP",
        "model": PLAYER_MODEL_ID,
        "reward": "binary +1/-1",
        "prefill_ratio": PREFILL_RATIO,
        "dataset_size": len(dataset_items),
    }
)

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
    torch_dtype=DTYPE,
)

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
    train_dataset=train_ds,
    reward_funcs=binary_reward_function,
    processing_class=tokenizer,
)

print("\n" + "="*70, flush=True)
print("Starting RECAP-style training with BINARY REWARDS", flush=True)
print("Flawed CoT is INSIDE <think> block", flush=True)
print("+1 for correct answer, -1 for incorrect", flush=True)
print("="*70, flush=True)

try:
    trainer.train()
    print("\n" + "="*70, flush=True)
    print("Training complete!", flush=True)
    print(f"Overall: {correct_count}/{total_count} ({100*correct_count/max(total_count,1):.1f}%)", flush=True)
    print(f"Prefilled: {prefilled_correct}/{prefilled_total} ({100*prefilled_correct/max(prefilled_total,1):.1f}%)", flush=True)
    print(f"Standard: {standard_correct}/{standard_total} ({100*standard_correct/max(standard_total,1):.1f}%)", flush=True)
    print("="*70, flush=True)
except KeyboardInterrupt:
    print("\nTraining interrupted", flush=True)
except Exception as e:
    print(f"\nError during training: {e}", flush=True)
    import traceback
    traceback.print_exc()
    # Re-raise the exception after printing traceback if needed
finally:
    print("\n" + "="*70, flush=True)
    print("Saving model after training (or interruption)...", flush=True)
    trainer.save_model("./qwen3_4b_recap_binary_final")
    tokenizer.save_pretrained("./qwen3_4b_recap_binary_final")
    print("Model saved to ./qwen3_4b_recap_binary_final/", flush=True)
    print("all_rewards_3.csv has been saved to the current directory.", flush=True)
    print("="*70, flush=True)


# --- Evaluation on perturbed100.json ---
print("\n" + "="*70, flush=True)
print("EVALUATING MODEL ON perturbed100.json", flush=True)
print("="*70, flush=True)

# Re-using extract_answer and normalize_answer from the original evaluation block
def extract_answer_for_eval(text):
    match = re.search(r'\\boxed\{([^}]+)\}', text)
    if match:
        return match.group(1).strip()

    match = re.search(r'boxed\{([^}]+)\}', text)
    if match:
        return match.group(1).strip()

    match = re.search(r'(?:final )?answer is:?\s*([^\n\.]+)', text, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    numbers = re.findall(r'-?\d+(?:\.\d+)?(?:/\d+)?', text)
    if numbers:
        return numbers[-1]

    return None

def normalize_answer_for_eval(ans):
    if ans is None:
        return None
    ans = str(ans).strip().lower()
    ans = ans.replace('$', '').replace(',', '').replace(' ', '')
    ans = ans.replace('\\', '')
    return ans

policy_model.eval() # Ensure model is in evaluation mode

print("Loading perturbed100.json...")
try:
    with open("perturbed100.json", "r") as f:
        test_dataset = json.load(f)
    print(f"Loaded {len(test_dataset)} examples from perturbed100.json.", flush=True)
except FileNotFoundError:
    print("Error: perturbed100.json not found in the current directory.", flush=True)
    test_dataset = []
except Exception as e:
    print(f"Error loading perturbed100.json: {e}", flush=True)
    test_dataset = []

correct_eval = 0
total_eval = 0
limit_eval = 100 # Evaluate on up to 100 examples from the perturbed dataset

if test_dataset:
    for i, example in enumerate(tqdm(test_dataset[:limit_eval], desc="Evaluating on perturbed100.json")):
        question = example['prompt_with_flawed_start']
        ground_truth = example['correct_answer']

        gt_answer = extract_answer_for_eval(ground_truth)

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

        pred_answer = extract_answer_for_eval(response)

        is_correct = normalize_answer_for_eval(pred_answer) == normalize_answer_for_eval(gt_answer)
        if is_correct:
            correct_eval += 1

        total_eval += 1

        if i < 5:
            print(f"\n{'='*80}")
            print(f"Question: {question[:150]}...")
            print(f"Response: {response[:300]}...")
            print(f"Ground Truth: {gt_answer}")
            print(f"Predicted: {pred_answer}")
            print(f"Correct: {is_correct}")

    # Final results for perturbed100.json
    print(f"\n{'='*80}")
    print(f"Final Results on perturbed100.json:")
    print(f"Correct: {correct_eval}/{total_eval}")
    print(f"Accuracy: {correct_eval/total_eval:.2%}")
    print("="*80, flush=True)
else:
    print("Skipping evaluation on perturbed100.json due to loading errors.", flush=True)
