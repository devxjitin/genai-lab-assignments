"""Lab 3: Text generation with GPT-2 (decoding comparison + last-block fine-tune).

Follows the lab manual steps 1 to 4. Raw outputs are saved to lab_outputs/outputs.json
so that scoring in the README is done from the saved text. Run:  python lab3_gpt2.py
"""
import json
import random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
out = Path("lab_outputs")
out.mkdir(exist_ok=True)
print("Device:", device)

# step 1: load model
model_id = "openai-community/gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
model.eval()


# step 2: compare prompts and decoding
def continue_text(prompt, *, sample=False, temperature=0.8, seed=42):
    torch.manual_seed(seed)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    options = {"do_sample": sample, "max_new_tokens": 45,
               "pad_token_id": tokenizer.eos_token_id}
    if sample:
        options.update(temperature=temperature, top_p=0.9)
    with torch.no_grad():
        tokens = model.generate(**inputs, **options)
    return tokenizer.decode(tokens[0, inputs.input_ids.size(1):],
                            skip_special_tokens=True).strip()


prompts = [
    "A story about a robot begins like this:",
    "In a school library, a tiny robot found a lost book. It",
]
decoding = []
for prompt in prompts:
    print("\nPROMPT:", prompt)
    g = continue_text(prompt)
    print("GREEDY:", g)
    decoding.append({"prompt": prompt, "setting": "greedy", "seed": None, "text": g})
    for temp in (0.6, 1.0):
        for seed in (42, 43):
            t = continue_text(prompt, sample=True, temperature=temp, seed=seed)
            print(temp, seed, t)
            decoding.append({"prompt": prompt, "setting": f"temp {temp}", "seed": seed, "text": t})

# step 3: fine-tune last block on the six supplied examples
examples = [
    ("graphs", "Sketch a graph, then mark a path through three vertices."),
    ("Python loops", "Trace a loop by writing the counter after each step."),
    ("fractions", "Draw equal parts and label the fraction you shaded."),
    ("sets", "Draw a Venn diagram and label the intersection."),
    ("sorting", "Sort five values by hand and record each swap."),
    ("probability", "List the outcomes of two coin tosses before counting."),
]
tuned = AutoModelForCausalLM.from_pretrained(model_id).to(device)
tuned.requires_grad_(False)
tuned.transformer.h[-1].requires_grad_(True)
opt = torch.optim.AdamW(tuned.transformer.h[-1].parameters(), lr=2e-5)
n_train = sum(p.numel() for p in tuned.parameters() if p.requires_grad)
n_all = sum(p.numel() for p in tuned.parameters())
print(f"trainable params: {n_train:,} of {n_all:,}")
losses = []
for epoch in range(3):
    tuned.train()
    ep = []
    for topic, answer in examples:
        prefix = tokenizer.encode(f"Topic: {topic}\nStudy tip:")
        target = tokenizer.encode(" " + answer) + [tokenizer.eos_token_id]
        ids = torch.tensor([prefix + target], device=device)
        labels = torch.tensor([[-100] * len(prefix) + target], device=device)
        opt.zero_grad()
        loss = tuned(input_ids=ids, labels=labels).loss
        loss.backward()
        opt.step()
        ep.append(loss.item())
    print("Epoch:", epoch + 1, "last example loss:", round(loss.item(), 3),
          "| mean loss:", round(sum(ep) / len(ep), 3))
    losses.append({"epoch": epoch + 1, "last": round(loss.item(), 3), "mean": round(sum(ep) / len(ep), 3)})
tuned.save_pretrained(out / "study_tip_gpt2")
tokenizer.save_pretrained(out / "study_tip_gpt2")

# step 4: compare on held-out topics
held_out = ["recursion", "matrices", "binary search"]
base_model = model
heldout_rows = []
for topic in held_out:
    prompt = f"Topic: {topic}\nStudy tip:"
    model = base_model
    b = continue_text(prompt)
    model = tuned.eval()
    t = continue_text(prompt)
    print(topic, "\n  base:", b, "\n  tuned:", t)
    heldout_rows.append({"topic": topic, "base": b, "tuned": t})
model = base_model

json.dump({"seed": SEED, "device": str(device), "trainable_params": n_train, "total_params": n_all,
           "decoding": decoding, "finetune_losses": losses, "heldout": heldout_rows},
          open(out / "outputs.json", "w"), indent=2)
