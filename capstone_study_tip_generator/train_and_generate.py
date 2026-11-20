"""Capstone: fine-tune GPT-2 (last block only, Lab 3 workflow) on study tips, then
generate base and adapted outputs for the held-out topics. Run from this folder:
    python train_and_generate.py
Outputs: lab_outputs/study_tip_gpt2/ (model), lab_outputs/outputs.json, lab_outputs/train_log.json
"""
import json
import random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SEED = 42
EPOCHS = 10
LR = 1e-4  # chosen from training loss only; held-out topics were not used to tune it
random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
out = Path("lab_outputs")
out.mkdir(exist_ok=True)

train = [tuple(x) for x in json.load(open("data/train.json", encoding="utf-8"))]
held_out = json.load(open("data/heldout.json", encoding="utf-8"))
assert not {t.lower() for t, _ in train} & {h.lower() for h in held_out}, "train/held-out overlap"
print(f"device={device} train={len(train)} held_out={len(held_out)}")

model_id = "openai-community/gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
base = AutoModelForCausalLM.from_pretrained(model_id).to(device).eval()
tuned = AutoModelForCausalLM.from_pretrained(model_id).to(device)

tuned.requires_grad_(False)
tuned.transformer.h[-1].requires_grad_(True)
opt = torch.optim.AdamW(tuned.transformer.h[-1].parameters(), lr=LR)

log = []
for epoch in range(1, EPOCHS + 1):
    tuned.train()
    random.shuffle(train)
    losses = []
    for topic, answer in train:
        prefix = tokenizer.encode(f"Topic: {topic}\nStudy tip:")
        target = tokenizer.encode(" " + answer) + [tokenizer.eos_token_id]
        ids = torch.tensor([prefix + target], device=device)
        labels = torch.tensor([[-100] * len(prefix) + target], device=device)
        opt.zero_grad()
        loss = tuned(input_ids=ids, labels=labels).loss
        loss.backward()
        opt.step()
        losses.append(loss.item())
    mean = sum(losses) / len(losses)
    log.append({"epoch": epoch, "mean_loss": round(mean, 3)})
    print("epoch", epoch, "mean loss", round(mean, 3))
tuned.eval()
tuned.save_pretrained(out / "study_tip_gpt2")
tokenizer.save_pretrained(out / "study_tip_gpt2")


def generate(net, topic):
    inputs = tokenizer(f"Topic: {topic}\nStudy tip:", return_tensors="pt").to(device)
    with torch.no_grad():
        o = net.generate(**inputs, do_sample=False, max_new_tokens=45,
                         pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(o[0, inputs.input_ids.size(1):], skip_special_tokens=True).strip()


rows = [{"topic": t, "base": generate(base, t), "adapted": generate(tuned, t)} for t in held_out]
for r in rows:
    print(f"\n{r['topic']}\n  base:    {r['base']}\n  adapted: {r['adapted']}")

json.dump({"seed": SEED, "epochs": EPOCHS, "lr": LR, "train_size": len(train), "log": log},
          open(out / "train_log.json", "w"), indent=2)
json.dump(rows, open(out / "outputs.json", "w"), indent=2)
