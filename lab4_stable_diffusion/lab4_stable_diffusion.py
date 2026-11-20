"""Lab 4: Multimodal prompt-to-image generation with Stable Diffusion v1.5.

Follows the lab manual steps 1 and 2. Run from this folder:  python lab4_stable_diffusion.py
"""
import json
import random
import time
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
out = Path("lab_outputs")
out.mkdir(exist_ok=True)
print("Device:", device)

# step 1: load Stable Diffusion
assert device.type == "cuda", "Select a GPU runtime and rerun setup."
repo = "stable-diffusion-v1-5/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(repo, torch_dtype=torch.float16)
pipe = pipe.to(device)
pipe.enable_attention_slicing()

# step 2: compare prompt variants
prompts = [
    "A small library with a reading table, wide view, watercolor illustration",
    "A small library with a reading table, wide view, pencil sketch",
]
log = []
for i, prompt in enumerate(prompts, 1):
    for seed in (42, 43):
        rng = torch.Generator(device=device).manual_seed(seed)
        t0 = time.time()
        result = pipe(prompt, width=512, height=512,
                      num_inference_steps=25, guidance_scale=7.5, generator=rng)
        secs = round(time.time() - t0, 1)
        flags = result.nsfw_content_detected
        flagged = bool(flags and flags[0])
        if flagged:
            print("Image flagged. Review the classroom prompt.")
            log.append({"prompt": prompt, "seed": seed, "flagged": True, "seconds": secs})
            continue
        image = result.images[0]
        name = f"library_prompt_{i}_seed_{seed}.png"
        image.save(out / name)
        print("saved", name, f"({secs}s)")
        log.append({"prompt": prompt, "seed": seed, "flagged": False, "file": name, "seconds": secs})

json.dump({"repo": repo, "device": str(device), "gpu": torch.cuda.get_device_name(0),
           "settings": {"width": 512, "height": 512, "steps": 25, "guidance_scale": 7.5, "dtype": "float16"},
           "runs": log}, open(out / "results.json", "w"), indent=2)
