# Lab 4: Multimodal Prompt-to-Image Generation

Course: ETMMGA304 Generative AI and Its Applications (MCA, K.R. Mangalam University)

| Item | Value |
|---|---|
| Date run | 2026-11-20 |
| Aim | Generate images from text with Stable Diffusion, change one prompt element (the art style) while keeping seed and settings matched, and rate the results |
| Software | Python 3.14, PyTorch 2.14.1 (CUDA 12.6), Diffusers 0.35.2, Transformers 4.57.6 |
| Hardware | NVIDIA GeForce RTX 3050 Laptop GPU (6 GB), local generation (no Replicate or DALL-E API used) |
| Model | `stable-diffusion-v1-5/stable-diffusion-v1-5`, `StableDiffusionPipeline`, float16, attention slicing on |
| Generation settings | 512 x 512, 25 inference steps, guidance scale 7.5, generator seeds 42 and 43 |
| Safety checker | Left on. No image was flagged. |
| Run time | about 8 seconds per image (10.4 s for the first one, which includes warm-up) |

## What this experiment is about

Stable Diffusion starts from random noise and removes it over several steps, steered by the text prompt. Both the starting noise (set by the seed) and the wording affect the picture. To judge what the wording does, the seed and every other setting is held fixed, and only the style words change:

- Prompt 1: "A small library with a reading table, wide view, watercolor illustration"
- Prompt 2: "A small library with a reading table, wide view, pencil sketch"

Each prompt was run with seeds 42 and 43, giving four images.

## Files

| File | Purpose |
|---|---|
| `lab4_stable_diffusion.py` | The experiment (steps 1 and 2 of the manual) |
| `lab_outputs/library_prompt_1_seed_42.png` | Watercolor, seed 42 |
| `lab_outputs/library_prompt_1_seed_43.png` | Watercolor, seed 43 |
| `lab_outputs/library_prompt_2_seed_42.png` | Pencil sketch, seed 42 |
| `lab_outputs/library_prompt_2_seed_43.png` | Pencil sketch, seed 43 |
| `lab_outputs/results.json` | Settings, seeds, timing, safety flag status |

To reproduce: run `python lab4_stable_diffusion.py` in a fresh session on a GPU. The first run downloads several GB of model files. The manual suggests running this lab alone in a fresh session to avoid running out of GPU memory, and I did.

## Results

Scoring uses the manual scale (0 missing, 1 partial, 2 clear). Subject score asks "is it a small library with a reading table, in a wide view?". Style score asks "does it look like the requested medium?". They are rated separately. One rater, so a one point difference is a judgement call.

| Style requested | Seed | Subject score 0-2 | Style score 0-2 | Composition or errors |
|---|---|---|---|---|
| Watercolor | 42 | 1 | 2 | Convincing wash colours and ink outlines. A figure with long hair sits at a small desk facing the window, so there is a desk but not really a reading table, and the figure was not requested. A large blue oval (a rug or a pond-like shape) at the bottom is odd. |
| Watercolor | 43 | 2 | 2 | Cleanest result. Shelves on three walls, a wooden table with one chair, a window. Painterly washes. The chair back has a strange round cutout. |
| Pencil sketch | 42 | 1 | 1 | Grey line drawing with tall shelves, a window and an armchair, but no reading table. Lines are clean and flat like an ink or digital drawing, with little pencil shading. The window panes are scribbled and not consistent, and the armchair is cut off at the edge. |
| Pencil sketch | 43 | 2 | 0 | Right subject (shelves, table, chair, window) but it is a near-photographic colour render with warm wooden floor lighting. Nothing about it reads as a pencil sketch, so the style word was effectively ignored. |

### What changes with style, and what changes with seed

- Changes with style: the medium and colour treatment (soft colour washes with outlines for watercolor, grey linework for the pencil prompt at seed 42), and the amount of fine detail.
- Changes with seed: the whole layout. Both seed-42 images share the same arrangement (two tall bookcases flanking a central window, a figure or armchair in the foreground). Both seed-43 images share another one (a corner room with shelves on both sides, a table and a red chair in the middle). So the seed mostly decides composition, and the prompt wording decides how it is rendered, which is exactly why comparing variants at the same seed is useful.
- Not everything is controlled by the prompt. The "reading table" was missing at seed 42 in both styles, and the figure at watercolor seed 42 was not asked for.

## Interpretation

The model followed "watercolor" well at both seeds but followed "pencil sketch" at only one of the two, so the same style word can have different strength depending on the starting noise. That means a fair prompt comparison needs more than one seed, as the manual sets up here. A limit of this lab is that four images and one rater is very little evidence. A sensible next step would be 8 to 10 seeds per prompt, plus a negative prompt like "photo, color" for the pencil version, and measuring prompt match with an automatic score such as CLIP similarity.

## Application and risk

- Useful application: quickly producing illustration drafts for a library newsletter, a classroom poster or a storyboard, where the user picks the best of several seeds and a human edits it.
- Risk: Stable Diffusion was trained on large web-scraped image sets, so there are open questions about the permission of the source artists, and its outputs can reproduce social biases (for example, the unrequested figure here is a long-haired person, a stereotype for a library user). Images that look like real photos or art can also mislead if they are shown without saying they are AI generated.

## Viva questions (short answers)

1. **Why compare prompt variants at the same seed?** The seed fixes the starting noise, so any difference between two images comes from the wording and not from different random starts. The results above show how large the seed effect alone is.
2. **What is the purpose of the denoising steps?** Each step removes a bit of noise while following the text guidance, so the image gradually goes from random noise to a coherent picture. Too few steps leaves it rough, many steps cost time.
3. **How can subject and style adherence be rated independently?** Ask separate questions: "are the requested objects present?" and "does the medium match the request?". The pencil sketch at seed 43 shows why this matters: the subject score was 2 and the style score was 0.
4. **When should a generated image carry a disclosure?** Whenever it could be taken for a photograph, a real artwork or a real place, when it is published or used for teaching or news, and when it is used in a context where trust in the source matters.
