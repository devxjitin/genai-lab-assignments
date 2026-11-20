# Lab 3: Text Generation Using GPT-2

Course: ETMMGA304 Generative AI and Its Applications (MCA, K.R. Mangalam University)

| Item | Value |
|---|---|
| Date run | 2026-11-20 |
| Aim | Generate continuations with GPT-2, compare greedy and sampled decoding at two temperatures and two seeds, then fine-tune only the last Transformer block on short study tips and compare base and tuned models on held-out topics |
| Software | Python 3.14, PyTorch 2.14.1 (CUDA 12.6), Hugging Face Transformers 4.57.6 (the manual asks for >=4.51,<5) |
| Hardware | NVIDIA GeForce RTX 3050 Laptop GPU, device printed as `cuda` |
| Model | `openai-community/gpt2` (124,439,808 parameters) |
| Seed | 42 (sampling runs use 42 and 43 as the manual asks) |
| Generation settings | `max_new_tokens=45`, sampling uses `top_p=0.9` and temperature 0.6 or 1.0 |
| Fine-tune settings | Last block only (7,087,872 trainable parameters, 5.7 percent), AdamW lr 2e-5, 3 epochs over 6 examples, batch of 1, prompt tokens masked with label -100 |

## What this experiment is about

GPT-2 predicts a probability distribution over the next token, picks one, appends it and repeats. Greedy decoding always picks the most likely token, which is stable but tends to repeat. Sampling picks randomly from the distribution (here trimmed with top-p 0.9). Temperature reshapes the distribution: below 1 it sharpens, at 1 it is left as the model produced it.

For the fine-tuning part, the whole model is frozen except the last Transformer block. Each training example is `Topic: <topic>\nStudy tip:` followed by the tip and an end-of-text token. The loss is only computed on the tip and the end token, because the prompt part is labelled -100.

## Files

| File | Purpose |
|---|---|
| `lab3_gpt2.py` | The full experiment (steps 1 to 4) as one script |
| `lab_outputs/outputs.json` | Every raw generation, fine-tune losses and held-out comparisons |
| `lab_outputs/study_tip_gpt2/` | Saved tuned model and tokenizer from the six-example fine-tune (the capstone trains its own, larger version) |

To reproduce: `python lab3_gpt2.py` from this folder. The saved model (about 475 MB) is not committed to Git because of its size, and running the script recreates it.

## Results part 1: decoding comparison

Ratings use the manual's 0 / 1 / 2 scale (0 poor, 1 partial, 2 strong), scored by me from the saved text. Coherence is about flow within the output, relevance is about how well it follows the prompt, creativity is about specificity or novelty. This is one rater and the scale is coarse, so differences of one point should not be over-read.

### Robot prompt: "A story about a robot begins like this:"

| Decoding and seed | Coherence | Relevance | Creativity | Excerpt that supports the rating |
|---|---|---|---|---|
| Greedy | 1 | 1 | 0 | "A robot is a robot that is programmed to do something. It is programmed to do something. It is programmed to..." Grammatical but it loops, and it defines a robot instead of starting a story. |
| 0.6 and 42 | 1 | 2 | 1 | "I was in my basement, and a guy came in... we're going to be building a robot." A real narrative opening about a robot, but it starts repeating itself by the end. |
| 0.6 and 43 | 1 | 1 | 0 | "A robot is a computer program that... A robot is not a computer program." Contradicts itself and repeats the sentence. Not a story. |
| 1.0 and 42 | 1 | 1 | 1 | "It was supposed to be something big, but it never happened. It was called a toy..." Reads like a story but never mentions a robot. |
| 1.0 and 43 | 1 | 2 | 2 | "A robotic girl runs into a strange, alien world called 'Rome.'" Most inventive, though the details are odd and the quote marks drift. |

### Library prompt: "In a school library, a tiny robot found a lost book. It"

| Decoding and seed | Coherence | Relevance | Creativity | Excerpt that supports the rating |
|---|---|---|---|---|
| Greedy | 1 | 1 | 0 | "was a book that had been lost for years. 'I was like, Oh my God...' said the boy, who asked not to be identified." Switches to a news-style quote and forgets the robot. |
| 0.6 and 42 | 1 | 2 | 1 | "was the only book in the library... The robot was not the only one who had lost his book." On topic, a little repetitive. |
| 0.6 and 43 | 1 | 2 | 1 | "was a book with a name: 'The Book of the Dead.' ... the robot went missing." Keeps the robot, but repeats "the book of the dead". |
| 1.0 and 42 | 1 | 2 | 1 | "was the only book in its collection... A dozen books stood on the shelf, each a different class of books." Library detail is kept, meaning gets thin. |
| 1.0 and 43 | 1 | 1 | 2 | "knew that the book belonged to a man named David, who had been captured by aliens with a key." Most original, but has drifted far from a school library. |

### What the comparison shows

- Greedy decoding gave the least creative output both times, and the robot prompt looped on "It is programmed to do something." Repetition is the typical greedy failure.
- At the same seed, temperature 1.0 gave more specific and surprising text than 0.6, but also wandered further from the prompt (seed 43 on both prompts ends up with aliens). The 0.6 outputs stayed closer to the prompt but repeated phrases ("the book of the dead", "A robot is not a computer program").
- The same temperature with seed 42 versus 43 gave very different content (basement scene versus a computer-program definition at 0.6). So one sample per setting is not enough to say a setting is "better", and my scores mostly separate greedy from sampling, not 0.6 from 1.0.
- Coherence never reached 2 for any output. GPT-2 small loses the thread inside 45 tokens.

## Results part 2: fine-tuning on six study tips

Training loss (mean over the 6 examples in each epoch, plus the last example as the manual prints it):

| Epoch | Last example loss | Mean loss |
|---|---|---|
| 1 | 4.759 | 5.159 |
| 2 | 4.328 | 4.876 |
| 3 | 4.724 | 5.178 |

The loss did not go down in any consistent way. That makes sense: there were only 18 optimiser steps at a learning rate of 2e-5 with a single block unfrozen, which is a very small amount of adaptation.

### Held-out comparison (greedy decoding, topics never seen in training)

Scoring: 0 to 2 for topic relevance plus 0 to 2 for usefulness as a study tip, total out of 4.

| Held-out topic | Base score out of 4 | Tuned score out of 4 | Example difference |
|---|---|---|---|
| Recursion | 1 | 1 | Both start with "Use the recursion keyword to define a new type of recursion" and then spill into C++-like code. Only the code details differ (`Recursion(int i)` against `Recursion(int x, int y)`). Mentions the topic but is wrong and not usable advice. |
| Matrices | 1 | 1 | Word-for-word identical. "Use the matrices to create a matrix of the same size as the input", then repeats a sentence. On topic by name only. |
| Binary search | 0 | 0 | Word-for-word identical. "Use the search box to find the search term you want to search for." This is about a website search box, not binary search. |

Two things stand out. First, the base model already opens with "Use ..." for all three prompts, which suggests the prompt format `Topic: ...\nStudy tip:` is doing most of the work, not the fine-tune. Second, the tuned model did not become less useful, but it also did not become more useful. With six examples and 18 steps that is what I would expect: the manual itself says a six-example fine-tune only demonstrates the workflow and that consistent improvement needs more evidence.

## Interpretation

The decoding experiment gave clear lessons: greedy decoding is repetitive, sampling at temperature 1.0 is more creative but drifts, and a single seed hides a lot of variation. The fine-tuning experiment did not show an improvement, and I do not want to claim one. The workflow itself works (only the last block trains, prompt tokens are masked, the model saves and reloads). The limit is data and optimisation: six examples, three epochs, a tiny learning rate. The next step, which the capstone folder takes, is to use a larger, varied set of study-tip examples and many more passes, then score at least 10 held-out topics for both models.

## Viva questions (short answers)

1. **How does greedy decoding select the next token?** It takes the token with the highest probability at every step, with no randomness, so the same prompt always gives the same output.
2. **How does temperature affect sampling?** The logits are divided by the temperature before the softmax. Lower values sharpen the distribution so the likely tokens dominate. Higher values flatten it, so unlikely tokens get picked more often, which raises variety and the chance of drifting off topic.
3. **Which tokens contribute to the fine-tuning loss?** Only the tip text and the end-of-text token. The `Topic: ...\nStudy tip:` prefix is labelled -100, which the loss ignores.
4. **Why are the held-out topics excluded from training?** If recursion, matrices or binary search were in the training set the model could just recall them, and we could not tell whether it had learned the task or memorised the examples.
