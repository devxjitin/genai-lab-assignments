# Capstone: Study Tip Generator (Generative AI Creative Solution)

Course: ETMMGA304 Generative AI and Its Applications (MCA, K.R. Mangalam University)

| Item | Value |
|---|---|
| Date run | 2026-11-20 |
| Domain | Beginner study support for computing and maths topics |
| Model | `openai-community/gpt2` (124M), adapted with the Lab 3 workflow (only the last Transformer block trained, 7.09M of 124.4M parameters) |
| Software | Python 3.14, PyTorch 2.14.1 (CUDA 12.6), Transformers 4.57.6, Streamlit 1.67.0 |
| Hardware | NVIDIA GeForce RTX 3050 Laptop GPU (6 GB) |
| Seed | 42 |

## 1. Define: the user problem and success criteria

**Problem.** A beginner who types a topic (for example "stacks and queues") wants one short, concrete thing to do to start studying it, not a definition or a wall of text.

**Success criteria** (set before running the experiment):

1. Output is one short instruction a beginner can act on.
2. It is about the topic that was entered.
3. It is readable, with no repeated sentences, code dumps or stray web text.
4. The adapted model beats the plain GPT-2 base model on the mean score out of 6 over at least 10 held-out topics.

## 2. Prepare: the dataset

| Item | Detail |
|---|---|
| Source | Written by hand for this project in the same style as the six Lab 3 examples. No scraped or third-party data and no personal data. Because I wrote them without an instructor review, they still need the instructor approval the manual asks for before wider use. |
| Training set | 40 topic and tip pairs (`data/train.json`). The six from Lab 3 plus 34 more on Python, data structures basics, maths, logic, databases and web basics. |
| Held-out set | 12 topics (`data/heldout.json`): recursion, matrices, binary search, linked lists, SQL joins, stacks and queues, hash tables, Big O notation, Git branches, derivatives, standard deviation, object-oriented programming. |
| Split check | The training script asserts that no held-out topic appears in the training topics. Some held-out topics are related to a training topic (for example SQL joins and SQL SELECT, hash tables and dictionaries), which makes the test a bit easier. Unrelated ones (derivatives, Big O, OOP) are the harder part. |
| Format | `Topic: <topic>\nStudy tip: <tip><end-of-text>` |

## 3. Adapt: training

| Setting | Value |
|---|---|
| Trainable part | Last Transformer block only (`h[-1]`) |
| Optimiser | AdamW, learning rate 1e-4 |
| Epochs | 10 over 40 examples (400 update steps), examples reshuffled each epoch, batch of 1 |
| Loss | Cross-entropy on tip tokens and the end token only (prompt tokens masked with -100) |
| Learning rate choice | Lab 3 used 2e-5 for 18 steps and the loss barely moved, so I raised it to 1e-4. I picked it by watching training loss only. The held-out topics were not used to tune anything. |

Training loss (mean per epoch): 4.399, 3.771, 3.441, 3.069, 2.725, 2.397, 1.998, 1.811, 1.483, 1.227 (epochs 1 to 10). The loss fell steadily, so with 40 examples the block is learning the training tips, and a loss that low after 10 epochs also hints at memorisation, which shows up in the outputs below.

## 4. Evaluate: base versus adapted on the same held-out topics

Both models used greedy decoding with `max_new_tokens=45` on the same prompt `Topic: <topic>\nStudy tip:`. Each output was scored 0 (poor), 1 (partial) or 2 (strong) for topic relevance, readability and usefulness. Scores are mine, from the saved text in `lab_outputs/outputs.json`, entered in `data/scores.json`, and totalled by `score_summary.py`. One rater, so treat small gaps with care.

| Topic | Model | Relevance | Readability | Usefulness | Total /6 | Notes |
|---|---|---|---|---|---|---|
| recursion | Base | 1 | 1 | 0 | 2 | Names the topic, then slides into C++ style code. Not advice. |
| recursion | Adapted | 1 | 2 | 1 | 4 | Clean sentence about drawing a tree with children. Loosely fits recursion but copied from the training tip on trees. |
| matrices | Base | 1 | 1 | 0 | 2 | Vague and repeats one sentence. |
| matrices | Adapted | 0 | 1 | 0 | 1 | 'Draw the points as you see fit.' Nothing about matrices. |
| binary search | Base | 0 | 1 | 0 | 1 | Talks about a website search box and repeats itself. |
| binary search | Adapted | 1 | 1 | 0 | 2 | 'Look for two values' hints at searching but the rest is unrelated. |
| linked lists | Base | 0 | 1 | 0 | 1 | Boilerplate from a news article. |
| linked lists | Adapted | 1 | 1 | 0 | 2 | Mentions lists and first and last items but the sentence is muddled. |
| SQL joins | Base | 1 | 1 | 0 | 2 | Mentions join and table but loops on the same sentence. |
| SQL joins | Adapted | 2 | 2 | 1 | 5 | Short, on topic and actionable. Best output of the set, though 'join the table' is imprecise. |
| stacks and queues | Base | 1 | 1 | 0 | 2 | Talks about a queue, then repeats a sentence about data. |
| stacks and queues | Adapted | 0 | 2 | 0 | 2 | Readable but unrelated ('print the result'). |
| hash tables | Base | 1 | 1 | 0 | 2 | Circular definition of a hash table. |
| hash tables | Adapted | 2 | 2 | 1 | 5 | Stores names under a key, which is the right idea. 'Same key' would cause a collision though, so a reviewer should fix it. |
| Big O notation | Base | 0 | 0 | 0 | 0 | A run of '1.1.1.1...' characters. |
| Big O notation | Adapted | 0 | 1 | 0 | 1 | Nonsense about root letters and cells. |
| Git branches | Base | 2 | 1 | 1 | 4 | Right topic and plausible, but it repeats the 'Study tip:' line again and again. |
| Git branches | Adapted | 1 | 1 | 0 | 2 | Circular ('build a branch with one or two branches'). |
| derivatives | Base | 0 | 1 | 0 | 1 | Talks about the derivatives market and a source URL. |
| derivatives | Adapted | 0 | 0 | 0 | 0 | 'Swap one swap for one swap.' Meaningless. |
| standard deviation | Base | 1 | 1 | 0 | 2 | Uses the term but drifts into a health survey citation. |
| standard deviation | Adapted | 1 | 2 | 1 | 4 | Short and sensible as a first step ('sort the numbers'), though it does not mention spread. |
| object-oriented programming | Base | 1 | 0 | 0 | 1 | Repeating 'a class that will be used to create' loop. |
| object-oriented programming | Adapted | 0 | 0 | 0 | 0 | Rambling about opening new issues. Meaningless. |

| Model | Mean relevance | Mean readability | Mean usefulness | Mean total /6 |
|---|---|---|---|---|
| Base | 0.75 | 0.83 | 0.08 | 1.67 |
| Adapted | 0.75 | 1.25 | 0.33 | 2.33 |

Adapted scored higher on 7 topics, tied on 1, lower on 4.

### Raw outputs for a few topics

| Topic | Base | Adapted |
|---|---|---|
| SQL joins | "Use the join-by-name option to join a table. Study tip: Use the join-by-name option to join a table..." | "List the three columns you need to join the table." |
| hash tables | "Use the hash table to find the most common hash tables. The hash table is a set of tables that are used to store the data in a database..." | "Store three names with the same key before you open a hash table." |
| derivatives | "If you're looking for a way to make a profit, you might want to look at the derivatives market. Source: http://..." | "Swap one swap for one swap." |
| Big O notation | "Use the following notation to write your own notation: 1.1.1.1.1.1.1..." | "Draw the root letter of each cell you can mark." |

All 24 outputs are in `lab_outputs/outputs.json`.

### Findings

- The adapted model has a higher mean total (see the table), which meets criterion 4 on paper, but the absolute level is low. Even the adapted model averages about 2.3 out of 6, and its mean usefulness is only 0.33 out of 2.
- Most of the gain is format. The adapted model writes one short imperative sentence and then stops, while the base model repeats lines, quotes web text or dumps code. Readability went up from 0.83 to 1.25 on average. Relevance did not move (0.75 for both).
- On topics close to a training topic the adapted model does well (SQL joins, hash tables, standard deviation). On unrelated topics it produces fluent nonsense.
- It lost to the base model on 4 topics (matrices, Git branches, derivatives, OOP). Git branches is the clearest case: the base model gave a relevant, if repetitive, sentence and the adapted one gave a circular one.

### Failure analysis (required one failure)

**Topic: derivatives.** Adapted output: "Swap one swap for one swap." This is grammatical and short, so it looks like a valid tip, but it is meaningless. Likely cause: there is no calculus tip in the training set except one on limits, so the model falls back on the "swap" wording from the sorting example and strings it together. It is a hazard because the fluent format hides the error, and a student could waste time on it.

**How a reviewer should handle it.** Reject any tip that does not name or clearly relate to the topic, replace it with a hand-written tip, and add that topic and tip to the training set so the gap shrinks. Keep a log of rejected tips.

## 5. Demonstrate: Streamlit app

`app.py` is the manual's Streamlit form, with one addition: a disclosure line under every tip saying it is AI generated, must be reviewed, and that no personal information should be entered. It loads the model from `lab_outputs/study_tip_gpt2` (written by the training script) and uses greedy decoding.

The trained model (about 475 MB) is not committed to Git because of its size. Run the training script first to recreate it, then start the app.

Run it:

```
pip install streamlit torch transformers
python train_and_generate.py     # creates lab_outputs/study_tip_gpt2
streamlit run app.py
```

### Demonstration checks (run headlessly with Streamlit's AppTest, `test_app.py`)

| Check | Result |
|---|---|
| Submit an empty topic | The app shows the warning "Enter a study topic first." and generates nothing. |
| Whitespace-only topic | Same warning. |
| Held-out topic "recursion" | "Draw a small tree with a root and test it with three children." Understandable, loosely related, not very useful. |
| Held-out topic "SQL joins" | "List the three columns you need to join the table." Understandable and useful. |
| One weak output and its handling | See the "derivatives" failure analysis above. |

Raw results are in `lab_outputs/demo_checks.json`. I did not take a browser screenshot of the interface. The checks above ran the app logic through Streamlit's test runner instead, so a live walkthrough is still worth doing before presenting.

## 6. Conditions for use, privacy, bias and limits

- **Human review is required.** At this quality level only about 1 output in 3 (4 of 12) earned any usefulness credit, so tips must be reviewed by an instructor before students see them. The app should be used as a draft helper for instructors, not as a tool for students on their own.
- **Scope.** Only topics close to the 40 training topics should be offered. For other topics the app should say "no reviewed tip yet".
- **Privacy.** The app stores nothing and runs locally. Users are told not to enter personal information. Typed text is limited to 120 characters.
- **Bias and content risk.** GPT-2 was trained on web text and the base model's outputs above include news boilerplate, URLs and a financial-advice style sentence. The adapted model suppresses most of that but not all of it. Tips were written in English and assume a beginner with access to a computer and pen and paper.
- **Dataset limits.** 40 examples written by one person is a small and narrow set. The held-out score comes from 12 topics and one rater.

## 7. Next steps

1. Grow the dataset to a few hundred instructor-reviewed pairs, with at least a few tips for each major topic family.
2. Try training the last two or three blocks, or use LoRA, instead of only the last block.
3. Add a second rater and compute agreement on the scores.
4. Add a retrieval check that blocks tips that share no word with the topic.

## Files

| File | Purpose |
|---|---|
| `train_and_generate.py` | Fine-tunes the last block and generates base and adapted outputs |
| `score_summary.py` | Combines the manual scores into the table and means |
| `app.py` | Streamlit demo |
| `test_app.py` | Headless demonstration checks |
| `data/train.json`, `data/heldout.json` | Training pairs and held-out topics |
| `data/scores.json` | Manual scores with notes |
| `lab_outputs/outputs.json` | Raw base and adapted outputs |
| `lab_outputs/train_log.json` | Training settings and loss per epoch |
| `lab_outputs/eval_summary.md` | Generated evaluation table |
| `lab_outputs/demo_checks.json` | Results of the demonstration checks |
| `lab_outputs/study_tip_gpt2/` | Adapted model and tokenizer |

## Viva questions (short answers)

1. **Why this dataset and model?** Study tips are short, so a small hand-written set is enough to test the workflow, and GPT-2 small runs on a laptop GPU. The cost is that GPT-2 knows little about the topics, as the results show.
2. **Which results show whether adaptation helped?** The mean total score on 12 held-out topics (1.67 for base against 2.33 for adapted), the per-criterion means (the gain is in readability, not relevance) and the 4 topics where it did worse. Together they say adaptation changed the style a lot and the knowledge very little.
3. **What human review before students use it?** A subject instructor reads every tip for the topic, checks it is correct and actionable, can edit or reject it, and approves it before it is shown. The app also keeps the AI-generated notice visible.
