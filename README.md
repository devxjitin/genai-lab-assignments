# Generative AI and Its Applications: Lab Assignments

K.R. Mangalam University, Master of Computer Applications (MCA)
Course code: ETMMGA304

| | |
|---|---|
| Student | _your name here_ |
| Roll number | _your roll number here_ |
| Section / Semester | _your section and semester here_ |
| Instructor | _instructor name here_ |

This repository holds my work for the course laboratory manual: four lab experiments and one capstone project. Each experiment is in its own folder with the code, the saved outputs, and a README that explains the aim, the method, the results from the actual run, and short viva answers.

## Contents

| Folder | Experiment | Course outcomes | Headline result |
|---|---|---|---|
| [lab1_vae](lab1_vae) | Variational autoencoder on MNIST | CO1, CO2, CO3 | Loss per image fell from 195.2 to 160.5 in 5 epochs. 9 of 16 generated digits were recognizable. Digits 5 and 8 overlap in the 2D latent space. |
| [lab2_gan](lab2_gan) | GAN on Fashion-MNIST | CO1, CO2, CO3 | About 3 of 25 samples recognizable at epoch 1, about 15 at epoch 5 and about 19 at epoch 10. No full mode collapse, but limited variety. |
| [lab3_gpt2](lab3_gpt2) | GPT-2 decoding and last-block fine-tune | CO1, CO2, CO3 | Greedy decoding repeats itself. The six-example fine-tune gave no measurable gain on held-out topics. |
| [lab4_stable_diffusion](lab4_stable_diffusion) | Stable Diffusion v1.5 prompt comparison | CO2, CO3, CO4 | The seed sets the layout and the prompt sets the medium. "Pencil sketch" was ignored at seed 43. |
| [capstone_study_tip_generator](capstone_study_tip_generator) | Study-tip generator with a Streamlit app | CO2, CO3, CO4 | The adapted model scored 2.33 out of 6 against 1.67 for the base model on 12 held-out topics, mostly from better format rather than better content. |

Course outcomes: CO1 explain the theory of VAEs, GANs and Transformers. CO2 implement generative models with open-source tools. CO3 evaluate and compare generated outputs. CO4 analyze ethical and practical implications.

## Folder layout

```
.
|-- README.md
|-- requirements.txt
|-- lab1_vae/                      lab1_vae.py, README.md, lab_outputs/
|-- lab2_gan/                      lab2_gan.py, README.md, lab_outputs/
|-- lab3_gpt2/                     lab3_gpt2.py, README.md, lab_outputs/
|-- lab4_stable_diffusion/         lab4_stable_diffusion.py, README.md, lab_outputs/
`-- capstone_study_tip_generator/  train_and_generate.py, app.py, score_summary.py,
                                   test_app.py, data/, README.md, lab_outputs/
```

## How to run

Requirements: Python 3.10 or newer, and an NVIDIA GPU for Lab 4 (labs 1 to 3 also run on CPU, only slower). The results in these READMEs came from Python 3.14 on Windows 11 with an RTX 3050 Laptop GPU (6 GB) and CUDA 12.6.

```
python -m venv .venv
.venv\Scripts\activate                # Linux or macOS: source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

Then run any experiment from inside its own folder:

```
cd lab1_vae
python lab1_vae.py
```

Notes:

- MNIST and Fashion-MNIST download automatically into a shared `data` folder at the repository root (about 145 MB).
- Lab 4 downloads Stable Diffusion v1.5 (several GB) on first run and should be run on its own, in a fresh session, to avoid running out of GPU memory.
- The GPT-2 models saved by Lab 3 and the capstone are about 475 MB each, so they are not committed. Running the scripts recreates them. For the capstone, run `train_and_generate.py` first and then `streamlit run app.py`.
- Seeds are fixed at 42 (and 43 where the manual asks for a second seed). GPU runs can still differ slightly between machines, so expect similar but not identical numbers.

## About the scores in the READMEs

Losses, timings and parameter counts come straight from the runs. Counts of "recognizable" images and all 0 to 2 ratings for generated text and images are subjective, were made by a single rater from the saved outputs, and are labelled that way in each README. Small differences between scores should not be over-read.

## Reference

Lab manual: Generative AI and Its Applications, ETMMGA304, K.R. Mangalam University. Datasets and models used: MNIST, Fashion-MNIST (torchvision), GPT-2 (`openai-community/gpt2`), Stable Diffusion v1.5 (`stable-diffusion-v1-5/stable-diffusion-v1-5`).
