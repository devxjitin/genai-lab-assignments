# Lab 1: Variational Autoencoder for Image Generation

Course: ETMMGA304 Generative AI and Its Applications (MCA, K.R. Mangalam University)

| Item | Value |
|---|---|
| Date run | 2026-11-20 |
| Aim | Train a VAE on MNIST, generate digits, compare reconstructions with originals, and look at the 2D latent space |
| Software | Python 3.14, PyTorch 2.14.1 (CUDA 12.6), torchvision 0.29.1, Matplotlib |
| Hardware | NVIDIA GeForce RTX 3050 Laptop GPU (6 GB), device printed as `cuda` |
| Dataset | MNIST (60,000 train, 10,000 test), downloaded through `torchvision.datasets.MNIST` |
| Seed | 42 (Python and PyTorch) |
| Parameters | 784-256-2 encoder, 2-256-784 decoder, Adam lr 0.001, batch size 128, 5 epochs |

## What this experiment is about

A plain autoencoder squeezes an image into a code and rebuilds it. A VAE does the same thing, but the encoder outputs a mean and a log variance instead of a single point. A latent vector is then sampled with `z = mu + exp(0.5 * logvar) * eps`, and the decoder turns it back into an image. The loss has two parts:

- Reconstruction loss (binary cross-entropy summed over pixels): how far the rebuilt image is from the original.
- KL divergence: how far the encoder's distribution is from a standard normal. This is what keeps the latent space smooth enough that a random draw from N(0, I) decodes to something digit-like.

The latent space here is only two dimensional on purpose, so that it can be plotted directly. The price is that a lot of detail has to be thrown away.

## Files

| File | Purpose |
|---|---|
| `lab1_vae.py` | The full experiment (steps 1 to 4 of the manual) as one script |
| `lab_outputs/vae_reconstructed.png` | Top row: 8 test digits. Bottom row: their reconstructions from the latent mean |
| `lab_outputs/vae_generated.png` | 16 digits decoded from random draws of the prior |
| `lab_outputs/vae_latent.png` | Latent means of 1,024 held-out test images, coloured by label |
| `lab_outputs/vae_mnist.pt` | Trained weights |
| `lab_outputs/results.json` | Loss log and extra numbers quoted below |

To reproduce: `python lab1_vae.py` from this folder (the MNIST download goes to `../data`).

## Results

### Loss log (per image, averaged over the training set)

The manual asks for epochs 1, 3 and 5. I kept all five, the three requested rows are in bold.

| Epoch | Reconstruction loss per image | KL divergence per image | Total | Comments |
|---|---|---|---|---|
| **1** | 188.49 | 6.67 | 195.16 | Digits are mostly blurry averages at this stage. KL is at its highest because the encoder is still spreading points out. |
| 2 | 164.56 | 5.31 | 169.87 | Big drop in reconstruction error. |
| **3** | 159.61 | 5.38 | 164.99 | Reconstruction still improving, KL has stopped moving much. |
| 4 | 156.85 | 5.46 | 162.31 | |
| **5** | 154.99 | 5.53 | 160.52 | Gains are getting small (about 2 per epoch). |

Both terms stayed finite for the whole run (the script asserts this). On the full held-out set after epoch 5 the loss per image was 153.76 (reconstruction) and 5.70 (KL), which is close to the training numbers, so there is no sign of overfitting in 5 epochs.

One thing worth noticing: the total went down mostly because the reconstruction term went down. KL actually fell after epoch 1 and then crept back up slowly (5.31, 5.38, 5.46, 5.53). That is the usual trade-off: as the decoder gets better it wants the encoder to carry more information, which costs a little more KL.

### Generated digits (16 samples from the prior)

Recognisable digits: **9 out of 16**. Clear ones are three 1s plus one more thin slanted 1, two 0s, two 9s and one that reads as a 3 or 8. The other 7 are blobs that sit between a 4, a 9 and a 0 and I would not commit to a label for them. Because the decoder only gets 2 numbers, many random points land between digit regions and the output is a smeared mix.

### Reconstructions (8 test digits)

Only 4 of the 8 kept their original class: the 7, the two 1s and the 0. The two 4s came back as 9s, the 2 came back as a blurry 8-like shape, and the final 9 turned into a smudged 8/9. All reconstructions are visibly blurrier than the originals, which is expected when 784 pixels go through a 2-number bottleneck.

### Latent space

- Digit 1 (orange) forms a tight, well separated group on the right side.
- Digit 0 (blue) sits alone in the upper left.
- Two labels that overlap: **5 and 8**. Their centroids are only 0.04 apart in latent units, so they are effectively on top of each other. 4 and 9 (centroid distance 0.38) and 3 and 8 (0.38) are the next closest pairs. This lines up with the reconstruction errors above, where 4 became 9.
- The middle of the plot, around (-1, 0), is a dense mix of 2, 3, 5, 6 and 8.

## Interpretation

The VAE trained stably and learned a sensible layout: round or thin shapes like 0 and 1 get their own corners, while digits with similar strokes (4/9, 3/5/8) pile up in the centre. Samples drawn from the prior are digit-like about half the time. The main limitation is the 2D bottleneck: it makes the latent plot readable but costs a lot of detail, and a few digit classes simply cannot be separated in two dimensions. A natural next step would be to raise the latent size to 16 or 32 and project it with PCA or t-SNE for the plot, or add a `beta` weight on the KL term and see how the balance moves.

## Viva questions (short answers)

1. **What do the mean and log variance represent?** The encoder's guess for the centre and spread of a Gaussian over latent codes for that input. Log variance is used because it can be any real number, and `exp(0.5 * logvar)` always gives a positive standard deviation.
2. **Why is the KL term in the objective?** It pulls every encoded distribution towards N(0, I). Without it the encoder would place points far apart with tiny variance (basically a normal autoencoder) and random samples from the prior would decode to garbage.
3. **Reconstructed versus generated digit?** A reconstruction starts from the encoding of a real image and tries to rebuild that image. A generated digit starts from a random vector with no source image, so it only looks like a digit if the prior and the learned latent space match well.
4. **Why held-out images for the scatterplot?** Training images were seen by the model, so their clusters would look better than the model really is. Test images show how the encoder behaves on data it has not fitted.
