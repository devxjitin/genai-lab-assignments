# Lab 2: Image Synthesis Using GANs

Course: ETMMGA304 Generative AI and Its Applications (MCA, K.R. Mangalam University)

| Item | Value |
|---|---|
| Date run | 2026-11-20 |
| Aim | Train a basic GAN on Fashion-MNIST, watch how the same fixed noise vectors turn into images over the epochs, and judge training with both images and losses |
| Software | Python 3.14, PyTorch 2.14.1 (CUDA 12.6), torchvision 0.29.1, Matplotlib |
| Hardware | NVIDIA GeForce RTX 3050 Laptop GPU (6 GB), device printed as `cuda` |
| Dataset | Fashion-MNIST training split, 60,000 images, scaled to [-1, 1] |
| Seed | 42 |
| Parameters | Latent size 64, generator 64-256-512-784 with LeakyReLU(0.2) and tanh, discriminator 784-512-256-1 with LeakyReLU(0.2), BCE-with-logits loss, Adam lr 0.0002 and betas (0.5, 0.999) for both, batch size 128, 10 epochs |

## What this experiment is about

A GAN is two networks playing against each other. The generator turns random noise into an image. The discriminator looks at images and outputs a score for "real" or "generated". In each batch the discriminator is trained first, on real images (label 1) and on generated ones (label 0), then the generator is trained to make the discriminator say "real" for its fakes.

Two details in the code matter:

- In the discriminator step the fake batch is `detach()`ed, so the discriminator's loss does not send gradients back into the generator.
- In the generator step the discriminator's weights are frozen with `requires_grad_(False)`, but gradients still flow through it into the generator.

Before training I checked the shapes and ranges the manual lists: real pixels span -1 to 1, the generator output is N x 784 and bounded by tanh, and the discriminator outputs N x 1 logits. The fixed 25 noise vectors are reused for every saved grid, so the same grid position can be compared from epoch to epoch.

## Files

| File | Purpose |
|---|---|
| `lab2_gan.py` | The full experiment as one script |
| `lab_outputs/gan_epoch_01.png`, `gan_epoch_05.png`, `gan_epoch_10.png` | Sample grids from the same 25 noise vectors |
| `lab_outputs/gan_losses.png` | Mean discriminator and generator loss per epoch |
| `lab_outputs/fashion_generator.pt` | Generator weights |
| `lab_outputs/results.json` | Per-epoch losses |

To reproduce: `python lab2_gan.py` from this folder.

## Results

### Per-epoch losses

| Epoch | D loss | G loss |
|---|---|---|
| 1 | 0.356 | 1.837 |
| 2 | 0.329 | 2.624 |
| 3 | 0.396 | 2.128 |
| 4 | 0.376 | 2.190 |
| 5 | 0.438 | 1.909 |
| 6 | 0.432 | 1.888 |
| 7 | 0.429 | 1.955 |
| 8 | 0.467 | 1.744 |
| 9 | 0.470 | 1.680 |
| 10 | 0.481 | 1.717 |

### Observation table

The recognisable counts are my own eyeball counts from the saved grids, so treat them as approximate (plus or minus 2).

| Epoch | D loss | G loss | Recognisable out of 25 | Repeated shapes or artifacts |
|---|---|---|---|---|
| 1 | 0.356 | 1.837 | about 3 | Mostly noisy grey blobs. Two top-left cells are almost identical, and the same bright "tank top" silhouette appears in about 7 cells. Heavy salt-and-pepper speckle everywhere. |
| 5 | 0.438 | 1.909 | about 15 | Clear sneaker wedges, sweater and shirt shapes and a few trouser-like columns. About 6 cells are still faint grey ghosts. Speckle is still strong inside the garments. |
| 10 | 0.481 | 1.717 | about 19 | Sneakers and boots in about 7 cells, shirts and sweaters in about 9, trousers in 2 or 3. Shapes are sharper but still grainy. About 4 cells remain dim and shapeless. |

### Reading the losses

- The discriminator loss stays low (0.33 to 0.48) and the generator loss stays high (about 1.7 to 2.6). So the discriminator is the stronger player the whole time, which is normal for this simple setup.
- Neither loss shot to zero or diverged. A discriminator loss near 0 would mean it wins completely and the generator gets no useful signal. A generator loss that falls to 0 would suggest the generator is fooling a weak discriminator. Neither happened.
- The upward drift in D loss (0.36 to 0.48) together with the downward drift in G loss (2.6 to 1.7) after epoch 2 means the generator is slowly catching up.
- The losses on their own are hard to read. Between epoch 5 and 10 they hardly changed, yet the pictures clearly improved. This is why the manual asks for both.

### Diversity and mode collapse

Looking at the same grid positions across epochs: the "wedge" sneaker shapes at the top right stay where they are from epoch 5 to epoch 10, and get sharper, which shows the noise-to-class mapping is stable. There was no full mode collapse (the grid does not show one object repeated 25 times), but diversity is limited. The output is mostly tops, sneakers and a couple of trouser shapes. I could not find clear bags, sandals or ankle boots, and many shirts look very alike. So there is partial loss of diversity.

## Interpretation

After 10 epochs this small fully connected GAN produces shapes that are recognisable as clothes in about three quarters of the grid, up from almost none after epoch 1. Losses stayed in a healthy range, but they do not tell you how good the images are, only that nothing broke. The main limitation is texture: every garment has noisy, grainy pixels, which a plain MLP generator cannot clean up. A sensible next step is to switch to a DCGAN with convolutional layers and batch norm, train for 30 or more epochs, and add a quantitative metric such as FID instead of only counting by eye.

## Viva questions (short answers)

1. **What does each network learn?** The generator learns to map noise to images that look like the training data. The discriminator learns to tell real images from generated ones, and its mistakes are what teach the generator.
2. **Why is the generated batch detached in the discriminator step?** The discriminator update should only change the discriminator. Detaching stops the backward pass at the generator output, so no gradients are computed for the generator there, and the same fake batch can be reused for the generator step.
3. **Why are the grid inputs fixed?** Using the same 25 noise vectors every time means any change in a grid cell comes from the generator improving, not from different random input.
4. **How can a grid reveal mode collapse?** If many cells show nearly the same image, or the whole grid shows only one or two kinds of object even though the dataset has ten, the generator has collapsed onto a few modes. Here, the limited variety (mainly tops and sneakers) is a mild version of this.
