# Chapter 3: Convolutional Neural Networks (CNNs from Scratch)

[![Dataset: Fashion-MNIST](https://img.shields.io/badge/Dataset-Fashion--MNIST%20(70k%20items)-blueviolet.svg)](https://github.com/zalandoresearch/fashion-mnist)
[![Test Accuracy](https://img.shields.io/badge/Test%20Accuracy-88.6%25-brightgreen.svg)]()
[![Implementation](https://img.shields.io/badge/Implementation-Pure%20NumPy-013243.svg)](https://numpy.org/)

> **Welcome to Chapter 3!**
> In Chapters 1 and 2, we built fully connected Multi-Layer Perceptrons (MLPs). While they performed well, they had a fatal blind spot: **they flattened images into a 1D line of numbers, completely destroying the 2D geometry of the visual world.**
>
> In this chapter, we build a **Convolutional Neural Network (CNN)** from first principles in pure NumPy — the architecture behind modern computer vision (facial recognition, autonomous vehicles, medical imaging). Every 2D convolution, max pooling layer, `im2col` transformation, and backward gradient is implemented by hand.

---

## The Hook: The Cat Ear Dilemma & The Blind Spot of MLPs

Imagine you show a fully connected network a picture of a cat where the left ear is in the top-left corner of the image (pixels 50–70). The network carefully adjusts its weights until it learns: *"When pixels 50–70 are bright, there is an ear."*

```
Image A (Cat on the left):           Image B (Cat on the right):
┌─────────────────────────┐         ┌─────────────────────────┐
│  /\                     │         │                     /\  │
│ (..)                    │         │                    (..) │
│                         │         │                         │
└─────────────────────────┘         └─────────────────────────┘
  Ear at Pixels [50-70]               Ear at Pixels [700-720]
  MLP says: "I see an ear!"           MLP says: "What is that? I have
                                                no idea what an ear looks
                                                like over there!"
```

Now, what happens if the cat takes two steps to the right, placing the ear at pixels 700–720?
* To a human: It is obviously the exact same ear!
* To a fully connected MLP: **It is completely helpless.** Because every pixel has its own independent connection wire, the MLP has zero understanding of space or translation. It has to learn what an ear looks like at *every single pixel coordinate on the screen* from scratch!
* To make matters worse, connecting every pixel to every neuron creates an explosion of weights (millions of parameters), leading to severe overfitting.

**We need a network that understands that an edge, button, or ear is the same feature no matter where it appears on the image.** This property is called **Translation Invariance**.

---

## The Biological Inspiration: Hubel & Wiesel's 1959 Discovery

How does biological vision solve this?

In 1959, neurophysiologists **David Hubel and Torsten Wiesel** placed microelectrodes into the visual cortex (V1) of a cat while projecting slides of shapes onto a screen. For hours, nothing happened — the neurons remained completely silent. 

Suddenly, as they were inserting a new glass slide, the neuron fired wildly with electrical spikes! 

What had triggered the neuron? It wasn't the image on the slide — it was the **sharp edge of the glass slide's border moving across the cat's field of view at a specific angle!**

```
Biological Receptive Field in V1:
   [ . . . . . . . . ]  <-- Diffuse light: Neuron remains SILENT
   [ . . / / / . . . ]  <-- 45-degree edge: Neuron FIRES wildly!
```

Hubel and Wiesel won the Nobel Prize for discovering that mammalian vision is **hierarchical**:
1. **Primary Neurons (V1)**: Do not understand faces or cats. They only look at tiny local patches and fire when they detect simple lines, edges, and orientations.
2. **Intermediate Layers (V2/V4)**: Combine edge signals to detect shapes, curves, textures, and corners.
3. **High-Level Cortex (IT)**: Combines shapes into complex objects (faces, hands, clothing).

**Convolutional Neural Networks are designed to mimic this exact biological hierarchy.**

---

## Files in this Chapter

* **[notebook.ipynb](notebook.ipynb)**: Interactive Jupyter notebook with live demonstrations of hand-crafted edge filters, feature maps, and training.
* **[train.py](train.py)**: Standalone training script that trains the pure NumPy CNN on Fashion-MNIST and produces visual diagnostics.
* **[plots/](plots/)**: Output visualizations generated during training:
  * [`plots/filter_visualizations.png`](plots/filter_visualizations.png) — Classical edge filters (Sobel H, Sobel V, Laplacian) extracting feature maps from garments.
  * [`plots/cnn_feature_maps.png`](plots/cnn_feature_maps.png) — Learned internal feature maps from Conv1 and Conv2 for a test clothing item.
  * [`plots/cnn_training_curves.png`](plots/cnn_training_curves.png) — Training & validation loss and accuracy curves over time.
  * [`plots/cnn_confusion_matrix.png`](plots/cnn_confusion_matrix.png) — 10-class test confusion matrix showing improved garment separation.

---

## 1. How 2D Convolutions Work

### A. The Filter (Kernel) as a Feature Flashlight
A **convolutional filter** (or **kernel**) is a small square grid of numbers — typically $3 \times 3$ or $5 \times 5$.
Think of the filter as a tiny template or stamp. We slide this stamp across every position of the input image from top-left to bottom-right:

```
Input Image (5x5)                Kernel (3x3)           Output Feature Map (3x3)
┌───┬───┬───┬───┬───┐            ┌───┬───┬───┐          ┌───┬───┬───┐
│ 1 │ 0 │ 1 │ 0 │ 0 │            │ 1 │ 0 │ 1 │          │ 4 │ 3 │ 4 │
├───┼───┼───┼───┼───┤            ├───┼───┼───┤          ├───┼───┼───┤
│ 0 │ 1 │ 1 │ 1 │ 0 │     *      │ 0 │ 1 │ 0 │    =     │ 2 │ 4 │ 3 │
├───┼───┼───┼───┼───┤            ├───┼───┼───┤          ├───┼───┼───┤
│ 1 │ 0 │ 1 │ 1 │ 0 │            │ 1 │ 0 │ 1 │          │ 2 │ 3 │ 4 │
├───┼───┼───┼───┼───┤            └───┴───┴───┘          └───┴───┴───┘
│ 0 │ 1 │ 0 │ 0 │ 1 │
├───┼───┼───┼───┼───┤
│ 1 │ 0 │ 1 │ 0 │ 0 │
└───┴───┴───┴───┴───┘
```

At each position:
1. We compute the **element-wise product** between the $3 \times 3$ image patch and the $3 \times 3$ kernel.
2. We sum all 9 numbers together and add a bias: $z = \sum (X_{patch} \odot W) + b$.
3. The resulting single number is placed into the output grid, called the **Feature Map**.

* If the patch under the flashlight matches the pattern of the filter, the output value is large and positive.
* If the patch does not match, the output value is zero or negative.

---

### B. Seeing Convolutions in Action: Hand-Crafted Edge Detectors

Before letting the machine learn its own filters, we can inspect how human engineers used to design filters by hand:
* **Horizontal Sobel Filter**: Detects horizontal boundaries (top/bottom edges of sleeves, soles of shoes).
* **Vertical Sobel Filter**: Detects vertical boundaries (seams, trouser legs, zippers).
* **Laplacian Filter**: Detects all-direction outlines and high-contrast edges.

Applying these filters to real garments from Fashion-MNIST produces striking feature maps:

![Filter Visualizations](plots/filter_visualizations.png)

In modern deep learning, **we do not program these numbers by hand**. We initialize the filters with random noise, and the network uses calculus (backpropagation) to discover the best filters on its own!

---

### C. The Magic of Parameter Sharing
Consider an image of size $28 \times 28 = 784$ pixels:
* In a fully connected layer with 128 neurons: We needed **$784 \times 128 = 100,352$ weights**!
* In a $3 \times 3$ convolutional layer with 8 filters: We only need **$8 \times (3 \times 3) = 72$ weights**!

Because the exact same 9 weights are reused across every patch of the image, the parameter count drops by **over 99.9%**! This virtually eliminates overfitting while granting **Translation Invariance**: whether a button appears at the top, center, or bottom of a shirt, the same filter will detect it!

---

### D. Padding & Stride

```
Without Padding (Valid):                     With Padding = 1 (Same):
Image: 4x4, Kernel: 3x3                     Padded Image: 6x6, Kernel: 3x3
Output shrinks to: 2x2                      Output stays: 4x4

   x x x x                                     0 0 0 0 0 0
   x [x x x]                                   0 x x x x 0
   x [x x x]  --> Output: 2x2                  0 x x x x 0  --> Output: 4x4
   x [x x x]                                   0 x x x x 0
                                               0 x x x x 0
                                               0 0 0 0 0 0
```

* **Valid Padding (padding = 0)**: The filter is only placed where it completely fits inside the image. The output spatial size shrinks: $\text{Out} = \frac{H - K}{\text{stride}} + 1$.
* **Same Padding (padding = 1 for $3 \times 3$)**: We surround the image with a border of zeros. This keeps the output height and width identical to the input!
* **Stride**: The step size of the sliding window. A stride of 1 moves 1 pixel at a time; a stride of 2 skips every other pixel, cutting the spatial dimensions in half.

---

## 2. Spatial Max Pooling: Downsampling & Summarization

After detecting features, do we really need to know their exact sub-pixel coordinate?
No! If a shirt has a collar, we care *that* a collar exists in the upper region, not whether it is at coordinate $(4, 12)$ versus $(4, 13)$.

**Max Pooling** slides a $2 \times 2$ window with stride 2 across the feature map and selects only the **maximum value**:

```
Input Feature Map (4x4):                     Max-Pooled Output (2x2):
┌─────┬─────┬─────┬─────┐                     ┌─────┬─────┐
│  1  │  3  │  2  │  1  │                     │  6  │  8  │
├─────┼─────┼─────┼─────┤                     ├─────┼─────┤
│  6  │  2  │  8  │  4  │       ──────►       │  9  │  7  │
├─────┼─────┼─────┼─────┤                     └─────┴─────┘
│  5  │  9  │  3  │  1  │
├─────┼─────┼─────┼─────┤
│  2  │  4  │  7  │  2  │
└─────┴─────┴─────┴─────┘
```

### Why Max Pooling is Essential:
1. **Reduces Computation**: Shrinks height and width by half ($28 \times 28 \to 14 \times 14$), reducing the number of activations by **75%**.
2. **Translation Tolerance**: If a feature moves slightly by 1 pixel, the maximum in the $2 \times 2$ window remains unchanged.
3. **Expands the Receptive Field**: Subsequent layers see a much larger patch of the original image!

---

## 3. High Performance in Pure NumPy: The `im2col` Engine

If you write 2D convolution using raw Python `for` loops:
```python
for n in range(N):
    for c in range(C):
        for h in range(H_out):
            for w in range(W_out):
                # Elementwise multiplication ...
```
Processing one epoch would take **30 minutes** in Python!

### The Secret: `im2col` (Image to Column)
Every modern deep learning library (cuDNN, PyTorch, Caffe) implements convolutions by rearranging image patches into columns of a 2D matrix:

```
4D Input Tensor           Vectorized `im2col`          2D Matrix Multiplication
 (N, C, H, W)     ──►     Matrix X_col         ──►     Out = W_row @ X_col + b
                          Shape: (K*K*C, N*H*W)        Shape: (Out_C, N*H*W)
```

1. **Forward Pass**: Transforming sliding patches into a 2D column matrix turns the entire convolution into **a single BLAS matrix multiplication** (`W_row @ X_col`). In NumPy, this runs in **under 18 milliseconds per batch!**
2. **Backward Pass**: Gradients flow backward effortlessly:
   $$\nabla_W = d_{\text{out}} \cdot X_{\text{col}}^T, \quad \nabla_{X_{\text{col}}} = W_{\text{row}}^T \cdot d_{\text{out}}$$
   Then, `col2im` maps the column gradients back to the original 4D tensor shape $(N, C, H, W)$.

---

## 4. Our Architecture: The 2-Stage ConvNet

```
INPUT: (N, 1, 28, 28)
  │
  ▼
[ Conv2D: 1 -> 8 filters, 3x3, pad 1 ]  ──►  Shape: (N, 8, 28, 28)
[ ReLU Activation ]
[ MaxPool2D: 2x2, stride 2 ]            ──►  Shape: (N, 8, 14, 14)
  │
  ▼
[ Conv2D: 8 -> 16 filters, 3x3, pad 1 ] ──►  Shape: (N, 16, 14, 14)
[ ReLU Activation ]
[ MaxPool2D: 2x2, stride 2 ]            ──►  Shape: (N, 16, 7, 7)
  │
  ▼
[ Flatten ]                             ──►  Shape: (N, 784)
[ Dense Layer: 784 -> 64 ] + ReLU
[ Output Layer: 64 -> 10 ] + Softmax    ──►  Shape: (N, 10)
```

---

## 5. Experimental Results on Fashion-MNIST

We train our pure NumPy ConvNet on Fashion-MNIST for 5 epochs using the **Adam optimizer** ($\alpha = 0.002$).

### A. Learning Dynamics Over Time

![CNN Training Curves](plots/cnn_training_curves.png)

* **Rapid Optimization**: Within 3 epochs, training accuracy climbs to >90%, and validation loss descends smoothly.
* **Final Test Performance**:
  * **Test Accuracy**: **88.62%**
  * **Test Loss**: **0.3263**
  * Noticeable improvement over flat MLPs, particularly on structurally complex clothing.

---

### B. What Does the CNN Actually See? (Learned Feature Maps)

Passing an unseen test garment through the trained ConvNet lets us visualize what the filters in Conv1 and Conv2 have learned to detect:

![CNN Feature Maps](plots/cnn_feature_maps.png)

1. **Layer 1 (Conv1 — 8 channels, 28x28)**:
   * Acts as **edge and texture extractors**: distinct filters highlight vertical seams, horizontal hems, neck openings, and garment outlines.
2. **Layer 2 (Conv2 — 16 channels, 14x14)**:
   * Combines Layer 1 edges into **higher-level silhouettes and composite parts**: isolating sleeves, waistbands, and internal fabric density.

---

### C. Confusion Matrix: Did Convolutions Fix Misclassifications?

![CNN Confusion Matrix](plots/cnn_confusion_matrix.png)

Comparing the CNN confusion matrix to Chapter 2's flat MLP reveals clear visual improvements:
* **Trousers (Class 1) & Bags (Class 8)**: Remain near-perfect (>97% accuracy).
* **Footwear (Sandals vs Sneakers vs Ankle Boots)**: Distinct sole lines and ankle heights are cleanly separated by horizontal and diagonal edge filters.
* **T-shirts vs Shirts vs Coats**: Spatial awareness helps the network distinguish collars and open front coat seams much more reliably than a flattened MLP!

---

## 6. How to Run This Chapter

From the root of the repository:

```bash
# 1. Run the Chapter 3 ConvNet training script
python chapters/03-convolutional-neural-networks/train.py

# 2. Run unit tests for Conv2D, MaxPool2D, and ConvNet
pytest tests/test_chapter3.py

# 3. Launch the interactive Jupyter notebook
jupyter lab chapters/03-convolutional-neural-networks/notebook.ipynb
```
