# Replicating "Symmetry in Language Statistics Shapes the Geometry of Model Representations"

## A Didactic Notebook Guide

**Paper:** Karkada, Korchinski, Nava, Wyart & Bahri (2026). arXiv: 2602.15029

**Purpose:** This document is a detailed guide for building a Jupyter notebook that replicates the core empirical findings of the paper above, working directly with corpus statistics rather than training any models. The notebook should serve as a self-contained case study in how statistical symmetries in natural language give rise to geometric structure in representations.

**Scope:** We are NOT training word2vec, GloVe, or any neural model. We are NOT extracting embeddings from LLMs. We are working purely with co-occurrence counts from a text corpus, computing matrices from those counts, and analyzing their spectral properties. The claim we are testing is that the geometric structures people observe in learned representations (months forming circles, etc.) are already present in the raw statistics of language, before any model is trained.

---

## Part 1: Background and Motivation

### 1.1 What the paper claims

People have repeatedly observed that when you extract representations of calendar months from language models (word2vec, GloVe, BERT, GPT, etc.) and run PCA, the 12 months arrange themselves in a circle. Similarly, historical years form a smooth curve, and city representations encode latitude and longitude linearly. These observations have been treated as surprising or mysterious emergent properties of neural networks.

This paper argues there is nothing mysterious about it. The geometric structure is a direct, mathematically inevitable consequence of a symmetry in how words co-occur in natural text. Specifically:

1. The co-occurrence statistics of months exhibit **translation symmetry**: the probability that month i and month j appear near each other in text depends primarily on the temporal distance |i - j| (mod 12) between them, not on their absolute identities.
2. Translation symmetry makes the co-occurrence matrix approximately **circulant** (each row is a cyclic shift of the previous row).
3. Circulant matrices are diagonalized by the **Discrete Fourier Transform** (DFT), meaning their eigenvectors are sinusoidal.
4. Language models learn representations that approximate the top eigenvectors of a normalized version of this co-occurrence matrix.
5. The top eigenvectors of a circulant matrix are the lowest-frequency Fourier modes, which trace a **circle** when plotted in 2D.

Therefore: circle in data statistics → circle in eigenvectors → circle in learned representations. No mystery.

### 1.2 Why this matters for mechanistic interpretability

This is one of the rare cases where we can derive the geometry of internal representations from first principles. If you are building interpretability tools (linear probes, SAE feature analysis, activation steering), this gives you ground truth: you know what the representation *should* look like and why, so you can test whether your tools recover that structure correctly.

It also raises an important question: if representational geometry is largely determined by co-occurrence statistics, what does that imply about the features that sparse autoencoders discover? Are SAE features tracking statistical structure in the training corpus, or something deeper?

---

## Part 2: Key Quantities and How to Compute Them

### 2.1 The co-occurrence count matrix C

**What it is:** A matrix where entry C(i, j) counts how many times word i and word j appeared within the same text window across the entire corpus.

**How to compute it:**

1. Choose a window size w (e.g., w = 10 words).
2. Slide a window of size w across the corpus, one word at a time.
3. For each window position, look at all the words in the window. For every pair of distinct words (i, j) that both appear in the window, increment C(i, j) by 1.
4. Since co-occurrence is symmetric (if i is near j, then j is near i), the matrix C is symmetric: C(i, j) = C(j, i).

**Important:** C should be computed over the FULL vocabulary, not just the 12 months. This is crucial for later steps. So if the corpus has V unique words, C is a V x V matrix. In practice, you will want to restrict to a reasonable vocabulary size (e.g., the top 20,000-50,000 most frequent words) both for memory reasons and because very rare words have unreliable statistics.

**For Exercise 1 only**, we will look at the 12x12 submatrix of C restricted to month names, to visually check for circulant structure. But the full matrix is needed for Exercise 2.

### 2.2 The probabilities P(i) and P(i,j)

These are derived from the count matrix C and from word frequency counts.

**P(i,j)** is the probability that a randomly chosen word pair from the corpus is the pair (i, j). It is computed as:

```
P(i, j) = C(i, j) / (sum of all entries in C)
```

Here, "sum of all entries in C" is the total number of word pairs observed across all windows. This is a single number for the entire corpus.

**P(i)** is the marginal probability of word i. There are two equivalent ways to think about it:

- **From the co-occurrence matrix:** P(i) = (sum of row i of C) / (sum of all entries in C). This is the fraction of all co-occurrence events that involve word i.
- **From unigram counts:** P(i) = (number of times word i appears in the corpus) / (total number of words in the corpus). This is simpler to compute and is approximately proportional to the marginal of C.

Both definitions give the same M* matrix up to constant factors that cancel out. In practice, using unigram counts is easier because you don't need to sum over the full C matrix.

**Why we need P(i):** Raw co-occurrence counts are dominated by common words. "The" and "January" will co-occur a lot, not because they have a meaningful relationship, but because "the" appears everywhere. P(i) lets us control for this: we divide out the baseline expectation.

### 2.3 The normalized co-occurrence matrix M*

**What it is:** The matrix that measures how much more (or less) two words co-occur than you would expect if they were statistically independent.

**How to compute it:**

```
M*(i, j) = P(i, j) / (P(i) * P(j))
```

**Interpretation:**

- M*(i, j) = 1 means words i and j co-occur exactly as often as chance predicts.
- M*(i, j) > 1 means they co-occur MORE than expected (they are positively associated).
- M*(i, j) < 1 means they co-occur LESS than expected (they tend to avoid each other).

**Relationship to PMI:** The pointwise mutual information (PMI) is log(M*). The paper works with M* directly (the unlogged version), but both contain the same information. Some implementations use log(M*) or variants like PPMI (positive PMI, where negative values are clamped to 0). For this notebook, we will use M* directly.

**Size of M*:** It is the same size as C: V x V, where V is the vocabulary size. Each row of M* is a V-dimensional vector representing one word's normalized co-occurrence profile.

### 2.4 From M* to geometry: extracting month vectors

For the month geometry analysis, we do the following:

1. Compute the full V x V matrix M* (or at least the 12 rows of M* corresponding to the 12 months).
2. Extract the 12 rows corresponding to January through December. Each row is a V-dimensional vector.
3. These 12 vectors are our "month representations." They live in V-dimensional space, but we want to visualize them in 2D.
4. Run PCA on these 12 vectors (mean-center them, compute the 12x12 covariance matrix, find its eigenvectors).
5. Project the 12 vectors onto the top 2 principal components.
6. Plot. The prediction is a circle.

**Why PCA on 12 vectors in V dimensions reduces to a 12x12 problem:** PCA finds the directions of maximum variance. With only 12 data points, there can be at most 11 non-zero principal components (12 minus 1 for mean-centering). So regardless of how large V is, the problem is effectively 11-dimensional. The 12x12 covariance matrix (how similar is each month's profile to each other month's profile) determines everything. And the claim is that this 12x12 covariance matrix inherits circulant structure from the translation symmetry.

---

## Part 3: The Three Exercises

### Exercise 1: Verifying Translation Symmetry in Co-occurrence

**Goal:** Empirically confirm that month-month co-occurrence statistics exhibit translation symmetry, meaning the co-occurrence of month i and month j depends primarily on |i - j| (mod 12).

**Steps:**

1. **Get a corpus.** Wikipedia dumps work well. You want a reasonably large corpus (at least tens of millions of words) so that the co-occurrence counts for months are not too sparse. A preprocessed Wikipedia dump is ideal. Alternatively, a subset works if it is large enough.

2. **Preprocess.** Lowercase everything. Handle tokenization simply (split on whitespace and punctuation). You need to be able to identify the 12 month names: january, february, march, april, may, june, july, august, september, october, november, december. Note: "may" is ambiguous (it is also a common verb and a name). Consider handling this by only counting "may" when it is capitalized, or by flagging this ambiguity and noting it. The paper discusses this ambiguity explicitly.

3. **Build the 12x12 month co-occurrence matrix.** Choose a window size (start with w = 10). Slide the window across the corpus. For each window, if two different month names both appear, increment their count. The result is a 12x12 symmetric matrix C_months.

4. **Visualize as heatmap.** Plot C_months as a heatmap with months on both axes (in calendar order: Jan, Feb, ..., Dec). If translation symmetry holds, the heatmap should show a band-diagonal structure: high values along the main diagonal (months close together co-occur frequently), falling off symmetrically as you move away from the diagonal, and with wraparound at the edges (December and January should have high co-occurrence).

5. **Quantify circulant structure.** For a perfectly circulant matrix, the entry C(i,j) depends only on (i-j) mod 12. So group all entries by their distance d = |i-j| mod 12 (there are 7 unique distances for n=12: 0, 1, 2, 3, 4, 5, 6) and check: is the variance within each distance group small compared to the variance between groups? You can plot the average co-occurrence as a function of distance d, with error bars showing the spread. A tight plot with small error bars means strong circulant structure.

6. **Negative control.** Repeat the analysis for a set of 12 words that do NOT have periodic structure (e.g., 12 common animal names, or 12 color words). The prediction is that these will NOT show circulant structure.

**Expected result:** The month co-occurrence matrix should be approximately circulant, with co-occurrence highest for adjacent months and falling off with distance. The negative control should show no such pattern.

**Things to look out for:**
- "May" will be noisy because of its non-month uses. This is expected and worth noting.
- The symmetry will not be perfect. Some months have stronger associations than others (e.g., November-December because of holidays). The claim is that it is approximately circulant, not exactly.
- Window size affects the result. Smaller windows capture more syntactic relationships; larger windows capture more topical/semantic ones. Try w = 5, 10, 20 and compare.

### Exercise 2: From Statistics to Geometry

**Goal:** Show that the translation symmetry in co-occurrence statistics produces circular geometry when you analyze the eigenvectors, WITHOUT training any model.

**Steps:**

1. **Build the full co-occurrence matrix.** This is the computationally expensive step. You need C(i,j) for all word pairs in the vocabulary, not just months. To keep this manageable:
   - Restrict to a vocabulary of the top ~10,000-30,000 most frequent words.
   - Use sparse matrix representations (most word pairs never co-occur, so C is very sparse).
   - Same window size as Exercise 1.

2. **Compute M*.** For each entry:
   ```
   M*(i,j) = C(i,j) / (sum of all C) / (P(i) * P(j))
   ```
   Or equivalently:
   ```
   M*(i,j) = C(i,j) * N / (count(i) * count(j))
   ```
   where N is the total number of co-occurrence pairs and count(i) is the unigram count of word i. The second form avoids computing tiny probabilities.

   In practice, for memory reasons, you may not want to materialize the full V x V M* matrix. Instead, just compute the 12 rows of M* corresponding to the 12 months. Each row is a V-dimensional vector. You only need these 12 vectors for PCA.

3. **PCA on month vectors.** Take the 12 month vectors (each V-dimensional), mean-center them, compute the 12x12 covariance matrix, find its eigenvectors and eigenvalues. Project the 12 vectors onto the top 2 eigenvectors. Plot.

4. **Check for circularity.** The plot should show the 12 months arranged approximately in a circle, in calendar order. January should be next to February and December. July should be opposite January.

5. **The Fourier comparison (the interesting part).** This is where we connect the circle to Fourier analysis:

   a. Take the 12x12 submatrix of M* (just the month-month entries). Call it M_sub.
   
   b. If M_sub is circulant, it is fully determined by its first row (call it m, a vector of length 12).
   
   c. Compute the DFT of m. The DFT coefficients are the eigenvalues of M_sub, and the DFT basis vectors (sines and cosines at frequencies k = 0, 1, 2, ..., 6) are the eigenvectors.
   
   d. The predicted eigenvalues from the DFT should approximate the actual eigenvalues from PCA in step 3. Plot them side by side.
   
   e. The predicted eigenvectors from the DFT should approximate the actual principal components. The k=1 Fourier mode (one full cycle of sine/cosine over the 12 months) should correspond to the top principal component, which is the circle.
   
   f. **Why the k=1 mode gives a circle:** The k=1 sine and cosine evaluate at month i as sin(2*pi*i/12) and cos(2*pi*i/12). If you plot (cos(2*pi*i/12), sin(2*pi*i/12)) for i = 0, 1, ..., 11, you get 12 points equally spaced on a unit circle. The PCA projection onto the top 2 components should look like this, possibly rotated or scaled.
   
   g. **Higher harmonics:** The k=2 mode oscillates twice around the year; the k=3 mode oscillates three times; and so on. Each higher harmonic adds "ripples" to the circle. If you project onto 4 dimensions (keeping k=1 and k=2), the manifold becomes a circle with 2 bumps. You can visualize this by plotting pairs of principal components (PC1 vs PC2, PC3 vs PC4, etc.) and checking that each pair traces a circle at a different frequency.

6. **Vary the embedding dimension.** Repeat the PCA projection at d = 2, 4, 6, 8. At d = 2 you see a clean circle (just the fundamental mode). At higher d you include more harmonics. Compute the reconstruction error at each d: how well do the d-dimensional representations capture the structure of the full M* month vectors? The paper predicts that error decreases as d increases, with each step adding one Fourier pair.

**Expected result:** A clear circle in the top 2 PCA components. The DFT eigenvalues should approximately match the PCA eigenvalues. The k=1 Fourier mode should dominate (largest eigenvalue), producing the circle.

**Detailed note on the Fourier argument (for deeper understanding):**

A circulant matrix has a special algebraic property: it commutes with the cyclic permutation matrix P (the matrix that shifts every row down by one and wraps the last row to the top). Any matrix that commutes with P is diagonalized by the eigenvectors of P. The eigenvectors of P are the complex exponentials exp(2*pi*i*k*n/N) for k = 0, 1, ..., N-1, which are exactly the DFT basis. In real terms, these decompose into cosine and sine pairs at each frequency k.

The eigenvalue corresponding to frequency k is the k-th DFT coefficient of the first row of the circulant matrix. For co-occurrence data, the first row encodes how co-occurrence falls off with temporal distance. This falloff is typically smooth and monotonically decreasing (nearby months co-occur more than distant months), which means the DFT is dominated by low frequencies. The k=0 mode (constant offset) is typically removed by mean-centering in PCA. The k=1 mode (one full oscillation) has the largest eigenvalue among the remaining modes. This is why the circle dominates.

To make this concrete in the notebook: compute the first row of the (approximately) circulant M_sub, take its FFT, and plot the power spectrum (|DFT coefficient|^2 vs frequency k). You should see the power concentrated at low frequencies, with k=1 being the largest.

### Exercise 3: Robustness to Perturbation

**Goal:** Show that the circular geometry survives even when you remove all direct month-month co-occurrences from the corpus.

**Steps:**

1. **Create a filtered corpus.** Go through the original corpus sentence by sentence (or paragraph by paragraph). Remove any sentence that contains two or more month names. Keep sentences with zero or one month names.

2. **Rebuild C from the filtered corpus.** Use the same window size as before. Now C_months (the 12x12 month-month submatrix) should be all zeros or near-zero on the off-diagonal, because you have removed all sentences where two months co-occur.

3. **Compute M* from the filtered corpus** using the full vocabulary, and extract the 12 month rows as before.

4. **Run PCA on the 12 month vectors.** Project to 2D and plot.

**Expected result:** The circle should still appear, though possibly with more noise than in Exercise 2. The reason is that each month still co-occurs with many non-month words (January co-occurs with "winter," "cold," "new year"; February co-occurs with "valentine," "snow," etc.), and the translation symmetry is carried by these indirect relationships. January's profile across all 50,000 vocabulary words is still a shifted version of February's profile, even though the direct January-February co-occurrence has been zeroed out.

**What this tells us:** The circular geometry is not driven by the literal fact that months appear near each other in text. It is driven by the deeper pattern that each month's relationship to the rest of the vocabulary is a rotated version of every other month's relationship. Removing direct month-month co-occurrences removes the most obvious evidence of the symmetry, but the indirect evidence through shared context words is sufficient.

**Additional analysis:** Compare the eigenvalue spectrum from Exercise 2 and Exercise 3. The eigenvalues should be smaller in Exercise 3 (less signal), but the relative ranking should be preserved (k=1 still dominates). This shows the circulant structure is attenuated but not destroyed.

---

## Part 4: Practical Notes for Implementation

### 4.1 Corpus choice

Wikipedia is ideal because it is large, diverse, and freely available. You can use a preprocessed dump. Even a 10% sample of English Wikipedia should have enough month mentions for stable statistics. The full English Wikipedia is about 4 billion words, which is plenty.

If Wikipedia is too large to process locally, consider using a pre-built co-occurrence matrix. The GloVe project released co-occurrence matrices computed from various corpora. Alternatively, you could use a smaller corpus like WikiText-103 (~100M words) with the caveat that month co-occurrence counts may be noisier.

### 4.2 Handling "May"

"May" is ambiguous: it is a month, a modal verb ("you may go"), and a proper name. This is a known issue that the paper discusses explicitly. Options:

- **Simplest:** Include all occurrences of "may" (lowercase). This adds noise to the May row but the signal should still come through.
- **Better:** Only count "May" when it is capitalized and appears in contexts suggesting a month (e.g., near other months, dates, or temporal expressions). Even just requiring capitalization helps.
- **Best for didactic purposes:** Run the analysis both ways and show the difference. The paper actually uses the ambiguity of "May" as an interesting case study for how LLMs disambiguate in context.

### 4.3 Sparse matrices

The full V x V co-occurrence matrix C is very sparse (most word pairs never co-occur). Use scipy.sparse. You can build C as a dictionary of keys (DOK) matrix or coordinate (COO) matrix during counting, then convert to CSR for efficient row slicing.

You do NOT need to materialize the full M* matrix. To get the 12 month rows of M*, just take the 12 month rows of C and divide each entry C(month_i, j) by (count(month_i) * count(j) / N), where N is the total co-occurrence count. This can be done row by row without ever building the full V x V M*.

### 4.4 Computational cost

- Exercise 1 is cheap: you only need the 12x12 submatrix of C.
- Exercise 2 requires one pass through the corpus to build the full C (or at least the 12 rows of C corresponding to months, plus unigram counts for all words). This is the expensive step. For a corpus of 100M words with vocabulary 30,000, the pass takes minutes, not hours.
- Exercise 3 requires preprocessing the corpus to filter sentences, then repeating Exercise 2.

### 4.5 Visualization suggestions

- **Exercise 1:** Heatmap of the 12x12 C_months matrix. Line plot of average co-occurrence vs. distance d. Bar plot comparing within-distance variance to between-distance variance.
- **Exercise 2:** Scatter plot of months in PC1-PC2 space, labeled with month names, with lines connecting adjacent months to show the circular ordering. Stem plot of eigenvalue spectrum. Bar plot comparing DFT-predicted eigenvalues vs. PCA eigenvalues. Power spectrum of the first row of M_sub.
- **Exercise 3:** Same scatter plot as Exercise 2, overlaid or side by side, to show the circle persists. Eigenvalue comparison between perturbed and unperturbed cases.

### 4.6 Window size exploration

The paper doesn't commit to a single window size. It's worth trying a few (w = 5, 10, 20, 50) and showing how the results change. Smaller windows capture more local syntactic patterns; larger windows capture more topical patterns. The circulant structure should be present at all reasonable window sizes, but the signal-to-noise ratio may vary. This is a good thing to include in the notebook as an exploratory section.

---

## Part 5: Notebook Structure

The notebook should be organized as follows. Sections marked [SKELETON] should be pre-built with boilerplate code (data loading, plotting setup, etc.). Sections marked [FILL IN] should have clear instructions and empty cells for the user to implement the core logic.

### Section 0: Setup and Data Loading [SKELETON]
- Install/import dependencies (numpy, scipy, matplotlib, collections, etc.)
- Download or load a corpus (provide a helper function)
- Basic preprocessing (tokenization, lowercasing)
- Define the list of 12 month names
- Provide a helper function that yields sentences or windows from the corpus

### Section 1: Exercise 1 - Verifying Translation Symmetry
- **1.1 Build the 12x12 month co-occurrence matrix** [FILL IN]
  - Iterate over windows, count month-month co-occurrences
  - Store in a 12x12 numpy array
- **1.2 Visualize the matrix** [SKELETON - plotting code provided]
  - Heatmap with month labels
- **1.3 Quantify circulant structure** [FILL IN]
  - Group entries by distance d = |i-j| mod 12
  - Compute mean and variance within each group
  - Measure how much of the total variance is explained by distance alone
- **1.4 Negative control** [FILL IN]
  - Repeat with a set of non-periodic words (e.g., animals)
  - Show the contrast
- **1.5 Discussion cell** [SKELETON - markdown cell with guiding questions]
  - How close to circulant is the matrix?
  - Which month pairs deviate most from the circulant prediction, and can you explain why?
  - How does "May" behave?

### Section 2: Exercise 2 - From Statistics to Geometry
- **2.1 Build the full co-occurrence matrix** [FILL IN]
  - Iterate over windows, count all word-word co-occurrences
  - Use sparse matrix representation
  - Also compute unigram counts
- **2.2 Compute M* for the 12 months** [FILL IN]
  - Extract the 12 month rows from C
  - Normalize by P(i) * P(j) to get M*
  - Result: 12 vectors, each of dimension V
- **2.3 PCA on month vectors** [FILL IN]
  - Mean-center the 12 vectors
  - Compute the 12x12 covariance matrix
  - Eigendecompose
  - Project onto top 2 components
- **2.4 Plot the circle** [SKELETON - plotting code provided]
  - Scatter plot with month labels
  - Connect adjacent months with lines
- **2.5 The Fourier comparison** [FILL IN]
  - Extract the 12x12 submatrix of M*
  - Average entries at each distance to get the "circulant approximation" (first row of the best circulant fit)
  - Compute the DFT of this first row
  - Compare DFT coefficients (predicted eigenvalues) to actual PCA eigenvalues
  - Plot comparison
- **2.6 Visualize higher harmonics** [FILL IN]
  - Plot PC3 vs PC4 (should show k=2, a double loop)
  - Plot the eigenvalue spectrum (should decay with frequency)
  - Show how reconstruction error decreases as embedding dimension increases
- **2.7 Discussion cell** [SKELETON]
  - How well do the DFT eigenvalues match the PCA eigenvalues?
  - Which months deviate most from the predicted circle?
  - What does the eigenvalue spectrum tell you about how many "features" are needed to represent months?

### Section 3: Exercise 3 - Robustness to Perturbation
- **3.1 Filter the corpus** [FILL IN]
  - Remove all sentences containing two or more month names
  - Count how many sentences were removed
- **3.2 Rebuild M* from filtered corpus** [FILL IN]
  - Same procedure as Exercise 2
- **3.3 PCA and plot** [FILL IN]
  - Same PCA procedure
  - Overlay or compare with Exercise 2 results
- **3.4 Eigenvalue comparison** [FILL IN]
  - Compare eigenvalue spectra: original vs. perturbed
  - Show the circulant structure is attenuated but preserved
- **3.5 Discussion cell** [SKELETON]
  - Why does the circle survive?
  - What carries the translation symmetry if not direct month-month co-occurrence?
  - Can you identify specific non-month words that are most responsible for preserving the structure? (e.g., look at which vocabulary dimensions contribute most to the top PCA component)

### Section 4: Toy Illustration of the Fourier Argument [SKELETON + FILL IN]
This section is a standalone mathematical illustration, independent of the corpus.

- **4.1 Construct a synthetic circulant matrix** [SKELETON]
  - Define a circulant matrix by hand: first row = [10, 8, 5, 2, 1, 0.5, 0.5, 0.5, 1, 2, 5, 8] (or similar: high for nearby months, low for distant months)
  - Visualize as heatmap
- **4.2 Diagonalize it two ways** [FILL IN]
  - Method 1: numpy eigendecomposition
  - Method 2: DFT of the first row (numpy.fft.fft)
  - Show they give the same eigenvalues (up to ordering)
  - Show the eigenvectors are sinusoidal
- **4.3 Visualize the eigenvectors as functions** [FILL IN]
  - Plot each eigenvector as a function of month index (0 to 11)
  - Show that they are sines and cosines at different frequencies
  - Label the frequencies k = 0, 1, 2, ..., 6
- **4.4 Show how k=1 gives a circle** [FILL IN]
  - Take the k=1 sine and cosine eigenvectors
  - Plot (cos(2*pi*i/12), sin(2*pi*i/12)) for i = 0, ..., 11
  - This is the predicted circle
- **4.5 Show how adding harmonics changes the shape** [FILL IN]
  - Start with just k=1 (clean circle)
  - Add k=2 (circle with slight perturbation in 4D, project back to various 2D planes)
  - Add k=3, etc.
  - Show that the manifold gets more complex but the circle dominates
- **4.6 Perturb the circulant matrix and show robustness** [FILL IN]
  - Add random noise to the circulant matrix
  - Show that the top eigenvectors are still approximately sinusoidal
  - Show that the circle in PCA is preserved up to some noise level
  - Explore: how much noise can you add before the circle breaks?

---

## Part 6: Mathematical Details for Reference

### The DFT and Circulant Matrices

A circulant N x N matrix C is defined by a vector c = (c_0, c_1, ..., c_{N-1}):

```
C(i, j) = c_{(j - i) mod N}
```

The eigenvectors of C are the columns of the DFT matrix F:

```
F(n, k) = (1/sqrt(N)) * exp(2*pi*i*n*k/N)
```

The eigenvalue corresponding to eigenvector k is:

```
lambda_k = sum_{n=0}^{N-1} c_n * exp(-2*pi*i*n*k/N) = DFT(c)[k]
```

In real terms (since C is real and symmetric), the eigenvectors come in cosine/sine pairs:

```
v_k^cos(n) = cos(2*pi*k*n/N)
v_k^sin(n) = sin(2*pi*k*n/N)
```

For N = 12, the frequencies are k = 0, 1, 2, 3, 4, 5, 6. The k=0 mode is a constant (removed by mean-centering). The k=6 mode is alternating (+1, -1, +1, ...). The k=1 mode oscillates once around the year and produces the circle.

### Why the k=1 eigenvalue is largest

The first row c of the co-occurrence circulant encodes how co-occurrence depends on temporal distance. In natural language, nearby months co-occur more than distant months, so c is a smooth, peaked function. The DFT of a smooth peaked function has most of its energy at low frequencies. Since k=0 is removed by mean-centering, k=1 dominates.

More precisely: if c is a monotonically decreasing function of distance (which it roughly is for months), then the DFT magnitudes |lambda_k| decrease with k. This is a standard property of the Fourier transform: smooth signals have rapidly decaying Fourier coefficients.

### The PMI connection

The log of M* is the pointwise mutual information matrix:

```
PMI(i, j) = log(P(i,j) / (P(i) * P(j))) = log(M*(i, j))
```

Word2vec and GloVe implicitly factorize a matrix closely related to PMI. The paper works with M* (the unlogged version) for the theoretical analysis, but the log doesn't change the eigenvector structure qualitatively because log is a monotonic transformation that preserves the ordering and approximate circulant structure.

---

## Part 7: Extensions (for future notebooks)

These are not part of the current notebook but are natural next steps:

1. **Train word2vec on the same corpus** and compare its month embeddings to the PCA of M*. The paper predicts they should match.
2. **Extract month embeddings from an LLM** (e.g., using the representation of "January" at various layers of Llama) and compare to the theoretical prediction.
3. **Repeat for days of the week** (7 words, circulant structure with period 7) and **historical years** (linear rather than periodic symmetry, so you expect a smooth curve rather than a circle).
4. **Repeat for cities** with known latitudes and longitudes, to test the linear probing prediction.
5. **Investigate the "May" disambiguation** across LLM layers, replicating the paper's Figure 13.