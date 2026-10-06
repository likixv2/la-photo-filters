# Linear Algebra Mini Project
## Matrix Multiplication, Inversion, and Photo Filters

**Run:** `python photo_filters_la.py` (built-in demo photo) or `python photo_filters_la.py my_photo.jpg`
Needs only `numpy`, `matplotlib`, `Pillow`. Output goes to the console (saved in `run_output.txt`) and three figures.

## One-line story
A photo is a matrix. A photo filter is a matrix that multiplies it. Undoing a filter means inverting that matrix, and whether that works depends on the matrix's rank, its eigenvalues and its condition number.

## Workflow followed (matches the course diagram)

| Step | Template stage | What the code does | Section in output |
|---|---|---|---|
| 1 | Real-world data -> Matrix representation | 128x128 photo -> R, G, B matrices and a grayscale matrix X. Filters built as matrices (blur, sharpen, flip, edge, colour 3x3). | STEP 1 |
| 2 | Matrix simplification (RREF / LU) | RREF and LU written from scratch, tested on a 6x6 blur and a rank-deficient downsample matrix. | STEP 2 |
| 3 | Structure of the space | Rank and nullity of every filter and of the image. | STEP 3 |
| 4 | Remove redundancy | RREF pivot columns of X = a basis; effective rank. | STEP 4 |
| 5 | Orthogonalization | Gram-Schmidt on image columns -> Q, R; checks Q^T Q = I, X = QR. | STEP 5 |
| 6 | (Core topic) Inversion | Undo sharpen, warm tone, flip; grayscale fails; blur is undone by an LU solve. | STEP 6 |
| 7 | Prediction / approximation (least squares) | Normal equations (B^T B + lambda I) x = B^T y to deblur a noisy photo. | STEP 7 |
| 8 | Projection | Project onto top-k eigen-directions -> compression. | STEP 8 |
| 9 | Pattern discovery + diagonalization | Eigen-decomposition B = P D P^T, filter powers, eigenvalues of X^T X. | STEP 9 |
| 10 | Final output | Noise reduction by projection, summary table, figures. | STEP 10 |

## Key results (from `run_output.txt`)

| Experiment | Result |
|---|---|
| Blurred photo vs original | 29.9 dB |
| Exact inverse (LU solve), no noise | 109.9 dB (essentially perfect) |
| Exact inverse after adding tiny noise (sigma 0.002) | 5.0 dB (noise explodes) |
| Least-squares deblur (lambda = 1e-2) | 32.4 dB |
| Sharpen, then S^-1 / warm tone, then W^-1 | recovered to ~1e-15 |
| Grayscale matrix | rank 1, **cannot be inverted** |
| Projection compression k = 5 / 10 / 20 | 12.8x / 6.4x / 3.2x smaller; 26.0 / 30.4 / 35.8 dB |
| Noise reduction (sigma 0.10) | 20.3 dB -> 25.8 dB using top 8 eigen-directions |

Figures: `fig1_filters.png` (all filters), `fig2_inversion.png` (inversion and least squares), `fig3_eigen_projection.png` (eigenvalues, compression, denoising).

## Viva answers (Concept -> Purpose -> Outcome)

**What does your data matrix represent?**
Entry X[i,j] is the brightness of the pixel in row i, column j (0 black, 1 white). For colour, three such matrices (R, G, B).

**Matrix multiplication**
- Concept: Y = L X R^T. Left multiplication mixes pixels within each column, right multiplication mixes them within each row. A 3x3 matrix multiplies each pixel's (R,G,B) vector.
- Purpose: every linear photo filter (blur, sharpen, flip, brightness, warm tone, grayscale) can be written this way.
- Outcome: Fig 1 shows 10 filters, all made by matrix products.

**RREF / LU**
- Concept: Gauss-Jordan elimination reduces a matrix to RREF; LU splits PA = LU into triangular factors.
- Purpose: RREF shows pivots (so rank and invertibility); LU solves A x = b cheaply.
- Outcome: RREF of the blur matrix is the identity (invertible); RREF of the downsample matrix has 3 pivots for 6 columns (not invertible). LU is used to deblur.

**Rank, nullity, column space**
- Concept: rank = number of independent columns; nullity = columns - rank.
- Purpose: nullity 0 means no information lost, so an inverse exists.
- Outcome: blur, sharpen, flip, warm tone have nullity 0; grayscale has nullity 2 and the edge filter nullity 1, so they cannot be undone.

**Linear independence and basis**
- Concept: pivot columns of RREF form a basis of the column space.
- Purpose: find how much of the image is redundant.
- Outcome: the 128 columns of the photo span only 40 dimensions, and 4 directions already hold 99% of its energy.

**Gram-Schmidt**
- Concept: turn independent vectors into orthonormal ones.
- Purpose: orthonormal bases make projections simple (P = Q Q^T) and inverses simple (Q^-1 = Q^T).
- Outcome: Q^T Q = I to 1e-14 and X = QR. The flip matrix J is orthogonal, so J^-1 = J^T.

**Inversion**
- Concept: A^-1 undoes A when A is square with full rank.
- Purpose: recover the original photo from a filtered one.
- Outcome: sharpen, warm tone and flip are undone exactly; blur is undone to 110 dB; grayscale fails (singular). Blur's condition number is 3e11, so inversion is fragile.

**Least squares**
- Concept: minimise ||B x - y||^2 + lambda ||x||^2 using the normal equations (B^T B + lambda I) x = B^T y.
- Purpose: the exact inverse amplifies noise, so we want the best stable approximation.
- Outcome: naive inverse of the noisy blurred photo gives 5 dB (garbage); least squares gives 32.4 dB.
- Honest note: lambda was chosen by comparing with the original photo (a demo shortcut). In real use you would pick it with a validation set or an L-curve.

**Projection**
- Concept: orthogonal projection P = V_k V_k^T onto the span of the k most important eigenvectors; P^2 = P.
- Purpose: keep the main patterns, discard the rest.
- Outcome: compression (store k(2n) numbers per channel instead of n^2) and noise reduction (20.3 -> 25.8 dB). Noise spreads over all directions while the photo lives in a few.

**Eigenvalues, eigenvectors, diagonalization**
- Concept: the blur matrix is symmetric, so B = P D P^T with orthogonal P.
- Purpose: powers and inverses of B become powers and reciprocals of eigenvalues, and the eigenvalues show what the filter does to each pattern.
- Outcome: the smallest eigenvalue is 1.5e-4 (zig-zag patterns are almost erased), so un-blurring multiplies those components by a huge factor. That is exactly why noise explodes. For the photo, the first eigenvector of X^T X holds 97.4% of the energy.

**How do the steps connect?**
Data matrix -> filters as matrices -> RREF/LU tell us which are invertible and give a solver -> rank/nullity explain why -> orthogonal bases -> inversion works for well-conditioned matrices -> eigenvalues explain why blur is fragile -> least squares repairs it -> projection gives compression and denoising.

## Demo checklist (5 marks)
1. Run the script; show the console output step by step and say what each STEP means (the MEANING lines help).
2. Open the three figures.
3. Optionally run it with your own photo.
