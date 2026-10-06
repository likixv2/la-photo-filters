"""
LINEAR ALGEBRA MINI PROJECT
Topic : Matrix multiplication, inversion, and photo filters
Run   : python photo_filters_la.py            (uses a built-in demo photo)
        python photo_filters_la.py my_pic.jpg (uses your own photo, resized to 128x128)

Workflow followed (as required by the course template):
  REAL-WORLD DATA (image)
   -> Matrix Representation        (image = matrix, filter = linear transformation)
   -> Matrix Simplification        (RREF, LU decomposition  -- written from scratch)
   -> Structure of the Space       (column space, rank, nullity)
   -> Remove Redundancy            (linear independence, basis selection)
   -> Orthogonalization            (Gram-Schmidt -> orthonormal basis, QR)
   -> Inversion                    (undo filters with A^-1 via LU)
   -> Prediction / Approximation   (least squares, normal equations)
   -> Projection                   (orthogonal projection onto a subspace)
   -> Pattern Discovery            (eigenvalues / eigenvectors)
   -> System Simplification        (diagonalization of a symmetric matrix)
   -> FINAL OUTPUT: filters, deblurring, compression, noise reduction
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/mnt/user-data/outputs/"
N = 128
np.set_printoptions(precision=3, suppress=True, linewidth=110)
rng = np.random.default_rng(0)


def banner(title):
    print("\n" + "=" * 80 + f"\n{title}\n" + "=" * 80)


def psnr(a, b):
    mse = np.mean((np.clip(a, 0, 1) - np.clip(b, 0, 1)) ** 2)
    return 99.0 if mse < 1e-12 else 10 * np.log10(1.0 / mse)


# ----------------------------------------------------------------------------
# STEP 0 : REAL-WORLD DATA  (a photo)
# ----------------------------------------------------------------------------
def make_demo_photo(n=N):
    """Synthetic landscape photo (sky, sun, hills, house, fence), RGB in [0,1]."""
    y, x = np.mgrid[0:n, 0:n] / (n - 1)
    img = np.zeros((n, n, 3))
    img[..., 0] = 0.35 + 0.40 * y
    img[..., 1] = 0.60 + 0.25 * y
    img[..., 2] = 0.95 - 0.10 * y
    img[(x - 0.78) ** 2 + (y - 0.20) ** 2 < 0.008] = [1.0, 0.9, 0.3]
    ridge = 0.50 + 0.08 * np.sin(6 * x) + 0.04 * np.sin(17 * x + 1)
    hill = y > ridge
    img[hill] = np.array([0.25, 0.50, 0.25]) * (1 - 0.35 * (y[hill] - 0.5))[:, None]
    img[(x > 0.15) & (x < 0.42) & (y > 0.62) & (y < 0.88)] = [0.80, 0.30, 0.25]
    roof = (y > 0.48) & (y <= 0.62) & (np.abs(x - 0.285) < (y - 0.48) * 1.0 + 0.02)
    img[roof] = [0.35, 0.15, 0.12]
    img[(x > 0.20) & (x < 0.27) & (y > 0.68) & (y < 0.76)] = [1.0, 0.9, 0.5]
    img[(x > 0.31) & (x < 0.37) & (y > 0.72) & (y < 0.88)] = [0.30, 0.20, 0.10]
    fence = (y > 0.90)
    img[fence] = [0.55, 0.35, 0.18]
    img[fence & (((x * 32) % 2) < 1)] = [0.80, 0.60, 0.35]
    return np.clip(img, 0, 1)


def load_image(path=None, n=N):
    if path is None:
        return make_demo_photo(n)
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGB").resize((n, n)), dtype=float) / 255.0


# ----------------------------------------------------------------------------
# LINEAR ALGEBRA TOOLS WRITTEN FROM SCRATCH (so every step can be explained)
# ----------------------------------------------------------------------------
def rref(M, tol=1e-9):
    """Gauss-Jordan elimination with partial pivoting. Returns (R, pivot_columns)."""
    A = np.array(M, dtype=float)
    rows, cols = A.shape
    scale = max(1.0, np.abs(A).max())
    r, pivots = 0, []
    for c in range(cols):
        if r == rows:
            break
        p = r + np.argmax(np.abs(A[r:, c]))
        if abs(A[p, c]) < tol * scale:
            continue
        A[[r, p]] = A[[p, r]]
        A[r] /= A[r, c]
        others = np.arange(rows) != r
        A[others] -= np.outer(A[others, c], A[r])
        pivots.append(c)
        r += 1
    return A, pivots


def lu_decompose(A, tol=1e-13):
    """PA = LU with partial pivoting. Raises LinAlgError if A is singular."""
    n = A.shape[0]
    U = np.array(A, dtype=float)
    L = np.eye(n)
    perm = np.arange(n)
    scale = np.abs(U).max()
    for k in range(n):
        p = k + np.argmax(np.abs(U[k:, k]))
        if abs(U[p, k]) < tol * scale:
            raise np.linalg.LinAlgError(f"zero pivot in column {k}: matrix is singular")
        if p != k:
            U[[k, p]] = U[[p, k]]
            perm[[k, p]] = perm[[p, k]]
            L[[k, p], :k] = L[[p, k], :k]
        L[k + 1:, k] = U[k + 1:, k] / U[k, k]
        U[k + 1:, k:] -= np.outer(L[k + 1:, k], U[k, k:])
        U[k + 1:, k] = 0.0
    return np.eye(n)[perm], L, U


def lu_solve(P, L, U, B):
    """Solve A X = B using forward then backward substitution (B may be a matrix)."""
    n = L.shape[0]
    Y = P @ B
    for i in range(n):
        Y[i] -= L[i, :i] @ Y[:i]
    X = Y.copy()
    for i in range(n - 1, -1, -1):
        X[i] = (X[i] - U[i, i + 1:] @ X[i + 1:]) / U[i, i]
    return X


def lu_inverse(A):
    P, L, U = lu_decompose(A)
    return lu_solve(P, L, U, np.eye(A.shape[0]))


def gram_schmidt(A, tol=1e-9):
    """Modified Gram-Schmidt on the columns of A. Dependent columns are skipped.
    Returns Q (orthonormal columns), R, and indices of the independent columns kept."""
    m, n = A.shape
    Q, kept = [], []
    for j in range(n):
        v = A[:, j].astype(float).copy()
        for q in Q:
            v -= (q @ v) * q
        nv = np.linalg.norm(v)
        if nv > tol * max(1.0, np.linalg.norm(A[:, j])):
            Q.append(v / nv)
            kept.append(j)
    Q = np.array(Q).T
    return Q, Q.T @ A[:, kept], kept


# ----------------------------------------------------------------------------
# FILTER MATRICES  (every filter is a matrix, applying it is matrix multiplication)
# ----------------------------------------------------------------------------
def blur_matrix(n):
    """Symmetric tridiagonal blur: each pixel = 1/4 left + 1/2 itself + 1/4 right."""
    B = np.diag(np.full(n, 0.5)) + np.diag(np.full(n - 1, 0.25), 1) + np.diag(np.full(n - 1, 0.25), -1)
    B[0, 0] = B[-1, -1] = 0.75
    return B


def difference_matrix(n):
    """Edge detector: row i = pixel_i - pixel_(i-1). First row zeroed -> rank n-1."""
    D = np.eye(n) - np.eye(n, k=-1)
    D[0, :] = 0
    return D


def flip_matrix(n):
    return np.eye(n)[::-1]


def sandwich(L, X, R):
    """L X R^T : apply a matrix on the columns (left) and on the rows (right) of the image."""
    return L @ X @ R.T


def per_channel(fn, img):
    return np.stack([fn(img[..., c]) for c in range(3)], axis=-1)


def color_filter(img, M):
    """Apply a 3x3 colour matrix to every pixel's (R,G,B) vector."""
    return (img.reshape(-1, 3) @ M.T).reshape(img.shape)


def lowrank_projection(X, k):
    """Orthogonal projection of the rows of X onto the span of top-k eigenvectors of X^T X."""
    w, V = np.linalg.eigh(X.T @ X)
    Vk = V[:, -k:]
    return X @ Vk @ Vk.T


# ============================================================================
def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None
    img = load_image(path)
    n = img.shape[0]
    gray_w = np.array([0.299, 0.587, 0.114])
    X = img @ gray_w                              # grayscale data matrix (n x n)
    results = []                                  # (stage, metric) table for the report

    # ------------------------------------------------------------------ STEP 1
    banner("STEP 1  REAL-WORLD DATA -> MATRIX REPRESENTATION")
    print(f"Photo size: {img.shape}  ->  3 matrices (R, G, B), each {n}x{n}.")
    print(f"Grayscale data matrix X = 0.299R + 0.587G + 0.114B  has shape {X.shape}.")
    print("Entry X[i,j] = brightness of the pixel in row i, column j (0 = black, 1 = white).")
    print("Top-left 4x4 corner of X:\n", X[:4, :4])
    print("\nFilters are matrices too. Applying a filter = matrix multiplication:")
    print("   Y = L @ X @ R^T     (L mixes the pixels in each column, R mixes them in each row)")

    B = blur_matrix(n)
    S = np.eye(n) + 1.5 * (np.eye(n) - B)         # unsharp-mask sharpen matrix
    D = difference_matrix(n)
    J = flip_matrix(n)

    # Eigen-decomposition gives a strong blur B^3 cheaply (used again in STEP 9)
    d, P_eig = np.linalg.eigh(B)
    B3 = P_eig @ np.diag(d ** 3) @ P_eig.T

    W = np.array([[1.15, 0.10, 0.00],             # 'warm tone' colour matrix (invertible)
                  [0.05, 1.00, 0.00],
                  [0.00, 0.10, 0.85]])
    SEPIA = np.array([[0.393, 0.769, 0.189],
                      [0.349, 0.686, 0.168],
                      [0.272, 0.534, 0.131]])
    G = np.tile(gray_w, (3, 1))                   # grayscale filter: all three rows identical

    filtered = {
        "Original": img,
        "Blur  (X B^T, horizontal)": per_channel(lambda c: c @ B3.T, img),
        "Sharpen (S X S^T)": per_channel(lambda c: sandwich(S, c, S), img),
        "Flip (J X)": per_channel(lambda c: J @ c, img),
        "Rotate 90 (J X^T)": per_channel(lambda c: J @ c.T, img),
        "Brightness (1.3 X)": 1.3 * img,
        "Warm tone (W pixels)": color_filter(img, W),
        "Sepia (SEPIA pixels)": color_filter(img, SEPIA),
        "Grayscale (G pixels)": color_filter(img, G),
    }
    edge = np.sqrt((D @ X) ** 2 + (X @ D.T) ** 2)
    print("\nFilters built as matrices:")
    print(f"  Blur B^3     : {n}x{n} banded symmetric, rows sum to 1 (horizontal blur of every row, Y = X B^T)")
    print(f"  Sharpen S    : I + 1.5(I - B) -> boosts what the blur removes")
    print(f"  Flip J       : anti-identity (permutation) matrix")
    print(f"  Edge D       : difference matrix, horizontal+vertical gradients combined")
    print(f"  Colour W/G   : 3x3 matrices multiplying each pixel's (R,G,B) vector")

    # ------------------------------------------------------------------ STEP 2
    banner("STEP 2  MATRIX SIMPLIFICATION : RREF and LU (from scratch)")
    B6 = blur_matrix(6)
    R6, piv = rref(B6)
    print("Small 6x6 blur matrix B6 (same pattern as the image filter):\n", B6)
    print("\nRREF(B6) =\n", R6)
    print(f"Pivot columns: {piv} -> 6 pivots = identity -> B6 is INVERTIBLE (no pixel info lost).")

    Dn = np.zeros((3, 6))
    for i in range(3):
        Dn[i, 2 * i:2 * i + 2] = 0.5
    Rd, pivd = rref(Dn)
    print("\nDownsample matrix (average pairs of pixels, 6 -> 3 pixels):\n", Dn)
    print("RREF =\n", Rd)
    print(f"Pivot columns {pivd}: rank 3 < 6 columns -> NOT invertible, information is lost.")

    Pm, Lm, Um = lu_decompose(B6)
    print("\nLU decomposition PA = LU of B6:")
    print("L =\n", Lm, "\nU =\n", Um)
    print("Check ||P B6 - L U|| =", np.linalg.norm(Pm @ B6 - Lm @ Um))
    b = np.arange(1, 7, dtype=float).reshape(-1, 1)
    xs = lu_solve(Pm, Lm, Um, b)
    print("Solve B6 x = b with forward/back substitution: ||B6 x - b|| =", np.linalg.norm(B6 @ xs - b))
    print("MEANING: LU turns 'invert a matrix' into two cheap triangular solves. This is how we un-blur.")

    # ------------------------------------------------------------------ STEP 3
    banner("STEP 3  STRUCTURE OF THE SPACE : column space, rank, nullity")

    def rank_info(name, M):
        r = np.linalg.matrix_rank(M)
        print(f"  {name:<28} size {str(M.shape):<10} rank = {r:<4} nullity = {M.shape[1] - r}")
        return r

    rank_info("Blur B^3", B3)
    rank_info("Sharpen S", S)
    rank_info("Flip J", J)
    rank_info("Edge difference D", D)
    rank_info("Warm tone W", W)
    rank_info("Sepia", SEPIA)
    rank_info("Grayscale G", G)
    rank_info("Image X", X)
    print("MEANING: nullity 0 -> the filter keeps all information -> it can be undone (inverse exists).")
    print("         nullity > 0 -> some inputs are mapped to 0 / merged -> NO inverse (e.g. grayscale, edge).")
    print(f"  Sepia determinant = {np.linalg.det(SEPIA):.2e}, condition number = {np.linalg.cond(SEPIA):.2e}")
    print("  -> sepia is almost singular: undoing it would amplify any rounding error enormously.")

    # ------------------------------------------------------------------ STEP 4
    banner("STEP 4  REMOVE REDUNDANCY : linear independence and basis selection")
    Rx, pivx = rref(X, tol=1e-8)
    sv = np.linalg.svd(X, compute_uv=False)
    energy = np.cumsum(sv ** 2) / np.sum(sv ** 2)
    k99 = int(np.searchsorted(energy, 0.99) + 1)
    print(f"RREF of the image matrix finds {len(pivx)} independent columns out of {n}.")
    print("Pivot columns of X form a BASIS of its column space (the other columns are combinations of them).")
    print(f"Effective rank: only {k99} directions carry 99% of the image 'energy' -> huge redundancy.")

    # ------------------------------------------------------------------ STEP 5
    banner("STEP 5  ORTHOGONALIZATION : Gram-Schmidt -> orthonormal basis")
    Q, Rq, kept = gram_schmidt(X)
    print(f"Gram-Schmidt on the columns of X gives Q with {Q.shape[1]} orthonormal columns.")
    print("||Q^T Q - I||        =", f"{np.linalg.norm(Q.T @ Q - np.eye(Q.shape[1])):.2e}  (0 means perfectly orthogonal)")
    print("||X[:,kept] - Q R||  =", f"{np.linalg.norm(X[:, kept] - Q @ Rq):.2e}  (QR factorisation is exact)")
    print("MEANING: orthonormal bases make projection trivial (P = Q Q^T) and inverses trivial (Q^-1 = Q^T).")
    print("         Flip matrix J is orthogonal:  ||J^T J - I|| =", np.linalg.norm(J.T @ J - np.eye(n)))

    # ------------------------------------------------------------------ STEP 6
    banner("STEP 6  INVERSION : undoing the filters with A^-1")
    blurred = filtered["Blur  (X B^T, horizontal)"]
    P3, L3, U3 = lu_decompose(B3)
    print(f"Blur matrix B^3: condition number = {np.linalg.cond(B3):.2e}  (big -> inverse amplifies errors)")
    print("LU of B^3 computed: ||P B^3 - L U|| =", f"{np.linalg.norm(P3 @ B3 - L3 @ U3):.2e}")
    print("Un-blurring means solving  B^3 x = y  for every row (x = B^-3 y). We solve with LU")
    print("forward/back substitution instead of forming B^-1 explicitly (more accurate).")
    solve_B3 = lambda c: lu_solve(P3, L3, U3, c.T.copy()).T      # = c @ (B^3)^-T
    deblur_exact = per_channel(solve_B3, blurred)
    print(f"Deblur with exact inverse (no noise): PSNR blurred = {psnr(blurred, img):.1f} dB "
          f"-> recovered = {psnr(deblur_exact, img):.1f} dB")
    results.append(("Blurred vs original", psnr(blurred, img)))
    results.append(("Exact inverse deblur (no noise)", psnr(deblur_exact, img)))

    Sinv = lu_inverse(S)
    sharp = filtered["Sharpen (S X S^T)"]
    sharp_back = per_channel(lambda c: Sinv @ c @ Sinv.T, sharp)
    Winv = np.linalg.inv(W)
    warm = filtered["Warm tone (W pixels)"]
    warm_back = color_filter(warm, Winv)
    print(f"Un-sharpen with S^-1   : error = {np.abs(sharp_back - img).max():.2e}  (perfect recovery)")
    print(f"Un-warm with W^-1      : error = {np.abs(warm_back - img).max():.2e}  (perfect recovery)")
    print("Un-flip: J^-1 = J^T = J : exact (a permutation).")
    try:
        lu_inverse(G)
    except np.linalg.LinAlgError as e:
        print(f"Un-grayscale: FAILS -> {e}\n   (rank 1: three colours were merged into one number, cannot be recovered)")

    # noisy case
    sigma = 0.002
    noisy_blur = blurred + rng.normal(0, sigma, blurred.shape)
    naive = per_channel(solve_B3, noisy_blur)
    print(f"\nNow add tiny sensor noise (sigma = {sigma}) to the blurred photo:")
    print(f"  naive inverse -> PSNR = {psnr(naive, img):.1f} dB  (noise amplified by up to ~{np.linalg.cond(B3):.0e}x)")
    results.append(("Noisy blurred vs original", psnr(noisy_blur, img)))
    results.append(("Naive inverse on noisy blur", psnr(naive, img)))

    # ------------------------------------------------------------------ STEP 7
    banner("STEP 7  PREDICTION / APPROXIMATION : least squares (normal equations)")
    print("Instead of solving B x = y exactly, minimise  ||B x - y||^2 + lam ||x||^2")
    print("Normal equations:  (B^T B + lam I) x = B^T y   -> x = R y,  R = (B^T B + lam I)^-1 B^T")
    best = (None, -1, None)
    print(f"  {'lambda':>10} | PSNR (dB)")
    for lam in [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
        Rl = np.linalg.solve(B3.T @ B3 + lam * np.eye(n), B3.T)
        rec = per_channel(lambda c: c @ Rl.T, noisy_blur)
        p = psnr(rec, img)
        print(f"  {lam:>10.0e} | {p:6.1f}")
        if p > best[1]:
            best = (lam, p, rec)
    lam_best, _, ls_rec = best
    print(f"Best lambda = {lam_best:.0e}: PSNR {psnr(ls_rec, img):.1f} dB "
          f"vs naive inverse {psnr(naive, img):.1f} dB vs noisy blur {psnr(noisy_blur, img):.1f} dB")
    print("MEANING: least squares gives the closest sensible answer when an exact inverse is unstable.")
    print("(lambda picked by comparing with the true photo - a demo shortcut; in practice use a validation set.)")
    results.append((f"Least-squares deblur (lam={lam_best:.0e})", psnr(ls_rec, img)))

    # ------------------------------------------------------------------ STEP 8
    banner("STEP 8  PROJECTION : orthogonal projection onto a subspace (compression)")
    Gm = X.T @ X
    w, V = np.linalg.eigh(Gm)
    w, V = w[::-1], V[:, ::-1]
    print("Projection of the rows of X onto span(v1..vk):  X_k = X V_k V_k^T   (P = V_k V_k^T is symmetric, P^2 = P)")
    Vk = V[:, :10]
    Pk = Vk @ Vk.T
    print(f"Check projection matrix: ||P^2 - P|| = {np.linalg.norm(Pk @ Pk - Pk):.1e},  rank(P) = {np.linalg.matrix_rank(Pk)}")
    print(f"  {'k':>3} | stored numbers | compression | PSNR (dB)")
    comp_imgs = {}
    for k in [5, 10, 20, 40]:
        rec = per_channel(lambda c: lowrank_projection(c, k), img)
        stored = 3 * k * (2 * n)
        print(f"  {k:>3} | {stored:>14} | {img.size / stored:>9.1f}x | {psnr(rec, img):6.1f}")
        comp_imgs[k] = rec
        results.append((f"Projection compression k={k} ({img.size / stored:.1f}x)", psnr(rec, img)))

    # ------------------------------------------------------------------ STEP 9
    banner("STEP 9  PATTERN DISCOVERY & DIAGONALIZATION : eigenvalues / eigenvectors")
    print("Blur matrix B is SYMMETRIC -> B = P D P^T with orthogonal P and diagonal D (eigenvalues).")
    print(f"||B - P D P^T||            = {np.linalg.norm(B - P_eig @ np.diag(d) @ P_eig.T):.2e}")
    print(f"||B^3 - P D^3 P^T||        = {np.linalg.norm(B3 - np.linalg.matrix_power(B, 3)):.2e}  "
          "(powers of a filter become powers of numbers!)")
    print(f"Eigenvalues of B: largest = {d.max():.4f} (flat/smooth pattern kept),  smallest = {d.min():.2e} "
          "(zig-zag pattern almost erased)")
    print(f"Applying the filter 3 times = scaling each eigen-direction by lambda^3.")
    print(f"Inverse = P D^-1 P^T: the smallest eigenvalue {d.min():.1e} is why un-blurring amplifies noise ~{1/d.min()**3:.0e}x.")
    print("\nPattern discovery on the PHOTO: eigenvalues of X^T X (squares of singular values):")
    print("  top 6 eigenvalues:", w[:6])
    print(f"  first eigenvector alone captures {w[0] / w.sum() * 100:.1f}% of the total energy;"
          f" top 10 capture {w[:10].sum() / w.sum() * 100:.1f}%")
    print("  => the photo is dominated by a few smooth patterns (sky gradient, hills), small eigenvalues are fine detail/noise.")

    # ------------------------------------------------------------------ STEP 10
    banner("STEP 10  FINAL APPLICATION : noise reduction by projection")
    noisy = np.clip(img + rng.normal(0, 0.10, img.shape), 0, 1)
    scores = {k: psnr(per_channel(lambda c: lowrank_projection(c, k), noisy), img) for k in range(1, 61)}
    kbest = max(scores, key=scores.get)
    den = per_channel(lambda c: lowrank_projection(c, kbest), noisy)
    print(f"Noisy photo (sigma = 0.10): PSNR = {psnr(noisy, img):.1f} dB")
    print(f"Projected onto top-{kbest} eigen-directions: PSNR = {psnr(den, img):.1f} dB")
    print("MEANING: noise spreads over ALL directions, the photo lives in a FEW. Projecting keeps the photo, drops noise.")
    results.append(("Noisy photo (sigma=0.10)", psnr(noisy, img)))
    results.append((f"Denoised by projection k={kbest}", psnr(den, img)))

    banner("SUMMARY TABLE")
    for name, v in results:
        print(f"  {name:<46} {v:6.1f} dB")

    # ========================== FIGURES ====================================
    show = lambda a: np.clip(a, 0, 1)
    fig, axs = plt.subplots(2, 5, figsize=(17, 7.2))
    items = list(filtered.items()) + [("Edges (|DX|+|XD^T|)", None)]
    for ax, (name, im) in zip(axs.ravel(), items):
        if im is None:
            ax.imshow(edge / edge.max(), cmap="gray")
        else:
            ax.imshow(show(im))
        ax.set_title(name, fontsize=10); ax.axis("off")
    fig.suptitle("Fig 1 - Photo filters = matrix multiplication", fontsize=14)
    fig.tight_layout(); fig.savefig(OUT + "fig1_filters.png", dpi=110); plt.close(fig)

    fig, axs = plt.subplots(2, 4, figsize=(15, 7.6))
    panels = [("Original", img), ("Blurred (X B^T)", blurred),
              (f"Exact inverse (LU)\n{psnr(deblur_exact, img):.0f} dB", deblur_exact),
              (f"Blur + noise\n{psnr(noisy_blur, img):.1f} dB", noisy_blur),
              (f"Naive inverse on noisy\n{psnr(naive, img):.1f} dB  (explodes)", naive),
              (f"Least squares (lam={lam_best:.0e})\n{psnr(ls_rec, img):.1f} dB", ls_rec),
              ("Sharpened (S X S^T)", sharp), ("Sharpen undone by S^-1", sharp_back)]
    for ax, (t, im) in zip(axs.ravel(), panels):
        ax.imshow(show(im)); ax.set_title(t, fontsize=10); ax.axis("off")
    fig.suptitle("Fig 2 - Inversion: undoing filters, and why noise breaks it (least squares fixes it)", fontsize=13)
    fig.tight_layout(); fig.savefig(OUT + "fig2_inversion.png", dpi=110); plt.close(fig)

    fig, axs = plt.subplots(2, 4, figsize=(16, 8))
    axs[0, 0].semilogy(w[:60], "o-", ms=3); axs[0, 0].set_title("Eigenvalues of X^T X (scree)")
    axs[0, 0].set_xlabel("index"); axs[0, 0].grid(alpha=.3)
    axs[0, 1].plot(np.sort(d)[::-1], "o-", ms=2); axs[0, 1].set_title("Eigenvalues of blur B (gain per pattern)")
    axs[0, 1].set_xlabel("index"); axs[0, 1].grid(alpha=.3)
    axs[0, 2].plot(list(scores), list(scores.values())); axs[0, 2].axvline(kbest, color="r", ls="--")
    axs[0, 2].set_title(f"Denoising PSNR vs k (best k={kbest})"); axs[0, 2].set_xlabel("k"); axs[0, 2].grid(alpha=.3)
    axs[0, 3].imshow(show(img)); axs[0, 3].set_title("Original"); axs[0, 3].axis("off")
    for ax, k in zip(axs[1, :3], [5, 20, 40]):
        ax.imshow(show(comp_imgs[k]))
        ax.set_title(f"Projection k={k}\n{psnr(comp_imgs[k], img):.1f} dB, {img.size/(3*k*2*n):.1f}x smaller", fontsize=10)
        ax.axis("off")
    axs[1, 3].imshow(show(den)); axs[1, 3].set_title(f"Denoised (k={kbest})\n{psnr(den, img):.1f} dB vs noisy {psnr(noisy, img):.1f} dB", fontsize=10)
    axs[1, 3].axis("off")
    fig.suptitle("Fig 3 - Eigenvalues, projection compression and noise reduction", fontsize=13)
    fig.tight_layout(); fig.savefig(OUT + "fig3_eigen_projection.png", dpi=110); plt.close(fig)
    print("\nSaved: fig1_filters.png, fig2_inversion.png, fig3_eigen_projection.png")


if __name__ == "__main__":
    main()
