import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_eed(input_u, steps=40, K=0.05, dt=0.1):
    u = np.asarray(input_u, dtype=np.float64).copy()
    for _ in range(steps):
        # 1. Compute Gradients (ux, uy)
        ux = cv2.Sobel(u, cv2.CV_64F, 1, 0, ksize=3)
        uy = cv2.Sobel(u, cv2.CV_64F, 0, 1, ksize=3)
        
        # Squared gradient magnitude (must be non-negative)
        eps = 1e-12
        mag_sq = ux**2 + uy**2 + eps
        
        # 2. Design Eigenvalues for D
        # Lambda 1: Across the edge (minimum diffusion)
        l1 = np.exp(-(mag_sq / (K**2)))
        l1 = np.clip(l1, 0.0, 1.0)
        # Lambda 2: Along the edge (maximum diffusion)
        l2 = 1.0 
        
        # 3. Construct Diffusion Tensor D = P * Lambda * P_inv
        # From your notes: D = [[d11, d12], [d12, d22]]
        d11 = (l1 * ux**2 + l2 * uy**2) / mag_sq
        d12 = (l1 - l2) * (ux * uy) / mag_sq
        d22 = (l1 * uy**2 + l2 * ux**2) / mag_sq
        
        # 4. Compute Flux: j = D * grad(u)
        flux_x = d11 * ux + d12 * uy
        flux_y = d12 * ux + d22 * uy
        
        # 5. Update u: u_t = div(flux)
        # Divergence is d/dx(flux_x) + d/dy(flux_y)
        div = cv2.Sobel(flux_x, cv2.CV_64F, 1, 0, ksize=3) + \
              cv2.Sobel(flux_y, cv2.CV_64F, 0, 1, ksize=3)
        
        u += dt * div
        u = np.clip(u, 0, 1)
    return u

# --- Load and Setup ---
img = cv2.imread('image.png', cv2.IMREAD_GRAYSCALE)
ref = img.astype(float) / 255.0

# --- Generate 4 Noises ---
# 1. Additive
n_add = np.clip(ref + np.random.normal(0, 0.1, ref.shape), 0, 1)
# 2. Multiplicative
n_mult = np.clip(ref + ref * np.random.randn(*ref.shape) * 0.1, 0, 1)
# 3. Poisson
n_pois = np.random.poisson(ref * 255.0) / 255.0
# 4. Salt & Pepper
n_sp = ref.copy()
amt = 0.05
coords = [np.random.randint(0, i - 1, int(amt * ref.size * 0.5)) for i in ref.shape]
n_sp[tuple(coords)] = 1
coords = [np.random.randint(0, i - 1, int(amt * ref.size * 0.5)) for i in ref.shape]
n_sp[tuple(coords)] = 0

# --- Process ---
noises = [n_add, n_mult, n_pois, n_sp]
titles = ["Additive", "Multiplicative", "Poisson", "Salt & Pepper"]
results = [apply_eed(n) for n in noises]

# --- Visualization ---
plt.figure(figsize=(12, 16))
for i in range(4):
    plt.subplot(4, 2, 2*i + 1)
    plt.imshow(noises[i], cmap='gray')
    plt.title(f"Noisy {titles[i]}")
    plt.axis('off')
    
    plt.subplot(4, 2, 2*i + 2)
    plt.imshow(results[i], cmap='gray')
    plt.title(f"EED (Weickert) {titles[i]}")
    plt.axis('off')

plt.tight_layout()
plt.show()