import cv2
import numpy as np
import matplotlib.pyplot as plt

def forward_diff(img):
    p = np.pad(img, 1, mode='edge')
    gx = p[1:-1, 2:] - p[1:-1, 1:-1]
    gy = p[2:, 1:-1] - p[1:-1, 1:-1]
    return gx, gy

def backward_diff(img):
    p = np.pad(img, 1, mode='edge')
    gx = p[1:-1, 1:-1] - p[1:-1, :-2]
    gy = p[1:-1, 1:-1] - p[:-2, 1:-1]
    return gx, gy

def central_diff(img):
    p = np.pad(img, 1, mode='edge')
    gx = (p[1:-1, 2:] - p[1:-1, :-2]) / 2.0
    gy = (p[2:, 1:-1] - p[:-2, 1:-1]) / 2.0
    return gx, gy

def convolve(img, kernel):
    p = np.pad(img, 1, mode='edge')
    k = np.flip(kernel)
    out = np.zeros_like(img, dtype=float)
    for i in range(3):
        for j in range(3):
            out += k[i, j] * p[i:i+img.shape[0], j:j+img.shape[1]]
    return out

def sobel_kernel(img):
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=float)
    return convolve(img, kx), convolve(img, ky)

def prewitt_kernel(img):
    kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=float)
    ky = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=float)
    return convolve(img, kx), convolve(img, ky)

def magnitude_and_direction(gx, gy):
    mag = np.sqrt(gx**2 + gy**2)
    mag_norm = (mag / (mag.max() + 1e-8) * 255).astype(np.uint8)
    direction = np.arctan2(gy, gx)
    return mag_norm, direction

def compare_finite_diff(img_rgb, img_gray):
    img_f = img_gray.astype(float)
    fx, fy = forward_diff(img_f)
    bx, by = backward_diff(img_f)
    cx, cy = central_diff(img_f)

    fig, ax = plt.subplots(2, 4, figsize=(14, 7))
    items = [
        (img_rgb, 'Input', None), (fx, 'Forward Gx', 'gray'), (bx, 'Backward Gx', 'gray'), (cx, 'Central Gx', 'gray'),
        (img_rgb, 'Input', None), (fy, 'Forward Gy', 'gray'), (by, 'Backward Gy', 'gray'), (cy, 'Central Gy', 'gray')
    ]
    for a, (im, title, cmap) in zip(ax.flat, items):
        a.imshow(im, cmap=cmap)
        a.set_title(title)
        a.axis('off')
    plt.tight_layout()
    plt.show()

def compare_edge_kernels(img_rgb, img_gray):
    img_f = img_gray.astype(float)
    sx, sy = sobel_kernel(img_f)
    px, py = prewitt_kernel(img_f)

    fig, ax = plt.subplots(2, 3, figsize=(11, 7))
    items = [
        (img_rgb, 'Input', None), (sx, 'Sobel Gx', 'gray'), (px, 'Prewitt Gx', 'gray'),
        (img_rgb, 'Input', None), (sy, 'Sobel Gy', 'gray'), (py, 'Prewitt Gy', 'gray')
    ]
    for a, (im, title, cmap) in zip(ax.flat, items):
        a.imshow(im, cmap=cmap)
        a.set_title(title)
        a.axis('off')
    plt.tight_layout()
    plt.show()

def compare_mag_dir(img_rgb, img_gray):
    gx, gy = sobel_kernel(img_gray.astype(float))
    mag, direction = magnitude_and_direction(gx, gy)

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    items = [
        (img_rgb, 'Input', None),
        (mag, 'Magnitude [0, 255]', 'gray'),
        (direction, 'Direction [-pi, pi]', 'hsv')
    ]
    for a, (im, title, cmap) in zip(ax, items):
        a.imshow(im, cmap=cmap)
        a.set_title(title)
        a.axis('off')
    plt.tight_layout()
    plt.show()

def main():
    img_bgr = cv2.imread('gambar.jpg')
    if img_bgr is None:
        print("File 'gambar.jpg' tidak ditemukan.")
        return

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    while True:
        print("\n=== MENU ===")
        print("1. Finite Differences (Forward, Backward, Central)")
        print("2. Edge Kernels (Sobel, Prewitt)")
        print("3. Magnitude and Direction")
        print("4. Keluar")

        pilihan = input("Pilih menu (1-4): ").strip()
        if pilihan == '1':
            compare_finite_diff(img_rgb, img_gray)
        elif pilihan == '2':
            compare_edge_kernels(img_rgb, img_gray)
        elif pilihan == '3':
            compare_mag_dir(img_rgb, img_gray)
        elif pilihan == '4':
            break
        else:
            print("Pilihan tidak valid.")

if __name__ == '__main__':
    main()