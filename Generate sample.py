import numpy as np
import cv2
import matplotlib.pyplot as plt

def generate_greyscale_ring(
    size=512,
    inner_radius=120,
    outer_radius=200,
    greys=(220, 170, 100, 40),
    filename="pure_ring.png"
):
    """
    Generate a ring split into 4 greyscale regions on a perfectly white background.

    size         : image is size x size
    inner_radius : inner radius of ring (pixels)
    outer_radius : outer radius of ring (pixels)
    greys        : 4 greyscale values (0–255) for each quadrant, clockwise
    filename     : where to save the image
    """
    h = w = size

    # start with pure white background
    img = np.full((h, w), 255, dtype=np.uint8)

    # coordinate grid
    y, x = np.ogrid[:h, :w]
    cy, cx = h // 2, w // 2

    # squared distance from center
    r2 = (x - cx)**2 + (y - cy)**2
    inner2 = inner_radius**2
    outer2 = outer_radius**2

    # ring mask: between inner and outer radius
    ring_mask = (r2 >= inner2) & (r2 <= outer2)

    # angle of each pixel (0 at +x axis, increasing counterclockwise)
    angle = np.arctan2(-(y - cy), x - cx)  # [-pi, pi]
    angle[angle < 0] += 2 * np.pi          # [0, 2pi)

    # quadrant masks
    q0 = (angle >= 0)            & (angle < np.pi / 2)   # right-top
    q1 = (angle >= np.pi / 2)    & (angle < np.pi)       # left-top
    q2 = (angle >= np.pi)        & (angle < 3*np.pi/2)   # left-bottom
    q3 = (angle >= 3*np.pi/2)    & (angle < 2*np.pi)     # right-bottom

    # apply greyscale levels inside ring only
    img[ring_mask & q0] = greys[0]
    img[ring_mask & q1] = greys[1]
    img[ring_mask & q2] = greys[2]
    img[ring_mask & q3] = greys[3]

    # save and show
    cv2.imwrite(filename, img)

    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.title("Pure greyscale ring")
    plt.axis("off")
    plt.show()

    return img


if __name__ == "__main__":
    generate_greyscale_ring()