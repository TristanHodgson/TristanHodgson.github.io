# Note this code is fully vibe-coded as it was just a bit of fun rather than anything that matters. It should not be taken as a reflection of my skill (or lack thereof). The goal was simply to produce some images for linkedin.

"""
Social Preview Image Generator: "Mathematics through Computation"

Two Modes:
1. Standard (--mode standard): Original 1200x627 layout.
2. LinkedIn (--mode linkedin): 1584x396 layout with adapted safe-zones.

Run:
    python main.py --mode standard
    python main.py --mode linkedin
"""

# Mathematical Formulation of the Social Preview Image Generation Pipeline:

# 1. Synthetic Data & Network Architecture:
#     - Target function: y = sin(5x) sampled uniformly at x_i in [-1, 1] across N = 8 points.
#     - Neural network: A single-hidden-layer network with 5 hidden nodes, parameterized by 
#         theta in R^16 (weights and biases).
#     - Loss function (MSE):
#         $$L(\theta) = \frac{1}{N} \sum_{i=1}^{N} \left( y_{\text{pred}}(x_i; \theta) - y_i \right)^2$$

# 2. 2D Loss Landscape Slice (Parameter Plane):
#     - A random initial parameter vector theta_0 is chosen.
#     - Two orthogonal directions d_1, d_2 in R^16 are generated via Gaussian sampling followed 
#         by Gram-Schmidt orthogonalization:
#         $$\langle d_1, d_2 \rangle = 0, \quad \|d_1\| = \|d_2\| = 1$$
#     - The parameter plane is parameterized by coordinates (u, v):
#         $$\theta(u, v) = \theta_0 + u \cdot d_1 + v \cdot d_2$$

# 3. Critical Point Anchoring:
#     - A coarse evaluation grid over (u, v) identifies local minima.
#     - The closest pair of local minima, (u_1, v_1) and (u_2, v_2), is selected to define 
#         a canonical mathematical displacement vector:
#         $$\vec{v}_{\text{math}} = (u_2 - u_1, v_2 - v_1)$$

# 4. Affine Projection (Pixel Space to Mathematical Plane):
#     - Pixel space coordinates (x_img, y_img) are mapped to plane offsets (du, dv) using 
#         an anchor point (IMG1_X, IMG1_Y), a scale factor, and a rotation angle derived from 
#         the alignment between pixel anchors and the mathematical minima vector:
#         $$\begin{pmatrix} du \\ dv \end{pmatrix} = \text{scale} \cdot R(\theta_{\text{rot}}) \begin{pmatrix} x_{\text{img}} - X_1 \\ y_{\text{img}} - Y_1 \end{pmatrix}$$

# 5. Isoline Extraction & Rendering:
#     - A high-resolution evaluation grid (GRID_RES = 160) computes loss values across the image space.
#     - Marching Squares extracts contour lines at non-linear loss thresholds:
#         $$\text{thresh}_i = \text{min\_loss} + \left(\frac{i}{\text{NUM\_CONTOURS}}\right)^{1.5} (\text{max\_val} - \text{min\_loss})$$
#     - Contours are rendered with a positional linear fade mask and a cyclic color palette.

from pathlib import Path
import argparse
import math
from PIL import Image, ImageDraw, ImageFont
import random


# ---------------------------------------------------------------------------
# Argument Parsing for Modes
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Generate graphic banner.")
parser.add_argument("--mode", choices=["standard", "linkedin"], default="standard",
                    help="Select output resolution and layout mode.")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

if args.mode == "standard":
    WIDTH, HEIGHT = 1200, 627

    # Text positions, sizes, and alignment
    NAME_SIZE, SUB_SIZE, DETAIL_SIZE, URL_SIZE = 72, 36, 30, 25
    TEXT_ANCHOR = "lt"  # Left-Top alignment
    X_TEXT = 90
    Y_NAME, Y_BAR, Y_SUB, Y_DETAIL, Y_URL = 145, 255, 300, 360, 475

    # Anchor the blue accent rectangle to the left
    RECT_X1, RECT_X2 = X_TEXT, X_TEXT + 105

    # Projection anchoring (Mathematical Plane -> Pixels)
    IMG1_X, IMG1_Y = 1225, 653   # First min -> Bottom Right
    IMG2_X, IMG2_Y = 750, -50    # Second min -> Top Left

    # Evaluation Grid mapping
    X_MIN_IMG, X_MAX_IMG = 700, 1500
    Y_MIN_IMG, Y_MAX_IMG = -150, 900

    # Fade mask properties (Fades in from left to right)
    FADE_START, FADE_END = 900, 1050
    FADE_DIRECTION = "in"

    OUTPUT_FILE = "social-preview.png"

else:  # linkedin
    # LinkedIn recommended dimensions
    WIDTH, HEIGHT = 1584, 396

    # Text positions, sizes, and alignment (right padding ~48px)
    NAME_SIZE, SUB_SIZE, DETAIL_SIZE, URL_SIZE = 64, 32, 26, 22
    TEXT_ANCHOR = "rt"  # Right-Top alignment
    X_TEXT = 1536       # Right margin coordinate (1584 width - 48 padding)
    Y_NAME, Y_BAR, Y_SUB, Y_DETAIL, Y_URL = 70, 160, 195, 245, 315

    # Anchor the blue accent rectangle to the right
    RECT_X1, RECT_X2 = X_TEXT - 105, X_TEXT

    # Projection anchoring extended rightward by ~10% of total width
    IMG1_X, IMG1_Y = 550, 320
    IMG2_X, IMG2_Y = 100, -30

    # Evaluation Grid mapping extended further right while keeping clear of the text zone
    X_MIN_IMG, X_MAX_IMG = -150, 760
    Y_MIN_IMG, Y_MAX_IMG = -100, 450

    # Fade mask properties extended to utilize more banner space safely
    FADE_START, FADE_END = 550, 760
    FADE_DIRECTION = "out"

    OUTPUT_FILE = "linkedin-banner.png"


# Website / Nord colours
BG = "#e5e9f0"
FG = "#2e3440"
SECONDARY = "#4c566a"
ACCENT = "#5e81ac"

COMPONENT_COLOURS = [
    "#8fbcbb", "#88c0d0", "#81a1c1", "#5e81ac", "#b48ead",
]

REPO_ROOT = Path(__file__).resolve().parent.parent
FONT_PATH = REPO_ROOT / "fonts" / "firacode.ttf"
OUTPUT_PATH = REPO_ROOT / "img" / OUTPUT_FILE


def font(size):
    return ImageFont.truetype(str(FONT_PATH), size)


# ---------------------------------------------------------------------------
# Colour utilities
# ---------------------------------------------------------------------------

def hex_to_rgb(colour):
    return tuple(int(colour[i:i + 2], 16) for i in (1, 3, 5))


def rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(rgb[0], rgb[1], rgb[2])


def rgb_blend(fg_rgb, bg_rgb, strength):
    """Blend an RGB colour into an RGB background."""
    return tuple(
        round(strength * fg + (1 - strength) * bg)
        for fg, bg in zip(fg_rgb, bg_rgb)
    )


# ---------------------------------------------------------------------------
# Image setup & Text Content
# ---------------------------------------------------------------------------

image = Image.new("RGB", (WIDTH, HEIGHT), BG)
draw = ImageDraw.Draw(image)

name = "Tristan Hodgson"
subtitle = "Mathematics @ University of Oxford"
detail = "Mathematics through computation"
url = "tristanhodgson.com"

# Drawing text using the dynamic TEXT_ANCHOR parameter
draw.text((X_TEXT, Y_NAME), name, font=font(NAME_SIZE), fill=FG,
          stroke_width=1, stroke_fill=FG, anchor=TEXT_ANCHOR)
draw.rounded_rectangle(
    (RECT_X1, Y_BAR, RECT_X2, Y_BAR + 7), radius=3, fill=ACCENT)
draw.text((X_TEXT, Y_SUB), subtitle, font=font(
    SUB_SIZE), fill=FG, anchor=TEXT_ANCHOR)
draw.text((X_TEXT, Y_DETAIL), detail, font=font(
    DETAIL_SIZE), fill=SECONDARY, anchor=TEXT_ANCHOR)
draw.text((X_TEXT, Y_URL), url, font=font(
    URL_SIZE), fill=ACCENT, anchor=TEXT_ANCHOR)


# ---------------------------------------------------------------------------
# Computational Mathematics: SIREN ML Model & Loss Landscape
# ---------------------------------------------------------------------------

NUM_PARAMS = 16

# Synthetic high-frequency dataset to induce non-convexity
X_DATA = [-1.0 + i * (2.0 / 7.0) for i in range(8)]
Y_DATA = [math.sin(5.0 * x) for x in X_DATA]


def evaluate_loss(theta):
    w1, b1 = theta[0:5], theta[5:10]
    w2, b2 = theta[10:15], theta[15]

    loss = 0.0
    for x, y_true in zip(X_DATA, Y_DATA):
        hidden = [math.sin(w1[i] * x + b1[i]) for i in range(5)]
        y_pred = sum(w2[i] * hidden[i] for i in range(5)) + b2
        loss += (y_pred - y_true) ** 2

    return loss / len(X_DATA)


random.seed(42)
theta_0 = [random.uniform(-1, 1) for _ in range(NUM_PARAMS)]

d1 = [random.gauss(0, 1) for _ in range(NUM_PARAMS)]
d2 = [random.gauss(0, 1) for _ in range(NUM_PARAMS)]

norm1 = math.sqrt(sum(x*x for x in d1))
d1 = [x / norm1 for x in d1]

dot_product = sum(x*y for x, y in zip(d1, d2))
d2 = [d2[i] - dot_product * d1[i] for i in range(NUM_PARAMS)]

norm2 = math.sqrt(sum(x*x for x in d2))
d2 = [x / norm2 for x in d2]


# ---------------------------------------------------------------------------
# Algorithmic Framing
# ---------------------------------------------------------------------------

search_coords = [x * 0.5 for x in range(-40, 41)]
search_grid = []

for u_test in search_coords:
    row = []
    for v_test in search_coords:
        theta = [theta_0[i] + u_test * d1[i] + v_test * d2[i]
                 for i in range(NUM_PARAMS)]
        row.append(evaluate_loss(theta))
    search_grid.append(row)

local_minima = []
for i in range(1, len(search_coords) - 1):
    for j in range(1, len(search_coords) - 1):
        val = search_grid[i][j]
        if (val < search_grid[i-1][j] and val < search_grid[i+1][j] and
            val < search_grid[i][j-1] and val < search_grid[i][j+1] and
            val < search_grid[i-1][j-1] and val < search_grid[i+1][j+1] and
                val < search_grid[i-1][j+1] and val < search_grid[i+1][j-1]):
            local_minima.append((search_coords[i], search_coords[j], val))

min_dist = float('inf')
closest_pair = None

for i in range(len(local_minima)):
    for j in range(i + 1, len(local_minima)):
        u_w, v_w, val_w = local_minima[i]
        u_z, v_z, val_z = local_minima[j]
        d = math.hypot(u_w - u_z, v_w - v_z)

        if d < min_dist:
            min_dist = d
            if val_w < val_z:
                closest_pair = ((u_w, v_w), (u_z, v_z))
            else:
                closest_pair = ((u_z, v_z), (u_w, v_w))

if closest_pair:
    u1, v1 = closest_pair[0]
    u2, v2 = closest_pair[1]
else:
    u1, v1 = 0.0, 0.0
    u2, v2 = 2.0, 2.0


v_img_x, v_img_y = IMG2_X - IMG1_X, IMG2_Y - IMG1_Y
len_img = math.hypot(v_img_x, v_img_y)

v_math_u, v_math_v = u2 - u1, v2 - v1
len_math = math.hypot(v_math_u, v_math_v)

scale = len_math / len_img
angle_img = math.atan2(v_img_y, v_img_x)
angle_math = math.atan2(v_math_v, v_math_u)
rotation = angle_math - angle_img

cos_rot, sin_rot = math.cos(rotation), math.sin(rotation)


# ---------------------------------------------------------------------------
# Grid Evaluation & Marching Squares
# ---------------------------------------------------------------------------

GRID_RES = 160

img_x_coords = [X_MIN_IMG + c *
                (X_MAX_IMG - X_MIN_IMG) / (GRID_RES - 1) for c in range(GRID_RES)]
img_y_coords = [Y_MIN_IMG + r *
                (Y_MAX_IMG - Y_MIN_IMG) / (GRID_RES - 1) for r in range(GRID_RES)]

grid_losses = [[0.0 for _ in range(GRID_RES)] for _ in range(GRID_RES)]
all_losses = []

for r in range(GRID_RES):
    y_img = img_y_coords[r]
    dy = y_img - IMG1_Y

    for c in range(GRID_RES):
        x_img = img_x_coords[c]
        dx = x_img - IMG1_X

        du = scale * (dx * cos_rot - dy * sin_rot)
        dv = scale * (dx * sin_rot + dy * cos_rot)

        theta = [theta_0[i] + (u1 + du) * d1[i] + (v1 + dv) * d2[i]
                 for i in range(NUM_PARAMS)]
        loss = evaluate_loss(theta)

        grid_losses[r][c] = loss
        all_losses.append(loss)

all_losses.sort()
min_loss = all_losses[0]
max_contour_val = all_losses[int(len(all_losses) * 0.85)]


def get_contour_segments(grid, thresh, img_x, img_y):
    segments = []
    for r in range(GRID_RES - 1):
        for c in range(GRID_RES - 1):
            v0, v1 = grid[r][c], grid[r][c+1]
            v3, v2 = grid[r+1][c], grid[r+1][c+1]

            edges = []
            if (v0 > thresh) != (v1 > thresh):
                t = (thresh - v0) / (v1 - v0 + 1e-9)
                edges.append((img_x[c] + t*(img_x[c+1]-img_x[c]), img_y[r]))
            if (v1 > thresh) != (v2 > thresh):
                t = (thresh - v1) / (v2 - v1 + 1e-9)
                edges.append((img_x[c+1], img_y[r] + t*(img_y[r+1]-img_y[r])))
            if (v3 > thresh) != (v2 > thresh):
                t = (thresh - v3) / (v2 - v3 + 1e-9)
                edges.append((img_x[c] + t*(img_x[c+1]-img_x[c]), img_y[r+1]))
            if (v0 > thresh) != (v3 > thresh):
                t = (thresh - v0) / (v3 - v0 + 1e-9)
                edges.append((img_x[c], img_y[r] + t*(img_y[r+1]-img_y[r])))

            if len(edges) == 2:
                segments.append((edges[0], edges[1]))
            elif len(edges) == 4:
                center_val = (v0 + v1 + v2 + v3) / 4.0
                if (center_val > thresh) == (v0 > thresh):
                    segments.append((edges[0], edges[3]))
                    segments.append((edges[1], edges[2]))
                else:
                    segments.append((edges[0], edges[1]))
                    segments.append((edges[2], edges[3]))
    return segments


# ---------------------------------------------------------------------------
# Geometric rendering
# ---------------------------------------------------------------------------

NUM_CONTOURS = 15
bg_rgb = hex_to_rgb(BG)

palette_len = len(COMPONENT_COLOURS)
cycle_length = 2 * palette_len - 2

fade_range = FADE_END - FADE_START if FADE_END != FADE_START else 1.0

for i in range(1, NUM_CONTOURS):
    fraction = (i / NUM_CONTOURS) ** 1.5
    thresh = min_loss + fraction * (max_contour_val - min_loss)

    segments = get_contour_segments(
        grid_losses, thresh, img_x_coords, img_y_coords)

    pos = i % cycle_length
    colour_index = pos if pos < palette_len else cycle_length - pos

    base_colour = COMPONENT_COLOURS[colour_index]
    base_rgb = hex_to_rgb(base_colour)

    for pt1, pt2 in segments:
        x_mid = (pt1[0] + pt2[0]) / 2.0

        if FADE_DIRECTION == "in":
            fade = min(1.0, max(0.0, (x_mid - FADE_START) / fade_range))
        else:
            fade = min(1.0, max(0.0, 1.0 - (x_mid - FADE_START) / fade_range))

        if fade > 0.01:
            blended_rgb = rgb_blend(base_rgb, bg_rgb, fade)
            fill_hex = rgb_to_hex(blended_rgb)

            draw.line([pt1, pt2], fill=fill_hex, width=2)


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
image.save(OUTPUT_PATH, "PNG", optimize=True)

print(f"Saved {args.mode} preview to {OUTPUT_PATH}")
