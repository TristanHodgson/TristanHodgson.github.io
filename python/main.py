"""
Social Preview Image Generator: "Mathematics through Computation"

THE MATHEMATICS:
This script visualises a 2D slice of a high-dimensional neural network loss 
landscape. It is a genuine, mathematically precise topographical map computed 
in pure Python.

1. The Model: We define a SIREN (Sinusoidal Representation Network) with 
   1 input, 5 hidden units (using sine activations), and 1 output. Because 
   of the periodic activations, its loss landscape is highly non-convex, 
   featuring beautiful interference patterns, multiple basins, and complex 
   saddle points.
2. The Loss: We evaluate the Mean Squared Error (MSE) of this model attempting 
   to fit a high-frequency signal, creating a continuous scalar field in 
   16-dimensional space.
3. The Projection: We generate a random center point in this 16D space, and 
   two random direction vectors. We use Gram-Schmidt orthogonalisation to 
   ensure these vectors form a flat, Euclidean 2D plane. 
4. The Framing: We identify the set A of all true local minima in the plane. 
   We then find the pair w, z in A such that the Euclidean distance d(w, z) 
   is minimal. By mapping this closest pair of adjacent basins to the bottom-
   right and top-left (off-canvas), we perfectly frame the sharpest, most 
   prominent saddle structure in the landscape.

THE COLOUR MAPPING:
To avoid "business" and harsh transitions, we use a restricted 5-colour Nord 
palette. Instead of abruptly wrapping from the last colour back to the first, 
we use a triangular wave function to "bounce" back and forth through the 
palette (0→1→2→3→4→3→2→1→0), guaranteeing smooth colour gradients across the 
entire topography.
"""

from pathlib import Path
import math
import random

from PIL import Image, ImageDraw, ImageFont


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WIDTH = 1200
HEIGHT = 627

# Website / Nord colours
BG = "#e5e9f0"
FG = "#2e3440"
SECONDARY = "#4c566a"
ACCENT = "#5e81ac"

# Refined icy Nord palette ending in purple
COMPONENT_COLOURS = [
    "#8fbcbb",
    "#88c0d0",
    "#81a1c1",
    "#5e81ac",
    "#b48ead",
]

REPO_ROOT = Path(__file__).resolve().parent.parent
FONT_PATH = REPO_ROOT / "fonts" / "firacode.ttf"
OUTPUT_PATH = REPO_ROOT / "img" / "social-preview.png"


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

name_font = font(72)
subtitle_font = font(32)
detail_font = font(27)
url_font = font(23)

x_text = 90

draw.text((x_text, 145), name, font=name_font, fill=FG, stroke_width=1, stroke_fill=FG)
draw.rounded_rectangle((x_text, 255, x_text + 105, 262), radius=3, fill=ACCENT)
draw.text((x_text, 300), subtitle, font=subtitle_font, fill=FG)
draw.text((x_text, 360), detail, font=detail_font, fill=SECONDARY)
draw.text((x_text, 475), url, font=url_font, fill=ACCENT)


# ---------------------------------------------------------------------------
# Computational Mathematics: SIREN ML Model & Loss Landscape
# ---------------------------------------------------------------------------

# A tiny Sinusoidal Representation Network (SIREN): 1 input, 5 hidden units, 1 output.
NUM_PARAMS = 16

# Synthetic high-frequency dataset to induce non-convexity
X_DATA = [-1.0 + i * (2.0 / 7.0) for i in range(8)]
Y_DATA = [math.sin(5.0 * x) for x in X_DATA]

def evaluate_loss(theta):
    """Compute MSE loss for the tiny SIREN given parameter vector theta."""
    w1, b1 = theta[0:5], theta[5:10]
    w2, b2 = theta[10:15], theta[15]
    
    loss = 0.0
    for x, y_true in zip(X_DATA, Y_DATA):
        # Periodic sine activations induce complex, multi-basin landscapes
        hidden = [math.sin(w1[i] * x + b1[i]) for i in range(5)]
        y_pred = sum(w2[i] * hidden[i] for i in range(5)) + b2
        loss += (y_pred - y_true) ** 2
        
    return loss / len(X_DATA)

random.seed(42)

# Generate a random center point in parameter space
theta_0 = [random.uniform(-1, 1) for _ in range(NUM_PARAMS)]

# Generate two random direction vectors to define our 2D slice
d1 = [random.gauss(0, 1) for _ in range(NUM_PARAMS)]
d2 = [random.gauss(0, 1) for _ in range(NUM_PARAMS)]

# Gram-Schmidt orthogonalisation
norm1 = math.sqrt(sum(x*x for x in d1))
d1 = [x / norm1 for x in d1]

dot_product = sum(x*y for x, y in zip(d1, d2))
d2 = [d2[i] - dot_product * d1[i] for i in range(NUM_PARAMS)]

norm2 = math.sqrt(sum(x*x for x in d2))
d2 = [x / norm2 for x in d2]


# ---------------------------------------------------------------------------
# Algorithmic Framing: Find the Closest Pair of Local Minima
# ---------------------------------------------------------------------------

search_coords = [x * 0.5 for x in range(-40, 41)]
search_grid = []

# 1. Evaluate a coarse grid over the plane
for u_test in search_coords:
    row = []
    for v_test in search_coords:
        theta = [theta_0[i] + u_test * d1[i] + v_test * d2[i] for i in range(NUM_PARAMS)]
        row.append(evaluate_loss(theta))
    search_grid.append(row)

# 2. Identify the set A of all true local minima
local_minima = []
for i in range(1, len(search_coords) - 1):
    for j in range(1, len(search_coords) - 1):
        val = search_grid[i][j]
        # Check against 8-way neighbors
        if (val < search_grid[i-1][j] and val < search_grid[i+1][j] and
            val < search_grid[i][j-1] and val < search_grid[i][j+1] and
            val < search_grid[i-1][j-1] and val < search_grid[i+1][j+1] and
            val < search_grid[i-1][j+1] and val < search_grid[i+1][j-1]):
            local_minima.append((search_coords[i], search_coords[j], val))

# 3. Find w, z in A such that d(w, z) is minimal
min_dist = float('inf')
closest_pair = None

for i in range(len(local_minima)):
    for j in range(i + 1, len(local_minima)):
        u_w, v_w, val_w = local_minima[i]
        u_z, v_z, val_z = local_minima[j]
        d = math.hypot(u_w - u_z, v_w - v_z)
        
        if d < min_dist:
            min_dist = d
            # Order them so the deeper of the two goes to the bottom right
            if val_w < val_z:
                closest_pair = ((u_w, v_w), (u_z, v_z))
            else:
                closest_pair = ((u_z, v_z), (u_w, v_w))

if closest_pair:
    u1, v1 = closest_pair[0]
    u2, v2 = closest_pair[1]
else:
    # Fallback if the space lacks multiple distinct minima
    u1, v1 = 0.0, 0.0
    u2, v2 = 2.0, 2.0


# 4. Calculate mapping from Canvas Pixels directly to Mathematical Plane
IMG1_X, IMG1_Y = 1125, 553   # First min (w) -> Bottom Right
IMG2_X, IMG2_Y = 650, -150   # Second min (z) -> Top Left (Above canvas and behind fade mask)

v_img_x = IMG2_X - IMG1_X
v_img_y = IMG2_Y - IMG1_Y
len_img = math.hypot(v_img_x, v_img_y)

v_math_u = u2 - u1
v_math_v = v2 - v1
len_math = math.hypot(v_math_u, v_math_v)

# Linear transform parameters
scale = len_math / len_img
angle_img = math.atan2(v_img_y, v_img_x)
angle_math = math.atan2(v_math_v, v_math_u)
rotation = angle_math - angle_img

cos_rot = math.cos(rotation)
sin_rot = math.sin(rotation)


# ---------------------------------------------------------------------------
# Grid Evaluation & Marching Squares Algorithm
# ---------------------------------------------------------------------------

GRID_RES = 160

# We expand the evaluation grid slightly so that the structure around the 
# hidden top-left minimum is captured before it sweeps into the frame.
X_MIN_IMG, X_MAX_IMG = 600, 1400
Y_MIN_IMG, Y_MAX_IMG = -250, 800

img_x_coords = [X_MIN_IMG + c * (X_MAX_IMG - X_MIN_IMG) / (GRID_RES - 1) for c in range(GRID_RES)]
img_y_coords = [Y_MIN_IMG + r * (Y_MAX_IMG - Y_MIN_IMG) / (GRID_RES - 1) for r in range(GRID_RES)]

grid_losses = [[0.0 for _ in range(GRID_RES)] for _ in range(GRID_RES)]
all_losses = []

for r in range(GRID_RES):
    y_img = img_y_coords[r]
    dy = y_img - IMG1_Y
    
    for c in range(GRID_RES):
        x_img = img_x_coords[c]
        dx = x_img - IMG1_X
        
        # Apply transformation to map pixel back to mathematical coordinates
        du = scale * (dx * cos_rot - dy * sin_rot)
        dv = scale * (dx * sin_rot + dy * cos_rot)
        
        u = u1 + du
        v = v1 + dv
        
        theta = [theta_0[i] + u * d1[i] + v * d2[i] for i in range(NUM_PARAMS)]
        loss = evaluate_loss(theta)
        
        grid_losses[r][c] = loss
        all_losses.append(loss)

# Percentile cutoff smoothly clips off the steep outer walls of the basins
all_losses.sort()
min_loss = all_losses[0]
max_contour_val = all_losses[int(len(all_losses) * 0.85)]

def get_contour_segments(grid, thresh, img_x, img_y):
    """A compact marching squares algorithm to extract continuous level sets."""
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

NUM_CONTOURS = 12
bg_rgb = hex_to_rgb(BG)

palette_len = len(COMPONENT_COLOURS)
cycle_length = 2 * palette_len - 2

for i in range(1, NUM_CONTOURS):
    # Map the contour smoothly between the minimum and the 85th percentile.
    fraction = (i / NUM_CONTOURS) ** 1.5
    thresh = min_loss + fraction * (max_contour_val - min_loss)
    
    segments = get_contour_segments(grid_losses, thresh, img_x_coords, img_y_coords)
    
    # Calculate the bouncing index (triangular wave mapping)
    pos = i % cycle_length
    colour_index = pos if pos < palette_len else cycle_length - pos
    
    base_colour = COMPONENT_COLOURS[colour_index]
    base_rgb = hex_to_rgb(base_colour)
    
    for pt1, pt2 in segments:
        x_mid = (pt1[0] + pt2[0]) / 2.0
        
        # Absolute guarantee against text overlap.
        # Opacity remains strictly 0.0 until x = 800 (well past the text bounds).
        # It smoothly fades to full opacity by x = 950.
        fade = min(1.0, max(0.0, (x_mid - 800) / 150.0))
        
        if fade > 0.01:
            blended_rgb = rgb_blend(base_rgb, bg_rgb, fade)
            fill_hex = rgb_to_hex(blended_rgb)
            
            draw.line([pt1, pt2], fill=fill_hex, width=2)


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
image.save(OUTPUT_PATH, "PNG", optimize=True)

print(f"Saved preview to {OUTPUT_PATH}")