# Some code to generate the social image that appears on LinkedIn and other social media platforms when linking to the site.

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


# LinkedIn / Open Graph dimensions
WIDTH = 1200
HEIGHT = 627

# Colours matching the website
BG = "#e5e9f0"
FG = "#2e3440"
ACCENT = "#5e81ac"
SECONDARY = "#4c566a"

# main.py is located in <repo>/python/
REPO_ROOT = Path(__file__).resolve().parent.parent
FONT_PATH = REPO_ROOT / "fonts" / "firacode.ttf"
OUTPUT_PATH = REPO_ROOT / "img" / "social-preview.png"


def font(size):
    return ImageFont.truetype(str(FONT_PATH), size)


# Create image
image = Image.new("RGB", (WIDTH, HEIGHT), BG)
draw = ImageDraw.Draw(image)

# Typography
name_font = font(72)
subtitle_font = font(32)
detail_font = font(27)
url_font = font(23)

# Content
name = "Tristan Hodgson"
subtitle = "Mathematics @ University of Oxford"
detail = "Mathematics through Computation"
url = "tristanhodgson.com"

# Left-aligned content
x = 110
name_y = 155

# Name
draw.text(
    (x, name_y),
    name,
    font=name_font,
    fill=FG,
    stroke_width=1,
    stroke_fill=FG,
)

# Accent rule
rule_y = 260
draw.rounded_rectangle(
    (x, rule_y, x + 105, rule_y + 7),
    radius=3,
    fill=ACCENT,
)

# Subtitle
draw.text(
    (x, 305),
    subtitle,
    font=subtitle_font,
    fill=FG,
)

# Areas of interest
draw.text(
    (x, 365),
    detail,
    font=detail_font,
    fill=SECONDARY,
)

# Website
draw.text(
    (x, 475),
    url,
    font=url_font,
    fill=ACCENT,
)

# Save image
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
image.save(OUTPUT_PATH, "PNG", optimize=True)

print(f"Saved preview to {OUTPUT_PATH}")