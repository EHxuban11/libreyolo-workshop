#!/usr/bin/env python3
"""Write the workshop's training notebook: train_signs.ipynb at the root of this repository.

    python3 tools/make_notebook.py

The notebook is made for Google Colab's free GPU (students' laptops can be slow), and also runs in Jupyter.
Every cell is defined here, so review and edit the cells in this file, then run it again.
Needs nbformat (pip install nbformat).
"""
import pathlib

import nbformat as nbf

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "train_signs.ipynb"

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

cells = [
    md("""# Train your own sign detector with LibreYOLO

**LibreYOLO hands-on workshop, step 3: train and evaluate.**

You will fine-tune a detector on the photos your group collected and labelled, with **one change**
written on your experiment card, and report your score to the leaderboard.

**Before you start:** in the menu, **Runtime > Change runtime type > T4 GPU**, then **Save**.
Then run the cells from top to bottom: click a cell and press **Shift + Enter**.

| Step | What happens | Time |
|---|---|---|
| 1 | Install LibreYOLO | 1 min |
| 2 | Choose your experiment card | |
| 3 | Download the dataset | |
| 4 | Look at the labels | |
| 5 | Train | a few minutes on the T4 |
| 6 | Evaluate and look at the mistakes | 1 min |
| 7 | Your line for the leaderboard | |
| 8 | Try your own photo | |
| 9 | Download your model | |"""),

    code("""#@title Step 1: install LibreYOLO
%pip install -q "libreyolo[onnx]" gdown

import torch, libreyolo
print("LibreYOLO", libreyolo.__version__, "| PyTorch", torch.__version__)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("No GPU found. Runtime > Change runtime type > T4 GPU, then run this cell again.")"""),

    md("""## Step 2: your experiment card

Everyone trains on the same photos and changes **exactly one thing**, so the leaderboard shows what each change does.
Pick the letter on your card in the box on the right, then run the cell.

| Card | Change | The bet |
|---|---|---|
| A | nothing | The baseline everyone compares against |
| B | `flip_prob=0` | Mirroring turns a left arrow into a right arrow |
| C | `mosaic_prob=0` | With so few photos, stitching four into one may hurt |
| D | `imgsz=320` | Four times fewer pixels: faster, but far signs get lost |
| E | `epochs=10` | Ten passes are not enough to learn |
| F | `freeze="backbone"` | Keep the COCO features, train only the neck and head |
| G | `LibreYOLO9s.pt` | A bigger model: more accurate, slower |
| H | `lr0=0.001` | A learning rate ten times smaller: careful, and too slow |"""),

    code("""#@title Step 2: choose your card
CARD = "A"  #@param ["A", "B", "C", "D", "E", "F", "G", "H"]

CARDS = {
    "A": ("The baseline",    "LibreYOLO9t.pt", {}),
    "B": ("No flips",        "LibreYOLO9t.pt", {"flip_prob": 0}),
    "C": ("No mosaic",       "LibreYOLO9t.pt", {"mosaic_prob": 0}),
    "D": ("Small input",     "LibreYOLO9t.pt", {"imgsz": 320}),
    "E": ("Short training",  "LibreYOLO9t.pt", {"epochs": 10}),
    "F": ("Frozen backbone", "LibreYOLO9t.pt", {"freeze": "backbone"}),
    "G": ("Bigger model",    "LibreYOLO9s.pt", {}),
    "H": ("Tiny steps",      "LibreYOLO9t.pt", {"lr0": 0.001}),
}
TITLE, WEIGHTS, CHANGE = CARDS[CARD]
# The same base settings for everyone; the card changes one of them.
SETTINGS = {"epochs": 50, "imgsz": 640, "batch": 16, **CHANGE}
print(f"Card {CARD}: {TITLE}")
print(f"Model: {WEIGHTS}")
print(f"Settings: {SETTINGS}")"""),

    md("""## Step 3: the dataset

Paste the dataset link from the slide into the box on the right (a Google Drive share link or any direct link to a
`.zip`), then run the cell. It downloads the photos and labels and shows how many boxes each class has."""),

    code("""#@title Step 3: download the dataset
DATASET_URL = ""  #@param {type:"string"}

import pathlib, shutil, urllib.request, zipfile, collections

if not DATASET_URL.strip():
    raise ValueError("Paste the dataset link from the slide into DATASET_URL, then run this cell again.")

ZIP = pathlib.Path("dataset.zip")
if "drive.google.com" in DATASET_URL:
    import gdown
    gdown.download(DATASET_URL.strip(), str(ZIP), quiet=True, fuzzy=True)
else:
    urllib.request.urlretrieve(DATASET_URL.strip(), ZIP)

ROOT = pathlib.Path("dataset")
shutil.rmtree(ROOT, ignore_errors=True)
with zipfile.ZipFile(ZIP) as z:
    z.extractall(ROOT)
DATA = next(ROOT.rglob("data.yaml"))   # the dataset's description file
DATASET = DATA.parent

import yaml
NAMES = yaml.safe_load(DATA.read_text())["names"]
NAMES = dict(enumerate(NAMES)) if isinstance(NAMES, list) else {int(k): v for k, v in NAMES.items()}
for split in ("train", "val"):
    labels = sorted((DATASET / "labels" / split).glob("*.txt"))
    boxes = collections.Counter(NAMES[int(line.split()[0])] for f in labels for line in f.read_text().splitlines() if line.strip())
    empty = sum(1 for f in labels if not f.read_text().strip())
    print(f"{split:5}: {len(labels):3} photos ({empty} with no sign), boxes per class: {dict(sorted(boxes.items()))}")"""),

    code("""#@title Step 4: look at the labels
import random
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

COLORS = ["#22d3ee", "#f59e0b", "#a78bfa", "#34d399", "#f472b6", "#60a5fa", "#f87171", "#facc15"]

def label_boxes(image_path):
    \"\"\"The labelled boxes of one photo, in pixels: (class id, x1, y1, x2, y2).\"\"\"
    w, h = Image.open(image_path).size
    label = image_path.parent.parent.parent / "labels" / image_path.parent.name / (image_path.stem + ".txt")
    out = []
    for line in label.read_text().splitlines() if label.exists() else []:
        c, cx, cy, bw, bh = line.split()[:5]
        cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
        out.append((int(c), cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2))
    return out

def draw(image_path, boxes, scores=None, dashed=()):
    \"\"\"A photo with boxes drawn on it. boxes: (class id, x1, y1, x2, y2); scores: one per box or None.\"\"\"
    im = Image.open(image_path).convert("RGB")
    d = ImageDraw.Draw(im)
    width = max(2, im.width // 320)
    try:
        font = ImageFont.load_default(size=max(14, im.width // 36))
    except TypeError:  # Pillow older than 10.1
        font = ImageFont.load_default()
    for c, x1, y1, x2, y2 in dashed:  # the labels, thin and white, under the predictions
        d.rectangle([x1, y1, x2, y2], outline="white", width=max(1, width // 2))
    for i, (c, x1, y1, x2, y2) in enumerate(boxes):
        color = COLORS[c % len(COLORS)]
        d.rectangle([x1, y1, x2, y2], outline=color, width=width)
        text = NAMES[c] + (f" {scores[i]:.2f}" if scores is not None else "")
        l, t, r, b = d.textbbox((0, 0), text, font=font)
        top = max(0, y1 - (b - t) - 8)
        d.rectangle([x1, top, x1 + (r - l) + 8, top + (b - t) + 8], fill=color)
        d.text((x1 + 4, top + 4 - t), text, fill="black", font=font)
    return im

def show(images, titles):
    cols = 3
    rows = (len(images) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4.2 * rows))
    for ax in axes.flat:
        ax.axis("off")
    for ax, im, t in zip(axes.flat, images, titles):
        ax.imshow(im)
        ax.set_title(t, fontsize=10)
    plt.tight_layout()
    plt.show()

train_images = sorted((DATASET / "images" / "train").glob("*"))
sample = random.Random(0).sample(train_images, min(6, len(train_images)))
def count(n, word):
    return f"{n} {word}{'' if n == 1 else ('es' if word.endswith('x') else 's')}"

show([draw(p, label_boxes(p)) for p in sample], [count(len(label_boxes(p)), "box") for p in sample])"""),

    md("""## Step 5: train

This fine-tunes a model that already learned to see on COCO (118,287 photos) so that it learns our six classes.
Each **epoch** is one pass over the training photos; after each one the model is scored on the validation photos,
which it never trains on. Watch the loss go down and the mAP go up."""),

    code("""#@title Step 5: train
import time
from libreyolo import LibreYOLO

model = LibreYOLO(WEIGHTS)                 # pretrained on COCO
start = time.time()
RUN = model.train(
    data=str(DATA),
    **SETTINGS,
    eval_interval=1,                       # score the validation photos after every epoch
    project="runs", name=f"card-{CARD}", exist_ok=True,
)
MINUTES = (time.time() - start) / 60
print(f"Trained in {MINUTES:.1f} minutes. Best epoch: {RUN['best_epoch']}. Best model: {RUN['best_checkpoint']}")"""),

    code("""#@title The training curves
import csv
from matplotlib.ticker import MaxNLocator

rows = list(csv.DictReader(open(pathlib.Path(RUN["save_dir"]) / "results.csv")))
epochs = [int(r["epoch"]) for r in rows]
fig, (a, b) = plt.subplots(1, 2, figsize=(14, 4))
a.plot(epochs, [float(r["train/loss"]) for r in rows], color="#0891b2")
a.set_title("Training loss: how wrong on the training photos")
a.set_xlabel("epoch")
scored = [r for r in rows if r.get("metrics/mAP50-95")]
b.plot([int(r["epoch"]) for r in scored], [float(r["metrics/mAP50"]) for r in scored], label="mAP50", color="#059669")
b.plot([int(r["epoch"]) for r in scored], [float(r["metrics/mAP50-95"]) for r in scored], label="mAP50-95", color="#0891b2")
b.set_title("Validation: how good on photos it never trained on")
b.set_xlabel("epoch")
b.set_ylim(0, 1)
b.legend()
for ax in (a, b):
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.show()"""),

    md("""## Step 6: evaluate

The best epoch is scored once more on the validation photos. Then look at what it got right and wrong:
the thin white boxes are the labels, the coloured boxes are the model's predictions."""),

    code("""#@title Step 6: evaluate and look at the mistakes
best = LibreYOLO(RUN["best_checkpoint"])
metrics = best.val(data=str(DATA))
MAP50, MAP = metrics["metrics/mAP50"], metrics["metrics/mAP50-95"]
print(f"\\nmAP50 {MAP50:.3f}   mAP50-95 {MAP:.3f}")

val_images = sorted((DATASET / "images" / "val").glob("*"))
panels, titles = [], []
for p in val_images[:9]:
    r = best(str(p), conf=0.25)
    boxes = [(int(c), *xyxy) for c, xyxy in zip(r.boxes.cls.tolist(), r.boxes.xyxy.tolist())]
    truth = label_boxes(p)
    panels.append(draw(p, boxes, r.boxes.conf.tolist(), dashed=truth))
    titles.append(f"{len(truth)} labelled, {len(boxes)} predicted")
show(panels, titles)"""),

    code("""#@title Step 7: your line for the leaderboard
print(f"Card {CARD} ({TITLE}):  mAP50 {MAP50:.2f}   mAP50-95 {MAP:.2f}   {MINUTES:.1f} min")
print("Give these three numbers to the leaderboard. Then: why do you think your card scored like this?")"""),

    md("""## Step 8: try your own photo

Take a photo of one of the signs with your phone, upload it here, and see what your model finds."""),

    code("""#@title Step 8: try your own photo
try:
    from google.colab import files
    uploaded = list(files.upload())
except ImportError:
    uploaded = []
    print("Outside Colab: set uploaded = ['path/to/photo.jpg'] and run this cell again.")
for name in uploaded:
    r = best(name, conf=0.25)
    boxes = [(int(c), *xyxy) for c, xyxy in zip(r.boxes.cls.tolist(), r.boxes.xyxy.tolist())]
    show([draw(pathlib.Path(name), boxes, r.boxes.conf.tolist())], [f"{len(boxes)} found"])"""),

    code("""#@title Step 9: download your model
ONNX = best.export(format="onnx")          # the same model as one portable file
print("Exported", ONNX)
try:
    from google.colab import files
    files.download(RUN["best_checkpoint"])  # best.pt: runs with LibreYOLO("best.pt")
    files.download(str(ONNX))                # best.onnx: runs with LibreYOLO("best.onnx") or ONNX Runtime
except ImportError:
    print("Outside Colab: the files are", RUN["best_checkpoint"], "and", ONNX)"""),

    md("""## If something goes wrong

- **No GPU, or training is very slow:** Runtime > Change runtime type > T4 GPU, then run everything again from Step 1.
- **The dataset does not download:** check the link on the slide; a Google Drive file must be shared as
  "Anyone with the link".
- **Out of memory:** change `"batch": 16` to `"batch": 8` in Step 2.
- **Colab disconnected:** Runtime > Run all. Training starts again from the beginning.

Model code: MIT licensed, [github.com/LibreYOLO/libreyolo](https://github.com/LibreYOLO/libreyolo).
Documentation: [libreyolo.com/docs](https://www.libreyolo.com/docs)."""),
]

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "accelerator": "GPU",
    "colab": {"provenance": [], "gpuType": "T4", "toc_visible": True},
    "kernelspec": {"display_name": "Python 3", "name": "python3"},
    "language_info": {"name": "python"},
}
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print(f"Wrote {OUT.relative_to(ROOT)}: {len(cells)} cells")
