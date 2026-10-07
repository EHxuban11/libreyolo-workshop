#!/usr/bin/env python3
"""Write the workshop's training notebook: train_hands.ipynb at the root of this repository.

    python3 tools/make_notebook.py

The notebook is made for Google Colab's free GPU (students' laptops can be slow), and also runs in Jupyter.
Every cell is defined here, so review and edit the cells in this file, then run it again.
Needs nbformat (pip install nbformat).
"""
import pathlib

import nbformat as nbf

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "train_hands.ipynb"

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

cells = [
    md("""# Train your own rock, paper, scissors detector with LibreYOLO

**LibreYOLO hands-on workshop, step 3: train and evaluate.**

You will fine-tune a detector on the hand photos your group took and labelled, with **one change**
written on your experiment card, and report your score to the leaderboard.

**Before you start:** in the menu, **Runtime > Change runtime type > T4 GPU**, then **Save**.
Then run the cells from top to bottom: click a cell and press **Shift + Enter**.

| Step | What happens | Time |
|---|---|---|
| 1 | Install LibreYOLO | 1 min |
| 2 | Choose your experiment card | |
| 3 | Download the dataset | |
| 4 | Look at the labels | |
| 5 | Train | about 10 minutes on the T4 |
| 6 | Evaluate and look at the mistakes | 1 min |
| 7 | Your line for the leaderboard | |
| 8 | Try a photo of your hand | |
| 9 | Download your model | |"""),

    code("""#@title Step 1: install LibreYOLO
%pip install -q "libreyolo[onnx]" gdown pillow-heif

import torch, libreyolo
print("LibreYOLO", libreyolo.__version__, "| PyTorch", torch.__version__)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("No GPU found. Runtime > Change runtime type > T4 GPU, then run this cell again.")"""),

    md("""## Step 2: your experiment card

Everyone changes **exactly one thing** against card A, so the leaderboard shows what each change does.
Every card is scored on the same 100 validation photos, labelled by Roboflow's annotators, that nobody in the room labelled.
Pick the letter on your card in the box on the right, then run the cell.

| Card | Change | The bet |
|---|---|---|
| A | nothing | The baseline: our photos and our labels |
| B | Roboflow's labels | The same photos, but the Roboflow ones keep the labels Roboflow's annotators drew. How much are careful labels worth? |
| C | only our photos | Only the photos the room took, no Roboflow photos. Is a small dataset from the right people enough? |
| D | `imgsz=320` | Four times fewer pixels: faster, but small hands get lost |
| E | a third of the epochs | Too few passes to learn |
| F | `freeze="backbone"` | Keep the COCO features, train only the neck and head |
| G | `LibreYOLO9s.pt` | A bigger model: more accurate, slower |
| H | `lr0=0.001` | A learning rate ten times smaller: careful, and too slow |"""),

    code("""#@title Step 2: choose your card
CARD = "A"  #@param ["A", "B", "C", "D", "E", "F", "G", "H"]

CARDS = {
    "A": ("The baseline",      "LibreYOLO9t.pt", {}),
    "B": ("Roboflow's labels", "LibreYOLO9t.pt", {}),   # other labels: data_B.yaml, Step 3
    "C": ("Only our photos",   "LibreYOLO9t.pt", {}),   # fewer photos: data_C.yaml, Step 3
    "D": ("Small input",       "LibreYOLO9t.pt", {"imgsz": 320}),
    "E": ("Short training",    "LibreYOLO9t.pt", {}),   # a third of the epochs, set in Step 3
    "F": ("Frozen backbone",   "LibreYOLO9t.pt", {"freeze": "backbone"}),
    "G": ("Bigger model",      "LibreYOLO9s.pt", {}),
    "H": ("Tiny steps",        "LibreYOLO9t.pt", {"lr0": 0.001}),
}
TITLE, WEIGHTS, CHANGE = CARDS[CARD]
VARIANT = {"B": "data_B.yaml", "C": "data_C.yaml"}.get(CARD, "data.yaml")
print(f"Card {CARD}: {TITLE}")
print(f"Model: {WEIGHTS}")
print(f"Dataset file: {VARIANT}")"""),

    md("""## Step 3: the dataset

Leave the box empty and run the cell. It downloads the room's dataset, made at coffee time from everyone's labels.
If that is not published yet, it takes the backup dataset (Roboflow's photos and labels) and says so.
Only paste a link if the slide tells you to.

It then shows how many photos and boxes each class has, and sets the number of epochs so that training takes
about 10 minutes on Colab's T4.

The Roboflow photos come from the [Rock Paper Scissors dataset](https://universe.roboflow.com/rock-paper-scissors-2/rock-paper-scissors-bwev7)
on Roboflow Universe, CC BY 4.0."""),

    code("""#@title Step 3: download the dataset
DATASET_URL = ""  #@param {type:"string"}

import pathlib, shutil, urllib.request, urllib.error, zipfile, collections, yaml

RELEASE = "https://github.com/EHxuban11/libreyolo-workshop/releases/download/data/"
ROOM, BACKUP = RELEASE + "rps-room.zip", RELEASE + "rps-backup.zip"

def published(url):
    try:
        urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=30)
        return True
    except Exception:
        return False

url = DATASET_URL.strip()
if not url:
    url = ROOM if published(ROOM) else BACKUP
    print("Dataset:", "the room's photos and labels" if url == ROOM else
          "the backup, Roboflow's photos and labels (the room's dataset is not published yet)")
ZIP = pathlib.Path("dataset.zip")
if "drive.google.com" in url:
    import gdown
    gdown.download(url, str(ZIP), quiet=True, fuzzy=True)
else:
    urllib.request.urlretrieve(url, ZIP)

ROOT = pathlib.Path("dataset")
shutil.rmtree(ROOT, ignore_errors=True)
with zipfile.ZipFile(ZIP) as z:
    z.extractall(ROOT)
BASE = next(ROOT.rglob("data.yaml"))   # card A's description of the dataset
DATASET = BASE.parent
DATA = DATASET / VARIANT                # your card's
if not DATA.exists():
    print(f"This dataset has no {VARIANT}, so card {CARD} trains on data.yaml, like card A.")
    DATA = BASE

def folders(data_yaml):
    \"\"\"The training folders and the class names of a data.yaml.\"\"\"
    d = yaml.safe_load(data_yaml.read_text())
    train = d["train"] if isinstance(d["train"], list) else [d["train"]]
    names = d["names"]
    names = dict(enumerate(names)) if isinstance(names, list) else {int(k): v for k, v in names.items()}
    return [DATASET / t for t in train], names

def photos(dirs):
    return sorted(p for d in dirs for p in d.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"))

TRAIN_DIRS, NAMES = folders(DATA)
VAL_DIR = DATASET / "images" / "val"
for title, dirs in (("train", TRAIN_DIRS), ("val", [VAL_DIR])):
    for d in dirs:
        labels = sorted((d.parent.parent / "labels" / d.name).glob("*.txt"))
        boxes = collections.Counter(NAMES[int(line.split()[0])] for f in labels for line in f.read_text().splitlines() if line.strip())
        empty = sum(1 for f in labels if not f.read_text().strip())
        print(f"{title:5} {d.name:8} {len(labels):4} photos ({empty} with no hand), boxes: {dict(sorted(boxes.items()))}")

# Epochs: the same for every card, from card A's dataset, so that a run takes about 10 minutes on a T4.
# The estimate comes from the dry run on Colab's T4: about 7 seconds per epoch for 110 photos (train and val).
n_base, n_card, n_val = len(photos(folders(BASE)[0])), len(photos(TRAIN_DIRS)), len(photos([VAL_DIR]))
EPOCHS = max(8, min(50, round(10 * 60 / (7 * (n_base + n_val) / 110))))
epochs = max(3, EPOCHS // 3) if CARD == "E" else EPOCHS
SETTINGS = {"epochs": epochs, "imgsz": 640, "batch": 16, **CHANGE}
minutes = max(1, round(epochs * 7 * (n_card + n_val) / 110 / 60))
print(f"\\nSettings: {SETTINGS}")
print(f"Expected time on a T4: about {minutes} minute{'s' if minutes > 1 else ''}" + {
    "D": " or less (smaller photos)", "F": " or less (fewer layers learn)", "G": " or more (a bigger model)"}.get(CARD, ""))"""),

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

train_images = photos(TRAIN_DIRS)
sample = random.Random(0).sample(train_images, min(6, len(train_images)))
def count(n, word):
    return f"{n} {word}{'' if n == 1 else ('es' if word.endswith('x') else 's')}"

show([draw(p, label_boxes(p)) for p in sample], [count(len(label_boxes(p)), "box") for p in sample])"""),

    md("""## Step 5: train

This fine-tunes a model that already learned to see on COCO (118,287 photos) so that it learns our three classes:
rock, paper and scissors. Each **epoch** is one pass over the training photos; after each one the model is scored on
the validation photos, which it never trains on. Watch the loss go down and the mAP go up. The mAP can stay near zero
for the first few epochs: that is normal."""),

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

val_images = photos([VAL_DIR])
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

Take a photo of your hand showing rock, paper or scissors, upload it here, and see what your model finds.
Try a hand it has never seen: yours, a friend's, on a new background. No photo? Press **Cancel upload** and go on to Step 9."""),

    code("""#@title Step 8: try your own photo
try:
    from google.colab import files
    uploaded = list(files.upload())
except ImportError:
    uploaded = []
    print("Outside Colab: set uploaded = ['path/to/photo.jpg'] and run this cell again.")
except KeyboardInterrupt:
    uploaded = []
    print("No photo this time. Go on to Step 9.")
if not uploaded:
    print("Nothing uploaded; the cell does nothing without a photo.")
from PIL import ImageOps
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()                 # so iPhone photos (HEIC) open too
except ImportError:
    pass
for name in uploaded:
    photo = pathlib.Path(name).with_suffix(".upright.jpg")
    ImageOps.exif_transpose(Image.open(name)).convert("RGB").save(photo)   # upright JPEG, whatever the phone sent
    r = best(str(photo), conf=0.25)
    boxes = [(int(c), *xyxy) for c, xyxy in zip(r.boxes.cls.tolist(), r.boxes.xyxy.tolist())]
    show([draw(photo, boxes, r.boxes.conf.tolist())], [f"{len(boxes)} found"])"""),

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
- **The dataset does not download:** leave the box in Step 3 empty and run it again. A pasted Google Drive link must be
  shared as "Anyone with the link".
- **Out of memory:** change `"batch": 16` to `"batch": 8` in Step 3.
- **Colab disconnected:** Runtime > Run all. Training starts again from the beginning.

Model code: MIT licensed, [github.com/LibreYOLO/libreyolo](https://github.com/LibreYOLO/libreyolo).
Documentation: [libreyolo.com/docs](https://www.libreyolo.com/docs).
Roboflow photos and labels: [Rock Paper Scissors on Roboflow Universe](https://universe.roboflow.com/rock-paper-scissors-2/rock-paper-scissors-bwev7),
CC BY 4.0."""),
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
