#!/usr/bin/env python3
"""Write the workshop's simple training notebook: train_simple.ipynb at the root of this repository.

    python3 tools/make_simple_notebook.py

The short version of train_hands.ipynb (tools/make_notebook.py): no experiment cards, no options,
a few lines per cell. Made for Google Colab's free T4 GPU.
Every cell is defined here, so review and edit the cells in this file, then run it again.
Needs nbformat (pip install nbformat).
"""
import pathlib

import nbformat as nbf

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "train_simple.ipynb"

# About 10 minutes on a T4: about 7 s per epoch per 110 photos, train and val (dry run on Colab).
EPOCHS = 25

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

cells = [
    md("""# Train a rock, paper, scissors detector with LibreYOLO

**Runtime > Change runtime type > T4 GPU**, then **Runtime > Run all**."""),

    md("Install LibreYOLO."),
    code("""%pip install -q "libreyolo[onnx]" pillow-heif"""),

    md("""Download today's dataset: the photos we took and labelled this morning, from Hugging Face."""),
    code("""from huggingface_hub import snapshot_download

snapshot_download("Xuban11/rock-paper-scissors-mondragon", repo_type="dataset", local_dir="dataset")"""),

    md("""Train. The model already knows how to see from COCO, and now it learns rock, paper and scissors.
This takes about 10 minutes."""),
    code(f"""from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLO9t.pt")
results = model.train(data="dataset/data.yaml", epochs={EPOCHS}, imgsz=640)"""),

    md("Score the best model on the validation photos, which it never trained on. mAP50 goes from 0 to 1: higher is better."),
    code("""best = LibreYOLO(results["best_checkpoint"])
metrics = best.val(data="dataset/data.yaml")
print("mAP50:", round(metrics["metrics/mAP50"], 3))"""),

    md("Look at what it finds on a few validation photos."),
    code("""import glob

for photo in sorted(glob.glob("dataset/images/val/*"))[:4]:
    display(best(photo).plot(pil=True))"""),

    md("Download your model: best.pt for Python, best.onnx to run it anywhere."),
    code("""from google.colab import files

files.download(results["best_checkpoint"])
files.download(best.export(format="onnx"))"""),

    md("Take a photo of your hand and upload it to see what your model finds. iPhone photos work too."),
    code("""from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

register_heif_opener()
for name in files.upload():
    display(best(ImageOps.exif_transpose(Image.open(name)).convert("RGB")).plot(pil=True))"""),

    md("""The Roboflow photos and labels come from the [Rock Paper Scissors dataset](https://universe.roboflow.com/rock-paper-scissors-2/rock-paper-scissors-bwev7) on Roboflow Universe, CC BY 4.0."""),
]

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "accelerator": "GPU",
    "colab": {"provenance": [], "gpuType": "T4"},
    "kernelspec": {"display_name": "Python 3", "name": "python3"},
    "language_info": {"name": "python"},
}
nbf.write(nb, OUT)
print(f"Wrote {OUT.relative_to(ROOT)}: {len(cells)} cells")
