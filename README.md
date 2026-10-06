# LibreYOLO workshop

A hands-on workshop notebook: train a rock, paper, scissors hand detector on photos your group took and labelled,
compare experiments, and download the model. It runs in Google Colab on a free GPU, so any laptop with a browser works.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EHxuban11/libreyolo-workshop/blob/main/train_hands.ipynb)

## The notebook

[`train_hands.ipynb`](train_hands.ipynb), in nine steps:

1. Install [LibreYOLO](https://github.com/LibreYOLO/libreyolo), the MIT-licensed computer vision library.
2. Pick your experiment card: everyone changes exactly one thing against the baseline.
3. Download the dataset. With the box left empty, it takes the room's dataset if it is published, and the backup if not.
   It also sets the number of epochs from the dataset size, so a run takes about 10 minutes on a T4.
4. Look at the labels.
5. Fine-tune `LibreYOLO9t` (pretrained on COCO), scoring the validation photos after every epoch.
6. Evaluate: mAP50 and mAP50-95, and the validation photos with labels and predictions side by side.
7. Print your line for the leaderboard.
8. Try the model on a photo of your own hand.
9. Download the model as `best.pt` and as ONNX.

Before running it in Colab: **Runtime > Change runtime type > T4 GPU**.

## The dataset

The data lives in the [`data` release](https://github.com/EHxuban11/libreyolo-workshop/releases/tag/data):

| File | What it holds |
| --- | --- |
| `rps-backup.zip` | A ready dataset: 250 training photos and 100 validation photos with Roboflow's labels |
| `rps-kit.zip` | The photos the room labels in Label Studio, their original labels, and the 100 validation photos |
| `rps-room.zip` | Published on the day: the room's own photos and labels, built by `make_dataset.py` |

Classes: `0 rock`, `1 paper`, `2 scissors`. Every zip is a LibreYOLO dataset (`data.yaml` with no `path:` line,
`images/...`, `labels/...`, YOLO-format labels), so any detection dataset in this layout trains the same way:
paste its link in Step 3.

The Roboflow photos and labels come from the
[Rock Paper Scissors dataset](https://universe.roboflow.com/rock-paper-scissors-2/rock-paper-scissors-bwev7) on
Roboflow Universe (version 3), provided by a Roboflow user under CC BY 4.0. The photos on a green background and the
3D-rendered hands are left out; the class names are lowercased and reordered.

## Editing

The notebook is generated: edit the cells in `tools/make_notebook.py`, then run `python3 tools/make_notebook.py`
(needs `nbformat`).

## License

The code is MIT, see [LICENSE](LICENSE). LibreYOLO's code is MIT; pretrained weights keep the license their authors
chose, listed on each model page of the [LibreYOLO docs](https://www.libreyolo.com/docs). The photos and labels in the
data release are CC BY 4.0, from the Roboflow dataset above.

The earlier version of this workshop, with printed road signs, is at the tag `signs-version`.
