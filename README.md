# LibreYOLO workshop

A hands-on workshop notebook: train an object detector on photos your group collected and labelled, compare
experiments, and download the model. It runs in Google Colab on a free GPU, so any laptop with a browser works.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EHxuban11/libreyolo-workshop/blob/main/train_signs.ipynb)

## The notebook

[`train_signs.ipynb`](train_signs.ipynb), in nine steps:

1. Install [LibreYOLO](https://github.com/LibreYOLO/libreyolo), the MIT-licensed computer vision library.
2. Pick your experiment card: everyone trains on the same photos and changes exactly one setting.
3. Download the dataset from the link your workshop gives you.
4. Look at the labels.
5. Fine-tune `LibreYOLO9t` (pretrained on COCO), scoring the validation photos after every epoch.
6. Evaluate: mAP50 and mAP50-95, and the validation photos with labels and predictions side by side.
7. Print your line for the leaderboard.
8. Try the model on a photo of your own.
9. Download the model as `best.pt` and as ONNX.

Before running it in Colab: **Runtime > Change runtime type > T4 GPU**.

## The dataset

The notebook expects a link to a `.zip` holding a LibreYOLO dataset: `data.yaml` (with no `path:` line),
`images/train`, `images/val`, `labels/train` and `labels/val`, with YOLO-format labels. A Google Drive share link
("Anyone with the link") or any direct download link works.

The workshop's six classes are five road signs (stop, left, right, u-turn, park) and a target, but the notebook
reads the class names from `data.yaml`, so any detection dataset in this layout trains the same way.

## Editing

The notebook is generated: edit the cells in `tools/make_notebook.py`, then run `python3 tools/make_notebook.py`
(needs `nbformat`).

## License

MIT, see [LICENSE](LICENSE). LibreYOLO's code is MIT; pretrained weights keep the license their authors chose,
listed on each model page of the [LibreYOLO docs](https://www.libreyolo.com/docs).
