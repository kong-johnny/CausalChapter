# Processed AVLecture inputs

The JSONL files contain the exact processed AVLecture transcripts, utterance
timestamps, segmentation labels, and chapter annotations used by this project.
The original train/dev/test splits contain 245/35/70 videos. These files were
recovered from the author's experiment backup and their test-file hashes were
checked against the training server.

Source dataset and download documentation:
https://github.com/Darshansingh11/AVLectures

Please cite the AVLectures dataset paper, *Unsupervised Audio-Visual Lecture
Segmentation* (WACV 2023), as well as CausalChapter when using these inputs.
The CausalChapter code license does not grant rights to the original lecture
media. No raw video or audio is included here. Dataset materials and derived
features remain subject to their upstream terms.

`labels` uses 1 for a chapter-ending utterance. The last video endpoint is
included in that source field; evaluators must explicitly state whether the
endpoint is excluded from internal-boundary metrics.
