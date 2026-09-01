# Data directory

Download the source datasets from their official repositories:

- [AVLectures](https://github.com/Darshansingh11/AVLectures)
- [VidChapters-7M](https://github.com/antoyang/VidChapters)

The raw videos are not redistributed here. For AVLecture evaluation, combine
the official subtitles/ASR and segmentation annotations into the following
local layout:

- `avlecture/test.jsonl`: utterance-level transcript and timestamps
- `avlecture/test_subheading_abstract.jsonl`: chapter segments and descriptions
- `avlecture/vis2d_cache/avlecture/`: 512-dimensional visual features stored as
  `<video-id>-<index>.npy`

The feature index may be either a dense utterance index or an integer
timestamp. A processed directory can be linked into this location:

```bash
ln -s /path/to/avlecture ./data/avlecture
```
