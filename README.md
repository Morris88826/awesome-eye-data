# awesome-eye-data

A curated collection of real-world eye tracking datasets unified under a common format, with tools for visualization and analysis.

## Datasets

| Dataset | Description |
|---|---|
| **TEyeD** | Over 20 million eye images with pupil, eyelid, and iris 2D/3D segmentations, landmarks, gaze vectors, and eye movement types (Dikablis recordings) |
| **GazeinTheWild** | Eye and head coordination data captured during everyday activities |
| **LPW** | Labelled Pupils in the Wild — pupil detection in unconstrained environments |
| **HMC** | Head-mounted camera eye recordings |

## Repository Structure

```
data/           # Processed data (Dikablis, GazeinTheWild, HMC, LPW)
libs/           # Processor and visualization utilities
notebooks/      # Jupyter notebooks for stats and visualization
```

## Setup

```bash
pip install -r requirements.txt
pip install tqdm
conda install opencv
```

## Data

Processed data is available for download on [Google Drive](https://drive.google.com/drive/folders/1JZpXaR66MXBPIshuhSwY22wwnb1hQM7l?usp=sharing) and stored under `./data/`, organized by dataset and recording:

```
data/
└── <Dataset>/
    └── <recording_name>/
        ├── ANNOTATIONS/   # CSV files (one per chunk)
        ├── VIDEOS/        # Chunked eye video clips (.mp4)
        └── IRIS/          # Cropped iris video clips (.mp4)
```

Each annotation CSV has the following columns:

| Column | Description |
|---|---|
| `FRAME` | Frame index |
| `validity` | 1 = valid, -1 = invalid |
| `gaze_x/y/z` | Normalized 3D gaze vector |
| `pupil_center_x/y` | Pupil center in image coordinates |

## Visualization

Open [`notebooks/visualize.ipynb`](notebooks/visualize.ipynb) to browse sample frames from any recording. Set `video_path` to a clip under `./data/` and the notebook will display 10 evenly-spaced frames annotated with:

- Validity status (green = valid, red = invalid)
- Pupil center (blue dot)
- Gaze direction (yellow arrow)
- Iris mask overlay (if available)

## Citations

If you use this repository, please cite:

```bibtex
@article{ICML2021DS,
  title={TEyeD: Over 20 million real-world eye images with Pupil, Eyelid, and Iris 2D and 3D Segmentations, 2D and 3D Landmarks, 3D Eyeball, Gaze Vector, and Eye Movement Types},
  author={Fuhl, Wolfgang and Kasneci, Gjergji and Kasneci, Enkelejda},
  journal={arXiv preprint arXiv:2102.02115},
  year={2021}
}
```

For the individual datasets, also cite:

<details>
<summary>GazeinTheWild</summary>

```bibtex
@article{kothari2020gaze,
  title={Gaze-in-wild: A dataset for studying eye and head coordination in everyday activities},
  author={Kothari, Rakshit and Yang, Zhizhuo and Kanan, Christopher and Bailey, Reynold and Pelz, Jeff B and Diaz, Gabriel J},
  journal={Scientific reports},
  volume={10},
  number={1},
  pages={1--18},
  year={2020},
  publisher={Nature Publishing Group}
}
```
</details>

<details>
<summary>LPW</summary>

```bibtex
@inproceedings{tonsen2016labelled,
  title={Labelled pupils in the wild: a dataset for studying pupil detection in unconstrained environments},
  author={Tonsen, Marc and Zhang, Xucong and Sugano, Yusuke and Bulling, Andreas},
  booktitle={Proceedings of the ninth biennial ACM symposium on eye tracking research \& applications},
  pages={139--142},
  year={2016}
}
```
</details>
