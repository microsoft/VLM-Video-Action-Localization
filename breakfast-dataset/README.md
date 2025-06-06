# Breakfast dataset

This folder provides resources for evaluating action label predictions on videos from the Breakfast dataset. It includes ground-truth annotations and an evaluation script.

This dataset is provided as supplementary material for the paper:

> **Open-vocabulary action localization with iterative visual prompting**  
> *Naoki Wake, Atsushi Kanehira, Kazuhiro Sasabuchi, Jun Takamatsu, Katsushi Ikeuchi (2025), [IEEE Access, 5, 56908-56917](https://ieeexplore.ieee.org/abstract/document/10942370)*
>
> ```bibtex
>@article{wake2025open,
>  author={Wake, Naoki and Kanehira, Atsushi and Sasabuchi, Kazuhiro and Takamatsu, Jun and Ikeuchi, Katsushi},
>  journal={IEEE Access}, 
>  title={Open-vocabulary action localization with iterative visual prompting},
>  year={2025},
>  volume={13},
>  number={},
>  pages={56908--56917},
>  doi={10.1109/ACCESS.2025.3555167}}
> ```

The original data is derived from the paper below:

> **Human grasping database for activities of daily living with depth, color and kinematic data streams**  
> *Hilde Kuehne, Ali Arslan, and Thomas Serre (2014), CVPR, 780--787*
>
> ```bibtex
>@inproceedings{kuehne2014language,
>  title={The language of actions: Recovering the syntax and semantics of goal-directed human activities},
>  author={Kuehne, Hilde and Arslan, Ali and Serre, Thomas},
>  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
>  pages={780--787},
>  year={2014}
>}
> ```

## Directory and File Structure

- **label_data_gt_breakfast.json**  
  This JSON file holds the ground-truth annotations for the videos. Each entry in the JSON contains:
  - **action**: A sequence of action labels that occur in the video.  
  - **gt_time**: The frame index annotations corresponding to each action label (FPS=15.0).
  - **video_path**: The relative path to the corresponding video file.

- **label_data_estimate_baseline_breakfast.json**  
  This is an example file that contains estimated action labels. It is used as an input to the evaluation script.

- **compute_mof_iou_f1.py**  
  This evaluation script computes performance metrics (e.g., MOF, IoU, and F1 score) by comparing predicted action labels with the ground truth.
   ```bash
  python compute_mof_iou_f1.py --file label_data_estimate_baseline.json
   ```