# THUMOS14 dataset
s
This folder provides resources for evaluating action label predictions on videos from the THUMOS14 dataset. Most of the necessary code, including the evaluation script and ground truth labels, needs to be downloaded from the official [THUMOS14 page](https://www.crcv.ucf.edu/THUMOS14/).

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

The following is the citation for the THUMOS challenge, taken from the official [THUMOS14 page](https://www.crcv.ucf.edu/THUMOS14/):
> ```bibtex
>@misc{THUMOS14,
>   author = "Jiang, Y.-G. and Liu, J. and Roshan Zamir, A. and Toderici, G. and Laptev, I. and Shah, M. and Sukthankar, R.",
>   title = "{THUMOS} Challenge: Action Recognition with a Large Number of Classes",
>   howpublished = "\url{http://crcv.ucf.edu/THUMOS14/}",
>   Year = {2014}}
> ```

## Directory and File Structure

- **label_data_estimate_thumos14.txt**  
  This is an example file that contains estimated action labels. It is used as input to the evaluation script (see below). For details on the file format, please refer to the [THUMOS14 challenge documentation](https://www.crcv.ucf.edu/THUMOS14/THUMOS14_Evaluation.pdf).

## Usage Instructions

1. **Download the THUMOS14 Evaluation Toolkit**  
   - As described in the  [THUMOS14 challenge documentation](https://www.crcv.ucf.edu/THUMOS14/THUMOS14_Evaluation.pdf), download the evaluation toolkit (see "Section 4 Development kit").
   - Unzip the downloaded file (`THUMOS14_evalkit_20140818.zip`).
2. **Place the Label Data File**  
   - Move `label_data_estimate_thumos14.txt` into the `THUMOS14_evalkit_20140818/TH14evalkit/results` directory.  
   - Install MATLAB or Octave, and from within the `THUMOS14_evalkit_20140818/TH14evalkit` directory, run the following command to compute the evaluation metrics:

   ```matlab
   [pr_all, ap_all, map] = TH14evaldet('results/label_data_estimate_thumos14.txt', 'groundtruth', 'test')
   ```