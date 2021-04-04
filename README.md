# Conditional Risk Minimization(CRM) for making better mistakes
Simplified Code for our Paper **No Cost Likelihood Manipulation at Test Time for Making Better Mistakes in Deep Networks, ICLR 2021**
[Shyamgopal Karthik](https://github.com/sgk98), [Ameya Prabhu](https://drimpossible.github.io), [Puneet K. Dokania](https://puneetkdokania.github.io), [Vineet Gandhi](https://faculty.iiit.ac.in/~vgandhi/)

**Paper:** [https://openreview.net/pdf?id=193sEnKY1ij](https://openreview.net/pdf?id=193sEnKY1ij)

<a href="url"><img src="https://github.com/sgk98/CRM-Better-Mistakes/blob/main/CRM_figure.png" ></a>

We resort to the classical Conditional Risk Minimization (CRM) framework for hierarchy aware classification. Given a cost matrix and a reliable estimate of likelihoods (obtained from a trained network), CRM simply amends mistakes at inference time; it needs no extra parameters; it requires adding just one line of code to the standard cross-entropy baseline. It significantly outperforms the state-of-the-art and consistently obtains large reductions in the average hierarchical distance of top-k predictions across datasets, with very little loss in accuracy. Since CRM does not require retraining or fine-tuning of any hyperparameter, it can be used with any off-the-shelf cross-entropy trained model. 


## Installation 
Follow the setup presented in https://github.com/fiveai/making-better-mistakes to prepare the dataset splits as well the conda environment.

## Training and Pre-Trained Model
Train the models using the code available in https://github.com/fiveai/making-better-mistakes. 
The pre-trained models can be found here: 
tiered-ImageNet: https://drive.google.com/file/d/1gMcpBSKTuN_E5dPQm2ZFF_WUSOOUI9yO/view?usp=sharing

iNaturalist-19: https://drive.google.com/file/d/1gZX10ac03ZSFL9-3K_VmND6SnaZobdqw/view?usp=sharing
Alternatively, this codebase should work with most feedforward models in implemented in PyTorch with just a few modifications in `compute_results.py`

## Usage
 - Set the paths to the tiered-ImageNet and/or iNaturalist19 dataset in `data_paths.yml`
 - Given a folder with checkpoints, `python3 src/optimal_epoch.py <path_to_directory>` can be used to find the best 5 checkpoints to benchmark.
 - Set the paths to the checkpoint in the corresponding experiment file in `experiments/<experiment.sh>`
 - `bash experiments/experiment.sh` should reproduce the numbers present in the corresponding log file `logs/experiment.csv`

 Each row in the log file corresponds to the results of one checkpoint.
 The format of each row is: `top-1 accuracy, hierarchical distance@1, hierarchical distance@5, hierarchical distance@20, avg. mistake severity, ECE, MCE `


##### If you discover any bugs in the code please contact me, I will cross-check them with my nightmares.
 
## Citation

If you find our work useful/interesting, please consider citing:
```
@inproceedings{
karthik2021no,
title={No Cost Likelihood Manipulation at Test Time for Making Better Mistakes in Deep Networks},
author={Shyamgopal Karthik and Ameya Prabhu and Puneet K. Dokania and Vineet Gandhi},
booktitle={International Conference on Learning Representations},
year={2021},
url={https://openreview.net/forum?id=193sEnKY1ij}
}
```
If you find the dataset splits as well as the class hierarchies, please cite:
```
@InProceedings{bertinetto2020making,
author = {Bertinetto, Luca and Mueller, Romain and Tertikas, Konstantinos and Samangooei, Sina and Lord, Nicholas A.},
title = {Making Better Mistakes: Leveraging Class Hierarchies With Deep Networks},
booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
} 

```

## License
MIT

