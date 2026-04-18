# ScNucAdapt: Partial domain adaptation enables cross domain cell type annotation between scRNA-seq and snRNA-seq
>  Xiran Chen, Quan Zou, Qinyu Cai, Xiaofeng Chen, Weikai Li*, Yansu Wang*<br>
>  PLoS Computational Biology 2026<br>

Single-cell and single-nucleus RNA sequencing are two powerful technologies that allow
scientists to study gene activity in individual cells. However, comparing data between
these methods remains challenging because they capture different parts of the cell and
are often collected under different conditions. This makes it difficult to consistently
identify cell types across experiments, hindering our understanding of health and
disease.
We developed ScNucAdapt, a computational framework that can automatically
transfer cell type knowledge between these two types of datasets, even when they come
from different laboratories or tissue conditions. Our method learns to recognize shared
patterns while ignoring dataset-specific differences. Through testing on diverse tissues,
including bladder, kidney, tumors, and brain, we show that ScNucAdapt consistently
outperforms existing approaches.
By enabling reliable integration of single-cell and single-nucleus data, our work helps
researchers build more complete pictures of cellular diversity across tissues and disease
states. This capability is particularly valuable for studying archived frozen samples or
fragile cell types that are difficult to analyze with conventional methods, potentially
accelerating discoveries in various fields.

## Usage<a id="usage"></a>
Input your preprocessed numpy array data into array(source) and array2(target) in main.py

<img width="8891" height="3183" alt="Graphical_abstract" src="https://github.com/user-attachments/assets/c9fba1b6-b5da-4525-867d-e3712b3ff1dd" />
