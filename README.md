

# Scalable Differentiable Physics for Learning and Control

*Yi-Ling Qiao, Junbang Liang, Vladlen Koltun, Ming C. Lin*

(in progress)

## Usage
1. Create a conda virtual environment and activate it.
```bash
conda create -n diffsim python=3.6 -y
conda activate diffsim
```

2. Download and build the project.
```bash
git clone git@github.com:YilingQiao/diffsim.git
cd diffsim
bash script_build.sh
cd pysim
```
3. Run the examples

### Learn to drag a cube using a cloth
```bash
python exp_learn_cloth.py
```

<div align="center">
<img width="300px" src="https://github.com/YilingQiao/linkfiles/raw/master/icml20/darg.gif"> 
</div>


### Learn to hold a rigid body using a parallel gripper
```bash
python exp_learn_stick.py
```

<div align="center">
<img width="300px" src="https://github.com/YilingQiao/linkfiles/raw/master/icml20/stick.gif"> 
</div>


## Bibtex
```
@aritical{Qiao2020Scalable,
author  = {Qiao, Yiling and Liang, Junbang and Koltun, Vladlen and Lin, Ming C.},
title  = {Scalable Differentiable Physics for Learning and Control},
booktitle = {ICML},
year  = {2020},
}
```
