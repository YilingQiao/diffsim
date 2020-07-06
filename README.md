


# Scalable Differentiable Physics for Learning and Control

*Yi-Ling Qiao, Junbang Liang, Vladlen Koltun, Ming C. Lin*

(in progress)

## Setup
1. Create a conda virtual environment and activate it.
```bash
conda create -n diffsim python=3.6 -y
conda activate diffsim
```

2. Download and build the project.
```bash
git clone git@github.com:YilingQiao/diffsim.git
cd diffsim
pip install -r requirements.txt
bash script_build.sh
cd pysim
```
3. Run the examples
## Examples
### Optimize an inverse problem
```bash
python exp_inverse.py
```
By default, the simulation output would be stored in `pysim/default_out` directory. 
If you want to store the results in some other places, like `./test_out`, you can specify it by `python exp_inverse.py test_out`

To visualize the simulation results, use
```bash
python msim.py
```
You can change the source folder of the visualization in `msim.py`. More functionality of `msim.py` can be found in `arcsim/src/msim.cpp`.

The visualization is the same for all other experiments.
<div align="center">
<img width="300px" src="https://github.com/YilingQiao/linkfiles/raw/master/icml20/inverse.gif"> 
</div>


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

### Scalability experiments
Figure 3, first row.
```bash
bash script_multibody.sh
```

Figure 3, second row.
```bash
bash script_scale.sh
```

### Ablation study
Table 1, sparse collision handling.
```bash
bash script_absparse.sh
```

Table 2, fast differentiation.
```bash
bash script_abqr.sh
```

### Estimate the mass of a cube
```bash
python exp_momentum.py
```

<div align="center">
<img width="300px" src="https://github.com/YilingQiao/linkfiles/raw/master/icml20/momentum.gif"> 
</div>

### Two-way coupling - Trampoline
```bash
python exp_trampoline.py
```

<div align="center">
<img width="300px" src="https://github.com/YilingQiao/linkfiles/raw/master/icml20/trampoline.gif"> 
</div>


### Two-way coupling - Domino
```bash
python exp_domino.py
```
<div align="center">
<img width="300px" src="https://github.com/YilingQiao/linkfiles/raw/master/icml20/domino.gif"> 
</div>


### Two-way coupling - armadillo and bunny
```bash
python exp_bunny.py
```


### Domain transfer - motion control in MuJoCo

This experiment requires MuJoCo environment. Install [MuJoCo](http://www.mujoco.org/) and its python interface [mujoco_py](https://github.com/openai/mujoco-py) before running this script.
```bash
python exp_mujoco.py
```
<div align="center">
<img width="300px" src="https://github.com/YilingQiao/linkfiles/raw/master/icml20/mj_mismatch.gif"> 
</div>

## Bibtex
```
@inproceedings{Qiao2020Scalable,
author  = {Qiao, Yi-Ling and Liang, Junbang and Koltun, Vladlen and Lin, Ming C.},
title  = {Scalable Differentiable Physics for Learning and Control},
booktitle = {ICML},
year  = {2020},
}
```
