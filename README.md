# MoPE — Minimal Research Release (Offline + Online)
## Background
This repository provides an open-source implementation for the paper: ***MoPE: A Mixture of Password Experts for Improving Password Guessing***

As mentioned in the paper, this repository provides two sub-projects to support different attack scenarios.
## Environment

```bash
pip install -r requirement.txt
```
## Data formats

### A) Offline scenario
Example: mope-code\mope-offline\data\example.txt

Format:
```
password1
password2
password3
......
```
### B) Online scenario
Example: mope-code\mope-online\data\example.csv

Format:
```
source_password<TAB>target_password<TAB>[edit path]
```

## Usage
### A) Offline scenario
1) Train demo:
```bash
python main.py
```
2) Evaluate demo:
```bash
python inference.py
```
### B) Online scenario
1) Train demo:
```bash
python train.py
```
2) Evaluate demo:
```bash
python evaluate.py
```