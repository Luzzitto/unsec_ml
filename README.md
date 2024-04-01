# Unveiling Vulnerabilities: Assessing the Impact of Adversarial Models on Self-Driving Systems
The overall format of the dataset should be the following
```text
<dataset>
├───images
│   ├───test
│   ├───train
│   │   ├───img123.jpg
│   │   └───<...>.jpg
│   └───val
├───labels
│   ├───train
│   │   ├───img123.txt
│   │   └───<...>.txt
│   └───val
├───info.txt
└───<dataset>.yaml
```
Content for sample `img123.txt` should be the following:
```text
id <x1> <y1> <x2> <y2> <...> <...> <xn> <yn>
```
The image should be normalized to its respective dimension.

## BDD
```text
bdd100k
├───images
│   └───10k
│       ├───test
│       ├───train
│       └───val
└───labels
    └───sem_seg
        ├───colormaps
        │   ├───train
        │   └───val
        ├───masks
        │   ├───train
        │   └───val
        ├───polygons
        └───rles
```

## IDD
```text
├───gtFine
│   ├───train
│   │   ├───201
│   │   ├───...
│   │   └───579
│   └───val
│       ├───205
│       ├───...
│       └───580
└───leftImg8bit
    ├───test
    │   ├───200
    │   ├───...
    │   └───576
    ├───train
    │   ├───201
    │   ├───...
    │   └───579
    └───val
        ├───205
        ├───...
        └───580
```

# Command Samples
## BDD
```bash
py bdd.py --method t --clean /path/to/bdd100k
```
```bash
py bdd.py --method v --adversary --host car pole --target train --ratio 0.7 /path/to/bdd100k
```