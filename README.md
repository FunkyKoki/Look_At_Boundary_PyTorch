### Environment Requirement

+ PyTorch >= 1.0.0
+ Python >=  3.6 (numpy, scipy, matplotlib, tqdm)
+ OpenCV == 3.4.5
+ Platform: Linux

### Get program

```git clone git@github.com:FunkyKoki/Look_At_Boundary_PyTorch.git```

Program structure is as below:

```
.
├── dataset.py
├── evaluate.py
├── models
│ ├── __init__.py
│ ├── losses.py
│ └── models.py
├── README.md
├── train.py
├── utils
│ ├── args.py
│ ├── dataload.py
│ ├── dataset_info.py
│ ├── __init__.py
│ ├── pdb.py
│ ├── train_eval_utils.py
│ └── visual.py
└── weights
  └── ckpts
```

### Dataset Prepare

This program support 4 popular face landmark datasets: 300W, AFLW, COFW, WFLW. The dataset file folder structure is as below:

```angular2
.
├── 300W
│ ├── afw
│ ├── helen
│ │ ├── testset
│ │ └── trainset
│ ├── ibug
│ ├── lfpw
│ │ ├── testset
│ │ └── trainset
│ ├── test_imgs
│ └── testset
│   ├── 01_Indoor
│   └── 02_Outdoor
├── AFLW
│ ├── 0
│ ├── 2
│ └── 3
├── COFW
│ ├── test_imgs
│ └── train_imgs
└── WFLW
  └── WFLW_images
    ├── 0−−Parade
    ├── 10−−People_Marching
    ├── 11−−Meeting
    ├── 12−−Group
    ├── 13−−Interview
    ├── 14−−Traffic
    ├── 15−−Stock_Market
    ├── 16−−Award_Ceremony
    ├── 17−−Ceremony
    ├── 18−−Concerts
    ├── 19−−Couple
    ├── 1−−Handshaking
    ├── 20−−Family_Group
    ├── 21−−Festival
    ├── 22−−Picnic
```

Tips: Pay attention to the ```test_imgs``` folder and ```testset``` folder in 300W dataset, the ```test_imgs``` pics are human faces from COFW which are annotated with 68 landmarks, that's why it is put here. Some other things are written in readme.txt.

The annotation file can be download from https://pan.baidu.com/s/1hYFcz260IB0pMISbHbxoTg, the code is ```tuz9```, annotation format is \[x1, y1, x2, y2, …, xn, yn, bboxleft, bboxtop, bboxright, bboxbottom, picH, picW, pic_route\], which are coordinates, bounding box position, height and width of inital pic, and route of the pic in order.

### Model Evaluation

WFLW training model can be download from https://pan.baidu.com/s/1tM3oJFUHmP4kJA7enXVLjA, the code is ```tbgi``` and put at ```weights``` folder,  this model is trained with 900 epoch.

When evaluating, you can config the param in utils/args.py or just set the param by terminal, for example, if you want to evaluate at ```Pose Testset``` normalized in the way of ```inter_ocular```:

```python evaluate.py −−dataset WFLW −−split pose −−eval_epoch 900 −−norm_way inter_ocular```

### Model Training

Config almost everything in utils/args or set them by terminal:

```python train.py −−dataset WFLW −−split train −−loss_type L2```

Tips: This program integrates the ```Wingloss``` and ```Pose-based Date Balancing```, if you want to use them, just choose it ^_^.

### In the end

Fuck every LICENSE.
