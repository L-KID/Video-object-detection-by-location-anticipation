## Objects do not disappear: Video object detection by single-frame object location anticipation
The official implementation of our ICCV 2023 paper:
[Objects do not disappear: Video object detection by single-frame object location anticipation](https://arxiv.org/abs/2308.04770).

#### Abstract
*Objects in videos are typically characterized by continuous smooth motion. We exploit continuous smooth motion in three ways. 1) Improved accuracy by using object motion as an additional source of supervision, which we obtain by anticipating object locations from a static keyframe. 2) Improved efficiency by only doing the expensive feature computations on a small subset of all frames. Because neighboring video frames are often redundant, we only compute features for a single static keyframe and predict object locations in subsequent frames. 3) Reduced annotation cost, where we only annotate the keyframe and use smooth pseudo-motion between keyframes. We demonstrate computational efficiency, annotation efficiency, and improved mean average precision compared to the state-of-the-art on four datasets: ImageNet VID, EPIC KITCHENS-55, YouTube-BoundingBoxes, and Waymo Open dataset.*

<img src="overview5.jpg" alt="drawing">

#### Install
###### pip
Please first install the required dependencies. This can be done by:
```
conda create -n efficientVOD python=3.7
conda activate efficientVOD
conda install pytorch==1.7.0 torchvision==0.8.1 cudatoolkit=10.1 -c pytorch
pip install -r requirements.txt
```

#### Training
```
python3 train.py \
    --use-cuda \
    [--timestep 10] \
    --iters -1 \
    [--epochs 100] \
    --lr-steps 51  \
    [--dataset imagenetvid] \
    [--data-dir ../ILSVRC2015] \
```
#### Evaluation
```
python3 train.py \
    --use-cuda \
    --resume \
    [--timestep 10] \
    --iters -1 \
    [--dataset imagenetvid] \
    [--data-dir ../ILSVRC2015] \

```

### Cite
If you found this work useful in your research, please consider citing:
```
@article{liu2023objects,
  title={Objects do not disappear: Video object detection by single-frame object location anticipation},
  author={Liu, Xin and Nejadasl, Fatemeh Karimi and van Gemert, Jan C and Booij, Olaf and Pintea, Silvia L},
  journal={arXiv preprint arXiv:2308.04770},
  year={2023}
}
```

