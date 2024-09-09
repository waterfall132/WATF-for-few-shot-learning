# WATF in PyTorch 

We provide a PyTorch implementation of WATF for few-shot learning.
If you use this code, please cite: 

[Enhancing Few-Shot Image Classification through Weighted Adaptive Threshold Filtering of Local Descriptors](https://arxiv.org/abs/2408.15924).
(https://arxiv.org/abs/2408.15924), Bingchen Yan. In the Visual Computer.<br> 



## Prerequisites
- Linux
- Python 3.8
- Pytorch 1.7.0
- GPU + CUDA CuDNN
- pillow, torchvision, scipy, numpy

## Getting Started
### Installation

- Clone this repo:
```bash
git clone https://github.com/waterfall132/WATF-for-few-shot-learning.git
cd WATF
```

- Install [PyTorch](http://pytorch.org) 1.7.0 and other dependencies.

### Datasets
[Caltech-UCSD Birds-200-2011](https://data.caltech.edu/records/20098), [Standford Cars](https://ai.stanford.edu/~jkrause/cars/car_dataset.html), [Standford Dogs](http://vision.stanford.edu/aditya86/ImageNetDogs/main.html), [*mini*ImageNet](https://arxiv.org/abs/1606.04080v2) and [*tiered*ImageNet](https://arxiv.org/abs/1803.00676) are available at [Google Drive](https://drive.google.com/drive/u/1/folders/1SEoARH5rADckI-_gZSQRkLclrunL-yb0) and [百度网盘(提取码：yr1w)](https://pan.baidu.com/s/1M3jFo2OI5GTOpytxgtO1qA).


###  miniImageNet Few-shot Classification
- Train a 5-way 1-shot model based on Conv64:
```bash
python Train_WATF.py --dataset_dir ./path/to/miniImageNet --data_name miniImageNet --encoder_model Conv64F_Local --way_num 5 --shot_num 1
```
- Train a 5-way 1-shot model based on ResNet12:
```bash
python Train_WATF.py --dataset_dir ./path/to/miniImageNet --data_name miniImageNet --encoder_model ResNet12 --way_num 5 --shot_num 1
```
- Test the model (specify the dataset_dir, encoder_model, and data_name first):
```bash
python Test_WATF.py --resume ./results/SGD_Cosine_Lr0.05_WATF_Conv64F_Local_Epoch_30_miniImageNet_84_84_5Way_1Shot/ --encoder_model Conv64F_Local
```


## Latest results on CUB-200  (2024)
(Compared to the originally reported results in the paper. )
<table>
  <tr>
      <td rowspan="2">Method</td>
      <td rowspan="2">Backbone</td>
      <td colspan="1">5-way 1-shot</td>
      <td colspan="2">5-way 5-shot</td>
  </tr>
  <tr>
      <td>2024 Version</td>
      <td>2024 Version</td>
  </tr>

  <tr>
      <td rowspan="2">WATF</td>
      <td> Conv64F_Local </td>
      <td> 65.94 </td>
      <td> 79.96 </td>
  </tr>
  <tr>
      <td> ResNet12 </td>
      <td> 79.63 </td>
      <td> 91.18 </td>
  </tr>
</table>






## Citation
If you use this code for your research, please cite this paper.Our code based on Wenbin Li's code of DN4 for few shot learning:
```
@inproceedings{WATF_CVPR_2019,
  author       = {Wenbin Li and
                  Lei Wang and
                  Jinglin Xu and
                  Jing Huo and
                  Yang Gao and
                  Jiebo Luo},
  title        = {Revisiting Local Descriptor Based Image-To-Class Measure for Few-Shot Learning},
  booktitle    = {{IEEE} Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages        = {7260--7268},
  year         = {2019}
}
```
