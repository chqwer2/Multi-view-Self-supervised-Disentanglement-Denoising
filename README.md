# Multi-view Self-supervised Disentanglement for General Image Denoising
<!-- <br /> -->
<!-- <p align="center">
  <img src="https://chqwer2.github.io/assets/paper/2023/MeD/MeD.jpg" align="center" width="60%"> -->

<p align="center">

[//]: # (  <font size=5><strong>Panoptic Scene Graph Generation</strong></font>)
    <br>
      <a href="http://chqwer2.github.io/" target='_blank'>Hao Chen</a>,&nbsp;
      <a href="https://chenyuanqu.com/" target='_blank'>Chenyuan Qu</a>,&nbsp;
      <a href="" target='_blank'>Yu Zhang</a>,&nbsp;
      <a href="https://www.crcv.ucf.edu/chenchen/" target='_blank'>Chen Chen</a>,&nbsp;
      <a href="https://jianbojiao.com/" target='_blank'>Jianbo Jiao</a>
    <br>
  University of Birmingham & Shanghai Jiao Tong University & University of Central Florida
  </p>
</p>


---

![arc](assets/arc.png)

## Updates

- **Opt 11, 2023**: Training and Test code release 🚧
- **Sept 10, 2023**: MeD paper is available in [Arxiv](https://arxiv.org/abs/2309.05049), or [ICCV version](https://openaccess.thecvf.com/content/ICCV2023/html/Chen_Multi-view_Self-supervised_Disentanglement_for_General_Image_Denoising_ICCV_2023_paper.html) 📖
- **Aug 17, 2023**: Don’t forget to visit our [Website](https://chqwer2.github.io/MeD/) for more interactive visualization examples~ 🤣
- **July 14, 2023**: MeD is accepted by ICCV'23 🎉

## Train & Test

Run the following scripts for train and test

```
bash scripts/train/MeD_multi.sh              
bash scripts/test/test_from_path.sh
```

## Experiment Results

#### Gaussian Noise and Beyond Removal Table

####  ![table](assets/table.png)

#### Real Noise Removal Examples

![table](assets/real.png)

[//]: # (## Acknowledgements)



## Citation
If you find our repository useful for your research, please consider citing our paper:
```bibtex
@InProceedings{MeD_ICCV23,
    author    = {Chen, Hao and Qu, Chenyuan and Zhang, Yu and Chen, Chen and Jiao, Jianbo},
    title     = {Multi-view Self-supervised Disentanglement for General Image Denoising},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
}
```