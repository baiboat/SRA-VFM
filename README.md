# SRA-VFM
Pytorch codes of **SRA-VFM: Boosting Remote Sensing Change Detection via Slice-Reassembled Augmentation and Vision Foundation Model-Guided Dual Streams** 



The SRA-VFM adopts [FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM) as the visual encoder with some modifications.


## How to Use
1. Installation
   * Install [FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM) following the instructions.
   * Refer to SAMCD to modify the code(https://github.com/ggsDing/SAM-CD/blob/main/models/FastSAM/README.md). 

2. Dataset preparation.
   * Please split the data into training, validation and test sets and organize them as follows:
```
      YOUR_DATA_DIR
      ├── ...
      ├── train
      │   ├── A
      │   ├── B
      │   ├── label
      ├── val
      │   ├── A
      │   ├── B
      │   ├── label
      ├── test
      │   ├── A
      │   ├── B
      │   ├── label
```

   * Find change line 13 in [SRA-VFM/datasets/Levir_CD.py] change `/YOUR_DATA_ROOT/` to your local dataset directory.

3. Training
   
   
   training CD with the proposed SRA-VFM:
   `python train_SRA_VFM.py`
   
   line 16-45 are the major training args, which can be changed to load different datasets, models and adjust the training settings.

5. Inference and evaluation
   
   inference on test sets: set the chkpt_path and run
   
   `python pred_CD.py`
   
   evaluation of accuracy: set the prediction dir and GT dir, and run
   
   `python eval_CD.py`
   
(More details to be added...)


## Dataset Download

In the following, we summarize links to some frequently used CD datasets:

* [LEVIR-CD](https://justchenhao.github.io/LEVIR/)
* [WHU-CD](https://study.rsgis.whu.edu.cn/pages/download/) [(baidu)](https://pan.baidu.com/s/1A0_xbV4ZktWCbL3j94CInA?pwd=WHCD )
* [CLCD (Baidu)](https://pan.baidu.com/s/1iZtAq-2_vdqoz1RnRtivng?pwd=CLCD)
* [S2Looking](https://github.com/S2Looking/Dataset)
* [SYSU-CD](https://github.com/liumency/SYSU-CD)

## Pretrained Models

For readers to easily evaluate the accuracy, we provide the trained weights of the SRA-VFM.

[Baidu](https://pan.baidu.com/s/1cVR39l8XVQr5vvzvhhe7YQ?pwd=SRCD) (pswd: SRCD)

## Acknowledgement
Thanks [FastSAM](https://github.com/CASIA-LMC-Lab/FastSAM) and [SAMCD](https://github.com/ggsDing/SAM-CD) for serving as building blocks of SRA-VFM.



