# Neural Wavelet Flow

This repo provide code for the paper **Learning Non-linear Wavelet Transformation via Normalizing Flow**.

## Intro

Wavelet transformation stands as a cornerstone in modern data analysis and signal processing. Its mathematical essence is an invertible transformation that discerns slow patterns from fast patterns in the frequency domain, which repeats at each level. Such an invertible transformation can be learned by a designed normalizing flow model. With a factor-out scheme resembling the wavelet downsampling mechanism, a mutually independent prior, and parameter sharing along the depth of the network, one can train normalizing flow models to factor-out variables corresponding to fast patterns at different levels, thus extending linear wavelet transformations to non-linear learnable models. In this paper, a concrete way of building such flows is given. Then, a demonstration of the model’s ability in lossless compression task, progressive loading, and super-resolution (upsampling) task. Lastly, an analysis of the learned model in terms of low-pass/high-pass filters is given.

## Results



## How to

```shell
# train
python ./easymain.py -cuda 0 -epoch 600 -hchnl 350 -repeat 3 -nhidden 3 -target ImageNet32
```

```shell
# Flow model compress method
python ./easyEncode.py -batch 10 -earlyStop 5 -folder /Users/lili/Documents/MySpace/NeoNWL/opt/reoder/default_easyMera_ImageNet32_simplePrior_False_repeat_3_hchnl_350_nhidden_3_nMixing_5_sameDetail_True_clamp_-1_62efb58d8de7b1c7587776b9cb53cac2c741244a
```

```shell
# Flow model migrate compress method
python ./easyEncode.py -batch 10 -earlyStop 5 -target ImageNet64 -folder /Users/lili/Documents/MySpace/NeoNWL/opt/reoder/default_easyMera_ImageNet32_simplePrior_False_repeat_3_hchnl_350_nhidden_3_nMixing_5_sameDetail_True_clamp_-1_62efb58d8de7b1c7587776b9cb53cac2c741244a
```

```shell
# Freq migrate compress method
python ./easyEncodeTrans.py -batch 10 -earlyStop 5 -target ImageNet32 -folder /Users/lili/Documents/MySpace/NeoNWL/opt/reoder/default_easyMera_ImageNet32_simplePrior_False_repeat_3_hchnl_350_nhidden_3_nMixing_5_sameDetail_True_clamp_-1_62efb58d8de7b1c7587776b9cb53cac2c741244a
```

```shell
# Freq migrate compress method
python ./easyEncodeTrans.py -batch 10 -earlyStop 5 -target ImageNet64 -folder /Users/lili/Documents/MySpace/NeoNWL/opt/reoder/default_easyMera_ImageNet32_simplePrior_False_repeat_3_hchnl_350_nhidden_3_nMixing_5_sameDetail_True_clamp_-1_62efb58d8de7b1c7587776b9cb53cac2c741244a
```

```shell
# plot inplot of wavelet
python ./simpleBigWavelet.py -img ./etc/lena512color.tiff  -folder /Users/lili/Documents/MySpace/NeoNWL/opt/reoder/default_easyMera_ImageNet32_simplePrior_False_repeat_3_hchnl_350_nhidden_3_nMixing_5_sameDetail_True_clamp_-1_62efb58d8de7b1c7587776b9cb53cac2c741244a
```

```shell
# progressive loading and super resolution
python ./progressive.py -folder /Users/lili/Documents/MySpace/NeoNWL/opt/reoder/default_easyMera_ImageNet32_simplePrior_False_repeat_3_hchnl_350_nhidden_3_nMixing_5_sameDetail_True_clamp_-1_62efb58d8de7b1c7587776b9cb53cac2c741244a
```

```shell
# 2d FFT plot for low/high pass
python ./legall_lena.py -img target -deltaDepth 1 -folder /Users/lili/Documents/MySpace/NeoNWL/opt/reoder/default_easyMera_ImageNet32_simplePrior_False_repeat_3_hchnl_350_nhidden_3_nMixing_5_sameDetail_True_clamp_-1_62efb58d8de7b1c7587776b9cb53cac2c741244a
```

```shell
# FIR plot for wavelet kernal
python ./FIR.py -folder /Users/lili/Documents/MySpace/NeoNWL/opt/reoder/default_easyMera_ImageNet32_simplePrior_False_repeat_3_hchnl_350_nhidden_3_nMixing_5_sameDetail_True_clamp_-1_62efb58d8de7b1c7587776b9cb53cac2c741244a
```

```shell
# Image generation 
 python ./easyGenerate.py -folder /Users/lili/Documents/MySpace/NeoNWL/opt/reoder/default_easyMera_ImageNet32_simplePrior_False_repeat_3_hchnl_350_nhidden_3_nMixing_5_sameDetail_True_clamp_-1_62efb58d8de7b1c7587776b9cb53cac2c741244a
```

## Citation

```
@article{li2021neuralwavelet,
title={Learning Non-linear Wavelet Transformation via Normalizing Flow}
author={Li, Shuo-Hui}
year = {2021},
eprint = {arXiv:XXXX.XXXXX},
}
```

## Contact

For questions and suggestions, please contact Shuo-Hui Li at [contact_lish@iphy.ac.cn](mailto:contact_lish@iphy.ac.cn).