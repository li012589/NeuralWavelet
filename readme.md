# Neural Wavelet Transformation

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







## Deprecated


```bash
# python ./encodeTarget2.py -folder ./opt/default_CIFAR_depth_-1_repeat_2_nhidden_2_hdim_25_Sprior_False -target CIFAR
```

```bash
 python ./plotBigWavelet.py -folder ./opt/default_CIFAR_depth_-1_repeat_2_nhidden_2_hdim_25_Sprior_False -img ./etc/lena512color.tiff
```

```bash
python ./progressiveLoading.py -folder ./opt/default_CIFAR_depth_-1_repeat_2_nhidden_2_hdim_25_Sprior_False
```


```bash
 python ./encode.py -folder ./opt/default_CIFAR_depth_-1_repeat_2_nhidden_2_hdim_25_Sprior_False
```

```bash
ython ./encodeTarget.py -earlyStop 2 -batch 30 -nbins 6000 -target ImageNet64 -folder ./opt/default_Conv2dnet_CIFAR_depth_-1_repeat_4_nhidden_2_hdim_512_hchnl_512_nNICE_4_nMixing_5_Sprior_False_bigM_False
```

