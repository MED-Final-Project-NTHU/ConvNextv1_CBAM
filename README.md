Integrate ConVNext with CBAM but without pre-trained
Our upgraded ConvNeXt structure, CBAM-ConvNeXt, integrates ConvNeXt
Block modules, downsampling modules, and CBAM [16] attention modules to
address complex background interference. Despite the selection of CBAM for its
combined channel and spatial attention capabilities, experimental results show
suboptimal performance, with a test mAP of 0.30. This is attributed to the
modification of the ConvNeXt architecture with CBAM, rendering pre-trained
models unusable and resulting in an overall performance decrease.
