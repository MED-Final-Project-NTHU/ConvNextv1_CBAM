Integrate ConVNext with CBAM but without pre-trained:
Our upgraded ConvNeXt structure, CBAM-ConvNeXt, integrates ConvNeXt Block modules, downsampling modules, and CBAM [16] attention modules to address complex background interference. Despite the selection of CBAM for its combined channel and spatial attention capabilities, experimental results show suboptimal performance, with a test mAP of 0.30. This is attributed to the modification of the ConvNeXt architecture with CBAM, rendering pre-trained models unusable and resulting in an overall performance decrease.

Code Explaination:
Myloader_multiT.py : Load dataset and execute data transformation.
model_convnextv1_CBAM.py: ConvNext v1 model with CBAM, but no pre-trained
train_new_test_convnext_CBAM_tiny.py: Training and testing code
utils.py: utility code.
