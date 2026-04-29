#AdaNet Model

This repo includes the code for AdaNet semantic segmentation model. The model mimics a 3-unit U-Net with 3 encoder modules in the downsampling path and similarly 3 decoder modules in the upsampling path. The encoder module utilizes DiffStride layer as a smart downsampling mechanism that learns to downsample the featuremaps by learning the optimal stride of that specific module. The model is forced to choose higher strides so it pushes the model towards smaller featuremaps, consequenly reducing FLOPs and model activation memory. The decoder receives the downsampled featuremaps and upsamples them by calculating the ratio between the size of the featuremap it receives from the previous layer and the size of the featuremap it receives through the skip-connection.

The complete model works in total synergy, encoder downsampling the input in a smart manner and the decoder learning to predict the same resolution featuremaps in the upsampling phase. This makes the model end-to-end traininable. Some of the inner mechanisms and architecture can be seen from the images, its description is beyond the scope of this repo.

##AdaNet Model Architecture
<img width="1387" height="1106" alt="image" src="https://github.com/user-attachments/assets/cba84439-b43a-43ce-99ec-87f7bddc0bb9" />

###Diffstride Encoder Module & Diffstride Internal Architecture
<img width="1470" height="751" alt="image" src="https://github.com/user-attachments/assets/7215b0b3-88fb-4556-a65d-023f250889cd" />

###Bottleneck architecture
<img width="1016" height="572" alt="image" src="https://github.com/user-attachments/assets/5b09364c-45c3-4c10-a93a-dbb257e119e1" />

###CUF Decoder Block & CUF internal Architecture
<img width="1472" height="794" alt="image" src="https://github.com/user-attachments/assets/1aea9324-9456-4507-902f-648f13afae5d" />

##Model Performance & Metric Parameters
The model prioritizes higher strides to enforce lower FLOPs. The stride evolution can be seen below.
<img width="1509" height="1169" alt="image" src="https://github.com/user-attachments/assets/a7d2cde4-66f8-42cc-92fc-77d88a0b927c" />

###FLOPs reduction & comparison with Fixed Pooling based model
<img width="4122" height="1583" alt="Picture12" src="https://github.com/user-attachments/assets/1f976a42-00fd-4a17-b4d1-8465097c7be5" />
*Figure 6: FLOPs reduction during training | Learned Encoder Output Shapes and Resulting FLOPs Reduction (a) Cityscapes (256,256) (b)
Cityscapes (512,1024) and (c) CamVid (256,256)*

##AdaNet Qualitative Results


