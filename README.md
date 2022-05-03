# Learning Local Displacements for Point Cloud Completion

BSD 3-Clause License Copyright (c) 2022, Yida Wang All rights reserved.

## Abstrarct
| Completing a car |  |
| :-: | :-- |
![teaser](readme_imgs/CVPR_teaser.png#center) | From the input partial scan to our object completion, we visualize the amount of detail in our reconstruction.

We propose a novel approach aimed at object and semantic scene completion from a partial scan represented as a 3D point cloud.
Our architecture relies on three novel layers that are used successively within an encoder-decoder structure and specifically developed for the task at hand.
The first one carries out feature extraction by matching the point features to a set of pre-trained local descriptors.
Then, to avoid losing individual descriptors as part of standard operations such as max-pooling, we propose an alternative neighbor-pooling operation that relies on adopting the feature vectors with the highest activations. Finally, up-sampling in the decoder modifies our feature extraction in order to increase the output dimension.
While this model is already able to achieve competitive results with the state of the art, we further propose a way to increase the versatility of our approach to process point clouds. To this aim, we introduce a second model that assembles our layers within a transformer architecture.
We evaluate both architectures on object and indoor scene completion tasks, achieving state-of-the-art performance.

## Methodology
### Local displacement operator
| The operation |  |
| :-: | :-- |
![operator](readme_imgs/CVPR_graph_conv.png#center) | (a) *k*-nearest neighbor in reference to an anchor **f**; (b) displacement vectors around the anchor **f** + δ<sub>i</sub> and the corresponding weight σ<sub>i</sub>; and, (c) closest features for all i.

### Architectures
| The *direct* architectrue | The *transformer* architecture |
| :-: | :-: |
![direct](readme_imgs/CVPR_direct_architecture.png#center) | ![transformer](readme_imgs/CVPR_transformer_architecture.png#center)

### Qualitatives
#### Object completion
![objects](readme_imgs/CVPR_shapenet.png#center)

#### Semantic scene completion
![objects](readme_imgs/CVPR_scannet.png#center)

## Cite

If you find this work useful in your research, please cite:

```bash
@inproceedings{wang2022displacement,
  title={Learning Local Displacements for Point Cloud Completion},
  author={Wang, Yida and Tan, David Joseph and Navab, Nassir and Tombari, Federico},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2022}
}
```