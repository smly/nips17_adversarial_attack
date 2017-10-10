This is my final submission for NIPS'17 Defense Against Adversarial Attack Challenge.

**Appraoch**: Weighted majority voting. The weights are the prediction result of XGBoost model, which is trained to predict the accuracy.

NOTE: Unfortunately the improvement by the weighting is relatively small. The main improvement in my trials is majority voting of high-performance models.

## Acknowledgements

This product includes software developed by the Soumith Chintala, Ross Wightman, Remi Cadene and Fisher Yu.

* https://github.com/pytorch/vision/ (BSD-3)
* https://github.com/rwightman/pytorch-nips2017-attack-example (BSD-3)
* https://github.com/rwightman/pytorch-dpn-pretrained (BSD-3)
* https://github.com/Cadene/pretrained-models.pytorch (BSD-3)
* https://github.com/fyu/drn (BSD-3)
