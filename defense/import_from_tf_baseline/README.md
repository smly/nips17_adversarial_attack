TBD

This work is inspired by [tensorflow-model-zoo.torch](https://github.com/Cadene/tensorflow-model-zoo.torch).

## Usage

```
$ python ens_adv_inception_resnet_v2.py \
    --checkpoint /data/pretrained/ens_adv_inception_resnet_v2.ckpt \
    --outdir /tmp/ens_adv_inception_resnet_v2
```

## Requirements

Full list of required packages are listed on `env.yaml`.

* Python 3.6.0
* tensorflow 1.3.0
* pytorch 0.2.0
* numpy
* PIL
* click
* h5py

## Submodule

* tensorflow/models (91fbd5c)