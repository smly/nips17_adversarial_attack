TBD

This work is inspired by [tensorflow-model-zoo.torch](https://github.com/Cadene/tensorflow-model-zoo.torch).

## Usage

```
$ PYTHONPATH=${PYTHONPATH}:./models/slim \
    python ens_adv_inception_resnet_v2.py \
    --checkpoint  /data/pretrained/ens_adv_inception_resnet_v2.ckpt \
    --outdir      /tmp/ens_adv_inception_resnet_v2 \
    --export-path ../working_files/ensadv_inceptionresnetv2_state.pth

$ PYTHONPATH=${PYTHONPATH}:./models/slim:../models \
    python ens_adv_inception_v3.py \
    --checkpoint  /data/pretrained/ens3_adv_inception_v3.ckpt \
    --outdir      /tmp/ens3_adv_inception_v3 \
    --export-path ../working_files/ens3incepv3_fullconv_state.pth

$ PYTHONPATH=${PYTHONPATH}:./models/slim:../models \
    python ens_adv_inception_v3.py \
    --checkpoint  /data/pretrained/ens4_adv_inception_v3.ckpt \
    --outdir      /tmp/ens4_adv_inception_v3 \
    --export-path ../working_files/ens4incepv3_fullconv_state.pth

$ PYTHONPATH=${PYTHONPATH}:./models/slim:../models \
    python ens_adv_inception_v3.py \
    --checkpoint  /data/pretrained/adv_inception_v3.ckpt \
    --outdir      /tmp/adv_inception_v3 \
    --export-path ../working_files/adv_inception_v3_fullconv_state.pth
```

## Requirements

Full list of required packages are listed on `env.yaml`.

* Python 3.6.0
* tensorflow 1.3.0
* pytorch 0.2.0
* numpy
* click
* Pillow
* h5py

## Submodule

* [tensorflow/models @ 91fbd5c](https://github.com/tensorflow/models/tree/91fbd5c5717b95a1b3345e04f10600d74be531d3)
