<div align="center">
  <img src="https://raw.githubusercontent.com/open-mmlab/mmcv/master/docs/en/mmcv-logo.png" width="300"/>
  <div>&nbsp;</div>
  <div align="center">
    <b><font size="5">OpenMMLab website</font></b>
    <sup>
      <a href="https://openmmlab.com">
        <i><font size="4">HOT</font></i>
      </a>
    </sup>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <b><font size="5">OpenMMLab platform</font></b>
    <sup>
      <a href="https://platform.openmmlab.com">
        <i><font size="4">TRY IT OUT</font></i>
      </a>
    </sup>
  </div>
  <div>&nbsp;</div>
</div>

[![docs](https://img.shields.io/badge/docs-latest-blue)](https://mmcv.readthedocs.io/en/latest/)
[![platform](https://img.shields.io/badge/platform-Linux%7CWindows%7CmacOS-blue)](https://mmcv.readthedocs.io/en/latest/get_started/installation.html)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mmcv)](https://pypi.org/project/mmcv/)
[![PyPI](https://img.shields.io/pypi/v/mmcv)](https://pypi.org/project/mmcv)
[![badge](https://github.com/open-mmlab/mmcv/workflows/build/badge.svg)](https://github.com/open-mmlab/mmcv/actions)
[![codecov](https://codecov.io/gh/open-mmlab/mmcv/branch/master/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmcv)
[![license](https://img.shields.io/github/license/open-mmlab/mmcv.svg)](https://github.com/open-mmlab/mmcv/blob/master/LICENSE)

English | [简体中文](README_zh-CN.md)

## Introduction

MMCV is a foundational library for computer vision research and it provides the following functionalities:

- [Universal IO APIs](https://mmcv.readthedocs.io/en/latest/understand_mmcv/io.html)
- [Image/Video processing](https://mmcv.readthedocs.io/en/latest/understand_mmcv/data_process.html)
- [Image and annotation visualization](https://mmcv.readthedocs.io/en/latest/understand_mmcv/visualization.html)
- [Useful utilities (progress bar, timer, ...)](https://mmcv.readthedocs.io/en/latest/understand_mmcv/utils.html)
- [PyTorch runner with hooking mechanism](https://mmcv.readthedocs.io/en/latest/understand_mmcv/runner.html)
- [Various CNN architectures](https://mmcv.readthedocs.io/en/latest/understand_mmcv/cnn.html)
- [High-quality implementation of common CPU and CUDA ops](https://mmcv.readthedocs.io/en/latest/understand_mmcv/ops.html)

It supports the following systems:

- Linux
- Windows
- macOS

See the [documentation](http://mmcv.readthedocs.io/en/latest) for more features and usage.

Note: MMCV requires Python 3.6+.

## Installation

There are two versions of MMCV:

- **mmcv-full**: comprehensive, with full features and various CPU and CUDA ops out of the box. It takes longer time to build.
- **mmcv**: lite, without CPU and CUDA ops but all other features, similar to mmcv\<1.0.0. It is useful when you do not need those CUDA ops.

**Note**: Do not install both versions in the same environment, otherwise you may encounter errors like `ModuleNotFound`. You need to uninstall one before installing the other. `Installing the full version is highly recommended if CUDA is available`.

### Install mmcv-full

Before installing mmcv-full, make sure that PyTorch has been successfully installed following the [PyTorch official installation guide](https://github.com/pytorch/pytorch#installation).

The command to install mmcv-full on Linux or Windows platforms is as follows (if your system is macOS, please refer to [build mmcv-full from source](https://mmcv.readthedocs.io/en/latest/get_started/build.html#macos-mmcv-full))

```bash
pip install -U openmim
mim install mmcv-full
```

If you need to specify the version of mmcv-full, you can use the following command

```bash
mim install mmcv-full==1.5.0
```

If you find that the above installation command does not use a pre-built package ending with `.whl` but a source package ending with `.tar.gz`, you may not have a pre-build package corresponding to the PyTorch or CUDA or mmcv-full version, in which case you can [build mmcv-full from source](https://mmcv.readthedocs.io/en/latest/get_started/build.html).

<details>
<summary>Installation log using pre-built packages</summary>

Looking in links: https://download.openmmlab.com/mmcv/dist/cu102/torch1.8.0/index.html<br />
Collecting mmcv-full<br />
<b>Downloading https://download.openmmlab.com/mmcv/dist/cu102/torch1.8.0/mmcv_full-1.6.1-cp38-cp38-manylinux1_x86_64.whl</b>

</details>

<details>
<summary>Installation log using source packages</summary>

Looking in links: https://download.openmmlab.com/mmcv/dist/cu102/torch1.8.0/index.html<br />
Collecting mmcv-full==1.6.0<br />
<b>Downloading mmcv-full-1.6.0.tar.gz</b>

</details>

For more installation methods, please refer to the [Installation documentation](https://mmcv.readthedocs.io/en/latest/get_started/installation.html).

### Install mmcv

If you need to use PyTorch-related modules, make sure PyTorch has been successfully installed in your environment by referring to the [PyTorch official installation guide](https://github.com/pytorch/pytorch#installation).

```bash
pip install -U openmim
mim install mmcv
```

## Supported projects

- [MIM](https://github.com/open-mmlab/mim): MIM installs OpenMMLab packages.
- [MMClassification](https://github.com/open-mmlab/mmclassification): OpenMMLab image classification toolbox and benchmark.
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab detection toolbox and benchmark.
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): OpenMMLab's next-generation platform for general 3D object detection.
- [MMRotate](https://github.com/open-mmlab/mmrotate): OpenMMLab rotated object detection toolbox and benchmark.
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab semantic segmentation toolbox and benchmark.
- [MMOCR](https://github.com/open-mmlab/mmocr): OpenMMLab text detection, recognition, and understanding toolbox.
- [MMPose](https://github.com/open-mmlab/mmpose): OpenMMLab pose estimation toolbox and benchmark.
- [MMHuman3D](https://github.com/open-mmlab/mmhuman3d): OpenMMLab 3D human parametric model toolbox and benchmark.
- [MMSelfSup](https://github.com/open-mmlab/mmselfsup): OpenMMLab self-supervised learning toolbox and benchmark.
- [MMRazor](https://github.com/open-mmlab/mmrazor): OpenMMLab model compression toolbox and benchmark.
- [MMFewShot](https://github.com/open-mmlab/mmfewshot): OpenMMLab fewshot learning toolbox and benchmark.
- [MMAction2](https://github.com/open-mmlab/mmaction2): OpenMMLab's next-generation action understanding toolbox and benchmark.
- [MMTracking](https://github.com/open-mmlab/mmtracking): OpenMMLab video perception toolbox and benchmark.
- [MMFlow](https://github.com/open-mmlab/mmflow): OpenMMLab optical flow toolbox and benchmark.
- [MMEditing](https://github.com/open-mmlab/mmediting): OpenMMLab image and video editing toolbox.
- [MMGeneration](https://github.com/open-mmlab/mmgeneration): OpenMMLab image and video generative models toolbox.
- [MMDeploy](https://github.com/open-mmlab/mmdeploy): OpenMMLab model deployment framework.

## FAQ

If you face installation problems or runtime issues, you may first refer to this [Frequently Asked Questions](https://mmcv.readthedocs.io/zh_CN/latest/faq.html) to see if there is a solution. If the problem is still not solved, feel free to open an [issue](https://github.com/open-mmlab/mmcv/issues).

## Citation

If you find this project useful in your research, please consider cite:

```latex
@misc{mmcv,
    title={{MMCV: OpenMMLab} Computer Vision Foundation},
    author={MMCV Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmcv}},
    year={2018}
}
```

## Contributing

We appreciate all contributions to improve MMCV. Please refer to [CONTRIBUTING.md](CONTRIBUTING.md) for the contributing guideline.

## License

MMCV is released under the Apache 2.0 license, while some specific operations in this library are with other licenses. Please refer to [LICENSES.md](LICENSES.md) for the careful check, if you are using our code for commercial matters.
