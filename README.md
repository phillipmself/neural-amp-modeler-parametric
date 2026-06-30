# NAM: Neural Amp Modeler

[![Build](https://github.com/sdatkinson/neural-amp-modeler/actions/workflows/python-package.yml/badge.svg)](https://github.com/sdatkinson/neural-amp-modeler/actions/workflows/python-package.yml)

This repository handles training models and exporting them to .nam files.
For playing trained models in real time in a standalone application or plugin, see the partner repo,
[NeuralAmpModelerPlugin](https://github.com/sdatkinson/NeuralAmpModelerPlugin).

For more information about the NAM ecosystem please check out https://www.neuralampmodeler.com/.

## Documentation
Online documentation can be found here: 
https://neural-amp-modeler.readthedocs.io

To build the documentation locally on a Linux system:
```bash
cd docs
make html
```

Or on Windows,
```
cd docs
make.bat html
```

## Active-learning capture selection (parametric)
This fork adds a PANAMA-style ([arXiv 2509.26564v1](https://arxiv.org/html/2509.26564v1))
active-learning loop that proposes *which knob settings to capture next* for a parametric model: it
trains a disposable ConcatLSTM ensemble and finds the control settings where the members disagree
most. See [docs/active_learning_usage.md](docs/active_learning_usage.md) for the workflow, with
example configs in [nam_full_configs/active_learning/](nam_full_configs/active_learning/).
