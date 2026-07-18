# SpecFSL
This framework derives transferable spectral feature extractors via contrastive pre-training to realize few-shot identification of compounds from SERS spectra.

This repository implements SpecFSL, a few-shot learning framework built on Siamese networks for SERS spectrum recognition, containing two interactive notebooks for contrastive pre-training and downstream inference.
The core concepts and experimental results of this work will be compiled into a paper focusing on few-shot SERS compound identification. We will continuously improve code readability in subsequent updates and extend the framework to support more spectral measurement modes and recognition tasks for complex mixed samples.

If you aim to reproduce the few-shot identification experiments, infer.ipynb is the recommended entry point. It provides an end-to-end demonstration of extracting spectral features and completing compound classification with the pre-trained encoder. The following preparations are required before running:
    1. A functional Python environment with all dependent packages installed (one-click installation command is provided below);
    2. Pre-trained model checkpoint: 2trained_model_final_model.pt, which stores complete parameters of the trained Siamese network;
    3. Test dataset: The data/ folder stores all SERS spectrum samples of compounds for testing. Place all categorized subfolders inside the data/ directory before inference.
Once the checkpoint and test dataset are ready, run all cells in infer.ipynb sequentially to launch the full few-shot recognition pipeline and evaluate classification performance.


