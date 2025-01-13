# Overview

This project involves calculating the following scores:
- BLEU (with [sacrebleu](https://github.com/mjpost/sacrebleu))
- BLEURT (with [bleurt](https://github.com/google-research/bleurt))
- COMET (with [unbabel](https://github.com/Unbabel/COMET))

There is a webinterface using streamlit

# Installation instructions

Note: To use some COMET models such as Unbabel/wmt22-cometkiwi-da you must acknowledge it's license on Hugging Face Hub and [log-in into hugging face hub](https://huggingface.co/docs/huggingface_hub/quick-start#:~:text=Once%20you%20have%20your%20User%20Access%20Token%2C%20run%20the%20following%20command%20in%20your%20terminal%3A).

```
pip install --upgrade pip
conda create -n scoring --file requirements_conda.txt
conda activate scoring
```

```
pip install sacrebleu
```

```
git clone https://github.com/google-research/bleurt.git
cd bleurt
pip install .
```

