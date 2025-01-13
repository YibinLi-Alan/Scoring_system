# Overview

This project involves calculating the following scores:
- BLEU (with [sacrebleu](https://github.com/mjpost/sacrebleu))
- BLEURT (with [bleurt](https://github.com/google-research/bleurt))
- COMET (with [unbabel](https://github.com/Unbabel/COMET))

There is a webinterface using streamlit

# Installation instructions
Install the conda environment
```
pip install --upgrade pip
conda create -n scoring --file requirements_conda.txt
conda activate scoring
```
## Sacrebleu

```
pip install sacrebleu
```

## BLEURT

#### 1. Download the python package itself
```
git clone https://github.com/google-research/bleurt.git
cd bleurt
pip install .
```

#### 2. Download the checkpoint (~2GB)
```
wget https://storage.googleapis.com/bleurt-oss-21/BLEURT-20.zip .
unzip BLEURT-20.zip
rm BLEURT-20.zip
cd ..
```

# Comet
```
pip install unbabel-comet
```

Note: To use some COMET models such as Unbabel/wmt22-cometkiwi-da you must acknowledge it's license on Hugging Face Hub and [log-in into hugging face hub](https://huggingface.co/docs/huggingface_hub/quick-start#:~:text=Once%20you%20have%20your%20User%20Access%20Token%2C%20run%20the%20following%20command%20in%20your%20terminal%3A).

Create a token on hugging_face and run:
```
bash save_hugging_face_token.sh <your_hugging_face_token>
```

If you haven't already set up a GPG keypair, you'll be prompted to do so during the first run of the script. Follow the on-screen instructions to create a keypair.


# Sanity check

### check sacrebleu

```
sacrebleu -t wmt17 -l en-de -i sample_translations/wmt17.en-de.de
```
should produce:
```
{
 "name": "BLEU",
 "score": 100.0,
 "signature": "nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.5.1",
 "verbose_score": "100.0/100.0/100.0/100.0 (BP = 1.000 ratio = 1.000 hyp_len = 61287 ref_len = 61287)",
 "nrefs": "1",
 "case": "mixed",
 "eff": "no",
 "tok": "13a",
 "smooth": "exp",
 "version": "2.5.1"
```

### check BLEURT

```
python test_bleurt.py
```

should print at the end:
```
[0.792839527130127]
```

### check comet (COMET-XL is ~14GB)
```
python test_comet.py
```

should print at the end:
```
Prediction({'scores': [0.8417136073112488, 0.7745388746261597], 'system_score': 0.8081262409687042}
```
