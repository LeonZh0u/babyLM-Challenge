#!/bin/bash
python3 -m venv babylm
source activate babylm/bin/activate
pip3 install -r requirements.txt
# download training data
python3 data_download.py
# train model
python3 babylm_syllables.py
# inference to generate babytalk corpus
python3 generate_babytalk_corpus.py