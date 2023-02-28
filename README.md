# babyLM-Challenge

## Setup environment and do training
run.sh

## Inference only to generate bably talk corpus
python3 -m venv babylm
source activate babylm/bin/activate
pip3 install -r requirements.txt
### download model.pt file and vocab_obj.pth to the babylm_10M folder
python3 generate_babytalk_corpus.py