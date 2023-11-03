# LSTM FROM SCRATCH
## Goal of this project
- Understand how to implement a LSTM model from scratch
- Understand the flow of creating and using a model with special task

If you are interested in learning about the theory of LSTM, You can refer to the `documents/LSTM.pdf` file.
## Install python package
### Using `venv`:
```
python -m venv ./myenv
source ./myenv/bin/activate
pip install requirements.txt
```
### Using `pipenv`:
```
pipenv install
```
## Training
You can train a LSTM model with default setting:
- data: `data/test.txt`
- learning rate: `0.3`
- number of iterations: `100`
```
python experiments.py
```
You can change to your custom data by coping your data file to `data` folder and update `DATA_PATH` variable in `experiments.py` file.
```
DATA_PATH = "path/to/your/data/file"
```
You also can change `learning rate` and `number of iterations` with:
```
python experiments.py --learning_rate 0.3 --num_iter 100
```
## Prediction
After training, the trained model will be save at `saved_model/model.pkl`, that will help us can run prediction at anytime by load this model file. You run the prediction by run the following command:
```
python word_prediction.py --text "old man"
```

with `--text` param is the text you want to predict the next word

