# LSTM FROM SCRATCH
## Goal of this project
- Implement an LSTM model from scratch.
- Process text data and transform it into a form useful for our model for the prediction task.

If you are interested in the theoretical background of LSTM, you can read about it in our file `documents/LSTM.pdf`.
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
You can train an LSTM model with default setting:
- data: `data/test.txt`
- learning rate: `0.1`
- number of iterations: `101`
```
python experiments.py
```
You can switch to your own data by coping your data file to the `data` folder and update the `DATA_PATH` variable in the `experiments.py` file.
```
DATA_PATH = "path/to/your/data/file"
```
You also can change the `learning rate` and the `number of iterations` with this command:
```
python experiments.py --learning_rate 0.1 --num_iter 101
```
## Prediction
After training, the trained model is saved at `saved_model/model.pkl` so that we can run the prediction without retraining by simply loading the model file. You can run the prediction by the following command:
```
python word_prediction.py --text "old man"
```

with the `--text` param is the text you want to predict the next word.

