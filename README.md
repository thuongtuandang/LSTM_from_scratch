# LSTM FROM SCRATCH
## Goal of this project
- Implement an LSTM model from scratch.
- Provide a complete documentation about LSTM mechanism, especially the gate architecture and their importance.
- Explain carefully the backprop computation with codes.
- Process text data and transform it into a form useful for our model for the prediction task.

## Documentation
We highyly recommend you to look at our documentation `documents/LSTM.pdf`.
Interesting points contained in the file:
- LSTM mechanism has memory cells and it also learns to forget.
- Hadamard product of matrices can be seen as a filter.
- The backprop computation for LSTM is much more complicated than RNN.

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

