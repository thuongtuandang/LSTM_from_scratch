import pickle
from dataset import Dataset
import numpy as np
import argparse
from experiments import DATA_PATH

from lstm import LSTM

def main(args):
    with open('saved_model/model.pkl', mode='rb') as f:
        model: LSTM = pickle.load(f)
        text = args.text
        dataset = Dataset(input_length=args.input_length, output_length=args.output_length)
        dataset.load(DATA_PATH)
        try:
            v = dataset.word_embedding(text)
            y_pred = model.word_predict(v)
            index = np.argmax(y_pred)
            word = dataset.word_dict[index]
            print(f"Next word prediction: {word}")
        except ValueError:
            print(f"The text '{text}' has word that is not in dictionary of the data file ({DATA_PATH}). Please try another words.")
        

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="Experiment model"
    )
    parser.add_argument(
        "-t",
        "--text",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--input_length",
        type=int,
        required=False,
        default=2,
    )

    parser.add_argument(
        "--output_length",
        type=int,
        required=False,
        default=1,
    )

    args = parser.parse_args()
    main(args)