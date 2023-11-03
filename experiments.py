
import argparse
import pickle

from dataset import Dataset
from lstm import LSTM

DATA_PATH = "data/test.txt"


def main(args):
    dataset = Dataset(input_length=args.input_length, output_length=args.output_length)
    dataset.load(DATA_PATH)

    model = LSTM(input_size=dataset.vocab_size, output_size=dataset.vocab_size)
    model.fit(dataset.x_train, dataset.y_train, max_iter=args.num_iter, learning_rate=args.learning_rate)

    output, acc = model.predict(dataset.x_test, dataset.y_test)
    print(f"Accuracy on test set: {acc}")
    with open('saved_model/model.pkl', mode='wb') as file:
        pickle.dump(model, file)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="Experiment model"
    )

    parser.add_argument(
        "-ni",
        "--num_iter",
        type=int,
        required=False,
        default=1,
    )

    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        required=False,
        default=0.03,
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
    



      