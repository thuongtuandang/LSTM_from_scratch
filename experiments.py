
import argparse
import importlib
import pickle

from dataset import Dataset
from models.model import BaseModel

_DEFAULT_DATA_PATH = "data/robert_frost.txt"

def _import_class(module_and_class_name: str) -> type:
    module_name, class_name = module_and_class_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_

def main(args):
    dataset = Dataset(input_length=args.input_length, output_length=args.output_length)
    dataset.load(args.data_path)
    model_class = _import_class(f"models.{args.model}")

    model: BaseModel = model_class(input_size=dataset.vocab_size, output_size=dataset.vocab_size)
    model.fit(dataset.x_train, dataset.y_train, learning_rate=args.learning_rate)

    output, acc = model.predict(dataset.x_test, dataset.y_test)
    print(f"Accuracy on test set: {acc}")
    with open('model.pkl', mode='wb') as file:
        pickle.dump(model, file)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="Experiment model"
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        help="Selected model class",
    )
    parser.add_argument(
        "-d",
        "--data_path",
        type=str,
        required=False,
        default=_DEFAULT_DATA_PATH,
        help="Name of the local data text file",
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
    



      