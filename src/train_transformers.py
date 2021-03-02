import argparse
from simpletransformers.classification import ClassificationModel,ClassificationArgs
import logging
import sklearn
import pandas as pd

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)


def parse_arguments():
    """
    Function that parsers parameters
    The parameters actions is placed there in the case the file main.py is executed independently
    :return:
    parser parameters
    """
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--action', type=str, default="Train", help='Action to execute [Train/Evaluate/Predict]')
    parser.add_argument('--n_epoch', '-e', type=int, default=20, help='number of epochs')
    parser.add_argument('--train_data', type=str, default='data/dataset_single_train_transformer.csv', help='train corpus (.csv)')
    parser.add_argument('--val_data', type=str, default='data/dataset_single_test_transformer.csv', help='validation corpus (.csv)')
    parser.add_argument('--out_dir_models', '-o', type=str, default='models', help='output directory')
    parser.add_argument('--modelpath', type=str, default="models/model3_final.save", help='Model to load')
    parser.add_argument('--out_model_name', type=str, default='model3_final.save', help='output directory')

    return parser.parse_args()

def train_transformer():
    args = parse_arguments()
    print("Training Transformer")
    model = ClassificationModel('roberta', 'seyonec/PubChem10M_SMILES_BPE_396_250',
                                args={'evaluate_each_epoch': True, 'evaluate_during_training_verbose': True,
                                      'no_save': True, 'num_train_epochs': args.n_epoch, 'auto_weights': False})
    train_df = pd.read_csv(args.train_data, dtype={'label': int, 'text': str})
    valid_df = pd.read_csv(args.val_data, dtype={'label': int, 'text': str})
    model.train_model(train_df, eval_df=valid_df, output_dir="{}/{}".format(args.out_dir_models, args.out_model_name),
                      args={'wandb_project': 'project-name'})

    result, model_outputs, wrong_predictions = model.eval_model(valid_df, acc=sklearn.metrics.accuracy_score)

    print(result)

    predictions, raw_outputs = model.predict(['Cc1nn(-c2ccccc2)c(Cl)c1C1C(C#N)=C(N)OC2=C1C(=O)CCC2'])

    print(predictions)
    print(raw_outputs)

def main():
    train_transformer()




if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        print("[Program stopped]", e)