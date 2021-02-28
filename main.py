import argparse
from models import *
import feature_extractor as fe
from datasets import *
import torch.optim as optim
from sklearn.metrics import confusion_matrix,classification_report,multilabel_confusion_matrix
from sklearn.preprocessing import OneHotEncoder



def parse_arguments():
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--action', type=str, default="Train", help='Action to execute [Train/Evaluate/Predict]')
    parser.add_argument('--modelnumber', type=str, default=1, help='Chose model tu use')
    parser.add_argument('--n_epoch', '-e', type=int, default=30, help='number of epochs')
    parser.add_argument('--train_data', type=str, default='data/dataset_single_train.csv', help='train corpus (.csv)')
    parser.add_argument('--val_data', type=str, default='data/dataset_single_test.csv', help='validation corpus (.csv)')
    parser.add_argument('--out_dir_models', '-o', type=str, default='models', help='output directory')
    parser.add_argument('--out_file_results', '-of', type=str, default='results/output1.csv', help='output file for Evaluation')
    parser.add_argument('--batch_size', '-b', type=int, default=6, help='batch size')
    parser.add_argument('--lr', type=float, default=0.00005, help='Learning rate')
    parser.add_argument('--modelpath', type=str, default="models/model1_final.save", help='Model to load')
    parser.add_argument('--out_model_name', type=str, default='model1_final.save', help='output directory')

    return parser.parse_args()

def predict(args):
    """
    Function to predict the model selected in the parameteres (could be model 1, 2 or 3) with a string inserted as an
    input.
    The Output changes depending of the chosen model (model 3 has 9 outputs).
    :param args:
    :return:
    """
    if args.modelnumber < 1 or args.modelnumber > 3:
        print("Error: Number of the chosen model must be between 1 and 3")
        return
    sigmoid_v = np.vectorize(sigmoid)
    print("Insert Smiles to predict")
    smiles_input = str(input())
    print("Predicting ", smiles_input)
    print("Smiles Validity",validity(smiles_input))

    sl = SmilesDataset_singleline(model_number=args.modelnumber, input1=smiles_input)
    outputsize = 2 if args.modelnumber < 3 else 18
    model = LSTMModel(input_size=2048, embed_size=50, hidden_size=40, output_size=outputsize)
    hotencoder = OneHotEncoder()
    nn_input = sl.get_item()

    if args.modelnumber < 3:
        he = hotencoder.fit_transform([[0], [1]]).toarray()
    elif args.modelnumber == 3:
        he = hotencoder.fit_transform([[0,0,0,0,0,0,0,0,0],[1,1,1,1,1,1,1,1,1]]).toarray()

    if (torch.cuda.is_available()):
        model.cuda()

    model.load_state_dict(torch.load(args.modelpath))

    if (torch.cuda.is_available()):
        model.cuda()

    with torch.no_grad():
        model.eval()
        X_batch = nn_input.cuda()
        y_val, _ = model(X_batch.unsqueeze(0))
        y_val2 = y_val.cpu().detach().numpy()

        y_pred_classes = hotencoder.inverse_transform(y_val2)
        print(y_pred_classes)


def evaluate(args):
    """
    I Normally evaluate the model after each training epoch but the pdf demanded an independent evaluate function
    :param args:
    :return:
    """
    if args.modelnumber < 1 or args.modelnumber > 3:
        print("Error: Number of the chosen model must be between 1 and 3")
        return
    print("Evaluating model ", args.modelnumber)
    sigmoid_v = np.vectorize(sigmoid)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    hotencoder = OneHotEncoder()
    outputsize = (1 if args.modelnumber < 3 else 9)*2
    model = LSTMModel(input_size=2048, embed_size=30, hidden_size=50, output_size=outputsize)
    eval_dataset = SmilesDataset(model_number=args.modelnumber, hotencoder=hotencoder, file=args.val_data, input_type="Long")

    # Load model from file
    #train_set, evaluate_data = torch.utils.data.random_split(full_dataset, [4499, 500], generator=torch.Generator().manual_seed(25))
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)
    model.load_state_dict(torch.load(args.modelpath),strict=False)
    model.eval()
    outputs = []
    targets = []
    if (torch.cuda.is_available()):
        model.cuda()
    for X_batch, y_batch in eval_loader:
        X_batch = X_batch.cuda()
        y_batch = y_batch.cuda()
        model.eval()
        y_pred, _ = model(X_batch)
        outputs.append(y_pred.cpu().detach().numpy())
        targets.append(y_batch.cpu().detach().numpy())

    # Save targets in an output file
    outputs = np.concatenate(outputs)
    targets = np.concatenate(targets)

    model_output = pd.DataFrame()

    y_test_classes = hotencoder.inverse_transform(targets)
    y_pred_classes = hotencoder.inverse_transform(outputs)
    model_output["target"] = [j for i in y_pred_classes for j in i]
    model_output["predictions"] = [j for i in y_test_classes for j in i]
    print("Confusion Matrix")
    if args.modelnumber < 3 :
        print(confusion_matrix(y_test_classes, y_pred_classes))
    if args.modelnumber == 3:
        print(multilabel_confusion_matrix(y_test_classes, y_pred_classes))
    print("Classification report")
    print(classification_report(y_test_classes, y_pred_classes))

    model_output.to_csv(args.out_file_results, index=False)
    print("Results exported to ", args.out_file_results)

def train(args):
    """
    Method to train the chosen Model (chosen in the arguments)
    :param args:
    :return:
    """
    print("Training model ",args.modelnumber," with learning rate ",args.lr)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.modelnumber<1 or args.modelnumber> 3:
        print("Error: Number of the chosen model must be between 1 and 3")
        return
    hotencoder = OneHotEncoder()

    outputsize = (1 if args.modelnumber < 3 else 9) * 2
    # Rule of thumb is to have the number of hidden units  (hidden_size) be in-between the number of input units
    # (embed_size)(50) and output classes (2/18);
    model = LSTMModel(input_size=2048, embed_size=50, hidden_size=40, output_size=outputsize)
    trian_dataset = SmilesDataset(model_number=args.modelnumber, hotencoder=hotencoder, file=args.train_data,
                                 input_type="Long")
    val_dataset = SmilesDataset(model_number=args.modelnumber, hotencoder=hotencoder, file=args.val_data,
                                  input_type="Long")
    # As we are working with an unbalaced dataset, use the loss function to prioritize the importance of a class
    criterionvector = torch.FloatTensor([0.4, 0.6]).cuda() if args.modelnumber < 3 else None
    #BCEWithLogitsLoss is more numerically stable than using a plain Sigmoid followed by a BCELoss
    criterion = nn.BCEWithLogitsLoss(pos_weight=criterionvector)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    #train_set, evaluate_data = torch.utils.data.random_split(full_dataset, [7172, 500],
    #                                                         generator=torch.Generator().manual_seed(25))


    train_loader = DataLoader(dataset=trian_dataset, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False)
    if (torch.cuda.is_available()):
        model.cuda()
    for e in range(1, args.n_epoch + 1):
        epoch_loss = 0
        epoch_acc = 0
        hidden = None
        for i, (X, Y) in enumerate(train_loader):
            model.train()
            X_batch = X.cuda()
            y_batch = Y.cuda()
            optimizer.zero_grad()
            y_pred,hidden =model(X_batch,hidden)

            if args.modelnumber == 3:
                loss = criterion(y_pred, y_batch)

            else:
                loss = criterion(y_pred, y_batch)
                acc = binary_acc2(y_pred, y_batch)

            # Backward pass
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            if args.modelnumber != 3:
                epoch_acc += acc.item()

        if args.modelnumber == 3:
            print(
                f'Epoch {e + 0:03}: | Loss: {epoch_loss / len(train_loader):.5f}')
        else:
            print(
            f'Epoch {e + 0:03}: | Loss: {epoch_loss / len(train_loader):.5f} | Acc: {epoch_acc / (len(train_loader) ) :.3f}')

        #Check overfitting
        if e % 10 == 0:
            with torch.no_grad():
                model.eval()
                outputs = []
                targets = []
                for X_batch, y_batch in train_loader:
                    X_batch = X_batch.cuda()
                    y_batch = y_batch.cuda()
                    y_pred, _ = model(X_batch)
                    outputs.append(y_pred.cpu().detach().numpy())
                    targets.append(y_batch.cpu().detach().numpy())

                # Save targets in an output file
                outputs = np.concatenate(outputs)
                targets = np.concatenate(targets)

                model_output = pd.DataFrame()

                y_train_classes = hotencoder.inverse_transform(targets)
                y_pred_classes = hotencoder.inverse_transform(outputs)
                print("Epoch {} Train measures:".format(e))
                print(classification_report(y_train_classes, y_pred_classes))

                outputs = []
                targets = []
                for X_batch, y_batch in eval_loader:
                    X_batch = X_batch.cuda()
                    y_batch = y_batch.cuda()
                    y_pred, _ = model(X_batch)
                    outputs.append(y_pred.cpu().detach().numpy())
                    targets.append(y_batch.cpu().detach().numpy())

                outputs = np.concatenate(outputs)
                targets = np.concatenate(targets)
                y_test_classes = hotencoder.inverse_transform(targets)
                y_pred_classes = hotencoder.inverse_transform(outputs)
                print("Epoch {} Eval measures:".format(e))
                print(classification_report(y_test_classes, y_pred_classes))



    torch.save(model.state_dict(), "{}/{}".format(args.out_dir_models, args.out_model_name))

    print("Model Saved in: ", "{}/{}".format(args.out_dir_models, args.out_model_name))
def main():
    args = parse_arguments()

    assert torch.cuda.is_available()
    if args.action == "Train":
        train(args)
    elif args.action == "Evaluate":
        evaluate(args)
    elif args.action == "Predict":
        predict(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        print("[Program stopped]", e)