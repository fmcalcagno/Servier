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
    parser.add_argument('--modelnumber', type=str, default=2, help='Chose model tu use')
    parser.add_argument('--n_epoch', '-e', type=int, default=100, help='number of epochs')
    parser.add_argument('--data', type=str, default='data/dataset_multi.csv', help='train corpus (.csv)')
    parser.add_argument('--out_dir_models', '-o', type=str, default='models', help='output directory')
    parser.add_argument('--out_file_results', '-of', type=str, default='results/output2.csv', help='output file for Evaluation')
    parser.add_argument('--batch_size', '-b', type=int, default=10, help='batch size')
    parser.add_argument('--lr', type=float, default=0.005, help='Learning rate')
    parser.add_argument('--modelpath', type=str, default="models/model2.save", help='Model to load')
    parser.add_argument('--out_model_name', type=str, default='model2.save', help='output directory')

    return parser.parse_args()

def predict(args):
    sigmoid_v = np.vectorize(sigmoid)
    print("Insert Smiles to predict")
    smiles_input = str(input())
    print("Predicting ", smiles_input)

    if args.modelnumber == 1:
        model = LinearClassificationX(input_dim=2048, hidden_dim=256, tagset_size=1, dropout=0.7)
        features = fe.fingerprint_features(smiles_input)
        nn_input = torch.FloatTensor(np.asarray(features))
    elif args.modelnumber == 2:
        model = LSTMModel(input_size=2048, embed_size=50, hidden_size=150, output_size=2)

    elif args.modelnumber == 3:
        model = LSTMModel(input_size=2048, embed_size=50, hidden_size=150, output_size=18)

        #sl = SmilesDataset_singleline(smiles_input, vocab)
        #nn_input = sl.get_item()

    if (torch.cuda.is_available()):
        model.cuda()

    model.load_state_dict(torch.load(args.modelpath))

    model.eval()

    if (torch.cuda.is_available()):
        model.cuda()

    with torch.no_grad():
        X_batch = nn_input.cuda()

        y_val = model(X_batch)
        print(sigmoid_v(y_val.cpu().detach().numpy()).round())


def evaluate(args):
    print("Evaluating model ", args.modelnumber)
    sigmoid_v = np.vectorize(sigmoid)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.modelnumber < 1 or args.modelnumber > 3: return
    if args.modelnumber == 1:
        data = importfiles(args.data)
        X, Y = [], []
        for index, row in data.iterrows():
            P1, mol_id, smiles = row
            features = fe.fingerprint_features(smiles)
            numpy_features = np.asarray(features)
            X.append(numpy_features)
            Y.append(P1)

        full_dataset = molDataset(torch.FloatTensor(X ),
                                   torch.FloatTensor(Y))
        model = LinearClassificationX(input_dim=2048,hidden_dim=256,tagset_size=1,dropout=0.7)

    elif args.modelnumber == 2:

        hotencoder= OneHotEncoder()
        full_dataset = SmilesDatasetP1(hotencoder,args.data,input_type="Long")
        model = LSTMModel(input_size=2048, embed_size=50, hidden_size=150, output_size=2)

    elif args.modelnumber == 3:

        model = LSTMModel(input_size=2048, embed_size=50, hidden_size=150, output_size=18)
        hotencoder = OneHotEncoder()
        full_dataset = SmilesDatasetP1_P9(hotencoder, args.data, input_type="Long")


    # Load model from file
    train_set, evaluate_data = torch.utils.data.random_split(full_dataset, [4499, 500], generator=torch.Generator().manual_seed(42))
    eval_loader = DataLoader(evaluate_data, batch_size=args.batch_size, shuffle=False)
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
        # Test mode and zero gradients
        if args.modelnumber ==1:
            y_pred = model(X_batch)
        elif args.modelnumber >= 2:
            y_pred,_ = model(X_batch)
        outputs.append(y_pred.cpu().detach().numpy())
        targets.append(y_batch.cpu().detach().numpy())
        #print("Predictions:", torch.round(torch.sigmoid(y_pred)))
        #print("Target:", y_batch.unsqueeze(1))

    # Save targets in an output file
    outputs = np.concatenate(outputs)
    targets = np.concatenate(targets)

    model_output = pd.DataFrame()


    if args.modelnumber == 1 :
        model_output["target"] = targets
        #model_output["predictions"] = outputs
        sigmoid_v = np.vectorize(sigmoid)
        outputs[outputs < -300] = -300
        outputs[outputs > 300] = 300
        out= sigmoid_v(outputs).round()
        model_output["predictions"] = out
        print("Confusion Matrix")
        print(confusion_matrix(targets, out))
        print("Classification report")
        print(classification_report(targets, out))

    if args.modelnumber == 2:

        y_test_classes = hotencoder.inverse_transform(targets)
        y_pred_classes = hotencoder.inverse_transform(outputs)
        model_output["target"] = [j for i in y_pred_classes for j in i]
        model_output["predictions"] = [j for i in y_test_classes for j in i]
        print("Confusion Matrix")
        print(confusion_matrix(y_test_classes, y_pred_classes))
        print("Classification report")
        print(classification_report(y_test_classes, y_pred_classes))

    if args.modelnumber == 3:
        y_test_classes = hotencoder.inverse_transform(targets)
        y_pred_classes = hotencoder.inverse_transform(outputs)
        model_output["target"] = [j for i in y_pred_classes for j in i]
        model_output["predictions"] = [j for i in y_test_classes for j in i]
        print("Confusion Matrix")
        print(multilabel_confusion_matrix(y_test_classes, y_pred_classes))
        print("Classification report")
        print(classification_report(y_test_classes, y_pred_classes))


    model_output.to_csv(args.out_file_results, index=False)
    print("Results exported to ", args.out_file_results)

def train(args):
    print("Training model ",args.modelnumber)
    sigmoid_v = np.vectorize(sigmoid)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.modelnumber<1 or args.modelnumber> 3: return
    if args.modelnumber == 1:
        data = importfiles(args.data)

        #Preprocessing Smiles into Features
        X,Y = [],[]
        for index, row in data.iterrows():
            P1, mol_id, smiles = row
            features = fe.fingerprint_features(smiles)
            X.append(np.asarray(features))
            Y.append(P1)


        full_dataset = molDataset(torch.FloatTensor(X),
                                  torch.FloatTensor(Y))


        print("Training in ", device)
        model = LinearClassificationX(input_dim=2048,hidden_dim=256,tagset_size=1,dropout=0.7)
        criterion = nn.BCEWithLogitsLoss(pos_weight = torch.FloatTensor([0.6]).cuda())
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        train_set, evaluate_data = torch.utils.data.random_split(full_dataset, [4499, 500],
                                                                 generator=torch.Generator().manual_seed(42))
        train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True)
        if (torch.cuda.is_available()):
            model.cuda()
        for e in range(1, args.n_epoch + 1):
            epoch_loss = 0
            epoch_acc = 0
            for i, (X, Y) in enumerate(train_loader):
                X_batch = X.cuda()
                y_batch = Y.cuda()
                # Training mode and zero gradients
                optimizer.zero_grad()
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch.unsqueeze(1))
                acc = binary_acc(y_pred, y_batch.unsqueeze(1))

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                epoch_acc += acc.item()
            if args.modelnumber == 3:
                print(
                    f'Epoch {e + 0:03}: | Loss: {epoch_loss / len(train_loader):.5f}')
            else:
                print(
                    f'Epoch {e + 0:03}: | Loss: {epoch_loss / len(train_loader):.5f} | Acc: {epoch_acc / len(train_loader):.3f}')

        torch.save(model.state_dict(), "{}/{}".format(args.out_dir_models, args.out_model_name))

        print("Model Saved in: ", "{}/{}".format(args.out_dir_models, args.out_model_name))
        return
    elif args.modelnumber == 2:


        model = LSTMModel(input_size=2048, embed_size=50, hidden_size=150, output_size=2)
        criterion = nn.BCEWithLogitsLoss(pos_weight = torch.FloatTensor([0.4, 0.6]).cuda())
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        hotencoder = OneHotEncoder()
        full_dataset = SmilesDatasetP1(hotencoder,args.data,input_type="Long")
        targets = full_dataset.targets

    elif args.modelnumber == 3:
        #Loading Model , loss and optimizer

        model = LSTMModel(input_size=2048, embed_size=50, hidden_size=150, output_size=18)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        hotencoder = OneHotEncoder()
        full_dataset = SmilesDatasetP1_P9(hotencoder,args.data,input_type="Long")




    train_set, evaluate_data = torch.utils.data.random_split(full_dataset, [4499, 500],
                                                             generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=False)
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