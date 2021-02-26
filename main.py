import argparse
import torch
from models import *
import feature_extractor as fe
from dataset import *
from build_vocab import WordVocab
import numpy as np
import math
from utils import *
from datasets import *
from sklearn.model_selection import train_test_split
import torch.optim as optim
from sklearn.metrics import f1_score,accuracy_score,balanced_accuracy_score

def sigmoid(x):
  return 1 / (1 + math.exp(-x))


def parse_arguments():
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--action', type=str, default="Predict", help='Action to execute [Train/Evaluate/Predict]')
    parser.add_argument('--modelnumber', type=str, default=2, help='Chose model tu use')
    parser.add_argument('--n_epoch', '-e', type=int, default=20, help='number of epochs')
    parser.add_argument('--vocab', '-v', type=str, default='vocab/vocab.pkl', help='vocabulary (.pkl)')
    parser.add_argument('--data', type=str, default='data/dataset_single.csv', help='train corpus (.csv)')
    parser.add_argument('--out_dir_models', '-o', type=str, default='models', help='output directory')
    parser.add_argument('--out_dir_results', '-od', type=str, default='results', help='output directory')
    parser.add_argument('--batch_size', '-b', type=int, default=10, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Adam learning rate')
    parser.add_argument('--modelpath', type=str, default="models/model2.save", help='Model to load')
    parser.add_argument('--out_model_name', type=str, default='model2.save', help='output directory')

    return parser.parse_args()

def predict(args):
    sigmoid_v = np.vectorize(sigmoid)
    print("Insert Smiles to predict")
    smiles_input = str(input())
    print("Predicting ", smiles_input)
    if args.modelnumber == 1:
        model = LinearClassification1(2048, 1)
        features = fe.fingerprint_features(smiles_input)
        nn_input = torch.FloatTensor(fe.convert_to_numpy(features))
    elif args.modelnumber == 2:
        model = LinearClassification2(220, 1)
        vocab = WordVocab.load_vocab('vocab/vocab.pkl')
        sl = SmilesDataset_singleline(smiles_input, vocab)
        nn_input = sl.get_item()

    elif args.modelnumber == 3:
        model = LinearClassification3(220, 9)
        vocab = WordVocab.load_vocab('vocab/vocab.pkl')
        sl = SmilesDataset_singleline(smiles_input, vocab)
        nn_input = sl.get_item()

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
    pass

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
            numpy_features = fe.convert_to_numpy(features)
            X.append(numpy_features)
            Y.append(P1)
            ## Train/Val Division

        X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.20, random_state=15, shuffle=True,
                                                          stratify=Y)

        train_data = molDataset(torch.FloatTensor(X_train),
                                torch.FloatTensor(y_train))
        val_data = molDataset(torch.FloatTensor(X_val),
                              torch.FloatTensor(y_val))
        end_lr = 0.0005
        PATH = "model.save"

        train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(dataset=val_data, batch_size=args.batch_size, shuffle=True)

        print("Training in ", device)
        model = LinearClassification1(2048, 1)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        lr_lambda = lambda x: math.exp(x * math.log(end_lr / args.lr) / (EPOCHS * len(train_data)))
        step_size = 4 * len(train_loader)
        clr = cyclical_lr(step_size, min_lr=end_lr / 2, max_lr=end_lr)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [clr])
        if (torch.cuda.is_available()):
            model.cuda()

        for e in range(1, args.n_epoch + 1):
            epoch_loss = 0
            epoch_acc = 0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.cuda()
                y_batch = y_batch.cuda()
                # Training mode and zero gradients
                optimizer.zero_grad()
                y_pred = model(X_batch)

                # Get outputs to calc loss
                loss = criterion(y_pred, y_batch.unsqueeze(1))
                acc = binary_acc(y_pred, y_batch.unsqueeze(1))

                # Backward pass
                loss.backward()
                optimizer.step()

                # Update LR
                scheduler.step()

                epoch_loss += loss.item()
                epoch_acc += acc.item()

            epoch_acc_val = 0
            targets = []
            outputs = []
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_val = model(X_batch)
                outputs.append(y_val.cpu().detach().numpy())
                targets.append(y_batch.cpu().detach().numpy())
                acc_val = binary_acc(y_val, y_batch.unsqueeze(1))
                epoch_acc_val += acc_val.item()
            outputs = np.concatenate(outputs)
            targets = np.concatenate(targets)
            f1 = f1_score(sigmoid_v(outputs).round().squeeze(), targets, average='binary')

            print(
                f'Epoch {e + 0:03}: Learning Rate {get_lr(optimizer):.7f} Training: | Loss: {epoch_loss / len(train_loader):.5f} | Acc: {epoch_acc / len(train_loader):.3f} Val: Acc: {epoch_acc_val / len(val_loader):.3f}  | F1 Score: {f1}')


    elif args.modelnumber == 2:
        model = LinearClassification2(220, 1)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        vocab = WordVocab.load_vocab(args.vocab)
        dataset = SmilesDatasetP1(args.data, vocab)
        train_ids, val_ids = torch.utils.data.random_split(pd.read_csv(args.data)['P1'],[len(dataset) - 1000, 1000])
        train_sampler = torch.utils.data.SubsetRandomSampler(train_ids)
        valid_sampler = torch.utils.data.SubsetRandomSampler(val_ids)
        train_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=16, sampler=train_sampler)
        val_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=16, sampler=valid_sampler)

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

                # Backward pass
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                epoch_acc += acc.item()

            epoch_acc_val = 0
            targets = []
            outputs = []
            with torch.no_grad():
                for i, (X, Y) in enumerate(val_loader):
                    X_batch = X.cuda()
                    y_batch = Y.cuda()

                    y_val = model(X_batch)
                    outputs.append(y_val.cpu().detach().numpy())
                    targets.append(y_batch.cpu().detach().numpy())
                    acc_val = binary_acc(y_val, y_batch.unsqueeze(1))
                    epoch_acc_val += acc_val.item()
                outputs = np.concatenate(outputs)
                targets = np.concatenate(targets)
                f1 = f1_score(targets, sigmoid_v(outputs).round().squeeze(), average='binary')
                print(
                    f'Epoch {e + 0:03}: Learning Rate {get_lr(optimizer):.7f} Training: | Loss: {epoch_loss / len(train_loader):.5f} | Acc: {epoch_acc / len(train_loader):.3f} Val: Acc: {epoch_acc_val / len(val_loader):.3f}  | F1 Score: {f1}')


    elif args.modelnumber == 3:
        #Loading Model , loss and optimizer
        model = LinearClassification3(220, 9)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.SGD(model.parameters(), lr=args.lr)

        #Vocabulary generated with build_vocab.py
        vocab = WordVocab.load_vocab(args.vocab)

        dataset = SmilesDatasetP1_P9(args.data, vocab)

        train, test = torch.utils.data.random_split(dataset, [len(dataset) - len(dataset) //5, len(dataset) //5])

        train_loader = DataLoader(train, batch_size=args.batch_size, num_workers=16)
        val_loader = DataLoader(test, batch_size=args.batch_size, num_workers=16)

        if (torch.cuda.is_available()):
            model.cuda()

        for e in range(1, args.n_epoch + 1):
            epoch_loss = 0
            epoch_acc = 0
            for i, (X, Y) in enumerate(train_loader):
                X_batch = X.cuda()
                y_batch = Y.cuda().squeeze(1)
                # Training mode and zero gradients
                optimizer.zero_grad()
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)

                # Backward pass
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            epoch_acc_val = 0
            targets = []
            outputs = []
            with torch.no_grad():
                for i, (X, Y) in enumerate(val_loader):
                    X_batch = X.cuda()
                    y_batch = Y.cuda().squeeze(1)
                    y_val = model(X_batch)
                    outputs.append(y_val.cpu().detach().numpy())
                    targets.append(y_batch.cpu().detach().numpy())

                outputs = np.concatenate(outputs)
                targets = np.concatenate(targets)

                # F1 Score
                f1 = f1_score(targets, outputs > 0.5, average="samples")
                # accuracy Score
                ac1 = accuracy_score(targets > 0.5, outputs > 0.5)
                print(
                    f'Epoch {e + 0:03}: Learning Rate {get_lr(optimizer):.7f} Training: | Loss: {loss.item():.5f} || Val: Accuracy Score: {ac1:.3f}  | F1 Score: {f1}')

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