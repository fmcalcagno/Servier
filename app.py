from flask import Flask, jsonify, request
from src.datasets import  SmilesDataset_singleline
from sklearn.preprocessing import OneHotEncoder
import torch
from src.models import LSTMModel

app = Flask(__name__)
hotencoder = OneHotEncoder()
he = hotencoder.fit_transform([[0], [1]]).toarray()
model = LSTMModel(input_size=2048, embed_size=50, hidden_size=40, output_size=2)
model.load_state_dict(torch.load("models/model1_final.save"))
model.eval()


def transform_smiles(smiles_input):

    sl = SmilesDataset_singleline(model_number=1, input1=smiles_input)
    nn_input = sl.get_item()

    return nn_input.unsqueeze(0)


def get_prediction(str_bytes):

    tensor = transform_smiles(smiles_input=str_bytes)
    outputs, _ = model.forward(tensor)
    outputs2 = outputs.cpu().detach().numpy()

    y_pred_classes = hotencoder.inverse_transform(outputs2)
    return y_pred_classes


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        smiles = request.files['smiles']
        img_bytes = smiles.read()
        print(img_bytes)
        predicted = str(get_prediction(str_bytes=img_bytes))
        return jsonify({'Predicted': predicted})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7000)