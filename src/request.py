import requests

def main():
    print("Insert Smiles to predict")
    smiles_input = str(input())
    print("Predicting ", smiles_input)

    resp = requests.post("http://0.0.0.0:7000/predict",
                         files={"smiles": smiles_input})
    try:
        print(resp.json())
    except ValueError:
        print("Input not valid")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        print("[Program stopped]", e)
