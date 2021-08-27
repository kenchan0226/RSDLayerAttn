"""i is the id number of the corresponding image. In order to ensure that the data can be distinguished, we divide it
into 0-8348 for the train set, 8349-9511 for the verification set, and 9512-11958 for the test set. Therefore,
when running the script, you need to modify the size of the reference data set to modify the corresponding value,
and modify the name of the saved file and the name of the corresponding variable. """
import json
import argparse
import os


def construct_prediction_json(data_root, result_path):
    prediction_dict = {}
    with open(os.path.join(data_root, 'token_val.json'), encoding='utf-8') as f1:
        data_token = json.load(f1)
        with open(os.path.join(result_path, "val_prediction.json"), encoding='utf-8') as f2:
            data_prediction = json.load(f2)
            i = 8349
            cnt = 0
            while i < 8349 + 1163:
                token = data_token[cnt][str(i)]
                bbox = data_prediction[cnt][str(i)]
                i += 1
                cnt += 1
                prediction_dict[token] = bbox

    with open(os.path.join(data_root, 'token_test.json'), encoding='utf-8') as f1:
        data_token = json.load(f1)
        with open(os.path.join(result_path, "test_prediction.json"), encoding='utf-8') as f2:
            data_prediction = json.load(f2)
            i = 8349 + 1163
            cnt = 0
            while i < 8349 + 1163 + 2447:
                token = data_token[cnt][str(i)]
                bbox = data_prediction[cnt][str(i)]
                i += 1
                cnt += 1
                prediction_dict[token] = bbox

    with open(os.path.join(result_path, 'predictions_for_leaderboard.json'), 'w') as f:
        json.dump(prediction_dict, f)


"""
def construct_prediction_for_val_set():
    with open('./token_val.json', encoding='utf-8') as f1:
        prediction_dict = {}
        data_token = json.load(f1)
        with open("./val_prediction.json", encoding='utf-8') as f2:
            data_prediction = json.load(f2)

            i = 8349
            cnt = 0
            while i < 8349 + 1163:
                token = data_token[cnt][str(i)]
                bbox = data_prediction[cnt][str(i)]
                i += 1
                cnt += 1

                prediction_dict[token] = bbox

            with open('predictions_val.json', 'w') as f:
                json.dump(prediction_dict, f)


def construct_prediction_for_test_set():
    with open('./token_test.json', encoding='utf-8') as f1:
        prediction_dict = {}
        data_token = json.load(f1)
        with open("./test_prediction.json", encoding='utf-8') as f2:
            data_prediction = json.load(f2)

            i = 8349+1163
            cnt = 0
            while i < 8349 + 1163+2447:
                token = data_token[cnt][str(i)]
                bbox = data_prediction[cnt][str(i)]
                i += 1
                cnt += 1

                prediction_dict[token] = bbox

            with open('predictions_test.json', 'w') as f:
                json.dump(prediction_dict, f)
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Model
    parser.add_argument("--data_root", default="data/talk2car", type=str)
    parser.add_argument("--result_path", required=True, type=str)

    args = parser.parse_args()

    construct_prediction_json(args.data_root, args.result_path)
    #construct_prediction_for_val_set()
    #construct_prediction_for_test_set()
