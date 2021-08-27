"""i is the id number of the corresponding image. In order to ensure that the data can be distinguished, we divide it
into 0-8348 for the train set, 8349-9511 for the verification set, and 9512-11958 for the test set. Therefore,
when running the script, you need to modify the size of the reference data set to modify the corresponding value,
and modify the name of the saved file and the name of the corresponding variable. """

import json
import argparse
import os


def save_json(save_path, data):
    assert save_path.split('.')[-1] == 'json'
    with open(save_path, 'w') as file: json.dump(data, file)


def construct_token_mapping_for_val_set(data_root):
    with open(os.path.join(data_root, 'talk2car_w_rpn_no_duplicates.json'), encoding='utf-8') as f:
        data = json.load(f)
        dataa = []

        i = 8349
        cnt = 0
        while i < 8349 + 1163:
            talk2car = {i: data['val'][str(cnt)]['command_token']}
            i += 1
            cnt += 1
            json_data = json.dumps(talk2car)
            data_ = json.loads(json_data)
            dataa.append(data_)

        save_json("./token_val.json", dataa)

        talk2car = {}
        f.close()


def construct_token_mapping_for_test_set(data_root):
    with open(os.path.join(data_root, 'talk2car_w_rpn_no_duplicates.json'), encoding='utf-8') as f:
        data = json.load(f)
        dataa = []

        i = 8349 + 1163
        cnt = 0
        while i < 8349 + 1163 + 2447:
            talk2car = {i: data['test'][str(cnt)]['command_token']}
            i += 1
            cnt += 1
            json_data = json.dumps(talk2car)
            data_ = json.loads(json_data)
            dataa.append(data_)

        save_json("./token_test.json", dataa)

        talk2car = {}
        f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Model
    parser.add_argument("--data_root", default="data/talk2car", type=str)

    args = parser.parse_args()

    construct_token_mapping_for_val_set(args.data_root)
    construct_token_mapping_for_test_set(args.data_root)
