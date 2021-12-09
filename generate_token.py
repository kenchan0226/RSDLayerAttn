"""construct a mapping between the id of the sample and token in the talk2car dataset"""

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

        save_json(os.path.join(data_root, "token_val.json"), dataa)

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

        save_json(os.path.join(data_root, "token_test.json"), dataa)

        f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Model
    parser.add_argument("--data_root", default="data/talk2car", type=str)

    args = parser.parse_args()

    construct_token_mapping_for_val_set(args.data_root)
    construct_token_mapping_for_test_set(args.data_root)
