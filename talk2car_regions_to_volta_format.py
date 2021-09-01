import json


def save_json(save_path, data):
    assert save_path.split('.')[-1] == 'json'
    with open(save_path, 'w') as file: json.dump(data, file)


def main(talk2car_region_path):
    talk2car_js = json.load(open(talk2car_region_path, 'r'))
    total_num_images = 8349 + 1163 + 2447
    image_id = 0
    dets = []
    det_id = 0

    for split in ["train", "val", "test"]:
        image_cnt_in_split = len(talk2car_js[split].keys())
        for i in range(image_cnt_in_split):
            sample_dict = talk2car_js[split][str(i)]
            b_box_list = sample_dict['centernet']
            for j in len(b_box_list):
                b_box = b_box_list[j]
                # {'bbox': [879.3, 470.78, 12.34, 25.95], 'class': 'human.pedestrian.adult', 'score': 0.32}
                det = {'det_id': det_id,
                       'h5_id': det_id,  # we make h5_id == det_id
                       'box': b_box['bbox'],
                       'image_id': image_id,
                       'category_id': 1, # we do not use category id
                       'category_name': b_box['class'],
                       'score': b_box['score']}
                dets += [det]
                det_id += 1
            image_id += 1

    assert image_id == total_num_images

    out_json_path = "./talk2car_dets_centernet.json"
    with open(out_json_path, 'w') as f:
        json.dump(dets, f)


if __name__ == "__main__":
    talk2car_region_path = "data/talk2car/talk2car_w_rpn_no_duplicates.json"
    main(talk2car_region_path)
