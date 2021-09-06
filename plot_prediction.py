import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
import argparse


def main(bounding_box_json_path, image_dir, result_dir):
    with open(bounding_box_json_path, encoding='utf-8') as f:
        data_gt = json.load(f)
        # ------------ Plot validation result image with pred_bounding box and gt_bounding box --------------
        #fig, ax = plt.subplots(1)
        with open(os.path.join(result_dir, 'val_prediction.json'), encoding='utf-8') as f_val_pred:
            data_val_pred = json.load(f_val_pred)
            i = 8349
            cnt = 0
            os.makedirs(os.path.join(result_dir, './output_fig/val'))
            while i < 8349 + 1163:
                img_path = os.path.join(image_dir, data_gt['val'][str(cnt)]['img'])
                with open(img_path, 'rb') as f_img:
                    img = Image.open(f_img).convert('RGB')

                bbox = data_val_pred[cnt][str(i)]
                xl, yb, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
                fig, ax = plt.subplots(1)
                ax.imshow(img)
                rect = patches.Rectangle((xl, yb), w, h, fill=False, edgecolor='r')
                ax.add_patch(rect)
                gt_bbox = data_gt['val'][str(cnt)]['referred_object']
                xl, yb, w, h = gt_bbox[0], gt_bbox[1], gt_bbox[2], gt_bbox[3]
                rect = patches.Rectangle((xl, yb), w, h, fill=False, edgecolor='b')
                ax.add_patch(rect)
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(os.path.join(result_dir, './output_fig/val/{}_bboxes.png'.format(i)), bbox_inches='tight')
                plt.close()
                cnt += 1
                i += 1
            f_img.close()
        f_val_pred.close()

        # ------------Plot test result image with pred_bounding box --------------
        with open(os.path.join(result_dir, 'test_prediction.json')) as f_test_pred:
            data_test_pred = json.load(f_test_pred)
            i = 8349 + 1163
            cnt = 0
            os.makedirs(os.path.join(result_dir, './output_fig/test'))
            while i < 8349 + 1163 + 2447:
                img_path = os.path.join(image_dir, data_gt['test'][str(cnt)]['img'])
                with open(img_path, 'rb') as f_img:
                    img = Image.open(f_img).convert('RGB')

                bbox = data_test_pred[cnt][str(i)]
                xl, yb, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
                #fig, ax = plt.subplots(1)
                ax.imshow(img)
                rect = patches.Rectangle((xl, yb), w, h, fill=False, edgecolor='r')
                ax.add_patch(rect)
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(os.path.join(result_dir, './output_fig/test/{}_bboxes.png'.format(i)), bbox_inches='tight')
                plt.close()
                cnt += 1
                i += 1
            f_img.close()
        f_test_pred.close()

        # ------------Plot region image with bounding box --------------
        i = 8349
        image_id = 0
        os.makedirs(os.path.join(result_dir, './output_fig/val_centernet'))
        while i < 8349 + 1163:
            centernet_bbox = []
            centernet_score = []
            img_path = os.path.join(image_dir, data_gt['val'][str(image_id)]['img'])
            with open(img_path, 'rb') as f_img:
                img = Image.open(f_img).convert('RGB')
            centernet_list = data_gt['val'][str(image_id)]['centernet']
            for m in range(0, len(centernet_list)):
                centernet_bbox.append(centernet_list[m]['bbox'])
                centernet_score.append(centernet_list[m]['score'])
            sorted_centernet_dict = zip(centernet_bbox, centernet_score)
            sorted_centernet_dict = sorted(sorted_centernet_dict, key=lambda scores: scores[1], reverse=True)[:36]
            #fig, ax = plt.subplots(1)
            ax.imshow(img)
            for m in range(0, len(sorted_centernet_dict)):
                bbox = sorted_centernet_dict[m][0]
                xl, yb, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
                rect = patches.Rectangle((xl, yb), w, h, fill=False, edgecolor='r')
                ax.add_patch(rect)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(result_dir, './output_fig/val_centernet/{}_bboxes.png'.format(i)), bbox_inches='tight')
            plt.close()
            image_id += 1
            i += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='')
    # choose metric to evaluate
    parser.add_argument('--bounding_box_json_path', action='store',
                        default="./data/talk2car/talk2car_w_rpn_no_duplicates.json", help='')
    parser.add_argument('--image_dir', action='store',
                        default="./data/talk2car/images", help='')
    parser.add_argument('--result_dir', action='store', required=True, help='')
    args = parser.parse_args()
    main(args.bounding_box_json_path, args.image_dir, args.result_dir)
