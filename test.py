from model.utils import get_model
import os
import itk
import argparse
import yaml
from monai.inferers import sliding_window_inference
from monailabel.transform.post import LargestCCd
import numpy as np
import scipy.ndimage as ndimage
import torch


def resample_3d(img, target_size):
    imx, imy, imz = img.shape
    tx, ty, tz = target_size
    zoom_ratio = (float(tx) / float(imx), float(ty) / float(imy), float(tz) / float(imz))
    img_resampled = ndimage.zoom(img, zoom_ratio, order=0, prefilter=False)
    return img_resampled

def pathExist(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Conv-Trans Segmentation')
    parser.add_argument('--dataset', type=str, default='ixi', help='dataset name')
    parser.add_argument('--model', type=str, default='resunet', help='model name')
    parser.add_argument('--dimension', type=str, default='3d', help='2d model or 3d model')
    parser.add_argument('--pretrain', action='store_true', help='if use pretrained weight for init')

    parser.add_argument('--batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--load', type=str, default=False, help='load pretrained model'
    parser.add_argument('--unique_name', type=str, default='test', help='unique experiment name')

    parser.add_argument('--gpu', type=str, default='0')

    parser.add_argument('--sliding_window', type=str, default=True, help='sliding_window or not')

    parser.add_argument("--roi_x", default=128, type=int, help="roi size in x direction")
    parser.add_argument("--roi_y", default=128, type=int, help="roi size in y direction")
    parser.add_argument("--roi_z", default=128, type=int, help="roi size in z direction")
    parser.add_argument("--infer_overlap", default=0.5, type=float, help="sliding window inference overlap")

    args = parser.parse_args()

    config_path = 'config/%s/%s_%s.yaml' % (args.dataset, args.model, args.dimension)
    if not os.path.exists(config_path):
        raise ValueError("The specified configuration doesn't exist: %s" % config_path)

    print('Loading configurations from %s' % config_path)

    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    for key, value in config.items():
        setattr(args, key, value)

    return args

def get_stride(image_width, kernel_size):
    '''
    return proper stride that can slide all images with min number of steps (min number of patches)
           by the given image and kernel sizes
    '''
    n = image_width // kernel_size + 1
    stride = (image_width - kernel_size) // (n - 1)
    return stride


if __name__ == '__main__':

    args = get_parser()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    # torch.cuda.set_device(args.gpu)  # assign which gpu will be used (only linux works)

    model_path =
    model_name = 'ixi_resunet_best.pth'

    image_path =

    sample_list = []
    for idx,case in enumerate(os.listdir(image_path)):
        sample_list.append(case)

    # sample_image_filename = '{}.nii.gz'
    output_path =
    pathExist(output_path)

    num_classes = 2
    num_channels = 1

    # set model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(args, pretrain=args.pretrain)

    # load trained model
    checkpoint = torch.load(os.path.join(model_path, model_name), map_location='cpu')
    # print(checkpoint)
    # model.load_state_dict(checkpoint['model_state_dict'])
    model.load_state_dict(checkpoint)
    del checkpoint
    model = model.to(device, dtype=torch.float)

    # cudnn
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    # eval model first to check gpu memory
    print('Predicting')
    model.eval()
    with torch.no_grad():
        for idx,i_sample in enumerate(sample_list):
            if idx >= 0:
                print(idx,i_sample)

                print('Predicting Sample filename: {}'.format(i_sample))
                # read image and label (annotation)
                itk_image = itk.imread(os.path.join(image_path, i_sample))
                np_image = itk.array_from_image(itk_image)
                predict_label = np.zeros(np_image.shape)

                # normalized
                max99 = np.percentile(np_image, 99)
                np_image = np.clip(np_image, 0, max99)
                np_image = np_image / max99

                z, y, x = np_image.shape
                print(np_image.shape)

                val_inputs = torch.from_numpy(np_image).unsqueeze(0).unsqueeze(0).cuda().float()

                _, _, h, w, d = val_inputs.shape
                # target_shape = (h, w, d)

                val_outputs = sliding_window_inference(
                    val_inputs, (args.roi_x, args.roi_y, args.roi_z), 1, model, overlap=args.infer_overlap,
                    mode="gaussian"
                )
                val_outputs = torch.softmax(val_outputs, 1).cpu().numpy()
                val_outputs = np.argmax(val_outputs, axis=1).astype(np.uint8)[0]
                # val_outputs = resample_3d(val_outputs, target_shape)

                print(val_inputs.shape)
                print(val_outputs.shape)

                # val_outputs = LargestCCd.get_largest_cc(val_outputs)

                # # output result
                itk_predict_label = itk.image_view_from_array(val_outputs)
                itk.imwrite(itk_predict_label, os.path.join(output_path, i_sample))