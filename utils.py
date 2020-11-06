import matplotlib
import matplotlib.cm
import numpy as np
import cv2
from PIL import Image
import torch
from tqdm import tqdm

def DepthNorm(depth, maxDepth=1000.0):
    return maxDepth / depth



def predict(model, images, minDepth=10, maxDepth=1000, batch_size=2):
    # Support multiple RGBs, one RGB image, even grayscale
    if len(images.shape) < 3: images = np.stack((images, images, images), axis=2)
    if len(images.shape) < 4: images = images.reshape((1, images.shape[0], images.shape[1], images.shape[2]))
    # Compute predictions
    images = np.transpose(images, [0,3,1,2])
    images = torch.tensor(images, dtype=torch.float32).cuda()
    #print(images.size())
    predictions = model(images)
    predictions = predictions.cpu().detach().numpy()
    predictions = np.transpose(predictions, [0,2,3,1])
    # Put in expected range
    return np.clip(DepthNorm(predictions, maxDepth=1000), minDepth, maxDepth) / maxDepth


def scale_up(scale, images):
    from skimage.transform import resize
    scaled = []

    for i in range(len(images)):
        img = images[i]
        output_shape = (scale * img.shape[0], scale * img.shape[1])
        scaled.append(resize(img, output_shape, order=1, preserve_range=True, mode='reflect', anti_aliasing=True))

    return np.stack(scaled)


def load_images(image_files):
    loaded_images = []
    for file in image_files:
        x = np.clip(np.asarray(Image.open(file), dtype=float) / 255, 0, 1)
        x = cv2.resize(x, (480, 640))
        loaded_images.append(x)
    print(loaded_images[0].shape)
    if len(loaded_images) >1:
        return np.stack(loaded_images, axis=0)
    else:
        return np.array(loaded_images[0])


def to_multichannel(i):
    if i.shape[2] == 3: return i
    i = i[:, :, 0]
    return np.stack((i, i, i), axis=2)



def load_test_data(test_data_zip_file='nyu_test.zip'):
    print('Loading test data...', end='')
    import numpy as np
    from data import extract_zip
    data = extract_zip(test_data_zip_file)
    from io import BytesIO
    rgb = np.load(BytesIO(data['eigen_test_rgb.npy']))
    depth = np.load(BytesIO(data['eigen_test_depth.npy']))
    crop = np.load(BytesIO(data['eigen_test_crop.npy']))
    print('Test data loaded.\n')
    return {'rgb': rgb, 'depth': depth, 'crop': crop}


def evaluate(model, rgb, depth, crop, batch_size=6, verbose=True):
    # Error computaiton based on https://github.com/tinghuiz/SfMLearner

    def compute_errors(gt, pred):
        thresh = np.maximum((gt / pred), (pred / gt))

        a1 = (thresh < 1.25).mean()
        a2 = (thresh < 1.25 ** 2).mean()
        a3 = (thresh < 1.25 ** 3).mean()

        abs_rel = np.mean(np.abs(gt - pred) / gt)

        rmse = (gt - pred) ** 2
        rmse = np.sqrt(rmse.mean())

        log_10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()

        return a1, a2, a3, abs_rel, rmse, log_10

    depth_scores = np.zeros((6, len(rgb)))  # six metrics

    bs = batch_size

    for i in tqdm(range(len(rgb) // bs)):
        x = rgb[(i) * bs:(i + 1) * bs, :, :, :]

        # Compute results
        true_y = depth[(i) * bs:(i + 1) * bs, :, :]
        pred_y = scale_up(2, predict(model, x / 255, minDepth=10, maxDepth=1000, batch_size=bs)[:, :, :, 0]) * 10.0

        # Test time augmentation: mirror image estimate
        pred_y_flip = scale_up(2,
                               predict(model, x[..., ::-1, :] / 255, minDepth=10, maxDepth=1000, batch_size=bs)[:, :, :,
                               0]) * 10.0

        # Crop based on Eigen et al. crop
        true_y = true_y[:, crop[0]:crop[1] + 1, crop[2]:crop[3] + 1]
        pred_y = pred_y[:, crop[0]:crop[1] + 1, crop[2]:crop[3] + 1]
        pred_y_flip = pred_y_flip[:, crop[0]:crop[1] + 1, crop[2]:crop[3] + 1]

        # Compute errors per image in batch
        for j in range(len(true_y)):
            errors = compute_errors(true_y[j], (0.5 * pred_y[j]) + (0.5 * np.fliplr(pred_y_flip[j])))

            for k in range(len(errors)):
                depth_scores[k][(i * bs) + j] = errors[k]

    e = depth_scores.mean(axis=1)

    if verbose:
        print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('a1', 'a2', 'a3', 'rel', 'rms', 'log_10'))
        print("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(e[0], e[1], e[2], e[3], e[4], e[5]))

    return e