import numpy as np
from skimage.measure import label, regionprops, find_contours
from tensorflow.keras.utils import normalize


def inference(model,image):
    single_patch = np.expand_dims(image,axis=-1)
    single_patch_norm = normalize(np.array(single_patch), axis=1)
    single_patch_input=np.expand_dims(single_patch_norm, 0)
        
    single_patch_prediction_s1 = (model.predict(single_patch_input))
    single_patch_prediction_s1_img = np.argmax(single_patch_prediction_s1,axis=-1)[0,:,:]
    return single_patch_prediction_s1_img
def get_binary_mask(multi_mask,class_value):
    return np.where(multi_mask==class_value,255,0).astype(np.uint8)
def mask_to_border(mask):
    h, w = mask.shape
    border = np.zeros((h, w))
    contours = find_contours(mask, 128)
    for contour in contours:
        for c in contour:
            x = int(c[0])
            y = int(c[1])
            border[x][y] = 255
    return border
def mask_to_bbox(mask):
    bboxes = []
    mask = mask_to_border(mask)
    lbl = label(mask)
    props = regionprops(lbl)
    for prop in props:
        x1 = prop.bbox[1]
        y1 = prop.bbox[0]
        x2 = prop.bbox[3]
        y2 = prop.bbox[2]
        bboxes.append([x1, y1, x2, y2])
    return bboxes