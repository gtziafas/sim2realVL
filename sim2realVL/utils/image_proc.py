from ..types import * 

import cv2 
import numpy as np 

MORPHOLOGY_MAP = {
    'erode': cv2.MORPH_ERODE,
    'dilate': cv2.MORPH_DILATE,
    'open': cv2.MORPH_OPEN,
    'close': cv2.MORPH_CLOSE
}


def show(img: array, legend: Maybe[str] = None):
    legend = 'unlabeled' if legend is None else legend
    cv2.imshow(legend, img)
    while 1:
        key = cv2.waitKey(1) & 0xff
        if key == ord('q'):
            break
    cv2.destroyWindow(legend)


def show_many(imgs: List[array], legends: Maybe[List[str]] = None):
    assert len(imgs) == len(legends)
    legends = [l if l is not None else 'image' + str(i) for i, l in enumerate(legends)]
    print([f'{l}: {i.shape}'for l, i in zip(legends, imgs)])
    for i, l in zip(imgs, legends):
        cv2.imshow(l, i)
    while 1:
        key = cv2.waitKey(1) & 0xff
        if key == ord('q'):
            break
    destroy()


def destroy():
    cv2.destroyAllWindows()


def box_to_cv2(box: Tuple[int, ...]) -> Box:
    # convert np-style to cv2-style coordinates for box
    y1, y2, x1, x2 = box
    return Box(x = x1, y = y1, w = x2 - x1, h = y2 - y1)

def crop_box(img: array, box: Box):
    return img[box.y : box.y + box.h, box.x : box.x + box.w]

