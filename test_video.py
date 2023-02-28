import os
import numpy as np
import cv2
import time
from rknn.api import RKNN
import sys
import argparse

from tracker.byte_tracker import BYTETracker
from visualize import plot_tracking

from collections import deque
import math


PT_MODEL = 'yolov5s.torchscript'
RKNN_MODEL = 'yolov5s.rknn'
# IMG_PATH = 'test_video.mp4'
DATASET = './dataset.txt'

QUANTIZE_ON = False

BOX_THRESH = 0.5
NMS_THRESH = 0.6
IMG_SIZE = 640

CLASSES = ("person", "bicycle", "car","motorbike ","aeroplane ","bus ","train","truck ","boat","traffic light",
           "fire hydrant","stop sign ","parking meter","bench","bird","cat","dog ","horse ","sheep","cow","elephant",
           "bear","zebra ","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite",
           "baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife ",
           "spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza ","donut","cake","chair","sofa",
           "pottedplant","bed","diningtable","toilet ","tvmonitor","laptop	","mouse	","remote ","keyboard ","cell phone","microwave ",
           "oven ","toaster","sink","refrigerator ","book","clock","vase","scissors ","teddy bear ","hair drier", "toothbrush ")
print("CLASSES:", len(CLASSES))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def xywh2xyxy(x):
    # Convert [x, y, w, h] to [x1, y1, x2, y2]
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def process(input, mask, anchors):

    anchors = [anchors[i] for i in mask]
    grid_h, grid_w = map(int, input.shape[0:2])

    box_confidence = sigmoid(input[..., 4])
    box_confidence = np.expand_dims(box_confidence, axis=-1)

    box_class_probs = sigmoid(input[..., 5:])

    box_xy = sigmoid(input[..., :2])*2 - 0.5

    col = np.tile(np.arange(0, grid_w), grid_w).reshape(-1, grid_w)
    row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_h)
    col = col.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    row = row.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    grid = np.concatenate((col, row), axis=-1)
    box_xy += grid
    box_xy *= int(IMG_SIZE/grid_h)

    box_wh = pow(sigmoid(input[..., 2:4])*2, 2)
    box_wh = box_wh * anchors

    box = np.concatenate((box_xy, box_wh), axis=-1)

    return box, box_confidence, box_class_probs

def filter_boxes(boxes, box_confidences, box_class_probs):
    """Filter boxes with box threshold. It's a bit different with origin yolov5 post process!

    # Arguments
        boxes: ndarray, boxes of objects.
        box_confidences: ndarray, confidences of objects.
        box_class_probs: ndarray, class_probs of objects.

    # Returns
        boxes: ndarray, filtered boxes.
        classes: ndarray, classes for boxes.
        scores: ndarray, scores for boxes.
    """
    box_classes = np.argmax(box_class_probs, axis=-1)
    box_class_scores = np.max(box_class_probs, axis=-1)
    pos = np.where(box_confidences[...,0] >= BOX_THRESH)


    boxes = boxes[pos]
    classes = box_classes[pos]
    scores = box_class_scores[pos]

    return boxes, classes, scores

def nms_boxes(boxes, scores):
    """Suppress non-maximal boxes.

    # Arguments
        boxes: ndarray, boxes of objects.
        scores: ndarray, scores of objects.

    # Returns
        keep: ndarray, index of effective boxes.
    """
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    keep = np.array(keep)
    return keep


def yolov5_post_process(input_data):
    masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    anchors = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
              [59, 119], [116, 90], [156, 198], [373, 326]]

    boxes, classes, scores = [], [], []
    for input,mask in zip(input_data, masks):
        b, c, s = process(input, mask, anchors)
        b, c, s = filter_boxes(b, c, s)
        boxes.append(b)
        classes.append(c)
        scores.append(s)

    boxes = np.concatenate(boxes)
    boxes = xywh2xyxy(boxes)
    classes = np.concatenate(classes)
    scores = np.concatenate(scores)

    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]

        keep = nms_boxes(b, s)

        nboxes.append(b[keep])
        nclasses.append(c[keep])
        nscores.append(s[keep])

    if not nclasses and not nscores:
        return None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)

    return boxes, classes, scores

def draw(image, boxes, scores, classes):
    """Draw the boxes on the image.

    # Argument:
        image: original image.
        boxes: ndarray, boxes of objects.
        classes: ndarray, classes of objects.
        scores: ndarray, scores of objects.
        all_classes: all classes name.
    """
    for box, score, cl in zip(boxes, scores, classes):
        top, left, right, bottom = box
        # print('class: {}, score: {}'.format(CLASSES[cl], score))
        # print('box coordinate left,top,right,down: [{}, {}, {}, {}]'.format(top, left, right, bottom))
        top = int(top)
        left = int(left)
        right = int(right)
        bottom = int(bottom)

        if(CLASSES[cl] == "person"):
            cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
            cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
                    (top, left - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2)


def letterbox(im, new_shape=(640, 640), color=(0, 0, 0)):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


def filter_person(boxes, classes, scores):

    pos = np.where(classes[...] == 0)

    boxes = boxes[pos]
    scores = scores[pos]




    



    # filtered_boxes = []
    # filtered_scores = []
    # for box, score, cl in zip(boxes, scores, classes):
    #     if CLASSES[cl] == "person":
    #         print(box)
    #         print(type(box))
    #         filtered_boxes.append(box)
    #         filtered_scores.append(score)
    
    # sys.pause()

    # if len(filtered_scores) > 0:
    #     print(filtered_boxes)
    #     print(filtered_scores)
    #     return np.concatenate(filtered_boxes), np.concatenate(filtered_scores)
    
    return boxes, scores

def make_parser():
    parser = argparse.ArgumentParser("ByteTrack Demo!")

    parser.add_argument("--input", default=None, type=str, help="test input")


    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--fps", default=30, type=int, help="frame rate (fps)")



    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument(
        "--aspect_ratio_thresh", type=float, default=1.6,
        help="threshold for filtering out boxes of which aspect ratio are above the given value."
    )
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")


    return parser

def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

def intersect(A, B, C, D):
        return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

def tlwh_to_midpoint(point):
    x, y, w, h = point
    return (int(x + (w/2)), int(y + (h/2)))

def vector_angle(midpoint, previous_midpoint):
        x = midpoint[0] - previous_midpoint[0]
        y = midpoint[1] - previous_midpoint[1]
        return math.degrees(math.atan2(y, x))


if __name__ == '__main__':
    args = make_parser().parse_args()

    # Create RKNN object
    rknn = RKNN(verbose=False)

    ret = rknn.load_rknn(RKNN_MODEL)

    ret = rknn.init_runtime()

    IMG_PATH = args.input

    video_input = cv2.VideoCapture(IMG_PATH)
    fps = video_input.get(cv2.CAP_PROP_FPS)
    print("FPS:", fps)
    video_size = (int(video_input.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_input.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print("VIDEO_SIZE:", video_size)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_output = cv2.VideoWriter(IMG_PATH.split(".")[0] + "_output.avi", fourcc, int(fps), (IMG_SIZE, IMG_SIZE))

    ret, frame = video_input.read()
    tracker = BYTETracker(args, frame_rate=fps)  #追踪
    frame_id = 0

    memory = {}
    up_count = 0
    down_count = 0
    already_counted = deque(maxlen=50)

    while ret:
        time_start = time.perf_counter()
        frame, ratio, (dw, dh) = letterbox(frame, new_shape=(IMG_SIZE, IMG_SIZE))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        outputs = rknn.inference(inputs=[frame]) #yolov5 object detection
        
        # post process
        input0_data = outputs[0]
        input1_data = outputs[1]
        input2_data = outputs[2]

        input0_data = input0_data.reshape([3,-1]+list(input0_data.shape[-2:]))
        input1_data = input1_data.reshape([3,-1]+list(input1_data.shape[-2:]))
        input2_data = input2_data.reshape([3,-1]+list(input2_data.shape[-2:]))

        input_data = list()
        input_data.append(np.transpose(input0_data, (2, 3, 0, 1)))
        input_data.append(np.transpose(input1_data, (2, 3, 0, 1)))
        input_data.append(np.transpose(input2_data, (2, 3, 0, 1)))

        boxes, classes, scores = yolov5_post_process(input_data)    #如果没有检测出物体，返回None
            # print(boxes.shape)  #(个数，4)  （x1, y1, x2, y2）
            # print(classes.shape)    #（个数，）
            # print(scores.shape)     #（个数，）

        boxes, scores = filter_person(boxes, classes, scores)

        frame_output = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        line = [(0, int(0.5 * frame_output.shape[0])), (int(frame_output.shape[1]), int(0.5 * frame_output.shape[0]))]

        cv2.line(frame_output, line[0], line[1], (0, 255, 255), 2)

        if scores is not None:
                online_targets = tracker.update(boxes, scores, [IMG_SIZE, IMG_SIZE], (IMG_SIZE, IMG_SIZE)) #跟踪信息更新
                online_tlwhs = []   #(x1, y1, w, h)
                online_ids = []
                online_scores = []
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                    if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                        # print(
                        #     f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f}"
                        # )
                        # sys.stdout.flush()
                    
                        midpoint = tlwh_to_midpoint(tlwh)
                        # origin_midpoint = (midpoint[0], frame_output.shape[0] - midpoint[1])  # get midpoint respective to botton-left

                        if tid not in memory:
                            memory[tid] = deque(maxlen=2)
                            memory[tid].append(midpoint)
                            memory[tid].append(midpoint)

                        
                        previous_midpoint = (memory[tid])[0]
                        # origin_previous_midpoint = (previous_midpoint[0], frame.shape[0] - previous_midpoint[1])

                        (memory[tid])[1] = midpoint

                        # cv2.line(frame_output, midpoint, previous_midpoint, (255, 255, 0), 2)  #后期可删

                        if intersect(midpoint, previous_midpoint, line[0], line[1]) and tid not in already_counted:
                            already_counted.append(tid)
                            angle = vector_angle(midpoint, previous_midpoint)

                            if angle > 0:
                                down_count += 1

                            elif angle < 0:
                                up_count += 1
                
                frame_output = plot_tracking(frame_output, online_tlwhs, online_ids, frame_id=frame_id + 1, up_count=up_count, down_count=down_count)
        
        # if boxes is not None:
        #     draw(frame_output, boxes, scores, classes)
        time_end = time.perf_counter()
        print("Time Cost: %s ms" % ((time_end-time_start) * 1000) )
        sys.stdout.flush()


        video_output.write(frame_output)
    
        ret, frame = video_input.read()
    



    rknn.release()
