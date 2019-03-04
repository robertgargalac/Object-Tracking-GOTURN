import cv2
import sys
from multiprocessing.dummy import Pool as ThreadsPool

from bounding_box import BoundingBox
from gt_session import GtSession
from frame_manager import FrameManager

from utils import run_tracker, draw_bboxes

if __name__ == '__main__':
    video = cv2.VideoCapture('test_data/train_video.avi')
    if not video.isOpened():
        print('Could not open video')
        sys.exit()

    ok, frame = video.read()
    if not ok:
        print('Cannot read video file')

    bboxes = cv2.selectROIs('MULTI TRACKING VERSION', frame)
    # bboxes = [[1006, 184, 79, 81], [513, 309, 114, 150]]
    init_rois = [
        BoundingBox(bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3])
        for bbox in bboxes
    ]
    frame_managers = [
        FrameManager(frame, init_roi)
        for init_roi in init_rois
    ]
    # pool = ThreadsPool(len(bboxes))
    # sessions = pool.map(lambda x: GtSession(x), range(len(bboxes)))
    # time.sleep(1)

    gt_sess = GtSession(0)

    while True:
        ok, frame = video.read()
        if not ok:
            break
        data = [
            (frame_manager, gt_sess, frame)
            for frame_manager in frame_managers
        ]
        timer = cv2.getTickCount()
        pool = ThreadsPool(len(data))
        predicted_bboxes = pool.map(run_tracker, data)

        frame_with_pred = draw_bboxes(frame, predicted_bboxes)

        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        cv2.putText(frame_with_pred, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (50, 170, 50), 2)
        cv2.imshow('GOTURN TRACKER', frame_with_pred)
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break

