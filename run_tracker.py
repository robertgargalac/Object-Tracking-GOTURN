import cv2
import sys
import threading
import tkinter as tk
from tkinter import filedialog
from multiprocessing.dummy import Pool as ThreadsPool

from bounding_box import BoundingBox
from gt_session import GtSession
from frame_manager import FrameManager

from utils import run_tracker, draw_bboxes

class ObjectTracking(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        tk.Tk.wm_title(self, "Multi Object Tracking")
        self.geometry('600x600')
        init_sess_thread = threading.Thread(target=self.init_sess)
        init_sess_thread.daemon = True
        init_sess_thread.start()
        container = tk.Frame(self)
        container.pack(fill='both', expand=True)

        path_label = tk.Label(container, text='File Path', font=('Verdana', 12))
        path_label.grid(row=0, column=0, sticky='w')

        path_value = tk.StringVar()
        file_path = tk.Entry(container, textvariable=path_value, width=40)
        file_path.grid(row=0, column=1)

        file_button = tk.Button(container, text='Choose File', font=('Verdana', 12), command=self.open_file)
        file_button.grid(row=1, column=0)

        run_button = tk.Button(container, text='RUN TRACKER', font=('Verdana', 12), command=self.run)
        run_button.grid(row=1, column=1)

    def open_file(self):
        self.chosen_path = filedialog.askopenfilename(
            initialdir="/",
            title="Select file",
            filetypes=(("mp4 files", "*.mp4"),
                       ("all files", "*.*"))
        )


    def init_sess(self):
        self.gt_session = GtSession(1)

    def run(self):
        video = cv2.VideoCapture(self.chosen_path)
        if not video.isOpened():
            print('Could not open video')
            sys.exit()

        ok, frame = video.read()
        if not ok:
            print('Cannot read video file')

        bboxes = cv2.selectROIs('MULTI TRACKING VERSION', frame)
        if bboxes:
            cv2.destroyAllWindows()
        # bboxes = [[1006, 184, 79, 81], [513, 309, 114, 150]]
        init_rois = [
            BoundingBox(bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3])
            for bbox in bboxes
        ]
        frame_managers = [
            FrameManager(frame, init_roi)
            for init_roi in init_rois
        ]

        while True:
            ok, frame = video.read()
            if not ok:
                break
            data = [
                (frame_manager, self.gt_session, frame)
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
app = ObjectTracking()
app.mainloop()
