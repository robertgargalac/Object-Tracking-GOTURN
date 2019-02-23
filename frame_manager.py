import cv2

from bounding_box import BoundingBox


class FrameManager:

    def __init__(self, frame, roi):
        self.roi = roi
        self.frame = frame
        self.predicted_bbox = None
        self.width, self.height, self.chanels = frame.shape

    def get_image(self):
        frame_h, frame_w, _ = self.frame.shape
        if self.predicted_bbox:
            x_center = self.predicted_bbox.get_x_center()
            y_center = self.predicted_bbox.get_y_center()
            image_h = self.predicted_bbox.compute_output_height()
            image_w = self.predicted_bbox.compute_output_width()
            y1_img = max(1, int(y_center - (image_h / 2)))
            y2_img = int(y_center + (image_h / 2))
            x1_img = max(1, int(x_center - (image_w / 2)))
            x2_img = int(x_center + (image_w / 2))
            if y2_img >= frame_h:
                y2_img = frame_h
            if x2_img >= frame_w:
                x2_img = frame_w
            self.image = self.frame[
                         y1_img: y2_img,
                         x1_img: x2_img
                         ]
            return cv2.resize(self.image, (227, 227))

        x_center = self.roi.get_x_center()
        y_center = self.roi.get_y_center()
        image_h = self.roi.compute_output_height()
        image_w = self.roi.compute_output_width()
        y1_img = max(1, int(y_center - (image_h / 2)))
        y2_img = int(y_center + (image_h / 2))
        x1_img = max(1, int(x_center - (image_w / 2)))
        x2_img = int(x_center + (image_w / 2))

        if y2_img >= frame_h:
            y2_img = frame_h
        if x2_img >= frame_w:
            x2_img = frame_w
        self.image = self.frame[
                     y1_img: y2_img,
                     x1_img: x2_img
        ]
        return cv2.resize(self.image, (227, 227))

    def update_frame(self, frame):
        self.frame = frame

    def update_bbox(self, bbox):
        img_h, img_w, _ = self.image.shape
        offset_x1 = (bbox[0] - 0.25) * img_w
        offset_x2 = (bbox[2] - 0.75) * img_w
        offset_y1 = (bbox[1] - 0.25) * img_h
        offset_y2 = (bbox[3] - 0.75) * img_h
        if not self.predicted_bbox:
            new_x1 = self.roi.x1 + offset_x1
            new_x2 = self.roi.x2 + offset_x2
            new_y1 = self.roi.y1 + offset_y1
            new_y2 = self.roi.y2 + offset_y2
            self.predicted_bbox = BoundingBox(new_x1, new_y1, new_x2, new_y2)
        else:
            new_x1 = self.predicted_bbox.x1 + offset_x1
            new_x2 = self.predicted_bbox.x2 + offset_x2
            new_y1 = self.predicted_bbox.y1 + offset_y1
            new_y2 = self.predicted_bbox.y2 + offset_y2
            self.predicted_bbox = BoundingBox(new_x1, new_y1, new_x2, new_y2)
        # print('x coord new:', new_x1, new_x2)
        # print('y coord new:', new_y1, new_y2)

    def draw_bbox(self):
        p1 = (int(self.predicted_bbox.x1), int(self.predicted_bbox.y1))
        p2 = (int(self.predicted_bbox.x2), int(self.predicted_bbox.y2))
        cv2.rectangle(self.frame, p1, p2, (255, 0, 0), 2, 1)

