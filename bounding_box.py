class BoundingBox:
    def __init__(self, x1=None, y1=None, x2=None, y2=None):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.k = 2 # scaling factor

    def get_x_center(self):
        return(self.x1 + self.x2) / 2

    def get_y_center(self):
        return(self.y1 + self.y2) / 2

    def compute_output_height(self):
        bbox_height = self.y2 - self.y1
        output_height = self.k * bbox_height
        return max(1.0, output_height)

    def compute_output_width(self):
        bbox_width = self.x2 - self.x1
        output_width = self.k * bbox_width
        return max(1.0, output_width)

    def update_coordinates(self, x1, x2, y1, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
