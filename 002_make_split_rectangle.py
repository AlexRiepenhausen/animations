import cv2
import numpy as np 


class RectangleSplit:

    def __init__(self, img_path):
        self.point_arr = []
        self.rectangle_drawn = True
        self.current_num_points = 0
        self.sub_img_count = 0
        self.img_path = img_path
        self.img = cv2.imread(img_path)
        self.img_original = np.copy(self.img)


    def run_splitter(self):

        cv2.namedWindow("PAINT", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('PAINT', self.rectangle_draw)

        # save blank image first
        new_path = self.img_path.replace(".png", f"{self.sub_img_count:03d}.png")
        cv2.imwrite(new_path, np.full(self.img.shape, 255).astype(np.uint8))
        self.sub_img_count += 1

        while(1):

            cv2.imshow('PAINT', self.img)
            k = cv2.waitKey(1)

            if k == 27:
                break

            self.update_point_array()
            self.save_sub_img()

        cv2.destroyAllWindows()


    def rectangle_draw(self, event, former_x, former_y, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
            self.point_arr.append((former_x, former_y))
            cv2.line(self.img, (former_x, former_y), (former_x, former_y), (0, 0, 255), 2)


    def save_sub_img(self):

        if not self.rectangle_drawn:

            x0, y0 = self.point_arr[0]
            x1, y1 = self.point_arr[1]

            if y0 > y1:
                tmp = y1
                y1 = y0
                y0 = tmp

            if x0 > x1:
                tmp = x1
                x1 = x0
                x0 = tmp

            sub_img = np.full(self.img.shape, 255)
            sub_img[y0:y1, x0:x1] = self.img_original[y0:y1, x0:x1]

            new_path = self.img_path.replace(".png", f"{self.sub_img_count:03d}.png")
            cv2.imwrite(new_path, sub_img.astype(np.uint8))

            cv2.line(self.img, (x0, y0),(x0, y1), (0, 0, 255), 1)
            cv2.line(self.img, (x1, y0),(x1, y1), (0, 0, 255), 1)
            cv2.line(self.img, (x0, y0),(x1, y0), (0, 0, 255), 1)
            cv2.line(self.img, (x0, y1),(x1, y1), (0, 0, 255), 1)

            self.rectangle_drawn = True
            self.sub_img_count += 1


    def update_point_array(self):
        if len(self.point_arr) > self.current_num_points:
            self.current_num_points = len(self.point_arr)
            if self.current_num_points == 2:
                self.rectangle_drawn = False
            if self.current_num_points == 3:
                self.point_arr = [self.point_arr[2]]
                self.current_num_points = 1
        return self.current_num_points


if __name__ == "__main__":

    rctng = RectangleSplit(
        img_path=""
    )

    rctng.run_splitter()

