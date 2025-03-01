import cv2
import numpy as np 


class FlexibleSplit:

    def __init__(self, apply_magnet, pad, img_path):
        self.apply_magnet = apply_magnet
        self.pad = pad
        self.point_arr = []
        self.make_drawing = False
        self.current_num_points = 0
        self.sub_img_count = 0
        self.img_path = img_path
        self.img, self.dim0, self.dim1 = self.init_img(img_path)
        self.edge_mask = self.get_edge_mask()
        self.display_img = self.init_display_img()


    def run_splitter(self):

        cv2.namedWindow("PAINT", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('PAINT', self.draw_point)

        while(1):

            cv2.imshow('PAINT', self.display_img)
            k = cv2.waitKey(1)

            self.update_point_array()

            if k == 27: # Esc
                break

            if k == 13: # Enter
                if len(self.point_arr) > 2:
                    self.make_drawing = True

            if self.make_drawing:
                self.save_sub_img()

        cv2.destroyAllWindows()


    def draw_point(self, event, former_x, former_y, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
            self.point_arr.append((former_x, former_y))
            if len(self.point_arr) == 1:
                cv2.line(self.display_img, (former_x, former_y), (former_x, former_y), (0, 0, 255), 1)
            else:
                x0, y0 = self.point_arr[-2]
                x1, y1 = self.point_arr[-1]
                cv2.line(self.display_img, (x0, y0), (x1, y1), (0, 0, 255), 1)


    def update_point_array(self):
        if len(self.point_arr) > self.current_num_points:
            self.current_num_points = len(self.point_arr)
            print(self.point_arr)
        return self.current_num_points


    def save_sub_img(self):

        if self.apply_magnet:
            self.point_arr = self.magnetize_point_array(factor=16)

        boundary_img = self.get_boundary_img_from_point_arr()
        boundary_img = np.min(boundary_img, axis=2)
        boundary_img = boundary_img / np.max(boundary_img)
        boundary_img = 255 * boundary_img

        retval, labels_im, stats, centroids = cv2.connectedComponentsWithStats(boundary_img.astype(np.uint8))
        labels, counts = np.unique(labels_im, return_counts=True)
        object_idxs = np.where(labels_im != labels[1])

        sub_img = np.full(self.img.shape, 255)
        sub_img[object_idxs] = self.img[object_idxs]
        sub_img = sub_img[self.pad:self.dim0 + self.pad, self.pad:self.dim1 + self.pad]

        new_path = self.img_path.replace(".png", f"{self.sub_img_count:03d}.png")
        cv2.imwrite(new_path, sub_img.astype(np.uint8))

        # show display
        if self.apply_magnet:

            self.display_img = self.init_display_img()
            num_items = len(self.point_arr)

            for i in range(num_items):
                x0, y0 = self.point_arr[i - 1]
                x1, y1 = self.point_arr[i]
                cv2.line(self.display_img, (x0, y0), (x1, y1), (0, 0, 255), 1)

        x0, y0 = self.point_arr[-2]
        x1, y1 = self.point_arr[-1]
        cv2.line(self.display_img, (x0, y0), (x1, y1), (0, 0, 255), 1)

        self.make_drawing = False
        self.sub_img_count += 1
        self.point_arr = []


    def get_boundary_img_from_point_arr(self):

        self.point_arr.append(self.point_arr[0])
        boundary_img = np.full(self.img.shape, 255)

        for i in range(1, len(self.point_arr)):
            x0, y0 = self.point_arr[i - 1]
            x1, y1 = self.point_arr[i]
            cv2.line(boundary_img, (x0, y0), (x1, y1), (0, 0, 0), 2)

        return boundary_img


    def get_edge_mask(self):

        edges = []
        for i in range(3):
            img = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV).astype(np.float32)
            img = img[:,:,i]
            img = cv2.GaussianBlur(img, (5, 5), 5)
            img = 255 * img / np.max(img)
            edg_img = cv2.Canny(img.astype(np.uint8), 50, 200)
            edges.append(edg_img)

        edge_mask = np.max(edges, axis=0)
        edge_mask = edge_mask / np.max(edge_mask)
        edge_mask[edge_mask >= 0.1] = 1.0
        edge_mask[edge_mask < 0.1] = 0.0

        edge_mask = cv2.dilate(edge_mask, np.ones((3,3),np.uint8), iterations=2)
        edge_mask = cv2.erode(edge_mask, np.ones((3,3),np.uint8), iterations=1)

        return edge_mask


    def init_display_img(self):

        if not self.apply_magnet:
            return np.copy(self.img)

        mask_img = np.dstack([self.edge_mask, self.edge_mask, self.edge_mask])
        display_img = self.img.astype(np.float32) * mask_img

        return display_img.astype(np.uint8)

    def init_img(self, img_path):
        img = cv2.imread(img_path)
        dim0, dim1 = img.shape[0], img.shape[1]
        padded_img = np.full((dim0 + 2 * self.pad, dim1 + 2 * self.pad, 3), 255)
        padded_img[self.pad:dim0 + self.pad, self.pad:dim1 + self.pad] = img
        padded_img = padded_img.astype(np.uint8)
        return padded_img, dim0, dim1


    def magnetize_point_array(self, factor=4):

        def get_centers_of_mass(point_arr):
            centers_of_mass = []
            number_of_points = len(point_arr)
            window_size = number_of_points // 2
            for i in range(window_size, number_of_points, 4):
                c_o_m = np.mean(point_arr[i - window_size:i], axis=0)
                centers_of_mass.append(c_o_m)
            centers_of_mass = np.array(centers_of_mass)
            return centers_of_mass

        def expand_point_arr(point_arr, factor):
            exp_point_arr = []
            for i in range(len(point_arr)):
                x0, y0 = point_arr[i - 1]
                x1, y1 = point_arr[i]
                for j in range(factor):
                    new_x = int(x0 + (j / factor) * (x1 - x0))
                    new_y = int(y0 + (j / factor) * (y1 - y0))
                    exp_point_arr.append((new_x, new_y))
            return exp_point_arr

        exp_point_arr = expand_point_arr(self.point_arr, factor)
        exp_point_arr = np.asarray(exp_point_arr)
        centers_of_mass = get_centers_of_mass(exp_point_arr)

        # move towards center
        num_items = len(exp_point_arr)

        for epoch in range(1000):

            update_count = 0
            for i in range(num_items):

                edg_point = exp_point_arr[i]
                idxs = (int(edg_point[1]), int(edg_point[0]))

                if self.edge_mask[idxs] == 0.0:

                    # get the closest centerof mass
                    distances = centers_of_mass - edg_point
                    distances = np.sqrt(distances[:, 0] ** 2.0 + distances[:, 1] ** 2.0)
                    idx = np.argmin(distances)

                    movement = centers_of_mass[idx] - edg_point
                    if np.max(np.abs(movement)) != 0.0:
                        movement = movement / np.max(np.abs(movement))

                    new_edg_point = edg_point + movement
                    idxs = (int(new_edg_point[1]), int(new_edg_point[0]))

                    if self.edge_mask[idxs] == 0.0:
                        exp_point_arr[i] = new_edg_point
                        update_count += 1

            update_ratio = update_count / num_items
            if update_ratio == 0.0:
                break

        exp_point_arr = [(int(pnt[0]), int(pnt[1])) for pnt in exp_point_arr]

        return exp_point_arr


if __name__ == "__main__":
    rctng = FlexibleSplit(
        apply_magnet=True,
        pad=20,
        img_path=""
    )
    rctng.run_splitter()
