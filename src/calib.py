import numpy as np
import cv2 as cv
import glob
import os
import json
import sys


class Chessboard():
    def __init__(self, ):
        self.CHECKER_BOARD_WIDTH = 9
        self.CHECKER_BOARD_HEIGHT = 6
        self.SQUARE_SIZE = 3 * 0.01
        # 0.03 m
        self.criteria = (cv.TERM_CRITERIA_EPS
                         + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.check_pt = np.zeros(
            (self.CHECKER_BOARD_WIDTH*self.CHECKER_BOARD_HEIGHT, 3), np.float32)
        self.check_pt[:, :2] = np.mgrid[0:self.CHECKER_BOARD_WIDTH,
                                        0:self.CHECKER_BOARD_HEIGHT].T.reshape(-1, 2)*self.SQUARE_SIZE
        jpg_path = glob.glob(os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "..", "imgs", "*.jpg"))
        JPG_path = glob.glob(os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "..", "imgs", "*.JPG"))
        png_path = glob.glob(os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "..", "imgs", "*.png"))
        PNG_path = glob.glob(os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "..", "imgs", "*.PNG"))

        self.images_path = jpg_path + JPG_path + png_path + PNG_path

        print(f"Image count: {len(self.images_path)}")

    def calibration(self, vis: bool = False, save_image: bool = True, alpha: float = 0.95) -> None:
        obj_pts = []
        img_pts = []

        for img_path in self.images_path:
            img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
            ret, corners = cv.findChessboardCorners(
                img, (self.CHECKER_BOARD_WIDTH, self.CHECKER_BOARD_HEIGHT))
            corners = cv.cornerSubPix(
                    img, corners, (11, 11), (-1, -1), self.criteria)
            if ret:
                obj_pts.append(self.check_pt)
                img_pts.append(corners)
            if vis or save_image:
                img_result = cv.imread(img_path, cv.IMREAD_ANYCOLOR)
                img_result = cv.drawChessboardCorners(
                    img_result, (self.CHECKER_BOARD_WIDTH, self.CHECKER_BOARD_HEIGHT), corners, ret)
            if vis:
                cv.imshow(img_path.split()[-1], img_result)
                cv.waitKey(500)
                cv.destroyAllWindows()
            if save_image:
                cv.imwrite(img_path.replace("imgs", "results"), img_result)

        ret, self.mtx, self.dist, rvecs, tvecs = cv.calibrateCamera(
            obj_pts, img_pts, img.shape[::-1], None, None)
        h_, w_ = img.shape[:2]
        self.new_mtx, self.roi = cv.getOptimalNewCameraMatrix(
            self.mtx, self.dist, (w_, h_), alpha, (w_, h_))
        self.mapx, self.mapy = cv.initUndistortRectifyMap(
            self.mtx, self.dist, None, self.new_mtx, (w_, h_), 5)

        print("=" * 15 + "camera matrix" + "=" * 15)
        for r_ in self.mtx:
            for c_ in r_:
                print(f"{c_:7.2f}", end=" ")
            print()
        print()
        print("=" * 15 + "distortion coefficient" + "=" * 15)
        print(
            f"radial distortion: k1: {self.dist[0][0]:.3f}, k2: {self.dist[0][1]:.3f}, k3: {self.dist[0][4]:.3f}")
        print(
            f"tangential distortion: p1: {self.dist[0][2]:.6f}, p2: {self.dist[0][3]:.6f}")
        return None

    def undistorting(self, vis=False, save_image=True) -> None:
        for img_path in self.images_path:
            img = cv.imread(img_path, cv.IMREAD_ANYCOLOR)
            undist = cv.remap(img, self.mapx, self.mapy,
                              cv.INTER_LINEAR)

            if vis:
                cv.imshow("undist", undist)
                if cv.waitKey(2000) == 27:
                    cv.destroyAllWindows()
                    break
                cv.destroyAllWindows()
            if save_image:
                cv.imwrite(img_path.replace(
                    "imgs", "results").replace(".jpg", "_undist.jpg"), undist)
        return None

    def save_json(self, ):
        dict = {}
        dict["camera matrix"] = (self.mtx).tolist()
        # distortion k1, k2, p1, p2, k3
        dict["distortion"] = (self.dist[0]).tolist()
        result_dir = os.path.join(os.path.dirname(
            os.path.realpath(__file__)), "..", "results")

        if not os.path.exists(result_dir):
            os.path.mkdir(result_dir)

        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "results", "calib.json"), 'w') as f:
            json.dump(dict, f, indent=4)


if __name__ == "__main__":
    cb = Chessboard()
    cb.calibration(alpha=1.0)
    cb.undistorting()
    cb.save_json()
