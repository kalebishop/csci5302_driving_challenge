#!/usr/bin/env python3
import cv2
import numpy as np

class Detector:
    def __init__(self, lower_bound, upper_bound):
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound

    def get_lines(self, img, plot=False):
        """Use HoughLines algorithm to find white lines in the image.
            :param img: 3D rgb pixel numpy array
            :return: Array of four tuples where points (x1, y1) and (x2, y2) form
            a line. Ex. [(x1, y1, x2, y2) ... ]
        """
        filtered_image = self.filter_image(img)
        processed_image = cv2.Canny(filtered_image, threshold1=100, threshold2=200)

        lines = cv2.HoughLinesP(processed_image, 1, np.pi/180, 2)
        if lines is not None and len(lines) > 0:
            if plot:
                for i in range(lines.shape[0]):
                    for coord in lines[i]:
                        cv2.line(img, (coord[0], coord[1]), (coord[2], coord[3]), (0, 0, 255), 1)

                cv2.imshow("lines", img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            lines = lines.reshape(-1, 4)
        return lines

    def process_image(self, img):
        """ Main routine; returns keypoints of detected image blobs
            :param img: 3D rgb pixel numpy array
        """
        filtered_image = self.filter_image(img)
        keypoints = self.compute_keypoints_for_blobs(filtered_image)

        im_with_keypoints = cv2.drawKeypoints(filtered_image, keypoints, np.array([]), (0, 0, 255),
                                        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # result is largest keypoint
        kid = -1
        max_size = -1
        for i, k in enumerate(keypoints):
            if k.size > max_size:
                k.size = max_size
                kid = i

        # cv2.imshow("im_with_keypoints", im_with_keypoints)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        if kid >= 0:
            # draw the main target on the im_with_keypoints output image
            cv2.drawMarker(im_with_keypoints, (int(keypoints[kid].pt[0]), int(keypoints[kid].pt[1])), (0, 0, 255),
                           cv2.MARKER_CROSS, 100, 4)


            return keypoints[kid].pt
        return None

    def filter_image(self, img):
        """ Convert image to HSV color space and filter on a given hue range
            :param cv_image: input image
            :param lower_hue_value: min hue value
            :param higher_hue_value: max hue value
            :return: filtered image
        """
        hsv = cv2.cvtColor(img / 255.0, cv2.COLOR_RGB2HSV)
        # print(hsv[0, 0, :])
        maskHSV = cv2.inRange(hsv, self.lower_bound, self.upper_bound)
        return maskHSV
        # return hsv

    def compute_keypoints_for_blobs(self, filtered_image):
        """ Compute keypoints for blobs of a given color in the image.
        :param cv_image: input image
        :return: list of cv2.Keypoint objects (see https://docs.opencv.org/2.4/modules/features2d/doc/common_interfaces_of_feature_detectors.html?highlight=keypoint)
        """
        params = cv2.SimpleBlobDetector_Params()

        params.filterByArea = True
        params.minArea = 3 # TODO: tweak
        params.maxArea = 40

        params.filterByColor = True
        params.blobColor = 255

        params.filterByInertia = False
        params.filterByCircularity = False
        params.filterByConvexity = False

        ver = (cv2.__version__).split('.')
        if int(ver[0]) < 3 :
            detector = cv2.SimpleBlobDetector(params)
        else :
            detector = cv2.SimpleBlobDetector_create(params)

        keypoints = detector.detect(filtered_image)
        return keypoints

# if __name__ == "__main__":
#     test_img = np.load("test.npy", allow_pickle=False) / 255.0
#     test_img = cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR)
#     cv2.imshow("img", test_img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
