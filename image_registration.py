import argparse
from math import sqrt

import cv2
import numpy as np


MIN_MATCH_COUNT = 10
FLATTEN_INDEX_KDTREE = 1
TREES = 5
CHECKS = 50
K = 2
GOOD_THRESH = 0.7
RANSAC_REPROJ_THRESH = 5.0
MATCH_COLOR = (0, 255, 0)
FLAGS = 2


def get_args():
    parser = argparse.ArgumentParser(
        description="Code for AKAZE local features matching tutorial."
    )
    parser.add_argument(
        "-f",
        "--from-img",
        help="Path to input image 1.",
        default="./input/473_SubImage.png",
    )
    parser.add_argument(
        "-t", "--to-img", help="Path to input image 2.", default="./input/473_COLOR.png"
    )
    parser.add_argument(
        "--homography", help="Path to the homography matrix.", default="H1to3p.xml"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    from_img = cv2.imread(args.from_img, cv2.IMREAD_GRAYSCALE)
    to_img = cv2.imread(args.to_img, cv2.IMREAD_GRAYSCALE)

    # initiate SIFT detector
    sift = cv2.SIFT_create()

    # find the keypoints and descriptors with SIFT
    from_kp, from_des = sift.detectAndCompute(from_img, None)
    to_kp, from_des = sift.detectAndCompute(to_img, None)

    flann = cv2.FlannBasedMatcher(
        dict(algorithm=FLATTEN_INDEX_KDTREE, trees=TREES),
        dict(checks=CHECKS),
    )
    matches = flann.knnMatch(from_des, from_des, k=K)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < GOOD_THRESH * n.distance:
            good.append(m)

    # early return if not enough matches
    if len(good) <= MIN_MATCH_COUNT:
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
        exit()

    src_pts = np.float32([from_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([to_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, RANSAC_REPROJ_THRESH)
    matchesMask = mask.ravel().tolist()

    h, w = from_img.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)

    to_img = cv2.polylines(to_img, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

    draw_paramas = dict(
        matchColor=MATCH_COLOR,
        singlePointColor=None,
        matchesMask=matchesMask,
        flags=FLAGS,
    )
    result_img = cv2.drawMatches(
        from_img, from_kp, to_img, to_kp, good, None, **draw_paramas
    )
    cv2.imwrite("result.png", result_img)
