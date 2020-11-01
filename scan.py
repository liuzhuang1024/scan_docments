# USAGE
# python opencv_text_detection_image.py --image images/lebron_james.jpg
# --east frozen_east_text_detection.pb

# import the necessary packages
from imutils.object_detection import non_max_suppression
import numpy as np
import time
import cv2
from config import args
import logging
logging.basicConfig(
    level=logging.INFO
)
print = logging.info


class Scan:
    def __init__(self) -> None:

        # define the two output layer names for the EAST detector model that
        # we are interested -- the first is the output probabilities and the
        # second can be used to derive the bounding box coordinates of text
        self.layerNames = [
            "feature_fusion/Conv_7/Sigmoid",
            "feature_fusion/concat_3"]

        # load the pre-trained EAST text detector
        print("[INFO] loading EAST text detector...")
        self.net = cv2.dnn.readNet(args["east"])

    def run(self, image_path: str, save_flag: bool = False) -> np.array:

        # load the input image and grab the image dimensions
        image = cv2.imread(image_path)
        orig = image.copy()
        (H, W) = image.shape[:2]
        orign_WH = (W, H)
        # set the new width and height and then determine the ratio in change
        # for both the width and height
        (newW, newH) = (args["width"], args["height"])
        rW = W / float(newW)
        rH = H / float(newH)

        # resize the image and grab the new image dimensions
        image = cv2.resize(image, (newW, newH))
        (H, W) = image.shape[:2]

        # construct a blob from the image and then perform a forward pass of
        # the model to obtain the two output layer sets
        blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
                                     (123.68, 116.78, 103.94), swapRB=True, crop=False)

        start = time.time()
        self.net.setInput(blob)
        (scores, geometry) = self.net.forward(self.layerNames)
        end = time.time()

        # show timing information on text prediction
        print("[INFO] text detection took {:.6f} seconds".format(end - start))

        # grab the number of rows and columns from the scores volume, then
        # initialize our set of bounding box rectangles and corresponding
        # confidence scores
        (numRows, numCols) = scores.shape[2:4]
        rects = []
        confidences = []

        # loop over the number of rows
        for y in range(0, numRows):
            # extract the scores (probabilities), followed by the geometrical
            # data used to derive potential bounding box coordinates that
            # surround text
            scoresData = scores[0, 0, y]
            xData0 = geometry[0, 0, y]
            xData1 = geometry[0, 1, y]
            xData2 = geometry[0, 2, y]
            xData3 = geometry[0, 3, y]
            anglesData = geometry[0, 4, y]

            # loop over the number of columns
            for x in range(0, numCols):
                # if our score does not have sufficient probability, ignore it
                if scoresData[x] < args["min_confidence"]:
                    continue

                # compute the offset factor as our resulting feature maps will
                # be 4x smaller than the input image
                (offsetX, offsetY) = (x * 4.0, y * 4.0)

                # extract the rotation angle for the prediction and then
                # compute the sin and cosine
                angle = anglesData[x]
                cos = np.cos(angle)
                sin = np.sin(angle)

                # use the geometry volume to derive the width and height of
                # the bounding box
                h = xData0[x] + xData2[x]
                w = xData1[x] + xData3[x]

                # compute both the starting and ending (x, y)-coordinates for
                # the text prediction bounding box
                endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
                endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
                startX = int(endX - w)
                startY = int(endY - h)

                # add the bounding box coordinates and probability score to
                # our respective lists
                rects.append((startX, startY, endX, endY))
                confidences.append(scoresData[x])

        # apply non-maxima suppression to suppress weak, overlapping bounding
        # boxes
        boxes = non_max_suppression(np.array(rects), probs=confidences)
        points = []
        # loop over the bounding boxes
        for (startX, startY, endX, endY) in boxes:
            # scale the bounding box coordinates based on the respective
            # ratios
            startX = int(startX * rW)
            startY = int(startY * rH)
            endX = int(endX * rW)
            endY = int(endY * rH)
            points.append([startX, startY, endX, endY])

        points = np.array(points)
        bbox = (
            np.min(points[:, 0]), np.min(points[:, 1]),
            np.max(points[:, 2]), np.max(points[:, 3]),
        )

        bbox = [bbox[0]//2, bbox[1]//2,
                orign_WH[0] - (orign_WH[0] - bbox[2])//2, orign_WH[1] - (orign_WH[1] - bbox[3])//2]
        print(bbox)
        print(f"orig.shape={orig.shape}")
        if save_flag:
            print("Saving!!!!")
            cv2.imwrite("images/save_2.jpg",
                        orig[bbox[1]:bbox[3], bbox[0]:bbox[2], :])


        return {
            "status": "sucess",
            "result_image": orig[bbox[1]:bbox[3], bbox[0]:bbox[2], :],
            "save_path": "images/save_2.jpg",
            "text_bbox": bbox,
        }

    def heart(self):
        return "Right"


if __name__ == "__main__":
    print(Scan().run("images/微信图片_20201031175806.jpg", True)["text_bbox"])