from detection.detection import Detector
from embedding.embedding import Embedding
import glob
import cv2


def main():
    detector = Detector("detection/weights/whole_body.engine")
    extractor = Embedding("embedding/weights/whole_body.engine")

    img_files = glob.glob("../tracking/eval/data/gt/zjlab/iros2022-fisheye-tradition-test/02_origial_black_fisheye_head_front/img1/*.jpg")
    img_files.sort()

    for img_file in img_files:
        img = cv2.imread(img_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        box, _ = detector(img)
        _ = extractor(img, box)


if __name__ == "__main__":
    main()
