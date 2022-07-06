import argparse
import cv2
import sys


def make_cat_passport_image(input_image_path, haar_model_path):

    # Read image

    image = cv2.imread(input_image_path)

    # Convert image to grayscale

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Normalize image intensity

    gray = cv2.equalizeHist(gray)

    # Resize image

    resized = cv2.resize(gray, (640, 480), interpolation = cv2.INTER_AREA)

    # Detect cat faces using Haar Cascade

    detector = cv2.CascadeClassifier(haar_model_path)
    rects = detector.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(75, 75))
    # Draw bounding box

    for(i, (x, y, w, h)) in enumerate(rects):
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(image, "Cat #{}".format(i + 1), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    # Display result image

    cv2.imshow("Cat", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Crop image

    x, y, w, h = rects[0]
    image = image[y:y+h, x:x+w]

    # Save result image to file

    cv2.imwrite('cat_result.jpg', image)

    return image

def cat_passport(cat_image):
    image_passport = cv2.imread("pet_passport.png")

    resized = cv2.resize(cat_image, (164, 130), interpolation = cv2.INTER_AREA)

    image_passport[52:182,34:198] = resized[0:130, 0:164]

    cv2.putText(image_passport, "MyCat", (86,218), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    cv2.putText(image_passport, "Cat", (86,231), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    cv2.putText(image_passport, "CatBreed", (86,245), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    cv2.putText(image_passport, "CatSex", (86,259), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    cv2.putText(image_passport, "CatDOB", (111,272), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    cv2.putText(image_passport, "CatCoat", (86,285), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    cv2.putText(image_passport, "CatNumber", (272,89), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    cv2.putText(image_passport, "CatNumberDate", (272,127), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    cv2.putText(image_passport, "CatNumberLocation", (272,168), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    cv2.putText(image_passport, "CatTattooNumber", (272,208), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    cv2.putText(image_passport, "CatTattoo", (272,247), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    cv2.imshow("passport", image_passport)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite('passport_result.jpg', image_passport)

    return


     


def build_argparser():
    parser = argparse.ArgumentParser(
        description='Speech denoising demo', add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
                      help='Show this help message and exit.')
    args.add_argument('-m', '--model', type=str, required=True,
                      help='Required. Path to .XML file with pre-trained model.')
    args.add_argument('-i', '--input', type=str, required=True,
                      help='Required. Path to input image')
    return parser


def main():
    
    args = build_argparser().parse_args()
    image = make_cat_passport_image(args.input, args.model)
    cat_passport(image)

    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
