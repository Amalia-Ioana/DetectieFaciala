import cv2
import mediapipe as mp
import argparse

# functie care blureaza fata dintr-o imagine sau frame video
def procesare_imagine(img, face_detect):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    final = face_detect.process(img_rgb)

    H, W, _ = img.shape 

    if final.detections is not None:
        for d in final.detections:
            # primirea informatiilor legate de detectarea fetei --> bounding box 
            data_location = d.location_data
            bbox = data_location.relative_bounding_box

            x1 = int(bbox.xmin * W)
            y1 = int(bbox.ymin * H)
            w = int(bbox.width * W)
            h = int(bbox.height * H)

            # limitere imaginii
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(W, x1 + w)
            y2 = min(H, y1 + h)

            img[y1:y2, x1:x2] = cv2.blur(img[y1:y2, x1:x2], (50, 50))

            # bounding box in jurul fetei 
            cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 10)

    return img

# argumente CLI
parser = argparse.ArgumentParser()
parser.add_argument("--mode", default='image', choices=['image', 'video'], help="Modul: image sau video")
parser.add_argument("--filePath", required=True, help="Calea către imagine sau video")
args = parser.parse_args()

# detectare faciala
face_detection = mp.solutions.face_detection

# model_selection=1 --> se focalizeaza pe detectarea fetelor aflate si mai in spate,
#                       maxim 5 metrii, maxim 10 persoane
# model_selection=0 --> este folosit pentru pozele cu persoane apropiate de camera
with face_detection.FaceDetection(model_selection=1, min_detection_confidence=3.0) as face_detect:
    if args.mode == "image":
        img = cv2.imread(args.filePath)
        if img is None:
            print("Imaginea nu a putut fi încărcată.")
            exit(1)

        img = procesare_imagine(img, face_detect)
        cv2.imshow('Imagine cu fata blurata', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif args.mode == "video":
        cap = cv2.VideoCapture(args.filePath)
        if not cap.isOpened():
            print("Video-ul nu a putut fi încărcat.")
            exit(1)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = procesare_imagine(frame, face_detect)
            cv2.imshow('Video cu fete blurate', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

#python face_blurring_privacy.py --mode image --filePath "C:/Users/Amy/Downloads/imagine_test.png"
#python face_blurring_privacy.py --mode video --filePath "C:/Users/Amy/Downloads/video_fete.mp4"
