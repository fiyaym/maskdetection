# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os

def detect_and_predict_mask(frame, faceNet, maskNet):
    # ambil dimensi bingkai dan kemudian buat blob dari itu
  (h, w) = frame.shape[:2] #untuk mengcapture gambar dari video 
  blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),(104.0, 177.0, 123.0))
  #[blobFromImage] membuat blob 4 dimensi dari gambar. Secara opsional mengubah ukuran dan memotong gambar dari tengah, 
  #mengurangi nilai rata-rata, menskalakan nilai berdasarkan faktor skala, menukar saluran Biru dan Merah.
  #blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size, mean, swapRB=True)
  #karena diatas kita mengcapture frame maka kita memakai frame sebagai inputan
  #untuk memasukkan nilai skala kita memilih 1.0 perlu diingat untuk memasukkan nilai skala harus 1 /sigma karena dikalikan nilai input
  #selanjutnya untuk spatial size untuk jst biasanya 224×224, 227×227, atau 299×299.
  # 3 tuple RGB (Red , Green, Blue)
  #Penjelasan lebih lanjut tentang blob https://pyimagesearch.com/2017/11/06/deep-learning-opencvs-blobfromimage-works/

    # pass  blob untuk mendapatkan deteksi wajah
  faceNet.setInput(blob)
  detections = faceNet.forward()
  print(detections.shape)

    # inialisasi list wajah, list lokasi yang sesuai,
    # dan list prediksi wajah dari network masker wajah
  faces = []
  locs = []
  preds = []

    # loop deteksinya
  for i in range(0, detections.shape[2]):
        # ekstrak confidence yang terkait dengan deteksi
        
    confidence = detections[0, 0, i, 2]

        # menyaring deteksi yang lemah dengan memastikan cofidence lebih besar dari cofidence minimum 
        
    if confidence > 0.5:
            # hitung koordinat (x, y)dari kotak pembatas objek
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # pastikan kotak pembatas berada dalam bingkai dimensi
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # ekstrak ROI wajah, ubah dari saluran BGR ke RGB
            # perintahkan , mengubah ukurannya menjadi 224x224, dan memprosesnya terlebih dahulu
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            # tambahkan wajah dan kotak pembatas ke masing-masing list
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    # membuat prediksi jika terdeteksi setidaknya satu wajah terdeteksi
  if len(faces) > 0:
        # untuk inferensi yang lebih cepat, buat prediksi batch di *semua* wajah secara bersamaan 
        # tidak satu persatu pada loop diatas
    faces = np.array(faces, dtype="float32")
    preds = maskNet.predict(faces, batch_size=32)

    # kembalikan 2-tupel lokasi dan prediksi
    return (locs, preds)

# Muat detektor wajah dari disk file dibawah,
# dimana deploy.prototxt digunakan untuk arsitektur model
# res10_300x300_ssd_iter_140000.caffemodel untuk 300 dimensi dan bingkai ssd
prototxtPath = r"deploy.prototxt"
weightsPath = r"res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
# muat detektor face mask dari file dibawah dimana file tersebut sudah terdapat modul untuk mendetektor masker
maskNet = load_model("mask_detector.model")

# jalankan webcame untuk secara realtime mendapatkan capture wajah yang muncul
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()

# loop bingkai dari video stream atau wajah yang tertangkap dalam webcame
while True:
    # ambil frame dari video yang berjalan lalu ubah ukurannyauntuk mendapatkan maksimal width 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    # deteksi wajah dalam frame dan tentukan apakah mereka memakai masker atau tidak
    (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

    # loop di atas lokasi wajah yang terdeteksi dan lokasi yang sesuai
    for (box, pred) in zip(locs, preds):
        # bongkar kotak pembatas dan prediksi
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred

        # tentukan label kelas dan warna yang akan kita gunakan untuk ditampilakn
        # gabungkan text dan kotak (0, 255, 0) = hijau , (0, 0, 255) = merah
    label = "Menggunakan Masker" if mask > withoutMask else "Tidak Menggunakan Masker"
    color = (0, 255, 0) if label == "Menggunakan Masker" else (0, 0, 255)

        # untuk menampilkan persen seberapa yakin wajah yang dideteksi memakai masker atau tidak
    label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

        # untuk menampilkan label dan disatukan dengan kotak box dari keluaran frame , dengan jenis font nya FONT_HERSHEY_SIMPLEX
        
    cv2.putText(frame, label, (startX, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
    cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    # menampilkan output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # jika klik f maka loop akan berhenti
    if key == ord("f"):
        break

# untuk menutup semua jendela yang kita buat
cv2.destroyAllWindows()
#videostreaming stop
vs.stop()
