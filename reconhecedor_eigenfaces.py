import cv2

detectorFace = cv2.CascadeClassifier("resources/haarcascade_frontalface_default.xml")
reconhecedor = cv2.face.EigenFaceRecognizer_create()
reconhecedor.read("resources/classificadorEigen.yml")
largura,altura = 220,220
font = cv2.FONT_HERSHEY_COMPLEX_SMALL

camera = cv2.VideoCapture(0)

while True:
    conectado, image = camera.read()
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    facesDetectadas = detectorFace.detectMultiScale(imageGray,
                                                    scaleFactor=1.5,
                                                    minSize=(30,30))

    for (x,y,l,a) in facesDetectadas:
        imageFace = cv2.resize(imageGray[y:y + a, x:x + l], (largura,altura))
        cv2.rectangle(image, (x,y), (x + l,y + a), (0,0,255),2)
        id,confianca = reconhecedor.predict(imageFace)

        nome = ''
        if id == 1: nome = 'Vanderson'
        elif id == 2: nome = 'Ludmila'

        #cv2.putText(image, str(id), (x,y+(a+30)), font, 2, (0,0,255))
        cv2.putText(image, nome, (x, y + (a + 30)), font, 2, (0, 0, 255))
        cv2.putText(image,str(confianca), (x,y + (a+50)), font,1,(0,0,255))

    cv2.imshow("Face", image)
    if cv2.waitKey(1) == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()