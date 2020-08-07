import cv2
import numpy as np

classificador = cv2.CascadeClassifier("resources/haarcascade_frontalface_default.xml")
classificadorEye = cv2.CascadeClassifier("resources/haarcascade_eye.xml")

camera = cv2.VideoCapture(0)
amostra = 1
numeroAmostras = 25
id = input("Digite seu ID: ")
largura, altura = 220, 220
print("Capturando as faces...")

while True:
    conectado, image = camera.read()
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #print(np.average(imageGray))

    facesDetectadas = classificador.detectMultiScale(imageGray,
                                                     scaleFactor=1.5,
                                                     minSize=(150,150))
 
    for (x,y,l,a) in facesDetectadas:
        cv2.rectangle(image,(x,y),(x + l,y + a),(255,0,0),2)
        regiao = image[y:y+a, x:x+l]
        regiaoGray = cv2.cvtColor(regiao, cv2.COLOR_BGR2GRAY)

        eyesDetectado = classificadorEye.detectMultiScale(regiaoGray)
        for (ox,oy,ol,oa) in eyesDetectado:
            cv2.rectangle(regiao, (ox,oy), (ox + ol, oy + oa), (0,255,0), 2)

            if cv2.waitKey(1) == ord("q"):
                if np.average(imageGray) > 110: # valor 0 <-> 255 = Verifica o brilho da imagem (tirando a média dos pixels)
                    imageFace = cv2.resize(imageGray[y:y + a, x:x + l], (largura, altura))
                    cv2.imwrite("fotos/pessoa." + str(id) + "." + str(amostra) + ".jpg", imageFace)
                    print("[foto " + str(amostra) + " capturada com sucesso]")
                    amostra += 1

    cv2.imshow("Face", image)
    cv2.waitKey(1)
    if (amostra >= numeroAmostras + 1):
        break

print("Faces: DONE")
camera.release()
cv2.destroyAllWindows()