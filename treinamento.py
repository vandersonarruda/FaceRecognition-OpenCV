import cv2
import os
import numpy as np

# eigenface = cv2.face.EigenFaceRecognizer_create()
# parametros:
# - num_components -> Numeros de componentes que será analisado (50 é o máximo necessário)
# - threshold (limite de confiança distância de uma face a outra) - ex.: 2 (precisa testar com diversas fotos para chegar num resultado melhor)
# calcula a distância entre a nova face e as faces de treinamento (knn)
# faces desconhecidas, retorna -1
eigenface = cv2.face.EigenFaceRecognizer_create(num_components=50, threshold=0)

fisherface = cv2.face.FisherFaceRecognizer_create()
lbph = cv2.face.LBPHFaceRecognizer_create()

def getImagemComID():
    caminhos = [os.path.join("fotos", f) for f in os.listdir("fotos")]
    # lista todas as fotos que tem na pasta
    #print(caminhos)
    faces = []
    ids = []

    for caminhoImagem in caminhos:
        imagemFace = cv2.cvtColor(cv2.imread(caminhoImagem), cv2.COLOR_BGR2GRAY) # carrega e converte as imagems em escala de cinza
        id = int(os.path.split(caminhoImagem)[-1].split(".")[1]) # pegar somente os ID que está no nome das fotos
        #print(id)

        ids.append(id)
        faces.append(imagemFace)
        #cv2.imshow("Face", imagemFace)
        #cv2.waitKey(10)
    return np.array(ids), faces

ids, faces = getImagemComID()
#print(ids)
#print(faces)

print("Treinando...")
eigenface.train(faces, ids)
eigenface.write("resources/classificadorEigen.yml")

fisherface.train(faces, ids)
fisherface.write("resources/classificadorFisher.yml")

lbph.train(faces, ids)
lbph.write("resources/classificadorLBPH.yml")

print("Treinamento realizado")

