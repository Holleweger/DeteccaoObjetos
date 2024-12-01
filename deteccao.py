# Sobre o código, ele é uma junção/personalização de códigos
# já existentes, eu e o João fomos pesquisando, criando e 
# adicionando trexos de códigos até funcionar. Usamos como base
# o nosso projeto de IA que fizemos detecção de objetos na webcan
# caso queira olhar, segue link abaixo
# https://github.com/joaocn2/objdetectpy/tree/main?tab=readme-ov-file

import cv2
import numpy as np

# Aqui é o caminhos para os modelos de treino, já deixei na pasta raiz
# junto com o projeto, só ajustar a pasta
model_cfg = "./yolov4.cfg"
model_weights = "./yolov4.weights"
class_names_file = "./coco.names"

with open(class_names_file, "r") as f:
    classes = [line.strip() for line in f.readlines()]

net = cv2.dnn.readNet(model_weights, model_cfg)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# image_path é o nome da imagem, pode baixar uma ou usar as que eu
# disponibilizei na pasta img
image_path = "./img/imagem.jpg"
image = cv2.imread(image_path)
height, width, _ = image.shape

blob = cv2.dnn.blobFromImage(image, scalefactor=1/255.0, size=(416, 416), swapRB=True, crop=False)
net.setInput(blob)
outputs = net.forward(output_layers)

boxes = []
confidences = []
class_ids = []

# O código foi baseado em confiança, então só vai mostrar se
# a confiança for maior que o valor "confidence > x", se quiser
# testar, pode alterar.
for output in outputs:
    for detection in output:
        if detection.shape[0] >= 5: 
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5: 
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

# Essa linha não entendo direito o que faz, porém é para "suprimir"
# a quantidade de caixas, porque estava aparecendo mais do que
# deveria
indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# Aqui é para mostrar as linhas
for i in indices.flatten():
    x, y, w, h = boxes[i]
    label = str(classes[class_ids[i]])
    confidence = confidences[i]
    color = (0, 255, 0)
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    cv2.putText(image, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# E aqui para mostrar a imagem
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
