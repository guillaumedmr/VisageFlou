# Détection et floutage des visages détectés
# DAUMUR Guillaume SN2 C2

# Import des librairies
import cv2

# Utilisation d'un fichier xml pour la détection de visage
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Appel de la webcam
webcam = cv2.VideoCapture(0)

# Si la webcam est ouverte :
while True:
    # On récupère les images de la webcam
    ret, frame = webcam.read()
    # On convertit l'image en noir et blanc
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # On détecte les visages
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Pour chaque visage détecté :
    for (x,y,w,h) in faces:
        # On dessine un rectangle autour du visage
        roi = frame[y:y+h+40, x:x+w+20]
        # On floute le visage
        roi = cv2.GaussianBlur(roi, (23, 23), 50)
        # On affiche le visage flouté
        frame[y:y+roi.shape[0], x:x+roi.shape[1]] = roi

    # Création d'une fenêtre pour afficher la webcam et le visage flouté
    cv2.imshow('Face Detection',frame)
    
    # Si on appuie sur la touche 'échap' on quitte la boucle
    p = cv2.waitKey(28)
    if p == 27:
        break

# On ferme la webcam
webcam.release()
# On ferme la fenêtre
cv2.destroyAllWindows()