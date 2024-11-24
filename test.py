import cv2
import os
from ultralytics import YOLO

# Charger le modèle YOLOv8
model = YOLO('drone-backend/best_model.pt')

# Dossiers pour les fichiers d'entrée et de sortie
INPUT_PATH = 'drone-backend/input'
OUTPUT_PATH = 'drone-backend/output'
os.makedirs(INPUT_PATH, exist_ok=True)
os.makedirs(OUTPUT_PATH, exist_ok=True)

def process_image(input_file):
    # Charger et détecter dans l'image
    results = model(input_file)
    result = results[0]  # Obtenir le premier résultat (utile pour une seule image)

    # Construire le chemin de sauvegarde pour l'image annotée
    annotated_image_path = os.path.join(OUTPUT_PATH, f"result_{os.path.basename(input_file)}")

    # Enregistrer l'image annotée
    annotated_image = result.plot()  # Renvoie une image avec les annotations sous forme de numpy array
    cv2.imwrite(annotated_image_path, annotated_image)  # Sauvegarde l'image annotée sur le disque

    # Extraire les statistiques
    stats = []
    for box in result.boxes:
        stats.append({
            "class": int(box.cls),
            "confidence": float(box.conf),
            "bbox": box.xyxy.tolist()
        })

    return annotated_image_path, stats



def process_video(input_file):
    # Charger la vidéo
    cap = cv2.VideoCapture(input_file)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_file = os.path.join(OUTPUT_PATH, f"result_{os.path.basename(input_file)}")
    out = cv2.VideoWriter(output_file, fourcc, cap.get(cv2.CAP_PROP_FPS), 
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    
    stats = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        annotated_frame = results[0].plot()
        out.write(annotated_frame)

        # Collecter les statistiques frame par frame
        for box in results[0].boxes:
            stats.append({
                "class": int(box.cls),
                "confidence": float(box.conf),
                "bbox": box.xyxy.tolist()
            })

    cap.release()
    out.release()
    return output_file, stats

def main():
    print("Entrez le chemin du fichier (image ou vidéo) :")
    file_path = input().strip()

    if not os.path.exists(file_path):
        print("Fichier introuvable. Assurez-vous que le chemin est correct.")
        return

    file_extension = os.path.splitext(file_path)[-1].lower()

    if file_extension in ['.jpg', '.jpeg', '.png']:
        print("Traitement de l'image...")
        output_file, stats = process_image(file_path)
    elif file_extension in ['.mp4', '.avi', '.mov']:
        print("Traitement de la vidéo...")
        output_file, stats = process_video(file_path)
    else:
        print("Type de fichier non pris en charge. Veuillez utiliser une image ou une vidéo.")
        return

    print(f"Traitement terminé. Résultat sauvegardé dans : {output_file}")
    print("Statistiques :")
    for stat in stats:
        print(stat)

if __name__ == '__main__':
    main()
