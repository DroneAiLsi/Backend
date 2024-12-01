from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import cv2
from ultralytics import YOLO
from collections import Counter
import numpy as np
from scipy.spatial import distance

# Initialisation de Flask
app = Flask(__name__)
CORS(app)  # Permettre les requêtes cross-origin

# Charger le modèle YOLO
model = YOLO('drone-backend/models/best_model.pt')

# Dossiers pour les fichiers d'entrée et de sortie
INPUT_PATH = 'drone-backend/input'
OUTPUT_PATH = 'drone-ui/public/results'
os.makedirs(INPUT_PATH, exist_ok=True)
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Dictionnaire pour les noms des classes
class_names = {
    0: "pedestrian",
    1: "people",
    2: "bicycle",
    3: "car",
    4: "van",
    5: "truck",
    6: "tricycle",
    7: "awning-tricycle",
    8: "bus",
    9: "motor"
}

def process_image(input_file):
    results = model(input_file)
    result = results[0]

    annotated_image_path = os.path.join(OUTPUT_PATH, f"result_{os.path.basename(input_file)}")
    annotated_image = result.plot()
    cv2.imwrite(annotated_image_path, annotated_image)

    class_counts = {i: 0 for i in range(10)}  # Initialisation des compteurs de classes (0 à 9)
    for box in result.boxes:
        class_id = int(box.cls)
        class_counts[class_id] += 1  # Incrémentation du compteur pour la classe correspondante

    # Mapping des IDs de classe à leurs noms
    class_names = {
        0: "pedestrian",
        1: "people",
        2: "bicycle",
        3: "car",
        4: "van",
        5: "truck",
        6: "tricycle",
        7: "awning-tricycle",
        8: "bus",
        9: "motor"
    }

    # Construire les statistiques avec les noms des classes et leur nombre
    stats = []
    for class_id, count in class_counts.items():
        if count > 0:  # Ajouter la classe uniquement si elle a été détectée
            stats.append({
                "class": class_names[class_id],
                "count": count
            })

    return annotated_image_path, stats



# Tolerance for spatial similarity (e.g., for bounding box comparison)
TOLERANCE = 50  # Adjust based on object size and video resolution
CONFIDENCE_THRESHOLD = 0.5  # Filter out low-confidence detections
FRAME_INTERVAL = 5  # Process every nth frame


def process_video(input_file):
    cap = cv2.VideoCapture(input_file)
    fourcc = cv2.VideoWriter_fourcc(*'H264')  # Utilisation de H264 pour la compatibilité
    output_file = os.path.join(OUTPUT_PATH, f"result_{os.path.basename(input_file)}")
    
    # Vérifier que le fichier vidéo a bien été ouvert
    if not cap.isOpened():
        raise Exception("Erreur lors de l'ouverture du fichier vidéo.")

    out = cv2.VideoWriter(output_file, fourcc, cap.get(cv2.CAP_PROP_FPS),
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    class_counts = {i: 0 for i in range(10)}  # Initialisation des comptes de classes
    tracked_objects = {}  # Pour stocker les objets détectés
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Traiter chaque n-ième frame
        if frame_count % FRAME_INTERVAL != 0:
            frame_count += 1
            continue

        results = model(frame)  # Traitement par votre modèle
        annotated_frame = results[0].plot()  # Ajout des annotations
        out.write(annotated_frame)

        # Traitement des boîtes détectées
        for box in results[0].boxes:
            class_id = int(box.cls)
            confidence = float(box.conf)
            if confidence < CONFIDENCE_THRESHOLD:
                continue  # Ignorer les détections avec une faible confiance

            bbox = tuple(map(int, box.xyxy[0].tolist()))  # (x_min, y_min, x_max, y_max)

            # Vérifier si l'objet a déjà été suivi
            unique = True
            for tracked_bbox in tracked_objects.values():
                if is_similar_bbox(bbox, tracked_bbox):
                    unique = False
                    break

            if unique:
                tracked_objects[frame_count] = bbox  # Suivi de l'objet
                class_counts[class_id] += 1  # Incrémenter le compteur de classes

        frame_count += 1

    cap.release()
    out.release()

    # Générer des statistiques
    class_names = {
        0: "pedestrian", 1: "people", 2: "bicycle", 3: "car", 4: "van",
        5: "truck", 6: "tricycle", 7: "awning-tricycle", 8: "bus", 9: "motor"
    }
    
    stats = []
    for class_id, count in class_counts.items():
        if count > 0:
            stats.append({
                "class": class_names[class_id],
                "count": count
            })

    return output_file, stats



def is_similar_bbox(bbox1, bbox2):
    """
    Check if two bounding boxes are spatially similar.
    """
    center1 = ((bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2)
    center2 = ((bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2)
    return distance.euclidean(center1, center2) < TOLERANCE

@app.route('/process', methods=['POST'])
def process_file():
    # Vérifier si un fichier est envoyé
    if 'file' not in request.files:
        return jsonify({'error': 'Aucun fichier fourni'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Nom de fichier vide'}), 400

    # Enregistrer le fichier d'entrée
    input_file_path = os.path.join(INPUT_PATH, file.filename)
    file.save(input_file_path)

    # Vérifier le type de fichier
    file_extension = os.path.splitext(file.filename)[-1].lower()

    if file_extension in ['.jpg', '.jpeg', '.png']:
        output_file, stats = process_image(input_file_path)
    elif file_extension in ['.mp4', '.avi', '.mov']:
        output_file, stats = process_video(input_file_path)
    else:
        return jsonify({'error': 'Type de fichier non pris en charge'}), 400
    
    return jsonify({
        'output_file': f"/results/{os.path.basename(output_file)}",
        'stats': stats
    })


@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    file_path = os.path.join(OUTPUT_PATH, filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    else:
        return jsonify({'error': 'Fichier non trouvé'}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
