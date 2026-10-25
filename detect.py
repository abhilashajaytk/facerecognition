import sqlite3
import cv2
import face_recognition
import pickle

print("[INFO] Loading faces from database...")
conn = sqlite3.connect('villager_database.db')
cursor = conn.cursor()

cursor.execute("SELECT name, encoding FROM villagers")
rows = cursor.fetchall()

known_face_names = []
known_face_encodings = []

for row in rows:
    name = row[0]
    encoding_blob = row[1]
    encoding = pickle.loads(encoding_blob)
    
    known_face_names.append(name)
    known_face_encodings.append(encoding)

conn.close()
print(f"[INFO] Loaded {len(known_face_names)} villagers from database.")

# Webcam open cheyyunnu
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

    # Performance-inayi frame cherutakkunnu
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
        name = "Unknown"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        if len(face_distances) > 0:
            best_match_index = face_distances.argmin()
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

        face_names.append(name)

    # Mugamukku chuttum box-um perum kanikkunnu
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.6, (255, 255, 255), 1)

    cv2.imshow('Village Face Detection System', frame)

    # ESC press cheythal exit aakam
    if cv2.waitKey(1) & 0xFF == 27:
        print("Exiting...")
        break

cap.release()
cv2.destroyAllWindows()