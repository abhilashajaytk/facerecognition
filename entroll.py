import cv2
import face_recognition
import sqlite3
import pickle

# Database connection sthapikkunnu
conn = sqlite3.connect('villager_database.db')
cursor = conn.cursor()

# Table create cheyyunnu
cursor.execute('''
    CREATE TABLE IF NOT EXISTS villagers (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        encoding BLOB NOT NULL
    )
''')
conn.commit()

# Alude peru keyboard-il ninnu vangunnu
name = input("Enter the person's name: ").strip()
if not name:
    print("Name cannot be empty!")
    conn.close()
    exit()

cap = cv2.VideoCapture(0)
print("[INFO] Position the face in front of the camera and press 's' to capture, or 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    cv2.imshow('Enrollment - Press S to Capture', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        # Frame-il ninnu face encoding edukkunnu
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_encodings = face_recognition.face_encodings(rgb_frame)

        if len(face_encodings) > 0:
            encoding = face_encodings[0]
            encoding_blob = pickle.dumps(encoding)

            # Database-ilekku insert cheyyunnu
            cursor.execute("INSERT INTO villagers (name, encoding) VALUES (?, ?)", (name, encoding_blob))
            conn.commit()
            print(f"[SUCCESS] {name} successfully enrolled and saved to database!")
            break
        else:
            print("[WARNING] No face detected! Please try again with clear lighting.")
    elif key == ord('q'):
        print("Enrollment cancelled.")
        break

cap.release()
cv2.destroyAllWindows()
conn.close()