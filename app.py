import os
import cv2
import face_recognition

# ഫോട്ടോകൾ സൂക്ഷിച്ചിരിക്കുന്ന ഫോൾഡറിന്റെ പാത്ത്
KNOWN_FACES_DIR = "photos"

print("[INFO] Loading known faces...")
known_face_encodings = []
known_face_names = []

# ഫോൾഡറിലെ ഓരോ ചിത്രവും ലോഡ് ചെയ്ത് എൻകോഡ് ചെയ്യുന്നു
for filename in os.listdir(KNOWN_FACES_DIR):
    if filename.endswith((".jpg", ".jpeg", ".png")):
        image_path = os.path.join(KNOWN_FACES_DIR, filename)
        image = face_recognition.load_image_file(image_path)
        
        # ഫയലിന്റെ പേര് (എക്സ്റ്റൻഷൻ ഇല്ലാതെ) ആളുടെ പേരായി എടുക്കുന്നു
        name = os.path.splitext(filename)[0]
        
        encodings = face_recognition.face_encodings(image)
        if len(encodings) > 0:
            known_face_encodings.append(encodings[0])
            known_face_names.append(name)
        else:
            print(print(f"[WARNING] Could not find face in {filename}, skipping."))

print(f"[INFO] Loaded {len(known_face_names)} known faces: {known_face_names}")

# വെബ്‌ക്യാം ഓപ്പൺ ചെയ്യുന്നു
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        break

    # പെർഫോമൻസിനായി ഫ്രെയിമിന്റെ സൈസ് ചെറുതാക്കുന്നു (0.25 തോതിൽ)
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    
    # BGR (OpenCV) ഇമേജിനെ RGB-ലേക്ക് മാറ്റുന്നു
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # നിലവിലെ ഫ്രെയിമിലെ മുഖങ്ങളും അവയുടെ എൻകോഡിംഗും കണ്ടെത്തുന്നു
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        # ഫോട്ടോകളുമായി മാച്ച് ചെയ്യുന്നുണ്ടോ എന്ന് നോക്കുന്നു
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.4)
        name = "Unknown"

        # ഏറ്റവും അധികം സാദൃശ്യമുള്ള ആളെ കണ്ടെത്തുന്നു
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        if len(face_distances) > 0:
            best_match_index = face_distances.argmin()
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

        face_names.append(name)

    # കണ്ടെത്തിയ മുഖങ്ങൾക്ക് ചുറ്റും ബോക്സും പേരും കാണിക്കുന്നു
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # സൈസ് ചെറുതാക്കിയതുകൊണ്ട് വീണ്ടും പഴയ അളവിലേക്ക് മാറ്റുന്നു
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # ബോക്സ് വരയ്ക്കുന്നു (പച്ച നിറം)
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # പേര് എഴുതാനുള്ള ബാക്ക്ഗ്രൗണ്ട് ബോക്സ്
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.6, (255, 255, 255), 1)

    cv2.imshow('Video Face Recognition', frame)

    # ESC അമർത്തിയാൽ ക്ലോസ് ആകും
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
