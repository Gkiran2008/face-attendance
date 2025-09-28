import cv2
import pickle
import numpy as np
import os

# ----------------- robust path handling -----------------
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
# If there's a subfolder named "data" use that, otherwise assume files are in the same folder as the script
if os.path.isdir(os.path.join(BASE_PATH, "data")):
    DATA_PATH = os.path.join(BASE_PATH, "data")
else:
    DATA_PATH = BASE_PATH

BACKGROUND_PATH = os.path.join(DATA_PATH, "background.png")
CASCADE_PATH = os.path.join(DATA_PATH, "haarcascade_frontalface_default.xml")
NAMES_FILE = os.path.join(DATA_PATH, "names.pkl")
FACES_FILE = os.path.join(DATA_PATH, "faces_data.pkl")

# ----------------- helper utilities -----------------
def safe_load_pickles():
    if not os.path.exists(NAMES_FILE) or not os.path.exists(FACES_FILE):
        return None, None
    with open(NAMES_FILE, "rb") as f:
        names = pickle.load(f)
    with open(FACES_FILE, "rb") as f:
        faces = pickle.load(f)
    return names, faces

def list_registered():
    names, faces = safe_load_pickles()
    if names is None:
        print("No registered faces yet.")
        return
    # normalize names to strings
    if isinstance(names, np.ndarray):
        names_list = [n.decode() if isinstance(n, bytes) else str(n) for n in names.tolist()]
    else:
        names_list = [n.decode() if isinstance(n, bytes) else str(n) for n in names]
    unique, counts = np.unique(names_list, return_counts=True)
    print("\nRegistered people:")
    for idx, (u, c) in enumerate(zip(unique, counts), start=1):
        print(f"{idx}. {u}  —  {c} samples")
    print(f"Total samples: {len(names_list)}")

# ----------------- Register face -----------------
def register_face():
    # attempt to read background; if missing we'll still show camera feed
    background = cv2.imread(BACKGROUND_PATH)
    if background is None:
        print("⚠️ background.png not found — running without UI mockup.")
    else:
        bg_height, bg_width, _ = background.shape

    facedetect = cv2.CascadeClassifier(CASCADE_PATH)
    if facedetect.empty():
        print("❌ Haarcascade file not found or invalid:")
        print(CASCADE_PATH)
        print("Put 'haarcascade_frontalface_default.xml' in the data folder.")
        return

    video = cv2.VideoCapture(0)
    faces_data = []
    i = 0
    name = input("Enter Your Name: ").strip()
    if not name:
        print("Name cannot be empty.")
        video.release()
        return

    # viewport defaults (works with your mockup). If no background, these are ignored.
    viewport_x, viewport_y, viewport_w, viewport_h = 147, 350, 470, 243

    print("Press 'q' to stop early. Move your face so detector gets good samples.")
    while True:
        ret, frame = video.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            crop_img = frame[y:y+h, x:x+w, :]
            resized_img = cv2.resize(crop_img, (50, 50))
            if len(faces_data) < 100 and i % 10 == 0:
                faces_data.append(resized_img)
            i += 1
            cv2.putText(frame, str(len(faces_data)), (50, 50),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)

        if background is not None:
            frame_resized = cv2.resize(frame, (viewport_w, viewport_h))
            display_frame = background.copy()
            display_frame[viewport_y:viewport_y+viewport_h,
                          viewport_x:viewport_x+viewport_w] = frame_resized
            cv2.imshow("Secure Attendance System - Register", display_frame)
        else:
            cv2.imshow("Secure Attendance System - Register", frame)

        k = cv2.waitKey(1)
        if k == ord('q') or len(faces_data) >= 100:
            break

    video.release()
    cv2.destroyAllWindows()

    if not faces_data:
        print("⚠️ No faces captured. Try again.")
        return

    faces_data = np.array(faces_data).reshape(len(faces_data), -1)

    # ensure data folder exists
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)

    # save names (strip whitespace)
    names_to_add = [name.strip()] * len(faces_data)
    if not os.path.exists(NAMES_FILE):
        with open(NAMES_FILE, "wb") as f:
            pickle.dump(names_to_add, f)
    else:
        with open(NAMES_FILE, "rb") as f:
            existing_names = pickle.load(f)
        # normalize to python list
        if isinstance(existing_names, np.ndarray):
            existing_names = [n.decode() if isinstance(n, bytes) else str(n) for n in existing_names.tolist()]
        names_combined = existing_names + names_to_add
        with open(NAMES_FILE, "wb") as f:
            pickle.dump(names_combined, f)

    # save faces
    if not os.path.exists(FACES_FILE):
        with open(FACES_FILE, "wb") as f:
            pickle.dump(faces_data, f)
    else:
        with open(FACES_FILE, "rb") as f:
            existing_faces = pickle.load(f)
        existing_faces = np.array(existing_faces)
        faces_combined = np.append(existing_faces, faces_data, axis=0)
        with open(FACES_FILE, "wb") as f:
            pickle.dump(faces_combined, f)

    print(f"✅ {name} registered successfully with {len(faces_data)} samples.")


# ----------------- Delete face (robust) -----------------
def delete_face():
    names, faces = safe_load_pickles()
    if names is None:
        print("❌ No saved data found!")
        return

    # normalize names into list[str]
    if isinstance(names, np.ndarray):
        names_list = [n.decode() if isinstance(n, bytes) else str(n) for n in names.tolist()]
    else:
        names_list = [n.decode() if isinstance(n, bytes) else str(n) for n in names]

    faces_arr = np.array(faces)

    # handle length mismatch
    if len(names_list) != len(faces_arr):
        print(f"⚠️ Warning: names length ({len(names_list)}) != faces length ({len(faces_arr)}). Proceeding with min length.")
        minlen = min(len(names_list), len(faces_arr))
        names_list = names_list[:minlen]
        faces_arr = faces_arr[:minlen]

    unique, counts = np.unique(names_list, return_counts=True)
    print("\nRegistered people:")
    for idx, (u, c) in enumerate(zip(unique, counts), start=1):
        print(f"{idx}. {u}  —  {c} samples")

    sel = input("\nEnter the name OR the index to delete (or 'cancel'): ").strip()
    if sel.lower() == "cancel" or not sel:
        print("Cancelled.")
        return

    target_name = None
    if sel.isdigit():
        idx = int(sel) - 1
        if 0 <= idx < len(unique):
            target_name = unique[idx]
        else:
            print("Invalid index. Aborting.")
            return
    else:
        sel_norm = sel.strip().lower()
        # exact match (case-insensitive)
        for u in unique:
            if u.strip().lower() == sel_norm:
                target_name = u
                break
        if target_name is None:
            # substring matches
            matches = [u for u in unique if sel_norm in u.strip().lower()]
            if len(matches) == 1:
                target_name = matches[0]
            elif len(matches) > 1:
                print("Multiple name matches:", matches)
                choice = input("Type exact name from the matches (or index): ").strip()
                if choice.isdigit():
                    idx = int(choice) - 1
                    if 0 <= idx < len(unique):
                        target_name = unique[idx]
                    else:
                        print("Invalid index. Aborting.")
                        return
                else:
                    for u in matches:
                        if u.strip().lower() == choice.strip().lower():
                            target_name = u
                            break
                    if target_name is None:
                        print("No valid selection. Aborting.")
                        return
            else:
                print(f"No match found for '{sel}'. Aborting.")
                return

    confirm = input(f"Are you SURE you want to delete ALL records for '{target_name}'? (y/n): ").strip().lower()
    if confirm != "y":
        print("Abort delete.")
        return

    keep_indexes = [i for i, n in enumerate(names_list) if n != target_name]
    names_updated = [names_list[i] for i in keep_indexes]
    faces_updated = faces_arr[keep_indexes]

    # if nothing left, remove files
    if len(names_updated) == 0:
        try:
            os.remove(NAMES_FILE)
            os.remove(FACES_FILE)
            print("All records removed; data files deleted.")
        except Exception as e:
            print("Removed but error deleting files:", e)
        return

    with open(NAMES_FILE, "wb") as f:
        pickle.dump(names_updated, f)
    with open(FACES_FILE, "wb") as f:
        pickle.dump(faces_updated, f)

    removed_count = sum(1 for n in names_list if n == target_name)
    print(f"🗑️ Deleted {removed_count} samples for '{target_name}' successfully.")


# ----------------- Main menu -----------------
def main():
    while True:
        print("\n--- Secure Attendance System ---")
        print("1. Register New Face")
        print("2. List Registered Names")
        print("3. Delete Existing Face")
        print("4. Exit")
        choice = input("Enter your choice: ").strip()

        if choice == "1":
            register_face()
        elif choice == "2":
            list_registered()
        elif choice == "3":
            delete_face()
        elif choice == "4":
            print("Exiting...")
            break
        else:
            print("Invalid choice! Please try again.")


if __name__ == "__main__":
    main()
