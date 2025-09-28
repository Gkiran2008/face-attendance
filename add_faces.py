from flask import Flask, render_template, request, redirect, url_for, flash, session
import os, cv2, pickle, numpy as np, datetime
from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__)
app.secret_key = "supersecretkey"

# ---------------- Paths ----------------
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_PATH, "data")
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)

TEACHERS_FILE = os.path.join(DATA_PATH, "teachers.pkl")
NAMES_FILE = os.path.join(DATA_PATH, "names.pkl")
FACES_FILE = os.path.join(DATA_PATH, "faces_data.pkl")
CASCADE_PATH = os.path.join(DATA_PATH, "haarcascade_frontalface_default.xml")
ATTENDANCE_FILE = os.path.join(DATA_PATH, "attendance.csv")
UPLOAD_FOLDER = os.path.join(DATA_PATH, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------- Teacher Utilities ----------------
def load_teachers():
    if os.path.exists(TEACHERS_FILE):
        with open(TEACHERS_FILE, "rb") as f:
            return pickle.load(f)
    return {}

def save_teachers(teachers):
    with open(TEACHERS_FILE, "wb") as f:
        pickle.dump(teachers, f)

# ---------------- Face Data Utilities ----------------
def load_data():
    if os.path.exists(NAMES_FILE) and os.path.exists(FACES_FILE):
        with open(NAMES_FILE, "rb") as f:
            names = pickle.load(f)
        with open(FACES_FILE, "rb") as f:
            faces = pickle.load(f)
        return np.array(names), np.array(faces)
    return None, None

# ---------------- Login Required ----------------
def login_required(func):
    def wrapper(*args, **kwargs):
        if 'teacher' not in session:
            flash("Please login first", "danger")
            return redirect(url_for('login'))
        return func(*args, **kwargs)
    wrapper.__name__ = func.__name__
    return wrapper

# ---------------- Routes ----------------

# -------- Login --------
@app.route("/", methods=["GET", "POST"])
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username").strip()
        password = request.form.get("password").strip()
        teachers = load_teachers()
        if username in teachers and teachers[username] == password:
            session['teacher'] = username
            flash(f"Welcome {username}!", "success")
            return redirect(url_for("index"))
        else:
            flash("Invalid username or password", "danger")
            return redirect(url_for("login"))
    return render_template("login.html")

# -------- Register Teacher --------
@app.route("/register_teacher", methods=["GET", "POST"])
def register_teacher():
    if request.method == "POST":
        username = request.form.get("username").strip()
        password = request.form.get("password").strip()
        teachers = load_teachers()
        if username in teachers:
            flash("Username already exists", "danger")
            return redirect(url_for("register_teacher"))
        teachers[username] = password
        save_teachers(teachers)
        flash("Teacher registered successfully! Please login.", "success")
        return redirect(url_for("login"))
    return render_template("register_teacher.html")

# -------- Forgot Password --------
@app.route("/forgot_password", methods=["GET", "POST"])
def forgot_password():
    if request.method == "POST":
        username = request.form.get("username").strip()
        new_password = request.form.get("new_password").strip()
        teachers = load_teachers()
        if username in teachers:
            teachers[username] = new_password
            save_teachers(teachers)
            flash("Password updated successfully! Please login.", "success")
            return redirect(url_for("login"))
        else:
            flash("Username not found", "danger")
            return redirect(url_for("forgot_password"))
    return render_template("forgot_password.html")

# -------- Logout --------
@app.route("/logout")
def logout():
    session.pop('teacher', None)
    flash("Logged out successfully", "success")
    return redirect(url_for("login"))

# -------- Home / Dashboard --------
@app.route("/index")
@login_required
def index():
    return render_template("index.html")

# -------- Register Face --------
@app.route("/register", methods=["GET", "POST"])
@login_required
def register():
    if request.method == "POST":
        name = request.form.get("name").strip()
        if not name:
            flash("Name cannot be empty", "danger")
            return redirect(url_for("register"))

        facedetect = cv2.CascadeClassifier(CASCADE_PATH)
        if facedetect.empty():
            flash("Haarcascade file missing", "danger")
            return redirect(url_for("register"))

        video = cv2.VideoCapture(0)
        faces_data = []
        i = 0
        while True:
            ret, frame = video.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = facedetect.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                crop_img = frame[y:y+h, x:x+w, :]
                resized_img = cv2.resize(crop_img, (50, 50))
                if len(faces_data) < 5 and i % 10 == 0:
                    faces_data.append(resized_img)
                i += 1
                cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
                cv2.putText(frame, str(len(faces_data)), (50, 50),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
            cv2.imshow("Register Face", frame)
            if cv2.waitKey(1) == ord("q") or len(faces_data) >= 10:
                break
        video.release()
        cv2.destroyAllWindows()

        if not faces_data:
            flash("No faces captured. Try again.", "danger")
            return redirect(url_for("register"))

        faces_data = np.array(faces_data).reshape(len(faces_data), -1)

        # Save names
        if os.path.exists(NAMES_FILE):
            with open(NAMES_FILE, "rb") as f:
                existing_names = pickle.load(f)
            names = existing_names + [name]*len(faces_data)
        else:
            names = [name]*len(faces_data)
        with open(NAMES_FILE, "wb") as f:
            pickle.dump(names, f)

        # Save faces
        if os.path.exists(FACES_FILE):
            with open(FACES_FILE, "rb") as f:
                existing_faces = pickle.load(f)
            faces_combined = np.append(existing_faces, faces_data, axis=0)
        else:
            faces_combined = faces_data
        with open(FACES_FILE, "wb") as f:
            pickle.dump(faces_combined, f)

        flash(f"{name} registered successfully with {len(faces_data)} samples", "success")
        return redirect(url_for("index"))

    return render_template("register.html")

# -------- List Faces --------
@app.route("/list")
@login_required
def list_faces():
    names, faces = load_data()
    if names is None:
        people = []
    else:
        unique, counts = np.unique(names, return_counts=True)
        people = list(zip(unique, counts))
    return render_template("list.html", people=people)

# -------- Delete Face --------
@app.route("/delete", methods=["GET", "POST"])
@login_required
def delete_face():
    if request.method == "POST":
        target = request.form.get("name").strip()
        names, faces = load_data()
        if names is None or target not in names:
            flash(f"No records found for '{target}'", "danger")
            return redirect(url_for("delete_face"))

        names_arr = np.array(names)
        faces_arr = np.array(faces)
        keep_idx = [i for i, n in enumerate(names_arr) if n != target]
        names_updated = names_arr[keep_idx].tolist()
        faces_updated = faces_arr[keep_idx]
        with open(NAMES_FILE, "wb") as f:
            pickle.dump(names_updated, f)
        with open(FACES_FILE, "wb") as f:
            pickle.dump(faces_updated, f)

        flash(f"Deleted all records for '{target}'", "success")
        return redirect(url_for("index"))

    return render_template("delete.html")

# -------- Mark Attendance Options Page --------
@app.route("/mark_attendance")
@login_required
def mark_attendance_page():
    return render_template("mark_attendance.html")

# -------- Attendance via Live Camera --------
@app.route("/open_camera")
@login_required
def open_camera():
    names, faces = load_data()
    if names is None or len(names) == 0:
        flash("No registered faces. Register first!", "danger")
        return redirect(url_for("index"))

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, names)

    facedetect = cv2.CascadeClassifier(CASCADE_PATH)
    cap = cv2.VideoCapture(0)
    recognized_names = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces_detected = facedetect.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces_detected:
            crop_img = frame[y:y+h, x:x+w, :]
            resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
            try:
                name = knn.predict(resized_img)[0]
            except:
                name = "Unknown"
            if name != "Unknown":
                recognized_names.add(name)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Attendance Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Save attendance
    if recognized_names:
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(ATTENDANCE_FILE, "a") as f:
            for name in recognized_names:
                f.write(f"{name},{now}\n")
        flash(f"Attendance marked for: {', '.join(recognized_names)}", "success")
    else:
        flash("No faces recognized", "danger")

    return redirect(url_for("attendance_list"))

# -------- Attendance via Group Photo Upload --------
@app.route("/upload_group_photo", methods=["POST"])
@login_required
def upload_group_photo():
    if 'group_image' not in request.files:
        flash("No file uploaded", "danger")
        return redirect(url_for("mark_attendance_page"))

    file = request.files['group_image']
    if file.filename == "":
        flash("No selected file", "danger")
        return redirect(url_for("mark_attendance_page"))

    # Read image
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        flash("Invalid image file", "danger")
        return redirect(url_for("mark_attendance_page"))

    names, faces = load_data()
    if names is None or len(names) == 0:
        flash("No registered faces. Register first!", "danger")
        return redirect(url_for("index"))

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, names)

    facedetect = cv2.CascadeClassifier(CASCADE_PATH)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces_detected = facedetect.detectMultiScale(gray, 1.3, 5)

    recognized_names = set()
    for (x, y, w, h) in faces_detected:
        crop_img = img[y:y+h, x:x+w, :]
        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
        try:
            name = knn.predict(resized_img)[0]
        except:
            name = "Unknown"
        if name != "Unknown":
            recognized_names.add(name)
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img, name, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Show preview for 3 seconds
    cv2.imshow("Group Photo Recognition", img)
    cv2.waitKey(3000)
    cv2.destroyAllWindows()

    # Save attendance
    if recognized_names:
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(ATTENDANCE_FILE, "a") as f:
            for name in recognized_names:
                f.write(f"{name},{now}\n")
        flash(f"Attendance marked for: {', '.join(recognized_names)}", "success")
    else:
        flash("No faces recognized", "danger")

    return redirect(url_for("attendance_list"))

# -------- Attendance List --------
@app.route("/attendance_list")
@login_required
def attendance_list():
    if not os.path.exists(ATTENDANCE_FILE):
        records = []
    else:
        with open(ATTENDANCE_FILE, "r") as f:
            records = [line.strip().split(",") for line in f.readlines()]
    return render_template("attendance_list.html", records=records)

# ---------------- Run ----------------
if __name__ == "__main__":
    app.run(debug=True)
