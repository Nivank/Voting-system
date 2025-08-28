import os
import io
import csv
import time
import base64
import pickle
from datetime import datetime

from flask import Flask, request, jsonify, send_from_directory, session
from flask_cors import CORS

import numpy as np
import cv2
from sklearn.neighbors import KNeighborsClassifier
from openpyxl import Workbook


app = Flask(__name__, static_folder="web", static_url_path="/")
CORS(app)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-me")


DATA_DIR = "data"
FACES_PATH = os.path.join(DATA_DIR, "faces_data.pkl")
NAMES_PATH = os.path.join(DATA_DIR, "names.pkl")
VOTES_CSV = "Votes.csv"
PROFILES_JSON = os.path.join(DATA_DIR, "profiles.json")
AUDIT_LOG = os.path.join(DATA_DIR, "audit.log")

USERS = {
    "nivank": "nivankclaps",
    "superior": "yougotnogame",
}


def ensure_data_dir() -> None:
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)


def load_training_data():
    ensure_data_dir()
    if not (os.path.exists(FACES_PATH) and os.path.exists(NAMES_PATH)):
        return None, None
    with open(NAMES_PATH, "rb") as f:
        labels = pickle.load(f)
    with open(FACES_PATH, "rb") as f:
        faces = pickle.load(f)
    return faces, labels


def build_classifier():
    faces, labels = load_training_data()
    if faces is None or labels is None or len(labels) == 0:
        return None
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    return knn


def require_admin():
    if not session.get("admin_user"):
        return False
    return True


def load_profiles():
    ensure_data_dir()
    if os.path.exists(PROFILES_JSON):
        try:
            import json
            with open(PROFILES_JSON, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_profiles(profiles):
    ensure_data_dir()
    import json
    with open(PROFILES_JSON, "w", encoding="utf-8") as f:
        json.dump(profiles, f, ensure_ascii=False, indent=2)


def append_audit(event: str, detail: dict | None = None) -> None:
    ensure_data_dir()
    try:
        import json
        now = datetime.utcnow().isoformat() + "Z"
        user = session.get("admin_user")
        payload = {
            "ts": now,
            "user": user,
            "event": event,
            "detail": detail or {},
        }
        with open(AUDIT_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        # Best-effort; do not break flow
        pass


def decode_base64_image(data_url: str):
    # data_url expected like: "data:image/jpeg;base64,<base64>" or raw base64
    if "," in data_url:
        data = data_url.split(",", 1)[1]
    else:
        data = data_url
    try:
        img_bytes = base64.b64decode(data)
    except Exception:
        return None
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/config", methods=["GET", "POST"])
def config():
    # Placeholder endpoint for customizable options (e.g., framesTotal, neighbors, parties)
    if request.method == "POST":
        return jsonify({"ok": True})
    return jsonify({
        "neighbors": 5,
        "framesTotal": 5,
        "maxFrames": 5,
        "captureEveryNFrames": 2,
        "parties": ["BJP", "CONGRESS", "AAP", "NOTA"]
    })


@app.route("/api/register", methods=["POST"])
def register_face():
    payload = request.get_json(silent=True) or {}
    aadhar = payload.get("aadhar")
    name = payload.get("name", "")
    images = payload.get("images", [])
    if not aadhar or not images:
        return jsonify({"error": "aadhar and images are required"}), 400

    ensure_data_dir()

    # Enforce max 5 frames for storage efficiency
    collected = []
    for data_url in images[:5]:
        img = decode_base64_image(data_url)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            crop_img = img[y:y+h, x:x+w]
            resized_img = cv2.resize(crop_img, (50, 50))
            collected.append(resized_img)
            break
    if not collected:
        return jsonify({"error": "no faces detected"}), 400

    new_faces = np.asarray(collected)
    new_faces = new_faces.reshape((len(collected), -1))

    # Duplicate-check: if any incoming face matches an existing different aadhar, block
    existing_faces, existing_labels = load_training_data()
    if existing_faces is not None and existing_labels is not None and len(existing_labels) > 0:
        knn = build_classifier()
        if knn is not None:
            preds = knn.predict(new_faces)
            # If any predicted label belongs to a different aadhar, reject
            for pred in preds:
                if str(pred) != str(aadhar):
                    return jsonify({
                        "error": "duplicate_face",
                        "message": "This face appears to be already registered to a different ID.",
                        "registered_to": str(pred)
                    }), 409

    # Save/Update profile name for this aadhar
    profiles = load_profiles()
    if name:
        profiles[str(aadhar)] = {"name": str(name)}
        save_profiles(profiles)

    if not os.path.exists(NAMES_PATH):
        names = [aadhar] * len(collected)
        with open(NAMES_PATH, "wb") as f:
            pickle.dump(names, f)
    else:
        with open(NAMES_PATH, "rb") as f:
            names = pickle.load(f)
        names = names + [aadhar] * len(collected)
        with open(NAMES_PATH, "wb") as f:
            pickle.dump(names, f)

    if not os.path.exists(FACES_PATH):
        with open(FACES_PATH, "wb") as f:
            pickle.dump(new_faces, f)
    else:
        with open(FACES_PATH, "rb") as f:
            faces = pickle.load(f)
        faces = np.append(faces, new_faces, axis=0)
        with open(FACES_PATH, "wb") as f:
            pickle.dump(faces, f)

    append_audit("register", {"aadhar": str(aadhar), "frames": len(collected)})
    return jsonify({"ok": True, "added": len(collected)})


@app.route("/api/predict", methods=["POST"])
def predict_face():
    payload = request.get_json(silent=True) or {}
    image = payload.get("image")
    if not image:
        return jsonify({"error": "image is required"}), 400

    knn = build_classifier()
    if knn is None:
        return jsonify({"error": "no training data"}), 400

    img = decode_base64_image(image)
    if img is None:
        return jsonify({"error": "bad image"}), 400

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return jsonify({"error": "no face detected"}), 400
    (x, y, w, h) = faces[0]
    crop_img = img[y:y+h, x:x+w]
    resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
    output = knn.predict(resized_img)
    return jsonify({"label": str(output[0])})


def has_already_voted(voter_id: str) -> bool:
    if not os.path.exists(VOTES_CSV):
        return False
    try:
        with open(VOTES_CSV, "r", newline="") as f:
            r = csv.reader(f)
            for row in r:
                if row and row[0] == voter_id:
                    return True
    except Exception:
        return False
    return False


@app.route("/api/vote", methods=["POST"])
def cast_vote():
    payload = request.get_json(silent=True) or {}
    voter = payload.get("voter")
    party = payload.get("party")
    if not voter or not party:
        return jsonify({"error": "voter and party are required"}), 400

    if has_already_voted(voter):
        return jsonify({"error": "already_voted"}), 409

    exists = os.path.exists(VOTES_CSV)
    ts = time.time()
    date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
    timestamp = datetime.fromtimestamp(ts).strftime("%H:%M-%S")

    with open(VOTES_CSV, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["NAME", "VOTE", "DATE", "TIME"])
        w.writerow([voter, party, date, timestamp])
    append_audit("vote_cast", {"voter": str(voter), "party": str(party), "date": date, "time": timestamp})
    return jsonify({"ok": True})


@app.route("/api/reset-faces", methods=["POST"])
def reset_faces():
    # Optional simple guard
    if not require_admin():
        return jsonify({"error": "unauthorized"}), 401
    payload = request.get_json(silent=True) or {}
    confirm = str(payload.get("confirm", "")).lower()
    if confirm not in ("yes", "true", "confirm"):
        return jsonify({"error": "confirmation_required"}), 400

    ensure_data_dir()
    removed = []
    for path in (FACES_PATH, NAMES_PATH):
        try:
            if os.path.exists(path):
                os.remove(path)
                removed.append(os.path.basename(path))
        except Exception as e:
            return jsonify({"error": "delete_failed", "detail": str(e)}), 500
    append_audit("reset_faces", {"removed": removed})
    return jsonify({"ok": True, "removed": removed})


@app.route("/api/admin/export", methods=["GET"])
def admin_export():
    if not require_admin():
        return jsonify({"error": "unauthorized"}), 401
    ensure_data_dir()
    wb = Workbook()

    # Sheet 1: Registered Faces (counts per aadhar)
    ws1 = wb.active
    ws1.title = "RegisteredFaces"
    ws1.append(["AADHAR", "COUNT_FRAMES"])
    faces, labels = load_training_data()
    if labels is not None and faces is not None:
        # Each row in faces is a frame; labels aligns 1:1
        counts = {}
        for lab in labels:
            counts[lab] = counts.get(lab, 0) + 1
        for aadhar, cnt in counts.items():
            ws1.append([aadhar, cnt])

    # Sheet 2: Votes
    ws2 = wb.create_sheet("Votes")
    ws2.append(["NAME", "VOTE", "DATE", "TIME"])
    if os.path.exists(VOTES_CSV):
        try:
            with open(VOTES_CSV, "r", newline="") as f:
                r = csv.reader(f)
                for i, row in enumerate(r):
                    if i == 0:
                        # assume header present; skip duplicating
                        continue
                    ws2.append(row)
        except Exception:
            pass

    # Sheet 3: Summary
    ws3 = wb.create_sheet("Summary")
    total_voters = 0
    votes_by_party = {}
    if os.path.exists(VOTES_CSV):
        with open(VOTES_CSV, "r", newline="") as f:
            r = csv.reader(f)
            header_seen = False
            for row in r:
                if not row:
                    continue
                if not header_seen and row[0].upper() == "NAME":
                    header_seen = True
                    continue
                total_voters += 1
                party = row[1] if len(row) > 1 else ""
                votes_by_party[party] = votes_by_party.get(party, 0) + 1
    ws3.append(["Metric", "Value"])
    ws3.append(["Total Registered IDs", len(set(labels)) if labels else 0])
    ws3.append(["Total Votes Cast", total_voters])
    ws3.append(["--", "--"])
    ws3.append(["Votes by Party", "Count"])
    for party, cnt in votes_by_party.items():
        ws3.append([party, cnt])

    # Save to bytes and send
    from flask import send_file
    bio = io.BytesIO()
    wb.save(bio)
    bio.seek(0)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"voting_export_{ts}.xlsx"
    append_audit("export")
    return send_file(bio, as_attachment=True, download_name=filename, mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')


@app.route("/api/admin/stats", methods=["GET"])
def admin_stats():
    if not require_admin():
        return jsonify({"error": "unauthorized"}), 401
    # Aggregate stats for dashboard
    faces, labels = load_training_data()
    total_registered_ids = len(set(labels)) if labels else 0

    total_votes = 0
    votes_by_party = {}
    recent = []
    if os.path.exists(VOTES_CSV):
        with open(VOTES_CSV, "r", newline="") as f:
            r = list(csv.reader(f))
            # remove header if present
            rows = r[1:] if r and r[0] and r[0][0].upper() == "NAME" else r
            for row in rows:
                if not row:
                    continue
                total_votes += 1
                party = row[1] if len(row) > 1 else ""
                votes_by_party[party] = votes_by_party.get(party, 0) + 1
            for row in rows[-10:]:
                if len(row) >= 4:
                    recent.append({"name": row[0], "vote": row[1], "date": row[2], "time": row[3]})

    return jsonify({
        "totalRegistered": total_registered_ids,
        "totalVotes": total_votes,
        "votesByParty": votes_by_party,
        "recent": recent
    })


@app.route("/api/admin/votes", methods=["GET"])
def admin_list_votes():
    if not require_admin():
        return jsonify({"error": "unauthorized"}), 401
    rows = []
    if os.path.exists(VOTES_CSV):
        with open(VOTES_CSV, "r", newline="") as f:
            r = list(csv.reader(f))
            data_rows = r[1:] if r and r[0] and r[0][0].upper() == "NAME" else r
            for i, row in enumerate(data_rows):
                if len(row) >= 4:
                    rows.append({"index": i, "name": row[0], "vote": row[1], "date": row[2], "time": row[3]})
    return jsonify({"rows": rows})


@app.route("/api/admin/votes/<int:index>", methods=["DELETE"])
def admin_delete_vote(index: int):
    if not require_admin():
        return jsonify({"error": "unauthorized"}), 401
    if not os.path.exists(VOTES_CSV):
        return jsonify({"error": "not_found"}), 404
    with open(VOTES_CSV, "r", newline="") as f:
        r = list(csv.reader(f))
    header = ["NAME", "VOTE", "DATE", "TIME"]
    data_rows = r[1:] if r and r[0] and r[0][0].upper() == "NAME" else r
    if index < 0 or index >= len(data_rows):
        return jsonify({"error": "index_out_of_range"}), 400
    del data_rows[index]
    with open(VOTES_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for row in data_rows:
            w.writerow(row)
    append_audit("delete_vote", {"index": index})
    return jsonify({"ok": True})


@app.route("/api/admin/reset-votes", methods=["POST"])
def admin_reset_votes():
    if not require_admin():
        return jsonify({"error": "unauthorized"}), 401
    payload = request.get_json(silent=True) or {}
    confirm = str(payload.get("confirm", "")).lower()
    if confirm not in ("yes", "true", "confirm"):
        return jsonify({"error": "confirmation_required"}), 400
    header = ["NAME", "VOTE", "DATE", "TIME"]
    with open(VOTES_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
    append_audit("reset_votes")
    return jsonify({"ok": True})


@app.route("/api/admin/login", methods=["POST"])
def admin_login():
    data = request.get_json(silent=True) or {}
    username = str(data.get("username", ""))
    password = str(data.get("password", ""))
    if USERS.get(username) == password:
        session["admin_user"] = username
        append_audit("admin_login", {"user": username})
        return jsonify({"ok": True, "user": username})
    return jsonify({"error": "invalid_credentials"}), 401


@app.route("/api/admin/logout", methods=["POST"])
def admin_logout():
    session.pop("admin_user", None)
    append_audit("admin_logout")
    return jsonify({"ok": True})


@app.route("/api/admin/me", methods=["GET"])
def admin_me():
    user = session.get("admin_user")
    return jsonify({"user": user})


@app.route("/api/admin/audit", methods=["GET"])
def admin_audit():
    if not require_admin():
        return jsonify({"error": "unauthorized"}), 401
    ensure_data_dir()
    rows = []
    if os.path.exists(AUDIT_LOG):
        try:
            import json
            with open(AUDIT_LOG, "r", encoding="utf-8") as f:
                for line in f.readlines()[-200:]:
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        continue
        except Exception:
            pass
    return jsonify({"rows": rows[-100:]})


def compute_votes_checksum() -> str:
    import hashlib
    h = hashlib.sha256()
    if not os.path.exists(VOTES_CSV):
        return h.hexdigest()
    with open(VOTES_CSV, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


@app.route("/api/integrity", methods=["GET"])
def integrity():
    checksum = compute_votes_checksum()
    return jsonify({"votesChecksum": checksum})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)


