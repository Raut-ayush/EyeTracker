import json
import os


PROFILE_DIR = "profiles"


class ProfileManager:
    def __init__(self):
        os.makedirs(PROFILE_DIR, exist_ok=True)

    def save_profile(self, name, calibration_data):
        filename = f"{name.strip()}.json"
        path = os.path.join(PROFILE_DIR, filename)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(calibration_data, f, indent=4)

    def load_profile(self, name):
        filename = f"{name.strip()}.json"
        path = os.path.join(PROFILE_DIR, filename)

        if not os.path.exists(path):
            return None

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def list_profiles(self):
        if not os.path.exists(PROFILE_DIR):
            return []

        return sorted(
            os.path.splitext(name)[0]
            for name in os.listdir(PROFILE_DIR)
            if name.lower().endswith(".json")
        )