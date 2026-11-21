"""Access-control and prediction-history tests.

Run from the project root:  python -m unittest discover -s tests -v
Uses a throwaway SQLite file, so the real database.db is never touched.
"""
import io
import os
import sys
import tempfile
import unittest

_DB_DIR = tempfile.mkdtemp()
os.environ["DATABASE_URL"] = "sqlite:///" + os.path.join(_DB_DIR, "test.db")
os.environ.pop("WEATHER_API_KEY", None)  # never call the live weather API from tests
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as cultivai  # noqa: E402
from PIL import Image  # noqa: E402

CROP_FORM = {"nitrogen": "90", "phosphorous": "42", "pottasium": "43",
             "ph": "6.5", "rainfall": "200", "city": ""}


class AccessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cultivai.app.config.update(TESTING=True, WTF_CSRF_ENABLED=False)

    def setUp(self):
        with cultivai.app.app_context():
            cultivai.db.drop_all()
            cultivai.db.create_all()
        self.client = cultivai.app.test_client()

    # -- helpers ---------------------------------------------------------
    def make(self, model, username, password="secret123"):
        with cultivai.app.app_context():
            obj = model(username=username, password=cultivai.bcrypt.generate_password_hash(password))
            cultivai.db.session.add(obj)
            cultivai.db.session.commit()
            return obj.id

    def login(self, username, password="secret123", admin=False):
        url = "/AdminLogin" if admin else "/login"
        return self.client.post(url, data={"username": username, "password": password})

    def history_rows(self):
        with cultivai.app.app_context():
            return [(p.user.username, p.prediction_type, p.result) for p in cultivai.Prediction.query.all()]

    # -- per-user history ------------------------------------------------
    def test_each_user_sees_only_their_own_history(self):
        self.make(cultivai.User, "asha01")
        self.make(cultivai.User, "ravi01")

        self.login("asha01")
        self.assertEqual(self.client.post("/crop-predict", data=CROP_FORM).status_code, 200)
        page = self.client.get("/history").get_data(as_text=True)
        self.assertIn("Crop", page)
        self.client.get("/logout")

        self.login("ravi01")
        page = self.client.get("/history").get_data(as_text=True)
        self.assertIn("No predictions yet", page)

        rows = self.history_rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][:2], ("asha01", "crop"))

    def test_disease_and_fertilizer_results_are_recorded(self):
        self.make(cultivai.User, "asha01")
        self.login("asha01")
        img = io.BytesIO()
        Image.new("RGB", (300, 300), (60, 140, 60)).save(img, "JPEG")
        img.seek(0)
        self.client.post("/disease-predict", data={"file": (img, "leaf.jpg")},
                         content_type="multipart/form-data")
        self.client.post("/fertilizer-predict", data={"cropname": "rice", "nitrogen": "10",
                                                      "phosphorous": "40", "pottasium": "40"})
        kinds = sorted(kind for _, kind, _ in self.history_rows())
        self.assertEqual(kinds, ["disease", "fertilizer"])

    def test_history_requires_login(self):
        resp = self.client.get("/history")
        self.assertEqual(resp.status_code, 302)
        self.assertIn("/login", resp.headers["Location"])

    def test_fertilizer_prediction_requires_login(self):
        resp = self.client.post("/fertilizer-predict", data={"cropname": "rice", "nitrogen": "1",
                                                             "phosphorous": "1", "pottasium": "1"})
        self.assertEqual(resp.status_code, 302)

    # -- admin separation ------------------------------------------------
    def test_regular_user_cannot_open_admin_dashboard(self):
        self.make(cultivai.User, "asha01")
        self.login("asha01")
        self.assertEqual(self.client.get("/admindashboard").status_code, 403)

    def test_admin_session_is_never_loaded_as_a_user(self):
        user_id = self.make(cultivai.User, "asha01")
        admin_id = self.make(cultivai.UserAdmin, "boss01")
        self.assertEqual(user_id, admin_id)  # same numeric id in both tables: the risky case

        self.login("boss01", admin=True)
        self.assertEqual(self.client.get("/admindashboard").status_code, 200)
        resp = self.client.get("/history")  # must not show asha01's history
        self.assertEqual(resp.status_code, 302)
        self.assertIn("/admindashboard", resp.headers["Location"])

    def test_only_admins_can_register_admins_after_the_first(self):
        resp = self.client.post("/reg", data={"username": "first01", "password": "secret123"})
        self.assertEqual(resp.status_code, 302)  # first admin may bootstrap
        self.assertEqual(self.client.get("/reg").status_code, 403)  # anonymous now refused

        self.make(cultivai.User, "asha01")
        self.login("asha01")
        self.assertEqual(self.client.get("/reg").status_code, 403)  # regular user refused


if __name__ == "__main__":
    unittest.main()
