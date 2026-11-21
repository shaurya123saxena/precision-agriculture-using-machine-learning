import string
import bcrypt
from functools import wraps
from flask import Flask, redirect, render_template, url_for, request, Markup, abort
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user, current_user
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import InputRequired, Length, ValidationError
from flask_wtf import FlaskForm
from flask_bcrypt import Bcrypt
from datetime import datetime
import requests
import numpy as np
import pandas as pd
import config
import pickle
import io
import torch
from torchvision import transforms
from PIL import Image
from utils.model import ResNet9
from utils.fertilizer import fertilizer_dic
from utils.disease import disease_dic

# -------------------------LOADING THE TRAINED MODELS -----------------------------------------------

# Loading crop recommendation model (trained by train_model.py)
CROP_FEATURES = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
DEFAULT_TEMPERATURE, DEFAULT_HUMIDITY = 25, 60  # used only when live weather is unavailable
crop_recommendation_model_path = 'models/new_model.pkl'
crop_recommendation_model = pickle.load(
    open(crop_recommendation_model_path, 'rb'))

# Loading plant disease classification model

disease_classes = ['Apple___Apple_scab',
                   'Apple___Black_rot',
                   'Apple___Cedar_apple_rust',
                   'Apple___healthy',
                   'Blueberry___healthy',
                   'Cherry_(including_sour)___Powdery_mildew',
                   'Cherry_(including_sour)___healthy',
                   'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                   'Corn_(maize)___Common_rust_',
                   'Corn_(maize)___Northern_Leaf_Blight',
                   'Corn_(maize)___healthy',
                   'Grape___Black_rot',
                   'Grape___Esca_(Black_Measles)',
                   'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                   'Grape___healthy',
                   'Orange___Haunglongbing_(Citrus_greening)',
                   'Peach___Bacterial_spot',
                   'Peach___healthy',
                   'Pepper,_bell___Bacterial_spot',
                   'Pepper,_bell___healthy',
                   'Potato___Early_blight',
                   'Potato___Late_blight',
                   'Potato___healthy',
                   'Raspberry___healthy',
                   'Soybean___healthy',
                   'Squash___Powdery_mildew',
                   'Strawberry___Leaf_scorch',
                   'Strawberry___healthy',
                   'Tomato___Bacterial_spot',
                   'Tomato___Early_blight',
                   'Tomato___Late_blight',
                   'Tomato___Leaf_Mold',
                   'Tomato___Septoria_leaf_spot',
                   'Tomato___Spider_mites Two-spotted_spider_mite',
                   'Tomato___Target_Spot',
                   'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                   'Tomato___Tomato_mosaic_virus',
                   'Tomato___healthy']

# disease prediction
disease_model_path = 'models/plant_disease_model.pth'
disease_model = ResNet9(3, len(disease_classes))
disease_model.load_state_dict(torch.load(
    disease_model_path, map_location=torch.device('cpu')))
disease_model.eval()



def weather_fetch(city_name):
    """Return (temperature in °C, humidity %) for a city, or None if live weather is unavailable."""
    if not config.weather_api_key or not city_name:
        return None
    try:
        response = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={"q": city_name, "appid": config.weather_api_key},
            timeout=5,
        )
        x = response.json()
    except (requests.RequestException, ValueError):
        return None
    if response.status_code != 200 or "main" not in x:
        return None
    return round(x["main"]["temp"] - 273.15, 2), x["main"]["humidity"]


def predict_image(img, model=disease_model):
    """Return (disease class, confidence %) for an uploaded leaf image.

    The ResNet9 was trained on 256x256 images scaled to [0, 1] with no extra
    normalization, so inference must use the same transform.
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
    ])
    image = Image.open(io.BytesIO(img)).convert("RGB")
    img_u = torch.unsqueeze(transform(image), 0)

    with torch.no_grad():
        probs = torch.nn.functional.softmax(model(img_u)[0], dim=0)

    top_prob, top_idx = torch.max(probs, dim=0)
    return disease_classes[top_idx.item()], round(top_prob.item() * 100, 1)


app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = config.database_url
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SECRET_KEY"] = config.secret_key
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)



login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"



@login_manager.user_loader
def load_user(session_id):
    # Session ids carry the role ("user:3" / "admin:1") so an admin session can
    # never be loaded as the regular user who happens to share the same numeric id.
    role, _, raw_id = session_id.rpartition(":")
    if not raw_id.isdigit():
        return None
    model = UserAdmin if role == "admin" else User
    return model.query.get(int(raw_id))

class User(db.Model,UserMixin):
    id = db.Column(db.Integer,primary_key=True)
    username = db.Column(db.String(20), nullable=False, unique=True)
    password = db.Column(db.String(80), nullable=False)
    predictions = db.relationship("Prediction", backref="user", lazy="dynamic", cascade="all, delete-orphan")

    def get_id(self):
        return f"user:{self.id}"

class UserAdmin(db.Model,UserMixin):
    id = db.Column(db.Integer,primary_key=True)
    username = db.Column(db.String(20), nullable=False, unique=True)
    password = db.Column(db.String(80), nullable=False)

    def get_id(self):
        return f"admin:{self.id}"

class Prediction(db.Model):
    """One crop, fertilizer or disease result, owned by the user who asked for it."""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False, index=True)
    prediction_type = db.Column(db.String(20), nullable=False)  # crop / fertilizer / disease
    result = db.Column(db.String(120), nullable=False)
    confidence = db.Column(db.Float)                              # percent, if the model gives one
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False, index=True)


def is_admin():
    return current_user.is_authenticated and isinstance(current_user._get_current_object(), UserAdmin)


def admin_required(view):
    @wraps(view)
    def wrapper(*args, **kwargs):
        if not current_user.is_authenticated:
            return login_manager.unauthorized()
        if not is_admin():
            abort(403)
        return view(*args, **kwargs)
    return wrapper


def record_prediction(kind, result, confidence=None):
    """Save a result to the signed-in user's history (admins don't have one)."""
    if is_admin():
        return
    db.session.add(Prediction(user_id=current_user.id, prediction_type=kind, result=str(result)[:120],
                              confidence=None if confidence is None else round(float(confidence), 2)))
    db.session.commit()

class RegisterForm(FlaskForm):
    username=StringField(validators=[InputRequired(),Length(min=5,max=20)],render_kw={"placeholder":"username"})
    password=PasswordField(validators=[InputRequired(),Length(min=5,max=20)],render_kw={"placeholder":"password"})
    submit = SubmitField("Register")

    def validate_username(self, username):
        existing_user_username = User.query.filter_by(username=username.data).first()
        if existing_user_username:
            raise ValidationError("That username already exist. please choose different one.")

class LoginForm(FlaskForm):
    username=StringField(validators=[InputRequired(),Length(min=5,max=20)],render_kw={"placeholder":"username"})
    password=PasswordField(validators=[InputRequired(),Length(min=5,max=20)],render_kw={"placeholder":"password"})
    submit = SubmitField("Login")


class ContactUs(db.Model):
    sno = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    email = db.Column(db.String(500), nullable=False)
    text = db.Column(db.String(900), nullable=False)
    date_created = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self) -> str:
        return f"{self.sno} - {self.title}"


@app.route("/")
def hello_world():
    return render_template("index.html")
    

@app.route("/aboutus")
def aboutus():
    return render_template("aboutus.html")

@app.route("/contact", methods=['GET', 'POST'])
def contact():
    if request.method=='POST':
        name = request.form['name']
        email = request.form['email']
        text = request.form['text']
        contacts = ContactUs(name=name, email=email, text=text)
        db.session.add(contacts)
        db.session.commit()
    
    return render_template("contact.html")

@app.route("/login", methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if current_user.is_authenticated:
         return redirect(url_for('dashboard'))

    elif form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user:
            if bcrypt.check_password_hash(user.password,form.password.data):
                login_user(user)
                return redirect(url_for('dashboard'))

    return render_template("login.html", form=form)

@ app.route('/dashboard',methods=['GET', 'POST'])
@login_required
def dashboard():
    if is_admin():
        return redirect(url_for('admindashboard'))
    title = 'dashboard'
    return render_template('dashboard.html',title=title)

@app.route('/history')
@login_required
def history():
    if is_admin():
        return redirect(url_for('admindashboard'))
    page = request.args.get('page', 1, type=int)
    # Only the signed-in user's own rows: the filter on user_id is the access control.
    records = (Prediction.query.filter_by(user_id=current_user.id)
               .order_by(Prediction.created_at.desc(), Prediction.id.desc())
               .paginate(page=page, per_page=10, error_out=False))
    return render_template('history.html', records=records, title='- Prediction History')

@ app.route('/logout',methods=['GET', 'POST'])
@login_required
def logout():
    logout_user()
    return redirect(url_for('hello_world'))


@app.route("/signup",methods=['GET', 'POST'])
def signup():
    form = RegisterForm()

    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data)
        new_user = User(username=form.username.data, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))


    return render_template("signup.html", form=form)

@ app.route('/crop-recommend')
@login_required
def crop_recommend():
    title = 'crop-recommend - Crop Recommendation'
    return render_template('crop.html', title=title)

@ app.route('/fertilizer')
@login_required
def fertilizer_recommendation():
    title = '- Fertilizer Suggestion'
    return render_template('fertilizer.html', title=title)

@app.route('/disease-predict', methods=['GET', 'POST'])
@login_required
def disease_prediction():
    title = '- Disease Detection'

    if request.method == 'POST':
        file = request.files.get('file')
        if not file or not file.filename:
            return render_template('disease.html', title=title)

        try:
            disease, confidence = predict_image(file.read())
        except (OSError, ValueError):
            # Not a readable image file
            return render_template('try_again.html', title=title)

        record_prediction('disease', disease, confidence)
        prediction = Markup(str(disease_dic[disease]))
        return render_template(
            'disease-result.html',
            prediction=prediction,
            confidence=confidence,
            title=title,
        )

    return render_template('disease.html', title=title)


# ===============================================================================================

# RENDER PREDICTION PAGES

# render crop recommendation result page

def parse_range(value):
    value = value.strip()

    if '-' in value:
        try:
            low, high = value.split('-')
            return (float(low) + float(high)) / 2
        except:
            return None
    else:
        try:
            return float(value)
        except:
            return None


@app.route('/crop-predict', methods=['POST'])
@login_required
def crop_prediction():
    title = '- Crop Recommendation'

    if request.method == 'POST':

        # ✅ Use range parsing
        N = parse_range(request.form['nitrogen'])
        P = parse_range(request.form['phosphorous'])
        K = parse_range(request.form['pottasium'])
        ph = parse_range(request.form['ph'])
        rainfall = parse_range(request.form['rainfall'])
        city = request.form.get("city")

        # ✅ Validation (VERY IMPORTANT)
        if None in [N, P, K, ph, rainfall]:
            return "❌ Invalid input. Use numbers or ranges like 10-20"

        weather = weather_fetch(city)
        if weather is None:
            # No live weather (no API key, unknown city or network error):
            # fall back to typical values and say so on the result page.
            temperature, humidity = DEFAULT_TEMPERATURE, DEFAULT_HUMIDITY
            weather_note = (f"Live weather for {city or 'your city'} was unavailable, so typical values "
                            f"were used ({temperature}°C, {humidity}% humidity).")
        else:
            temperature, humidity = weather
            weather_note = f"Live weather for {city}: {temperature}°C, {humidity}% humidity."

        data = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]],
                            columns=CROP_FEATURES)

        probs = crop_recommendation_model.predict_proba(data)[0]
        classes = crop_recommendation_model.classes_
        top3_idx = probs.argsort()[-3:][::-1]
        top3 = [(classes[i], round(probs[i] * 100, 2)) for i in top3_idx]
        final_prediction, confidence = top3[0]
        record_prediction('crop', final_prediction, confidence)

        return render_template(
            'crop-result.html',
            prediction=final_prediction,
            confidence=confidence,
            top3=top3,
            weather_note=weather_note,
            title=title
        )

# render fertilizer recommendation result page
FERTILIZER_RESULT = {"NHigh": "nitrogen high", "Nlow": "nitrogen low", "PHigh": "phosphorus high",
                     "Plow": "phosphorus low", "KHigh": "potassium high", "Klow": "potassium low"}


@ app.route('/fertilizer-predict', methods=['POST'])
@login_required
def fert_recommend():
    title = '- Fertilizer Suggestion'

    crop_name = str(request.form['cropname'])
    N = int(request.form['nitrogen'])
    P = int(request.form['phosphorous'])
    K = int(request.form['pottasium'])
    # ph = float(request.form['ph'])

    df = pd.read_csv('Data/fertilizer.csv')

    nr = df[df['Crop'] == crop_name]['N'].iloc[0]
    pr = df[df['Crop'] == crop_name]['P'].iloc[0]
    kr = df[df['Crop'] == crop_name]['K'].iloc[0]

    n = nr - N
    p = pr - P
    k = kr - K
    temp = {abs(n): "N", abs(p): "P", abs(k): "K"}
    max_value = temp[max(temp.keys())]
    if max_value == "N":
        if n < 0:
            key = 'NHigh'
        else:
            key = "Nlow"
    elif max_value == "P":
        if p < 0:
            key = 'PHigh'
        else:
            key = "Plow"
    else:
        if k < 0:
            key = 'KHigh'
        else:
            key = "Klow"

    record_prediction('fertilizer', f"{crop_name}: {FERTILIZER_RESULT[key]}")
    response = Markup(str(fertilizer_dic[key]))

    return render_template('fertilizer-result.html', recommendation=response, title=title)


@app.route("/AdminLogin", methods=['GET', 'POST'])
def AdminLogin():

    form = LoginForm()
    if is_admin():
         return redirect(url_for('admindashboard'))

    elif form.validate_on_submit():
        user = UserAdmin.query.filter_by(username=form.username.data).first()
        if user:
            if bcrypt.check_password_hash(user.password,form.password.data):
                login_user(user)
                return redirect(url_for('admindashboard'))

    return render_template("adminlogin.html", form=form)


    # return render_template("adminlogin.html")

@app.route("/admindashboard")
@admin_required
def admindashboard():
    alltodo = ContactUs.query.all()
    alluser = User.query.all()
    return render_template("admindashboard.html",alltodo=alltodo, alluser=alluser)

@app.route("/reg",methods=['GET', 'POST'])
def reg():
    # Anyone could create an admin account here before. Now only a signed-in admin can,
    # except for the very first admin when none exists yet.
    if UserAdmin.query.count() > 0 and not is_admin():
        abort(403)
    form = RegisterForm()

    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data)
        new_user = UserAdmin(username=form.username.data, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('AdminLogin'))

    return render_template("reg.html", form=form)


with app.app_context():
    db.create_all()  # creates database.db and its tables on first run


if __name__ == "__main__":
    app.run(debug=config.debug, port=8000)
