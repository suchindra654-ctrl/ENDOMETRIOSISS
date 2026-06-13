from flask import Flask, render_template, request, session
from datetime import datetime

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

app = Flask(__name__)
device = torch.device("cpu")

class MultiModalModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.cnn = models.resnet18(weights=None)

        self.cnn.fc = nn.Linear(512, 128)

        self.num_fc = nn.Sequential(
            nn.Linear(12, 64),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(192, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, image, numerical):

        img_features = self.cnn(image)

        num_features = self.num_fc(numerical)

        combined = torch.cat(
            [img_features, num_features],
            dim=1
        )

        return self.classifier(combined)

model = MultiModalModel()

model.load_state_dict(
    torch.load(
        "models/endometriosis_model.pth",
        map_location=device
    )
)

model.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# ==========================
# CONFIGURATION
# ==========================

app.secret_key = "endopredict_secret_key"




# ==========================
# HOME PAGE
# ==========================

@app.route("/")
def dashboard():
    return render_template("dashboard.html")


# ==========================
# NUMERICAL PAGE
# ==========================

@app.route("/numerical")
def numerical():
    return render_template("numerical.html")


# ==========================
# ULTRASOUND PAGE
# ==========================

@app.route("/ultrasound")
def ultrasound():
    return render_template("ultrasound.html")


# ==========================
# NUMERICAL PREDICTION
# ==========================

@app.route("/predict_numerical", methods=["POST"])
def predict_numerical():

    try:

        age = float(request.form["age"])
        bmi = float(request.form["bmi"])
        cycle_length = float(request.form["cycle_length"])
        pain_score = float(request.form["pain_score"])
        hormone_level = float(request.form["hormone_level"])

        previous_diagnosis = int(request.form["previous_diagnosis"])
        family_history = int(request.form["family_history"])
        pelvic_pain = int(request.form["pelvic_pain"])
        infertility = int(request.form["infertility"])
        fatigue = int(request.form["fatigue"])

        # ==========================
        # DEMO RISK CALCULATION
        # Replace with ML Model
        # ==========================

        score = 0

        if pain_score >= 6:
            score += 20

        if bmi >= 25:
            score += 10

        if family_history == 1:
            score += 20

        if pelvic_pain == 1:
            score += 20

        if infertility == 1:
            score += 15

        if previous_diagnosis == 1:
            score += 15

        confidence = min(score, 100)

        if confidence >= 80:
            risk_level = "High Risk"
        elif confidence >= 50:
            risk_level = "Moderate Risk"
        else:
            risk_level = "Low Risk"

        if confidence >= 50:
            prediction = "Endometriosis"
        else:
            prediction = "No Endometriosis"

        patient_data = {
            "Age": age,
            "BMI": bmi,
            "Cycle Length": cycle_length,
            "Pain Score": pain_score,
            "Hormone Level": hormone_level,
            "Previous Diagnosis": "Yes" if previous_diagnosis else "No",
            "Family History": "Yes" if family_history else "No",
            "Pelvic Pain": "Yes" if pelvic_pain else "No",
            "Infertility": "Yes" if infertility else "No",
            "Fatigue": "Yes" if fatigue else "No"
        }

        model_name = "EndoPredict AI v1.0"

        current_date = datetime.now().strftime("%d-%m-%Y %H:%M")

        report_id = "EPR-" + datetime.now().strftime("%Y%m%d%H%M%S")

        # ==========================
        # SAVE TO SESSION
        # ==========================

        session["prediction"] = prediction
        session["confidence"] = confidence
        session["risk_level"] = risk_level
        session["model_name"] = model_name
        session["patient_data"] = patient_data
        session["current_date"] = current_date
        session["report_id"] = report_id

        return render_template(
            "result.html",
            prediction=prediction,
            confidence=confidence,
            risk_level=risk_level,
            patient_data=patient_data,
            image_path=None,
            model_name=model_name,
            current_date=current_date
        )

    except Exception as e:
        return f"Error: {str(e)}"


# ==========================
# ULTRASOUND PREDICTION
# ==========================

@app.route("/predict_ultrasound", methods=["POST"])
def predict_ultrasound():

    try:

        image = request.files["image"]

        image_pil = Image.open(
            image.stream
        ).convert("RGB")

        image_tensor = transform(
            image_pil
        ).unsqueeze(0)

        numerical_features = [

            float(request.form["age"]),
            float(request.form["menstrual_irregularity"]),
            float(request.form["chronic_pain"]),
            float(request.form["hormone_abnormality"]),
            float(request.form["infertility"]),
            float(request.form["bmi"]),
            float(request.form["height"]),
            float(request.form["weight"]),
            float(request.form["bp_systolic"]),
            float(request.form["bp_diastolic"]),
            float(request.form["estrogen"]),
            float(request.form["progesterone"])

        ]

        numerical_tensor = torch.tensor(
            [numerical_features],
            dtype=torch.float32
        )

        with torch.no_grad():

            outputs = model(
                image_tensor,
                numerical_tensor
            )

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            predicted_class = torch.argmax(
                probabilities,
                dim=1
            ).item()

            confidence = round(
                probabilities[
                    0,
                    predicted_class
                ].item() * 100,
                2
            )

        prediction = (
            "Endometriosis Detected"
            if predicted_class == 1
            else "No Endometriosis Detected"
        )

        if confidence >= 80:
            risk_level = "High Risk"
        elif confidence >= 50:
            risk_level = "Moderate Risk"
        else:
            risk_level = "Low Risk"

        if risk_level == "High Risk":

            recommendations = [
                "Consult a gynecologist immediately",
                "Perform MRI evaluation",
                "Hormonal assessment recommended",
                "Follow-up within 2 weeks"
            ]

        elif risk_level == "Moderate Risk":

            recommendations = [
                "Schedule specialist consultation",
                "Monitor symptoms",
                "Repeat ultrasound examination",
                "Maintain symptom diary"
            ]

        else:

            recommendations = [
                "Continue regular monitoring",
                "Maintain healthy lifestyle",
                "Annual gynecological check-up"
            ]

        patient_data = {
            "Age": numerical_features[0],
            "BMI": numerical_features[5],
            "Height": numerical_features[6],
            "Weight": numerical_features[7],
            "Estrogen": numerical_features[10],
            "Progesterone": numerical_features[11]
        }

        current_date = datetime.now().strftime(
            "%d-%m-%Y %H:%M"
        )

        report_id = "EPR-" + datetime.now().strftime(
            "%Y%m%d%H%M%S"
        )

        session["prediction"] = prediction
        session["confidence"] = confidence
        session["risk_level"] = risk_level
        session["recommendations"] = recommendations
        session["patient_data"] = patient_data
        session["report_id"] = report_id
        session["current_date"] = current_date
        session["model_name"] = "EndoPredict AI v1.0"

        return render_template(
            "result.html",
            prediction=prediction,
            confidence=confidence,
            risk_level=risk_level,
            patient_data=patient_data,
            recommendations=recommendations,
            current_date=current_date,
            model_name="EndoPredict AI v1.0",
            image_path=None
        )

    except Exception as e:
        return f"Error: {str(e)}"
# ==========================
# PROFESSIONAL REPORT PAGE
# ==========================

@app.route("/report")
def report():

    return render_template(
        "report.html",
        prediction=session.get("prediction", "N/A"),
        confidence=session.get("confidence", 0),
        risk_level=session.get("risk_level", "N/A"),
        model_name=session.get("model_name", "EndoPredict AI"),
        patient_data=session.get("patient_data", {}),
        current_date=session.get(
            "current_date",
            datetime.now().strftime("%d-%m-%Y %H:%M")
        ),
        report_id=session.get("report_id", "N/A")
    )


# ==========================
# RESULT PAGE
# ==========================

@app.route("/result")
def result():

    return render_template(
        "result.html",
        prediction="No Result",
        confidence=0,
        risk_level="Low Risk",
        patient_data={},
        image_path=None,
        model_name="EndoPredict AI v1.0",
        current_date=datetime.now().strftime("%d-%m-%Y %H:%M")
    )
@app.route("/reports")
def reports():

    return render_template(
        "reports.html",
        prediction=session.get("prediction", "No Report Available"),
        confidence=session.get("confidence", 0),
        risk_level=session.get("risk_level", "N/A"),
        report_id=session.get("report_id", "N/A"),
        current_date=session.get("current_date", "N/A")
    )

# ==========================
# MAIN
# ==========================

if __name__ == "__main__":

    app.run()