from flask import Flask, render_template, request, session
from datetime import datetime

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

app = Flask(__name__)
app.secret_key = "endopredict_secret_key"

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
    transforms.Resize((64, 64)),
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
        menstrual_irregularity = int(request.form["menstrual_irregularity"])
        chronic_pain = float(request.form["chronic_pain"])
        hormone_abnormality = int(request.form["hormone_abnormality"])
        infertility = int(request.form["infertility"])

        bmi = float(request.form["bmi"])
        height = float(request.form["height"])
        weight = float(request.form["weight"])

        bp_systolic = float(request.form["bp_systolic"])
        bp_diastolic = float(request.form["bp_diastolic"])

        estrogen = float(request.form["estrogen"])
        progesterone = float(request.form["progesterone"])

        # =====================================
        # CLINICAL RISK SCORING
        # =====================================

        score = 0

        if menstrual_irregularity == 1:
            score += 15

        if chronic_pain >= 7:
            score += 20
        elif chronic_pain >= 4:
            score += 10

        if hormone_abnormality == 1:
            score += 15

        if infertility == 1:
            score += 15

        if bmi >= 30:
            score += 10
        elif bmi >= 25:
            score += 5

        if estrogen > 250:
            score += 10

        if progesterone < 10:
            score += 10

        if age >= 35:
            score += 5

        confidence = min(score, 100)

        # =====================================
        # PREDICTION
        # =====================================

        if confidence >= 60:

            prediction = "Endometriosis Suspected"

            if confidence >= 80:
                risk_level = "High Risk"
            else:
                risk_level = "Moderate Risk"

            recommendations = [
                "Consult a gynecologist",
                "Perform detailed ultrasound examination",
                "Hormonal assessment recommended",
                "Schedule follow-up evaluation"
            ]

        else:

            prediction = "Low Endometriosis Risk"

            risk_level = "Low Risk"

            recommendations = [
                "Continue routine health monitoring",
                "Maintain healthy lifestyle",
                "Seek medical advice if symptoms persist",
                "Annual gynecological checkup recommended"
            ]

        patient_data = {
            "Age": age,
            "Menstrual Irregularity": "Yes" if menstrual_irregularity else "No",
            "Chronic Pain Level": chronic_pain,
            "Hormone Abnormality": "Yes" if hormone_abnormality else "No",
            "Infertility": "Yes" if infertility else "No",
            "BMI": bmi,
            "Height": height,
            "Weight": weight,
            "BP Systolic": bp_systolic,
            "BP Diastolic": bp_diastolic,
            "Estrogen Level": estrogen,
            "Progesterone Level": progesterone
        }

        current_date = datetime.now().strftime("%d-%m-%Y %H:%M")

        report_id = "EPR-" + datetime.now().strftime("%Y%m%d%H%M%S")

        session["prediction"] = prediction
        session["confidence"] = confidence
        session["risk_level"] = risk_level
        session["patient_data"] = patient_data
        session["recommendations"] = recommendations
        session["report_id"] = report_id
        session["current_date"] = current_date
        session["model_name"] = "EndoPredict AI Clinical Assessment"

        return render_template(
            "result.html",
            prediction=prediction,
            confidence=confidence,
            risk_level=risk_level,
            patient_data=patient_data,
            recommendations=recommendations,
            current_date=current_date,
            model_name="EndoPredict AI Clinical Assessment",
            image_path=None
        )

    except Exception as e:
        return f"Error: {str(e)}"

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
model = MultiModalModel()

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
        if predicted_class == 1:

            prediction = "Endometriosis Detected"

            if confidence >= 80:
                risk_level = "High Risk"
            elif confidence >= 60:
                risk_level = "Moderate Risk"
            else:
                risk_level = "Low Risk"

            if confidence >= 80:

                recommendations = [
                    "Consult a gynecologist immediately",
                    "Perform MRI evaluation",
                    "Hormonal assessment recommended",
                    "Follow-up within 2 weeks"
                ]

            else:

                recommendations = [
                    "Schedule specialist consultation",
                    "Repeat ultrasound examination",
                    "Monitor symptoms carefully"
                ]

        else:

            prediction = "No Endometriosis Detected"

            risk_level = "No Significant Risk"

            recommendations = [
                "No significant indicators detected",
                "Maintain regular health checkups",
                "Continue healthy lifestyle",
                "Consult physician if symptoms persist"
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
        recommendations=session.get("recommendations", []),
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
    app.run(host="0.0.0.0", port=5000)