from flask import Flask, render_template, request, session
import os
from datetime import datetime

app = Flask(__name__)

# ==========================
# CONFIGURATION
# ==========================

app.secret_key = "endopredict_secret_key"

UPLOAD_FOLDER = "static/uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


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

        if "image" not in request.files:
            return "No image uploaded"

        image = request.files["image"]

        if image.filename == "":
            return "No image selected"

        file_path = os.path.join(
            app.config["UPLOAD_FOLDER"],
            image.filename
        )

        image.save(file_path)

        # ==========================
        # DEMO PREDICTION
        # Replace with .pth Model
        # ==========================

        prediction = "Endometriosis"
        confidence = 92

        if confidence >= 80:
            risk_level = "High Risk"
        elif confidence >= 50:
            risk_level = "Moderate Risk"
        else:
            risk_level = "Low Risk"

        image_path = "/" + file_path.replace("\\", "/")

        model_name = "EndoPredict AI v1.0"

        current_date = datetime.now().strftime("%d-%m-%Y %H:%M")

        report_id = "EPR-" + datetime.now().strftime("%Y%m%d%H%M%S")

        patient_data = {
            "Analysis Type": "Ultrasound Image Analysis"
        }

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
            image_path=image_path,
            model_name=model_name,
            current_date=current_date
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