
import streamlit as st 
import altair as alt
import pandas as pd
import joblib 
from datetime import datetime
#Loads our saved machine-learning model


#st.set_page_config() must appear before other Streamlit commands such as st.title().
st.set_page_config(
    page_title="Job Placement Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)




model = joblib.load('placement_prediction_model.pkl')

#st.title("Job Placement Prediction")
#This displays the heading Job Placement Prediction at the top of your Streamlit web application.

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []


#st.title("🎓 Job Placement Prediction System")
#st.title("🎓 Job Placement Prediction Dashboard")

st.markdown(
    """
    <div style="
        background-image: linear-gradient(
            120deg,
            rgba(124, 58, 237, 0.82),
            rgba(236, 72, 153, 0.82)
        ), url('https://upload.wikimedia.org/wikipedia/commons/0/08/GUVI_logo.png');
        background-repeat: no-repeat;
        background-position: center, center;
        background-size: cover, 200px auto;
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        color: white;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    ">
        <h1 style="color: white; margin-bottom: 10px;">
            🎓 Job Placement Prediction Dashboard
        </h1>
        <p style="font-size: 18px; margin: 0;">
            Smart Career Insights | AI-Powered Placement Prediction
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    ### Intelligent Career Placement Analytics

    Enter the candidate's academic, technical, and employment details
    to estimate their placement outcome.
    """
)

st.divider()

#st.write("Enter candidate details below to predict the placement status.")
#It displays text, explanations, and other content on your Streamlit webpage. 

#col1, col2 = st.columns(2)

#with col1:
#    st.metric("🤖 ML Algorithm", "Random Forest")

#with col2:
 #   st.metric("📊 Input Features", "25")

#col3, col4 = st.columns(2)
#with col3:
 #   st.metric("🎯 Output Classes", "2")

#with col4:
 #   st.metric("📈 Model Accuracy", "89.55%")


col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div style="background:#F3E8FF; padding:20px;
    border-radius:12px; text-align:center; border-bottom:5px solid #9333EA;">
        <h3>🤖</h3>
        <h4 style="color:#6B21A8;">ML Algorithm</h4>
        <h2 style="color:#6B21A8;">Random Forest</h2>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="background:#DBEAFE; padding:20px;
    border-radius:12px; text-align:center; border-bottom:5px solid #2563EB;">
        <h3>📊</h3>
        <h4 style="color:#1D4ED8;">Input Features</h4>
        <h2 style="color:#1D4ED8;">25</h2>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style="background:#DCFCE7; padding:20px;
    border-radius:12px; text-align:center; border-bottom:5px solid #16A34A;">
        <h3>🎯</h3>
        <h4 style="color:#15803D;">Output Classes</h4>
        <h2 style="color:#15803D;">2</h2>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div style="background:#FCE7F3; padding:20px;
    border-radius:12px; text-align:center; border-bottom:5px solid #DB2777;">
        <h3>📈</h3>
        <h4 style="color:#BE185D;">Model Accuracy</h4>
        <h2 style="color:#BE185D;">89.55%</h2>
    </div>
    """, unsafe_allow_html=True)


st.divider()

#st.subheader("📊 Model Performance")

#st.write("Comparison of model evaluation metrics")

st.markdown("""
<h2 style="color:#7C3AED; text-align:center;">
    📊 Model Performance
</h2>
<p style="text-align:center; color:gray; font-size:16px;">
    Evaluation of our Machine Learning Model
</p>
""", unsafe_allow_html=True)

#performance_data = pd.DataFrame({
 #   "Metric": ["Precision", "Recall", "F1-Score", "Accuracy"],
  #  "Score": [81, 86, 83, 89.55]
#})

#st.bar_chart(
 #   performance_data,
  #  x="Metric",
   # y="Score"
#)


performance_data = pd.DataFrame({
    "Metric": ["Precision", "Recall", "F1-Score", "Accuracy"],
    "Score": [81, 86, 83, 89.55]
})

performance_chart = alt.Chart(performance_data).mark_bar(
    cornerRadius=8,
    height=25
).encode(
    x=alt.X(
        "Score:Q",
        scale=alt.Scale(domain=[0, 100]),
        title="Performance Score (%)"
    ),
    y=alt.Y(
        "Metric:N",
        sort=None,
        title=None
    ),
    color=alt.Color(
        "Metric:N",
        scale=alt.Scale(
            range=["#7C3AED", "#EC4899", "#06B6D4", "#F59E0B"]
        ),
        legend=None
    ),
    tooltip=["Metric", "Score"]
)

labels = alt.Chart(performance_data).mark_text(
    align="left",
    dx=8,
    fontSize=14,
    fontWeight="bold",
    color="#334155"
).encode(
    x=alt.X("Score:Q"),
    y=alt.Y("Metric:N", sort=None),
    text=alt.Text("Score:Q", format=".2f")
)

final_chart = (performance_chart + labels).properties(
    height=300,
    title="🎯 Model Performance Analysis"
)

st.altair_chart(final_chart, use_container_width=True)






perf1, perf2 = st.columns(2)

with perf1:
    st.metric("Precision (Placed)", "81%")
    st.metric("Recall (Placed)", "86%")

with perf2:
    st.metric("F1-Score (Placed)", "83%")
    st.metric("Accuracy", "89.55%")


st.divider()

st.subheader("🔍 Feature Importance Analysis")
#feature_names = model.named_steps['preprocessor'].get_feature_names_out()
feature_names = [
    name.replace("categorical__", "").replace("numerical__", "")
    for name in model.named_steps['preprocessor'].get_feature_names_out()
]


feature_importance = model.named_steps['classifier'].feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": feature_importance
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

importance_df = importance_df.head(10)

#st.bar_chart(importance_df,x="Feature",y="Importance",horizontal=True)
feature_chart = alt.Chart(importance_df).mark_bar(
    color="#2563EB"
).encode(
    x=alt.X("Importance:Q", title="Importance"),
    y=alt.Y("Feature:N", sort="-x", title="Feature"),
    tooltip=["Feature", "Importance"]
).properties(
    height=400
).configure_axis(
    labelFontWeight="bold",
    titleFontWeight="bold",
    labelFontSize=13,
    titleFontSize=14
)

st.altair_chart(feature_chart, use_container_width=True)

with st.sidebar:
    st.title("About This Project")

    st.write(
        "This application predicts whether a candidate "
        "is likely to be placed using machine learning."
    )

    st.divider()

    st.subheader("Model Information")

    st.write("**Algorithm:** Random Forest Classifier")
    st.write("**Model:** Hyperparameter Tuned")
    st.write("**Input Features:** 25")
    st.write("**Output Classes:**")
    st.write("- Placed")
    st.write("- Not Placed")

    st.divider()

    st.caption("Developed using Python, Pandas, Scikit-learn and Streamlit.")


st.header("Candidate Details") 
#It displays a section heading on the webpage. We will place all candidate input fields under this heading.
 
#with st.form("candidate_form"):
st.subheader("1. Personal Information")  

candidate_name = st.text_input("Candidate Name")
#dummy column 

age_years = st.number_input("Age (Years)", min_value=18, max_value=60, value=25) 
#age_years stores the age entered by the user.
#st.number_input() creates a numeric input box.
#min_value=18 means the minimum age is 18.
#max_value=60 means the maximum age is 60.
#   value=25 means the default age displayed is 25. 

gender = st.selectbox("Gender", ["Male", "Female"])


st.divider()



st.subheader("2. Academic Details")

#ssc_percentage = st.number_input("SSC Percentage", min_value=0.0, max_value=100.0, value=75.0) 

#hsc_percentage = st.number_input("HSC Percentage", min_value=0.0, max_value=100.0, value=75.0) 

#degree_percentage = st.number_input("Degree Percentage", min_value=0.0, max_value=100.0, value=75.0) 

#degree_specialization = st.selectbox("Degree Specialization", ["Computer Science", "Electronics", "Mechanical", "Information Technology", "Others"])


col1, col2 = st.columns(2)

with col1:
    ssc_percentage = st.number_input(
        "SSC Percentage",
        min_value=0.0,
        max_value=100.0,
        value=75.0
    )

    degree_percentage = st.number_input(
        "Degree Percentage",
        min_value=0.0,
        max_value=100.0,
        value=75.0
    )

with col2:
    hsc_percentage = st.number_input(
        "HSC Percentage",
        min_value=0.0,
        max_value=100.0,
        value=75.0
    )

    degree_specialization = st.selectbox(
        "Degree Specialization",
        [
            "Computer Science",
            "Electronics",
            "Mechanical",
            "Information Technology",
            "Others"
        ]
    )


st.divider()



st.subheader("3. Skills and Assessment")

#technical_score = st.number_input("Technical Score", min_value=0.0, max_value=100.0, value=50.0) 


# Numerical Inputs

#aptitude_score = st.number_input("Aptitude Score", min_value=0.0, max_value=100.0, value=50.0)

#communication_score = st.number_input("Communication Score", min_value=0.0, max_value=100.0, value=50.0)

#skills_match_percentage = st.number_input("Skills Match Percentage", min_value=0.0, max_value=100.0, value=50.0)

#internship_experience = st.selectbox("Internship Experience", ["Yes", "No"])

#job_role_match = st.selectbox("Job Role Match", ["Matched", "Not Matched", "Unknown"])

col1, col2 = st.columns(2)

with col1:
    technical_score = st.number_input(
        "Technical Score",
        min_value=0.0,
        max_value=100.0,
        value=50.0
    )

    communication_score = st.number_input(
        "Communication Score",
        min_value=0.0,
        max_value=100.0,
        value=50.0
    )

    internship_experience = st.selectbox(
        "Internship Experience",
        ["Yes", "No"]
    )

with col2:
    aptitude_score = st.number_input(
        "Aptitude Score",
        min_value=0.0,
        max_value=100.0,
        value=50.0
    )

    skills_match_percentage = st.number_input(
        "Skills Match Percentage",
        min_value=0.0,
        max_value=100.0,
        value=50.0
    )

    job_role_match = st.selectbox(
        "Job Role Match",
        ["Matched", "Not Matched", "Unknown"]
    )

st.divider()



st.subheader("4. Employment Details")

#certifications_count = st.number_input("Number of Certifications", min_value=0, max_value=50, value=0)

#years_of_experience = st.number_input("Years of Experience", min_value=0.0, max_value=50.0, value=0.0)

#previous_ctc_lpa = st.number_input("Previous CTC (LPA)", min_value=0.0, max_value=1000.0, value=0.0)

#expected_ctc_lpa = st.number_input("Expected CTC (LPA)", min_value=0.0, max_value=1000.0, value=5.0)

#notice_period_days = st.selectbox("Notice Period (Days)", [0, 15, 30, 60, 90])

#employment_gap_months = st.number_input("Employment Gap (Months)", min_value=0, max_value=240, value=0)

#company_tier = st.selectbox("Company Tier", ["Tier 1", "Tier 2", "Tier 3"])

#layoff_history = st.selectbox("Layoff History", ["Yes", "No", "Unknown"])

col1, col2 = st.columns(2)

with col1:
    certifications_count = st.number_input(
        "Number of Certifications",
        min_value=0,
        max_value=50,
        value=0
    )

    previous_ctc_lpa = st.number_input(
        "Previous CTC (LPA)",
        min_value=0.0,
        max_value=1000.0,
        value=0.0
    )

    notice_period_days = st.selectbox(
        "Notice Period (Days)",
        [0, 15, 30, 60, 90]
    )

    company_tier = st.selectbox(
        "Company Tier",
        ["Tier 1", "Tier 2", "Tier 3"]
    )

with col2:
    years_of_experience = st.number_input(
        "Years of Experience",
        min_value=0.0,
        max_value=50.0,
        value=0.0
    )

    expected_ctc_lpa = st.number_input(
        "Expected CTC (LPA)",
        min_value=0.0,
        max_value=1000.0,
        value=5.0
    )

    employment_gap_months = st.number_input(
        "Employment Gap (Months)",
        min_value=0,
        max_value=240,
        value=0
    )

    layoff_history = st.selectbox(
        "Layoff History",
        ["Yes", "No", "Unknown"]
    )


# Categorical Inputs
#st.selectbox()?
#It creates a dropdown menu. The user selects one of the categories instead of typing manually.

#gender = st.selectbox(
#    "Gender",
#    ["Male", "Female"]
#)

#degree_specialization = st.selectbox(
 #   "Degree Specialization",
  #  ["Computer Science", "Electronics", "Mechanical",
   #  "Information Technology", "Others"]
#)

#internship_experience = st.selectbox(
 #   "Internship Experience",
  #  ["Yes", "No"]
#)


st.divider()


st.subheader("5. Placement Preferences")
career_switch_willingness = st.selectbox(
    "Career Switch Willingness",
    ["Unknown", "Not Willing", "Willing"]
)

relevant_experience = st.selectbox(
    "Relevant Experience",
    ["Not Relevant", "Relevant", "Unknown"]
)

#company_tier = st.selectbox(
 #   "Company Tier",
  #  ["Tier 1", "Tier 2", "Tier 3"]
#)

#job_role_match = st.selectbox(
 #   "Job Role Match",
  #  ["Matched", "Not Matched", "Unknown"]
#)

competition_level = st.selectbox(
    "Competition Level",
    ["High", "Medium", "Low"]
)

bond_requirement = st.selectbox(
    "Bond Requirement",
    ["Required", "Not Required"]
)

#layoff_history = st.selectbox(
 #   "Layoff History",
  #  ["Yes", "No", "Unknown"]
#)

relocation_willingness = st.selectbox(
    "Relocation Willingness",
    ["Willing", "Not Willing", "Unknown"]
)

# Prediction Button
#Now we'll create the button that the user clicks to get a placement prediction.

if st.button("Predict Placement"):    #st.button() creates a clickable button.  if checks whether the user clicked it.

    #st.success("Candidate details submitted successfully!")  #st.success() displays a green success message.
    
    candidate_data = {
        'age_years': age_years,
        'ssc_percentage': ssc_percentage,
        'hsc_percentage': hsc_percentage,
        'degree_percentage': degree_percentage,
        'technical_score': technical_score,
        'aptitude_score': aptitude_score,
        'communication_score': communication_score,
        'skills_match_percentage': skills_match_percentage,
        'certifications_count': certifications_count,
        'years_of_experience': years_of_experience,
        'previous_ctc_lpa': previous_ctc_lpa,
        'expected_ctc_lpa': expected_ctc_lpa,
        'notice_period_days': notice_period_days,
        'employment_gap_months': employment_gap_months,
        'gender': gender,
        'degree_specialization': degree_specialization,
        'internship_experience': internship_experience,
        'career_switch_willingness': career_switch_willingness,
        'relevant_experience': relevant_experience,
        'company_tier': company_tier,
        'job_role_match': job_role_match,
        'competition_level': competition_level,
        'bond_requirement': bond_requirement,
        'layoff_history': layoff_history,
        'relocation_willingness': relocation_willingness
    }




    candidate_df = pd.DataFrame([candidate_data]) 
    #candidate_data collects all 25 values entered by the user into a Python dictionary.
    #pd.DataFrame([candidate_data]) converts that dictionary into a one-row Pandas DataFrame.
    

    # Create a separate table for displaying candidate details
    display_df = candidate_df.copy()

    display_df.insert(0, "Candidate Name", candidate_name)



    st.write("Candidate Data:")
    st.dataframe(display_df, use_container_width=True) #The st.dataframe() line displays the candidate's submitted information on the webpage.

    
    prediction = model.predict(candidate_df)
#model.predict(candidate_df) asks your trained Random Forest model to classify the candidate.

    probabilities = model.predict_proba(candidate_df)
#model.predict_proba(candidate_df) calculates the model's estimated probabilities for both classes.    
    history_record = {
    "Candidate Name": candidate_name,
    "Age": age_years,
    "Prediction Date & Time": datetime.now().strftime("%d-%m-%Y %I:%M:%S %p"),
    "Prediction": "Placed" if prediction[0] == 1 else "Not Placed",
    "Not Placed Probability": round(probabilities[0][0] * 100, 2),
    "Placed Probability": round(probabilities[0][1] * 100, 2)
    }

    st.session_state.prediction_history.append(history_record)


    
    st.divider()

    if prediction[0] == 1:
        st.balloons()
        result_color = "#16A34A"
        result_background = "#DCFCE7"
        result_icon = "🎉"
        result_text = "PLACED"
        result_message = "Congratulations! Candidate is predicted to be placed."

    else:
        result_color = "#DC2626"
        result_background = "#FEE2E2"
        result_icon = "⚠️"
        result_text = "NOT PLACED"
        result_message = "Candidate is predicted to be not placed."

    st.markdown(
        f"""
        <div style="
            background: {result_background};
            border-left: 8px solid {result_color};
            padding: 25px;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0 4px 12px rgba(0,0,0,0.12);
            margin-top: 20px;
        ">
            <h2 style="color: {result_color};">
                {result_icon} {result_text}
            </h2>
            <h3 style="color: #334155;">
                Candidate: {candidate_name}
            </h3>
            <p style="font-size: 17px; color: #475569;">
                {result_message}
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )








    #st.divider()

    
    #st.header("Prediction Result")

    #if prediction[0] == 1:
    #    st.success(
     #       "🎉 Congratulations! Candidate is Predicted to be PLACED."
      #  )
    #else:
     #   st.warning(
      #      "Candidate is Predicted to be NOT PLACED."
       # )

    #if prediction[0] == 1:  #prediction[0] == 1 checks whether the predicted class is Placed.
     #   st.success("Congratulations! Candidate is Predicted to be PLACED.")
    #else:
     #   st.warning("Candidate is Predicted to be NOT PLACED.")

        

    st.subheader("Prediction Probabilities")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            label="Not Placed Probability",
            value=f"{probabilities[0][0] * 100:.2f}%"
        )
        st.progress(float(probabilities[0][0]))
    with col2:
        st.metric(
            label="Placed Probability",
            value=f"{probabilities[0][1] * 100:.2f}%"
        )
        st.progress(float(probabilities[0][1]))
    #st.write("Probability of Not Placed:", round(probabilities[0][0] * 100, 2), "%")
#probabilities[0][0] gives the probability of Not Placed.

    #st.write("Probability of Placed:", round(probabilities[0][1] * 100, 2), "%")
#probabilities[0][1] gives the probability of Placed. 




    st.divider()

    st.subheader("🎯 Placement Probability Gauge")

    placed_percentage = probabilities[0][1] * 100

    st.markdown(
        f"""
        <div style="display:flex; justify-content:center;
                    padding:20px;">
            <div style="
                width:200px;
                height:200px;
                border-radius:50%;
                background: conic-gradient(
                    #16A34A {placed_percentage}%,
                    #E5E7EB {placed_percentage}% 100%
                );
                display:flex;
                align-items:center;
                justify-content:center;
            ">
                <div style="
                    width:155px;
                    height:155px;
                    border-radius:50%;
                    background:white;
                    display:flex;
                    flex-direction:column;
                    align-items:center;
                    justify-content:center;
                ">
                    <h1 style="color:#16A34A; margin:0;">
                        {placed_percentage:.2f}%
                    </h1>
                    <p style="color:#475569; margin:5px;">
                        Placement Chance
                    </p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.subheader("📊 Placement Probability Level")

    if placed_percentage >= 75:
        st.success("🟢 HIGH — Strong placement probability")

    elif placed_percentage >= 40:
        st.warning("🟡 MODERATE — Average placement probability")

    else:
        st.error("🔴 LOW — Weak placement probability")

st.divider()

st.markdown(
    """
    <div style="text-align: center; color: gray;">
        <p>Job Placement Prediction System</p>
        <p>Powered by Python | Pandas | Scikit-learn | Streamlit</p>
        <p>Machine Learning Project</p>
    </div>
    """,
    unsafe_allow_html=True
)
st.divider()

st.subheader("📋 Prediction History")

if st.session_state.prediction_history:
    history_df = pd.DataFrame(st.session_state.prediction_history)

    if "Prediction Date & Time" not in history_df.columns:
        history_df["Prediction Date & Time"] = "Not Available"

    history_df = history_df[
    [
        "Candidate Name",
        "Age",
        "Prediction Date & Time",
        "Prediction",
        "Not Placed Probability",
        "Placed Probability"
    ]
]
    st.dataframe(history_df, use_container_width=True)

else:
    st.info("No prediction history available yet.")