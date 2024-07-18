from flask_wtf import FlaskForm
from wtforms import (
    IntegerField, 
    FloatField,
    SubmitField,
    StringField
)

from wtforms.validators import (
    DataRequired
)

class CustomEfficiencyForm(FlaskForm):
    rate = FloatField(
        label = "Rate of Increase",
        validators = [DataRequired()]
    )

    submit = SubmitField("Calculate")

class userPrediction(FlaskForm):
    financial_year = StringField(
        label = "Financial year",
        validators = [DataRequired()]
    )
    sales = IntegerField(
        label = "Sales",
        validators = [DataRequired()]
    )

    booked = IntegerField(
        label = "Booked Hours",
        validators = [DataRequired()]
    )

    submit = SubmitField("Predict")

class customUserPrediction(FlaskForm):
    financial_year = StringField(
        label = "Financial year",
        validators = [DataRequired()]
    )
    rate = FloatField(
        label = "Rate",
        validators = [DataRequired()]
    )
    sales = IntegerField(
        label = "Sales",
        validators = [DataRequired()]
    )
    booked = IntegerField(
        label = "Booked Hours",
        validators = [DataRequired()]
    )

    submit = SubmitField("Predict")

