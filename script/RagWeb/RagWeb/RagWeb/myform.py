from wtforms import StringField, BooleanField, PasswordField, SubmitField, IntegerField
from wtforms.validators import DataRequired
from wtforms.validators import Required, Length, Email, Regexp, EqualTo
from wtforms import ValidationError
from flask_wtf import FlaskForm

class message_form(FlaskForm):
    message = StringField(u'message', validators=[
                DataRequired(message= u'请输入一段诗文')]) #, Length(1, 128)])
    submit = SubmitField('generate-button')