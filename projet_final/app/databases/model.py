# coding: utf-8
from flask_sqlalchemy import SQLAlchemy


db = SQLAlchemy()



class Df(db.Model):
    __tablename__ = 'df'

    RowNumber = db.Column(db.Float(53), primary_key=True)
    CustomerId = db.Column(db.Float(53))
    CreditScore = db.Column(db.Float(53))
    Geography = db.Column(db.Text)
    Gender = db.Column(db.Text)
    Age = db.Column(db.Float(53))
    Tenure = db.Column(db.Float(53))
    Balance = db.Column(db.Float(53))
    NumOfProducts = db.Column(db.Float(53))
    HasCrCard = db.Column(db.Float(53))
    IsActiveMember = db.Column(db.Float(53))
    EstimatedSalary = db.Column(db.Text)
    Exited = db.Column(db.Float(53))

    def to_json(self):
        item = self.__dict__
        if "_sa_instance_state" in item:
            del item["_sa_instance_state"]
        return item