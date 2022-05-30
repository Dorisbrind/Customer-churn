from flask import request
from flask import jsonify

from flask_restful import Resource
from databases.model import db, Df

class ChurnAPI(Resource):
    def get(self):
        data_list = []
        datas = Df.query.all()
        for data in datas:
            data_list.append(data.to_json())
        return jsonify(data_list)
           