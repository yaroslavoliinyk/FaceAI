import sqlite3
import os

from Path import Path
from result import Result
# It has to be created only once!!
class ResultsDB_API:


    def __init__(self, path_to_db):
        self._connection       = sqlite3.connect(path_to_db)
        self._conn             = self._connection.cursor()
        self._images_list      = self.__get_images_list()
        self._scales_list      = self.__get_scales_list()
        self._assessments_list = self.__get_assessments_list()
        self._result_id        = 0


    @property
    def images_list(self):
        return self._images_list
  

    @property
    def scales_list(self):
        return self._scales_list
    

    @property
    def assessments_list(self):
        return self._assessments_list


    def write_images_to_db(self):
        image_pathes = self.files("photos/")
        # in sqlite we start from 1
        id           = 1
        for img_path in image_pathes:
            command  = "INSERT INTO ImagesTable VALUES(" + str(id) + ", " + "\'" + img_path + "\');"
            print(command)
            self._conn.execute(command)
            id      += 1
        self._connection.commit()

   
    def write_result(self, result):
        self._result_id += 1
        command = "INSERT INTO Results(ID, Img_ID, AssessmentMethod_ID, Scale_ID, Mark, Time) VALUES(" + str(self._result_id) + "," + str(result.img_path_id) + "," + str(result.assess_id) + "," + str(result.scale_id) + "," + str(result.mark) + "," + str(result.time) + ");"
        self._conn.execute(command)
        self._connection.commit()


    def get_result_data_by_assess(self, assess_name, mark_time):
        command = "SELECT s.ScaleName, " + "r." + mark_time + " FROM Results as r INNER JOIN ScaleTable AS s ON s.ID = r.Scale_ID INNER JOIN AssessmentTable AS a ON a.ID = r.AssessmentMethod_ID  WHERE a.AssessmentMethodName = \'" + assess_name + "\';"
        res     = self._conn.execute(command).fetchall()
        print(res)
        return res


    def close_connection(self):
        self._connection.close()

    @staticmethod
    def files(path):
        for file in os.listdir(path):
            if os.path.isfile(os.path.join(path, file)) and not file.startswith('.'):
                yield file


    def __get_images_list(self):
        command     = "SELECT * FROM ImagesTable;"
        imgs_list   = self._conn.execute(command).fetchall()
        return imgs_list
    

    def __get_scales_list(self):
        command      = "SELECT * FROM ScaleTable;"
        scale_list   = self._conn.execute(command).fetchall()
        return scale_list
    

    def __get_assessments_list(self):
        command       = "SELECT * FROM AssessmentTable;"
        assess_list   = self._conn.execute(command).fetchall()
        return assess_list


