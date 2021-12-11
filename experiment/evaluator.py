import assessment_methods
import time

from Path import Path
from ResultsDB_API import ResultsDB_API
from PIL import Image
from result import Result

class Evaluator:

    # Присвоєння посилання на АПІ до бази даних і заповення словників
    def __init__(self, res_db_api):
        self._db_api                  = res_db_api
        self._assessment_methods_dict = dict()
        self._scale_dict              = dict()
        self.__fulfill_am_dict(self._assessment_methods_dict)
        self.__fulfill_scale_dict(self._scale_dict)

    # Зменшення масштабу, залежно від коефіцієнтів в параметрі all_scales
    def descale(self, all_scales):
        print(all_scales)
        for scale in all_scales:
            coef = float(scale[1])
            scale_path = self._scale_dict[scale[1]]
            self.__descale_imgs_coef_many(coef, scale_path)

    # Оцінка сету зображень на кожному із методів для будь-якого масштабу
    def make_evaluation(self):
        print("images list: ", self._db_api.images_list)
        print("scales list: ", self._db_api.scales_list)
        print("assessments list: ", self._db_api.assessments_list)
        for img_item in self._db_api.images_list:
            for scale_item in self._db_api.scales_list:
                for assess_item in self._db_api.assessments_list:
                    self.__evaluate(img_item, scale_item, assess_item)

    # Оцінка зображення специфічним алгоритмом і запис до бази даних
    def __evaluate(self, img_item, scale_item, assess_item):
        # Set up algorithm usage with proper photo
        print("img item: ", str(img_item), "scale_item: ", str(scale_item), "assess_item: ", str(assess_item))
        eval_algorithm = self._assessment_methods_dict[assess_item[1]]
        path = self._scale_dict[scale_item[1]] + img_item[1]
        print(":::::PATH:::::", path)
        # Run
        start = time.time()
        mark = eval_algorithm(path)
        end = time.time()
        measured_time = end - start
        # Write results
        result = Result(img_item[0], scale_item[0], assess_item[0], mark, measured_time)
        self._db_api.write_result(result)
        return True

    # Заповнення словника меитодів
    def __fulfill_am_dict(self, am_dict):
        am_dict['entropy'] = assessment_methods.entropy
        am_dict['vollath'] = assessment_methods.vollath
        am_dict['energy'] = assessment_methods.energy
        am_dict['varience'] = assessment_methods.varience
        am_dict['SMD'] = assessment_methods.SMD
        am_dict['SMD2'] = assessment_methods.SMD2
        am_dict['brenner'] = assessment_methods.brenner

    # Заповнення словника масштабів
    def __fulfill_scale_dict(self, scale_dict):
        scale_dict["0.5"]   = Path.path_0_5
        scale_dict["0.75"]  = Path.path_0_75
        scale_dict["1"]     = Path.path_1_0

    # Зменшення масштабу за встановоленим коефіцієнтом
    def __descale_imgs_coef_many(self, coef, scale_path):
        img_names = ResultsDB_API.files(Path.path_1_0)
        for img_name in img_names:
            img = Image.open(Path.path_1_0 + img_name)
            width, height = img.size
            new_width     = int(width * coef)
            new_height    = int(height * coef)
            new_size      = (new_width, new_height)
            
            new_img       = img.resize(new_size)
            new_img_name  = scale_path + img_name
            new_img.save(new_img_name)


def do_ping(self, arg):
    return 'Pong, {0}!'.format(arg)

def do_ls(self, arg):
    return '\n'.join(os.listdir(arg))

dispatch = {
    'ping': do_ping,
    'ls': do_ls,
}

def process_network_command(command, arg):
    send(dispatch[command](arg))