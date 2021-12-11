

class Result:
    
    def __init__(self, img_path_id, scale_id, assess_id, mark, time):
        self._img_path_id = img_path_id
        self._scale_id    = scale_id
        self._assess_id   = assess_id
        self._mark        = mark
        self._time        = time


    @property
    def img_path_id(self):
        return self._img_path_id


    @property
    def scale_id(self):
        return self._scale_id 


    @property
    def assess_id(self):
        return self._assess_id
    

    @property
    def mark(self):
        return self._mark
    

    @property
    def time(self):
        return self._time

