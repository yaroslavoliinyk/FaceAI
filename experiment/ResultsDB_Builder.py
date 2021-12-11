
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.ticker import StrMethodFormatter

class ResultsDB_Builder:
    

    def __init__(self, results_db_api):
        self._db_api = results_db_api


    def build_mark_hist_by_assess(self, assess_name):
        result_table = self._db_api.get_result_data_by_assess(assess_name)
        result_dict  = dict()
        for coef, mark in result_table:
            if(coef not in result_dict.keys()):
                result_dict[coef] = list()
            result_dict[coef].append(mark)
        #print(result_dict)
        df = pd.DataFrame(data=result_dict)
        #print(df)
        elem_num = len(df.iloc[:, 0])
        # evenly sampled time at 200ms intervals
        img_number = np.arange(1, elem_num+1, 1)
        coef_1_arr = df.iloc[:, 0]
        coef_075_arr = df.iloc[:, 1]
        coef_05_arr = df.iloc[:, 2]

        coef_1_arr = np.asarray(coef_1_arr)
        #print(coef_1_arr)

        coef_075_arr = np.asarray(coef_075_arr)
        #print(coef_075_arr)

        coef_05_arr = np.asarray(coef_05_arr)
        #print(coef_05_arr)
        # red dashes, blue squares and green triangles
        plt.plot(img_number, coef_1_arr, 'orange', img_number, coef_075_arr, 'violet', img_number, coef_05_arr, 'brown')
        plt.ylabel("Mark")
        plt.xlabel("Image number")
        plt.title(assess_name + " algorithm")
        plt.show()
        # bins - number of lines; figsize - size of window; layout - apparently number of types
        # sharex - makes mandatory each plot has same x values; rwidth - width of lines
        #ax = df.hist(column='mark', by='user_type', bins=20, grid=False, figsize=(8,12), layout=(3,1), sharex=False, color='#00bf91', zorder=2, rwidth=0.75)

    
    def build_time_hist_by_assess(self, assess_name):
        pass


    def build_hist_assess_by(self, assess_name, mark_time):
        result_table = self._db_api.get_result_data_by_assess(assess_name, mark_time)
        result_dict  = dict()
        for coef, mt in result_table:
            if(coef not in result_dict.keys()):
                result_dict[coef] = list()
            result_dict[coef].append(mt)
        #print(result_dict)
        df = pd.DataFrame(data=result_dict)
        #print(df)
        elem_num = len(df.iloc[:, 0])
        # evenly sampled time at 200ms intervals
        img_number = np.arange(1, elem_num+1, 1)
        # a flaw of my program.. You cannot dynamically that simple change number of different coefs
        coef_1_arr = df.iloc[:, 0]
        ##
        #coef_075_arr = df.iloc[:, 1]
        coef_05_arr = df.iloc[:, 1]

        coef_1_arr = np.asarray(coef_1_arr)
        #print(coef_1_arr)

        ##
        #coef_075_arr = np.asarray(coef_075_arr)
        #print(coef_075_arr)

        coef_05_arr = np.asarray(coef_05_arr)
        if(mark_time == "Time"):
            color_1     = "red"
            ##
            #color_0_75  = "green"
            color_0_5   = "blue"
            ylabel = str(mark_time) + ", seconds"
        else:
            color_1     = "orange"
            ##
            #color_0_75  = "violet"
            color_0_5   = "brown"
            ylabel = str(mark_time) + ", normalized points"

        plt.plot(img_number, coef_1_arr, color_1, label="2x")
        ##
        #plt.plot(img_number, coef_075_arr, color_0_75, label="0.75x")
        plt.plot(img_number, coef_05_arr, color_0_5, label="1x")
        plt.ylabel(ylabel)
        plt.xlabel("Image number")
        plt.title(assess_name + " algorithm")
        plt.xticks(np.arange(1, elem_num+1, 1))
        leg = plt.legend(loc='best', ncol=1, shadow=True, fancybox=True)
        leg.get_frame().set_alpha(0.99)
        plt.show()
        '''ax = plt.subplot(111)
        t1 = np.arange(0.0, 1.0, 0.01)
        for n in [1, 2, 3, 4]:
            plt.plot(t1, t1**n, label=f"n={n}")

        leg = plt.legend(loc='best', ncol=2, mode="expand", shadow=True, fancybox=True)
        leg.get_frame().set_alpha(0.5)'''
