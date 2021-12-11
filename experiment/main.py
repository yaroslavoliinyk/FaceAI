
from ResultsDB_API import ResultsDB_API
from ResultsDB_Builder import ResultsDB_Builder
from evaluator import Evaluator

results_db_api = ResultsDB_API("ResultsBase.db")

class AssessBy:
    MARK = "Mark"
    TIME = "Time"


def main():
    # For adding new set of images 
    #results_db_api.write_images_to_db()    
    #results_db_api.close_connection()

    # For descale
    #e = Evaluator(results_db_api)
    #e.descale(results_db_api.scales_list)

    # For evaluation:
    #e = Evaluator(results_db_api)
    #e.make_evaluation()
    #results_db_api.close_connection()

    # For histogram
    hbuilder = ResultsDB_Builder(results_db_api)
    hbuilder.build_hist_assess_by("brenner", AssessBy.MARK)
    results_db_api.close_connection()


if __name__ == "__main__":
    main()