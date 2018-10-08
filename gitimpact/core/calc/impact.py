import time

from calc import NPNewCosineSimilarityCalculator, NewJaccardCoefficientCalculator


class ImpactAnalyser():
    def __init__(self):
        pass

    def impact_analysis_for_commit(self, item, space_items, results_count=None, debug=True, calculator_type=None):

        def impact_analysis(input_container, im_containers, impact_calculator_type, debug=True):
            impact_calculator = impact_calculator_type(im_containers + [input_container])
            if debug:
                print(input_container.name)
            results = [
                (s_container, impact_calculator.distance(input_container, s_container))
                for s_container in im_containers
            ]
            results = sorted(results, key=lambda tup: tup[1], reverse=False)
            first_non_null_distance = 0.0
            for s_commit, distance in results:
                ratio = distance / first_non_null_distance if first_non_null_distance != 0.0 else 0.0
                if debug:
                    print("\t%s | %s (%f)" % (s_commit.name, distance, ratio))

            return list(map(lambda x: (x[0], x[1]), results))

        if calculator_type is None:
            calculator_type = NewJaccardCoefficientCalculator

        if debug:
            start_time = time.time()

        analysis_results = impact_analysis(item, space_items, calculator_type, debug=debug)

        if debug:
            duration = time.time() - start_time
            print('%f sec' % duration)

        return analysis_results[:results_count]
