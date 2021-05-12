from coverage import Coverage
import unittest
from tests.lib import test_lib
from tests.yolomask import test_yolo
from tests.distance import test_distance


def run_unit_tests(runner):
    runner.run(test_lib.get_unit_test_suite())
    # runner.run(test_yolo.get_unit_test_suite())
    # runner.run(test_distance.get_unit_test_suite())


def run_integration_tests(runner):
    runner.run(test_lib.get_integration_test_suite())
    # runner.run(test_yolo.get_integration_test_suite())
    # runner.run(test_distance.get_integration_test_suite())


# ======================================================================================

def run_unit_test_with_coverage_report(runner):
    cov = Coverage()
    cov.start()
    run_unit_tests(runner)
    cov.stop()
    cov.save()
    percentage_covered = cov.html_report(title="OpenCovid Project Unit Test Report")
    print("Code Covered In Test = {}%".format(percentage_covered))


def run_all_tests(runner):
    run_unit_test_with_coverage_report(runner)
    # run_unit_tests(runner)
    # run_integration_tests(runner)


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    run_all_tests(runner)
