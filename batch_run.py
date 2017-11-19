# -*- coding: utf-8 -*-
#
# Author: taketoshi.kazusa
#

import subprocess


def run(command):
    subprocess.check_call("python {}".format(command), shell=True)


if __name__ == "__main__":
    run("run_extracter_eigen.py")
    run("run_extracter_median.py")
    print("fin")
