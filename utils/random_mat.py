import numpy as np
import os


def get_random_mat():
    pos = {"num": 30, "dim": 10, "max": 10, "min": 0}
    dp = {"num": 15, "dim": 10, "max": 10, "min": 0}
    head = {"num": 128, "dim": 5, "max": 10, "min": 0}

    posrand = np.random.uniform(pos["min"], pos["max"], (pos["num"], pos["dim"]))
    dprand = np.random.uniform(dp["min"], dp["max"], (dp["num"], dp["dim"]))
    headrand = np.random.uniform(head["min"], head["max"], (head["num"], head["dim"]))

    save_path = os.path.join(os.getcwd(), "../data")
    posfile = os.path.join(save_path, "pos_mat.npy")
    dpfile = os.path.join(save_path, "dp_mat.npy")
    headfile = os.path.join(save_path, "head_mat.npy")
    np.save(posfile, posrand)
    np.save(dpfile, dprand)
    np.save(headfile, headrand)

    loader = np.load(dpfile)
    print(loader)
    loader = np.load(headfile)
    print(loader)


get_random_mat()

