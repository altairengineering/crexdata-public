from numbers import Real

from py4j.java_gateway import JavaGateway, GatewayParameters, CallbackServerParameters
from numpy import tanh, sqrt, inf
import os


def build_gateway(python_port, java_port):
    gateway_parameters = GatewayParameters(port=java_port)
    callback_server_parameters = CallbackServerParameters(port=python_port)
    gateway = JavaGateway(gateway_parameters=gateway_parameters, callback_server_parameters=callback_server_parameters)
    return gateway


def calculate_mcc(tp, tn, fp, fn):
    tpfp = tp + fp
    tpfn = tp + fn
    tnfp = tn + fp
    tnfn = tn + fn
    total = tp + tn + fp + fn

    if tpfp == 0 or tpfn == 0 or tnfp == 0 or tnfn == 0:
        return 0.0

    else:
        precision = tp / tpfp if (tpfp != 0) else -1
        recall = tp / tpfn if (tpfn != 0) else -1
        specificity = tn / tnfp if (tnfp != 0) else -1
        accuracy = (tp + tn) / total if (total != 0) else -1
        npv = tn / tnfn if (tnfn != 0) else -1
        fdr = 1 - precision
        fnr = 1 - recall
        fpr = 1 - specificity
        fomr = 1 - npv
        return sqrt(precision * recall * specificity * npv) - sqrt(fdr * fnr * fpr * fomr)





def make_dir(path):
    """
    Create new directory

    :param path: path to the directory to be created
    """
    try:
        os.mkdir(path)
    except OSError:
        print("Warning: Directory {} already exists.".format(path))
    else:
        print("Successfully created the directory {}.".format(path))
