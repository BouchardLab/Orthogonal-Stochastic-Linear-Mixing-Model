from model.oslmm import *
from model.exp_analysis import *


if __name__ == "__main__":
    args = dict()
    # args["domain"] = "jura"
    # args["domain"] = "concrete"
    args["domain"] = "equity"
    # args["domain"] = "pm25"
    # args["domain"] = "neuron"
    args["kernel"] = "SEiso"
    args["Q"] = 2
    args["nsave"] = 300
    args["nburnin"] = 200
    args["interval"] = 10
    args["record_time"] = False

    for trail in range(5):
        args["trail"] = trail
        run(args)

    computing_predictive_measures(res_path="results", rank=2, model='oslmm', domains=["equity"], verbose=True)