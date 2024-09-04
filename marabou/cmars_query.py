
from maraboupy import Marabou
from maraboupy import MarabouNetworkONNX
from maraboupy import MarabouCore, MarabouUtils

import time


def run_query(model_path, model_input_size, model_output_size, 
    copy_one_argmax_candidate, copy_two_argmax_candidate, timeout):
    options = Marabou.createOptions(verbosity = 0, timeoutInSeconds=timeout)
    network = Marabou.read_onnx(model_path)

    inputVars = network.inputVars[0][0]
    outputVars = network.outputVars[0][0]

    for i in range(model_input_size):
        # print(f"setting bounds for input varaible {i}")
        network.setLowerBound(inputVars[i], 0.0)
        network.setUpperBound(inputVars[i], 1.0)

    for i in range(model_input_size, 2*model_input_size):
        # print(f"setting bounds for input varaible {i}")
        network.setLowerBound(inputVars[i], 0.0001)
        network.setUpperBound(inputVars[i], 0.0001)

    for i in range(model_output_size):
        if i == copy_one_argmax_candidate:
            continue
        # print("comparing output variable ", i)
        equation = MarabouUtils.Equation(EquationType=MarabouCore.Equation.GE)
        equation.addAddend(1, outputVars[copy_one_argmax_candidate])
        equation.addAddend(-1, outputVars[i])
        equation.setScalar(0)
        network.addEquation(equation)

    for i in range(model_output_size):
        if i == copy_two_argmax_candidate:
            continue
        # print("comparing output variable ", i+model_output_size)
        equation = MarabouUtils.Equation(EquationType=MarabouCore.Equation.GE)
        equation.addAddend(1, outputVars[copy_two_argmax_candidate+model_output_size])
        equation.addAddend(-1, outputVars[i+model_output_size])
        equation.setScalar(0)
        network.addEquation(equation)

    exitCode, vals, stats = network.solve(options = options)

    return exitCode

if __name__ == "__main__":
    layer_count = 2
    hidden_size = 32
    action_count = 15
    model_path = f"../applications/cmars/models/conv2d_based_onnx/model_l{layer_count}_h{hidden_size}_a{action_count}.onnx"
    model_input_size = 19
    model_output_size = action_count
    # indices = [6, 7, 8, 10, 13, 29, 30, 31, 33, 41, 42, 43, 45, 48, 52, 53, 54, 72, 75, 79, 80, 81, 83, 86, 87, 88, 90, 93]
    indices = None
    timeout = 3600
    with open(f"../applications/cmars/output{action_count}_multiplexing_gain_lower00upper10.csv", "r") as file:
        lines = file.readlines()
        result_dict = {}
        for idx, line in enumerate(lines):
            if not (indices is None):
                if not(idx in indices):
                    continue

            argmax_a_b = line.split("/")[-1].split(".")[0].split("_")[1:]
            argmax_a = int(argmax_a_b[0][1:])
            argmax_b = int(argmax_a_b[1][1:])
            print(f" ---- running query for property idx {idx} with argmax_a of {argmax_a} and argmax_b of {argmax_b}", flush=True)
            start_time = time.time()
            exitCode = run_query(model_path, model_input_size, model_output_size, argmax_a, argmax_b, timeout)
            end_time = time.time()

            if not (exitCode in result_dict):
                result_dict[exitCode] = []

            result_dict[exitCode].append(idx)

            print(f"execition time: {end_time - start_time}", flush=True)

    print("Results summary")
    for k, v in result_dict.items():
        print(f"{k} (total {len(v)}): index: {v}")
