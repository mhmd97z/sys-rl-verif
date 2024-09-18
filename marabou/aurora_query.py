
from maraboupy import Marabou
from maraboupy import MarabouNetworkONNX
from maraboupy import MarabouCore, MarabouUtils
import time


def run_query(model_path, vnnlib_path):
    options = Marabou.createOptions(verbosity=0, timeoutInSeconds=timeout, snc=True, numWorkers=28)
    network = Marabou.read_onnx(model_path)
    exitCode, _, _ = network.solve(options=options, propertyFilename=vnnlib_path)

    return exitCode

if __name__ == "__main__":
    indices = None
    timeout = 1000
    model_path = f"../applications/aurora/models/conv2d_based_onnx/model_conv2d.onnx"
    property_csv_file_path = f"../applications/aurora/capacity_utilization_eps001.csv"
    # capacity_utilization_eps001.csv
    # loss_avoidance_eps001.csv
    # robustness_multistep.csv
    
    with open(property_csv_file_path, "r") as file:
        lines = file.readlines()
        result_dict = {}
        for idx, line in enumerate(lines):
            line = line.strip()
            if "vnnlib" not in line:
                continue
            
            if not (indices is None):
                if not(idx in indices):
                    continue
            
            print(f" ---- running query for idx: {idx}, {line}", flush=True)
            vnnlib_path = "../applications/aurora/" + line
            start_time = time.time()
            exitCode = run_query(model_path, vnnlib_path)
            end_time = time.time()

            if not (exitCode in result_dict):
                result_dict[exitCode] = []
            result_dict[exitCode].append(idx)

            print(f"execition time: {end_time - start_time}", flush=True)

    print("Results summary")
    for k, v in result_dict.items():
        print(f"{k} (total {len(v)}): index: {v}")
