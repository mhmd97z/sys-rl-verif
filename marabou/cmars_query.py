
from maraboupy import Marabou
from maraboupy import MarabouNetworkONNX
from maraboupy import MarabouCore, MarabouUtils
import time


def run_query(model_path, vnnlib_path):
    options = Marabou.createOptions(verbosity=0, timeoutInSeconds=timeout)
    network = Marabou.read_onnx(model_path)
    exitCode, _, _ = network.solve(options=options, propertyFilename=vnnlib_path)

    return exitCode

if __name__ == "__main__":
    layer_count = 1
    hidden_size = 32
    action_count = 15
    model_input_size = 19
    model_output_size = action_count
    indices = [50, 51, 52, 53, 54, 55]
    indices = None
    timeout = 3600
    model_path = f"../applications/cmars/models/conv2d_based_onnx/model_l{layer_count}_h{hidden_size}_a{action_count}.onnx"
    property_csv_file_path = f"../applications/cmars/output{action_count}_robustness_lower00upper10.csv"
    
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
            vnnlib_path = "../applications/cmars/" + line
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
