import sys, time, os, argparse
import numpy as np
import torch
import torch.onnx
import torch.nn as nn
from model.RawNet3 import MainModel

if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True,
                        help="yaml for configuration")
    parser.add_argument('--default', type=str, required=True,
                    help="default configuration")
    parser.add_argument('--version',"-v", type=str, required=True)
    parser.add_argument('--chkpt',type=str,required=False,default=None)
    args = parser.parse_args()

    model = MainModel()
    model.load_state_dict(torch.load("best.pt", map_location="cpu"))
    model.eval()

    # torch to ONNX

    version = args.version

    ## tracing
    input = torch.rand(1,16000)
    output = model(input)
    print("{} -> {}".format(input.shape,output.shape))
 
    traced_model = torch.jit.trace(model, input)
    traced_model.save('rawnet3_traced.pt')
    torch.backends.cudnn.deterministic = True
 
    torch.onnx.select_model_mode_for_export(model, torch.onnx.TrainingMode.EVAL)

    print("ONXX Export")
    torch.onnx.export(
        model,         # model being run 
        input,       # model input (or a tuple for multiple inputs) 
        "./chkpt/rawnet3_{}.onnx".format(version),       # where to save the model  
        opset_version=12,
        do_constant_folding=False,
        keep_initializers_as_inputs=False,
        input_names = ['input'], 
        output_names = ['output'],
        dynamic_axes={'input' : {0 : 'batch_size',1: 'n_sample'},    # variable length axes
        'output' : {0 : 'batch_size'}},
        export_params=True,
        #verbose = True,
        training = torch.onnx.TrainingMode.EVAL

        )
