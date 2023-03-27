import numpy as np
import subprocess
import math
from numpy import sqrt
import argparse 
import time
 

def generateRandomArray(inputSize, input, w):
    w.write(str(inputSize))
    w.write(", 1> = [[")
    for n in range(0, inputSize - 1):
        w.write(str(input[n]))
        w.write(", ")
    w.write(str(input[inputSize-1]))
    w.write("]]; \n")
    
def generateFFTLang(inputSize, radix, inputR, inputI):
    with open("/proj/snic2022-22-1035/1205/mlir/test.txt", "w") as w:
        w.write("# RUN: toyc-ch7 %s -emit=jit -opt \n")
        w.write("def compute() { \n")
        w.write ("var InputReal <")
        generateRandomArray(inputSize, inputR, w)
        w.write ("var InputImg <")
        generateRandomArray(inputSize, inputI, w)
        w.write("var InputComplex = createComplex(InputReal, InputImg); \n")
        w.write("var IComplex = I(" + str(inputSize) + "); \n")
        w.write("var c = permute(InputComplex, " + str(inputSize) + ", " + str(radix) + "); \n")
        w.write("var result =  (DFT(" + str(radix) + ") ⊗ I(" + str(radix) + ")) · (twiddle(")
        w.write(str(inputSize) + ", " + str(radix) + ") · ((I(" + str(radix) + ") ⊗ DFT(" + str(radix) + ")) · c)); \n")
        w.write("var resultRe = reTensor(result); \n")
        w.write("var resultIm = imTensor(result); \n")
        w.write("print(resultRe); \n")
        w.write("print(resultIm); \n")
        w.write("}")

def generatestockhamFFTFormulaIterative(inputSize, radix):
    l = int(math.log(inputSize, radix))
    fftFormula = ""
    for n in range(0, l):
        fftFormula = ("(DFT(" + str(radix) + ") ⊗ I(" + str(int(pow(radix, l-1))) +
        ")) · stockhamTwiddle(" + str(inputSize) + ", " + str(int(radix)) + ", " + str(int(n)) + 
        ") · (Permute(" + str(int(pow(radix, n + 1))) + ", " + str(radix) + ") ⊗ I(" + 
        str(int(pow(radix, l - n - 1))) + ")) · " + fftFormula)
    return fftFormula[:-3]
    
def generateFFTFormulaRecursive(inputSize, radix):
    if inputSize == radix:
        return "DFT(" + str(radix) + ")"
    else:
        return ("((DFT(" + str(radix) + ") ⊗ I(" + str(int( pow(radix, 3))) + 
    ")) · twiddle(" + str(inputSize) + ", " + str(int(inputSize/radix)) + ") · (I(" + str(radix) + 
    ") ⊗ " + generateFFTFormulaRecursive(int(inputSize/radix), radix) + ") · Permute(" + str(inputSize) + 
    ", " + str(radix) + "))")
        
def generateFFTLangRecursive(inputSize, radix, inputR, inputI):
    with open("/proj/snic2022-22-1035/1205/mlir/recursive.fft", "w") as w:
        w.write("# RUN: toyc-ch7 %s -emit=jit -opt \n")
        w.write("def compute() { \n")
        w.write ("var InputReal <")
        generateRandomArray(inputSize, inputR, w)
        w.write ("var InputImg <")
        generateRandomArray(inputSize, inputI, w)
        w.write("var InputComplex = createComplex(InputReal, InputImg); \n")
        fftFormula = generateFFTFormulaRecursive(inputSize, radix)      
        w.write("var result = " + fftFormula + " · InputComplex; \n")
        w.write("var resultRe = reTensor(result); \n")
        w.write("var resultIm = imTensor(result); \n")
        w.write("print(resultRe); \n")
        w.write("print(resultIm); \n")
        w.write("}")
        
def generateStockhamFFTLang(inputSize, radix, inputR, inputI):
    with open("/proj/snic2022-22-1035/1205/stockhamFFT_" + str(radix) + "_" + str(inputSize) + ".fft", "w") as w:
        w.write("# RUN: toyc-ch7 %s -emit=jit -opt \n")
        w.write("def compute() { \n")
        w.write ("var InputReal <")
        generateRandomArray(inputSize, inputR, w)
        w.write ("var InputImg <")
        generateRandomArray(inputSize, inputI, w)
        w.write("var InputComplex = createComplex(InputReal, InputImg); \n")
        fftFormula = generatestockhamFFTFormulaIterative(inputSize, radix)      
        w.write("var result = " + fftFormula + " · InputComplex; \n")
        w.write("var resultRe = reTensor(result); \n")
        w.write("var resultIm = imTensor(result); \n")
        w.write("print(resultRe); \n")
        w.write("print(resultIm); \n")
        w.write("}")
def frontendMLIRPreprocessing(inputIR, inputSize, radix):
    
    fft_ir = inputIR.replace("toy", "fft" )
    fft_ir_func = fft_ir.replace("func", "func.func" )
    fft_ir_ret = fft_ir_func.replace("fft.return", "return" )

    with open("/proj/snic2022-22-1035/1205/stockhamFFT_fm_" + str(radix) + "_" + str(inputSize) + ".mlir", "w") as w:
        w.write(fft_ir_ret)
        
def isClose(a, b):
    return abs(a-b) <= 1e-06


def main():
    # Initialize parser
    parser = argparse.ArgumentParser()   
    # Adding optional argument
    parser.add_argument("-inputSize", type = int, help = "size of DFT")
    parser.add_argument("-radix", type = int, help = "radix for defactorization")   
   
    # Read arguments from command line
    args = parser.parse_args()

    if (args.inputSize and args.inputSize):
        inputSize = int(args.inputSize)
        radix = int(args.radix)
    else:
        print("inputSize and radix needed")
        return
    print("inputSize =", inputSize)
    print("radix =", radix)

    inputR = np.random.uniform(-1, 1, size=(inputSize))
    inputI = np.random.uniform(-1, 1, size=(inputSize))
    inputC = inputR + 1j * inputI

    #generateFFTLangRecursive(inputSize, radix, inputR, inputI)
    generateStockhamFFTLang(inputSize, radix, inputR, inputI)
    resultP = np.fft.fft(inputC)
    before_compile = time.time()


    useless_cat_call = subprocess.run(["/proj/snic2022-22-1035/1205/frontend/build_r/bin/toyc-ch7", "/proj/snic2022-22-1035/1205/stockhamFFT_" + str(radix) + "_" + str(inputSize) + ".fft"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    frontend_mlir = useless_cat_call.stderr
    frontendMLIRPreprocessing(frontend_mlir, inputSize, radix)
    after_frontend = time.time()

    opt = subprocess.run(["/proj/snic2022-22-1035/1205/llvm-project/build/bin/mlir-opt", "/proj/snic2022-22-1035/1205/stockhamFFT_fm_" + str(radix) + "_" + str(inputSize) + ".mlir", "-fft-shape-infer", "--canonicalize", "--cse", "--fft-shape-infer", "--fft-sparse-fusion", "--fft-shape-infer", "--convert-fft-to-affine", "--affine-loop-normalize", "--canonicalize", "--cse", "--convert-fft-affine-to-llvm"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    after_opt = time.time()

    opted_mlir = opt.stdout
    with open("/proj/snic2022-22-1035/1205/stockhamFFT_cg_" + str(radix) + "_" + str(inputSize) + ".mlir", "w") as w:
        w.write(opted_mlir)
    codeGen = subprocess.run(["/proj/snic2022-22-1035/1205/frontend/build/bin/mlir-cpu-runner", "-e", "compute", "-entry-point-result=void", "-O3", "-enable-vplan-native-path", "-force-vector-width=8", "-dump-object-file", "-object-filename=/proj/snic2022-22-1035/1205/stockhamFFT_" + str(radix) + "_" + str(inputSize) +".o",  "/proj/snic2022-22-1035/1205/stockhamFFT_cg_" + str(radix) + "_" + str(inputSize) + ".mlir"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) 
    after_cg = time.time()
    link = subprocess.run(["ar", "rcs", "/proj/snic2022-22-1035/1205/libstockhamFFT_" + str(radix) + "_" + str(inputSize) +".a", "/proj/snic2022-22-1035/1205/stockhamFFT_" + str(radix) + "_" + str(inputSize) + ".o"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) 
    before_aot = time.time()

    aotLink = subprocess.run(["g++", "compute.cpp", "-l" + "stockhamFFT_" + str(radix) + "_" + str(inputSize), "-L/proj/snic2022-22-1035/1205/", "-o", "/proj/snic2022-22-1035/1205/stockhamFFT_" + str(radix) + "_" + str(inputSize) + ".out"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) 
    aotRun = subprocess.run(["/proj/snic2022-22-1035/1205/stockhamFFT_" + str(radix) + "_" + str(inputSize) + ".out"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    after_aot = time.time()

    frontend = after_frontend - before_compile
    opt = after_opt - after_frontend
    cg = after_cg - after_opt
    aot = after_aot -before_aot
    print("Time in seconds for frontend:", frontend)
    print("Time in seconds for opt:", opt)
    print("Time in seconds for cg:", cg)
    print("Time in seconds for aot:", aot)
    print(aotRun.stdout)

    resultS = codeGen.stdout.split()
    resultF = [float(i) for i in resultS]

    resultR = np.array(resultF[0:inputSize]) 
    resultI = np.array(resultF[inputSize:2*inputSize])
    resultC = resultR + 1j * resultI


    for x, y in np.nditer ([resultP,resultC]):
        if (~(isClose(x,y))):
            print ("result from np: %f", x)
            print ("result from DSL: %f", y)
    print("It works!")        
if __name__ == "__main__":
    main() 