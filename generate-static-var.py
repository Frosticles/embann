configFile = open(".config", "r")
if "CONFIG_MEMORY_ALLOCATION_DYNAMIC=y" in configFile.read():
    raise SystemExit

configFile.seek(0)

outputFile = open("include/embann_static.h", "w+")
outputFile.write("/* File auto-generated by generate-static-var.py */\n")
outputFile.write("#include \"embann_data_types.h\"\n")
outputFile.write("#include \"embann_config.h\"\n\n")


numInputNeurons = 0
numHiddenNeurons = 0
numHiddenLayers = 0
numOutputNeurons = 0

#
# Read values from Kconfig
#
for line in configFile:
    if "CONFIG_NUM_INPUT_NEURONS=" in line:
        numInputNeurons = int(line.split('=')[1])
        print("Number of input neurons =", numInputNeurons)
    if "CONFIG_NUM_HIDDEN_NEURONS=" in line:
        numHiddenNeurons = int(line.split('=')[1])
        print("Number of hidden neurons =", numHiddenNeurons)
    if "CONFIG_NUM_HIDDEN_LAYERS=" in line:
        numHiddenLayers = int(line.split('=')[1])
        print("Number of hidden layers =", numHiddenLayers)
    if "CONFIG_NUM_OUTPUT_NEURONS=" in line:
        numOutputNeurons = int(line.split('=')[1])
        print("Number of output neurons =", numOutputNeurons)




#
# Input layer
#
outputFile.write("/*\n")
outputFile.write(" * Input Layer\n")
outputFile.write(" */\n")
outputFile.write("static activation_t inputNeurons[CONFIG_NUM_INPUT_NEURONS];\n")
outputFile.write("static inputLayer_t staticInputLayer = {\n")
outputFile.write("    .numNeurons = CONFIG_NUM_INPUT_NEURONS,\n")
outputFile.write("    .activation = inputNeurons\n")
outputFile.write("};\n\n\n\n\n")



#
# Hidden layer
#
for i in range(numHiddenLayers):
    outputFile.write("/*\n")
    outputFile.write(" * Hidden Layer %d\n" % i)
    outputFile.write(" */\n")
    outputFile.write("static activation_t hiddenNeuronsActivations_%d[CONFIG_NUM_HIDDEN_NEURONS];\n" % i)
    outputFile.write("static bias_t hiddenNeuronBias_%d[CONFIG_NUM_HIDDEN_NEURONS];\n\n" % i)

    for j in range(numHiddenNeurons):
        if (i == 0):
            outputFile.write("static weight_t hiddenNeuronWeights_%d_%d[CONFIG_NUM_INPUT_NEURONS];\n" % (i, j))
        else:
            outputFile.write("static weight_t hiddenNeuronWeights_%d_%d[CONFIG_NUM_HIDDEN_NEURONS];\n" % (i, j))
    
    outputFile.write("static weight_t* hiddenNeuronWeights_%d[CONFIG_NUM_HIDDEN_NEURONS] =\n{\n" % i)

    for j in range(numHiddenNeurons - 1):
        outputFile.write("    hiddenNeuronWeights_%d_%d,\n" % (i, j))
    outputFile.write("    hiddenNeuronWeights_%d_%d\n" % (i, (numHiddenNeurons - 1)))
    outputFile.write("};\n\n")

    outputFile.write("static hiddenLayer_t staticHiddenLayer_%d =\n{\n" % i)
    outputFile.write("    .numNeurons = CONFIG_NUM_HIDDEN_NEURONS,\n")
    outputFile.write("    .activation = hiddenNeuronsActivations_%d,\n" % i)
    outputFile.write("    .bias = hiddenNeuronBias_%d,\n" % i)
    outputFile.write("    .weight = hiddenNeuronWeights_%d,\n" % i)
    outputFile.write("};\n\n\n")

outputFile.write("static hiddenLayer_t* staticHiddenLayers[CONFIG_NUM_HIDDEN_LAYERS] =\n{\n")
for i in range(numHiddenLayers - 1):
    outputFile.write("    &staticHiddenLayer_%d,\n" % i)
outputFile.write("    &staticHiddenLayer_%d\n" % (numHiddenLayers - 1))
outputFile.write("};\n\n\n\n\n")


#
# Output Layer
#
outputFile.write("/*\n")
outputFile.write(" * Output Layer\n")
outputFile.write(" */\n")
outputFile.write("static activation_t outputNeuronsActivations[CONFIG_NUM_OUTPUT_NEURONS];\n")
outputFile.write("static bias_t outputNeuronBias[CONFIG_NUM_OUTPUT_NEURONS];\n\n")

for i in range(numOutputNeurons):
    outputFile.write("static weight_t outputNeuronWeights_%d[CONFIG_NUM_HIDDEN_NEURONS];\n" % i)

outputFile.write("static weight_t* outputNeuronWeights[CONFIG_NUM_OUTPUT_NEURONS] =\n{\n")

for i in range(numOutputNeurons - 1):
    outputFile.write("    outputNeuronWeights_%d,\n" % i)
outputFile.write("    outputNeuronWeights_%d\n" % (numOutputNeurons - 1))
outputFile.write("};\n\n")

outputFile.write("static outputLayer_t staticOutputLayer =\n{\n")
outputFile.write("    .numNeurons = CONFIG_NUM_OUTPUT_NEURONS,\n")
outputFile.write("    .activation = outputNeuronsActivations,\n")
outputFile.write("    .bias = outputNeuronBias,\n")
outputFile.write("    .weight = outputNeuronWeights\n")
outputFile.write("};\n\n\n\n\n")



#
# Network structure
#
outputFile.write("static network_t staticNetwork = {\n")
outputFile.write("    .inputLayer = &staticInputLayer,\n")
outputFile.write("    .hiddenLayer = staticHiddenLayers,\n")
outputFile.write("    .outputLayer = &staticOutputLayer\n")
outputFile.write("};\n")
