configFile = open(".config", "r")
if "CONFIG_MEMORY_ALLOCATION_DYNAMIC=y" in configFile.read():
    raise SystemExit

configFile.seek(0)

outputFile = open("include/embann_static.h", "w+")
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
for i in range(numInputNeurons):
    outputFile.write("static uNeuron_t inputLayer_%d;\n" % i)

outputFile.write("static inputLayer_t staticInputLayer = {\n")
outputFile.write("    .numNeurons = CONFIG_NUM_INPUT_NEURONS,\n")
outputFile.write("    .neuron = {\n")

for i in range(numInputNeurons - 1):
    outputFile.write("        &inputLayer_%d,\n" % i)

outputFile.write("        &inputLayer_%d\n" % (numInputNeurons - 1))
outputFile.write("    }\n")
outputFile.write("};\n\n\n\n")




#
# Hidden layer
#
for i in range(numHiddenLayers):
    for j in range(numHiddenNeurons):
        if (i == 0):
            for k in range(numInputNeurons):
                outputFile.write("static neuronParams_t hiddenLayer_%d_%d_%d;\n" % (i, j, k))
        else:
            for k in range(numHiddenNeurons):
                outputFile.write("static neuronParams_t hiddenLayer_%d_%d_%d;\n" % (i, j, k))

for i in range(numHiddenLayers):
    for j in range(numHiddenNeurons):
        outputFile.write("static wNeuron_t hiddenLayer_%d_%d = {\n" % (i, j))
        outputFile.write("    .activation = 0,\n")
        outputFile.write("    .params = {\n")

        if (i == 0):
            for k in range(numInputNeurons - 1):
                outputFile.write("        &hiddenLayer_%d_%d_%d,\n" % (i, j, k))
            outputFile.write("        &hiddenLayer_%d_%d_%d\n" % (i, j, numInputNeurons - 1))
        else:
            for k in range(numHiddenNeurons - 1):
                outputFile.write("        &hiddenLayer_%d_%d_%d,\n" % (i, j, k))
            outputFile.write("        &hiddenLayer_%d_%d_%d\n" % (i, j, numHiddenNeurons - 1))

        outputFile.write("    }\n")
        outputFile.write("};\n")


    outputFile.write("static hiddenLayer_t staticHiddenLayer_%d = {\n" % i)
    outputFile.write("    .numNeurons = CONFIG_NUM_HIDDEN_NEURONS,\n")
    outputFile.write("    .neuron = {\n")

    for j in range(numHiddenNeurons - 1):
        outputFile.write("        &hiddenLayer_%d_%d,\n" % (i, j))

    outputFile.write("        &hiddenLayer_%d_%d\n" % (i, (numHiddenNeurons - 1)))
    outputFile.write("    }\n")
    outputFile.write("};\n\n\n\n")




#
# Output Layer
#
for i in range(numOutputNeurons):
    for j in range(numHiddenNeurons):
        outputFile.write("static neuronParams_t outputLayer_%d_%d;\n" % (i, j))

for i in range(numOutputNeurons):
    outputFile.write("static wNeuron_t outputLayer_%d = {\n" % i)
    outputFile.write("    .activation = 0,\n")
    outputFile.write("    .params = {\n")

    for j in range(numHiddenNeurons - 1):
        outputFile.write("        &outputLayer_%d_%d,\n" % (i, j))
    outputFile.write("        &outputLayer_%d_%d\n" % (i, numHiddenNeurons - 1))

    outputFile.write("    }\n")
    outputFile.write("};\n")


outputFile.write("static outputLayer_t staticOutputLayer = {\n")
outputFile.write("    .numNeurons = CONFIG_NUM_OUTPUT_NEURONS,\n")
outputFile.write("    .neuron = {\n")

for i in range(numOutputNeurons - 1):
    outputFile.write("        &outputLayer_%d,\n" % (i))

outputFile.write("        &outputLayer_%d\n" % (numOutputNeurons - 1))
outputFile.write("    }\n")
outputFile.write("};\n\n\n\n")



#
# Network structure
#
outputFile.write("static network_t staticNetwork = {\n")
outputFile.write("    .inputLayer = &staticInputLayer,\n")
outputFile.write("    .hiddenLayer = {\n")

for i in range(numHiddenLayers - 1):
    outputFile.write("        &staticHiddenLayer_%d,\n" % i)
outputFile.write("        &staticHiddenLayer_%d\n" % (numHiddenLayers - 1))
outputFile.write("    },\n")

outputFile.write("    .outputLayer = &staticOutputLayer\n")
outputFile.write("};\n")
