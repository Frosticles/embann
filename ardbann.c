/*
  Ardbann.cpp - ARDuino Backpropogating Artificial Neural Network.
  Created by Peter Frost, February 9, 2017.
  Released into the public domain.
*/

#include "ardbann.h"


static network_t network;
static void ardbann_calculateInputNeurons(void);
#ifndef ARDUINO
static uint32_t millis(void);
#endif

void ardbann_initWithData(  uint16_t rawInputArray[], uint16_t maxInput,
                            char *outputArray, uint16_t numInputs,
                            uint16_t numInputNeurons,
                            uint16_t numHiddenNeurons, uint8_t numHiddenLayers,
                            uint16_t numOutputNeurons)
{
    float outputLayerWeightTable[numOutputNeurons][numHiddenNeurons];
    float outputLayerBiases[numOutputNeurons];

    for (uint8_t i = 0; i < numOutputNeurons; i++)
    {
        outputLayerBiases[i] = ((float)rand())/RAND_MAX;
        for (uint16_t j = 0; j < numHiddenNeurons; j++)
        {
            outputLayerWeightTable[i][j] = ((float)rand())/RAND_MAX;
        }
    }

    float hiddenLayerNeuronTable[numHiddenLayers][numHiddenNeurons];
    float hiddenLayerNeuronBiasTable[numHiddenLayers][numHiddenNeurons];
    float hiddenLayerNeuronWeightLayerTable[numHiddenLayers][numHiddenNeurons][numInputNeurons];

    for (uint8_t i = 0; i < numHiddenLayers; i++)
    {
        for (uint16_t j = 0; j < numHiddenNeurons; j++)
        {
            hiddenLayerNeuronTable[i][j] = 0;
            hiddenLayerNeuronBiasTable[i][j] = ((float)rand())/RAND_MAX;
            for (uint16_t k = 0; k < numInputNeurons; k++)
            {
                hiddenLayerNeuronWeightLayerTable[i][j][k] =
                    ((float)rand())/RAND_MAX;
            }
        }
    }

    network.numLayers = numHiddenLayers + 2;
    network.networkResponse = 0;

    network.inputLayer.numNeurons = numInputNeurons;
    network.inputLayer.numRawInputs = numInputs;
    network.inputLayer.neurons = (float*) malloc(numInputNeurons * sizeof(float));
    network.inputLayer.rawInputs = rawInputArray;
    network.inputLayer.maxInput = maxInput;
    network.inputLayer.groupThresholds = (uint16_t*) malloc(numInputNeurons * sizeof(uint16_t));
    network.inputLayer.groupTotal = (uint16_t*) malloc(numInputNeurons * sizeof(uint16_t));

    ardbann_calculateInputNeurons();

    network.outputLayer.numNeurons = numOutputNeurons;
    network.outputLayer.neurons = (float*) malloc(numOutputNeurons * sizeof(float));
    network.outputLayer.weightTable = outputLayerWeightTable;
    network.outputLayer.stringArray = outputArray;
    network.outputLayer.neuronBiasTable = outputLayerBiases;

    network.hiddenLayer.numNeurons = numHiddenNeurons;
    network.hiddenLayer.numLayers = numHiddenLayers;
    network.hiddenLayer.neuronTable = hiddenLayerNeuronTable;
    network.hiddenLayer.weightLayerTable = hiddenLayerNeuronWeightLayerTable;
    network.hiddenLayer.neuronBiasTable = hiddenLayerNeuronBiasTable;
}

void ardbann_initWithoutData(   uint16_t maxInput, char* outputArray,
                                uint16_t numInputNeurons,
                                uint16_t numHiddenNeurons, uint8_t numHiddenLayers,
                                uint16_t numOutputNeurons)
{

    float outputLayerWeightTable[numOutputNeurons][numHiddenNeurons];
    float outputLayerBiases[numOutputNeurons];

    for (uint8_t i = 0; i < numOutputNeurons; i++)
    {
        outputLayerBiases[i] = ((float)rand())/RAND_MAX;
        for (uint16_t j = 0; j < numHiddenNeurons; j++)
        {
            outputLayerWeightTable[i][j] = ((float)rand())/RAND_MAX;
        }
    }

    float hiddenLayerNeuronTable[numHiddenLayers][numHiddenNeurons];
    float hiddenLayerNeuronBiasTable[numHiddenLayers][numHiddenNeurons];
    float hiddenLayerNeuronWeightLayerTable[numHiddenLayers][numHiddenNeurons][numInputNeurons];

    for (uint8_t i = 0; i < numHiddenLayers; i++)
    {
        for (uint16_t j = 0; j < numHiddenNeurons; j++)
        {
            hiddenLayerNeuronTable[i][j] = 0;
            hiddenLayerNeuronBiasTable[i][j] = ((float)rand())/RAND_MAX;
            for (uint16_t k = 0; k < numInputNeurons; k++)
            {
                hiddenLayerNeuronWeightLayerTable[i][j][k] =
                    ((float)rand())/RAND_MAX;
            }
        }
    }

    network.numLayers = numHiddenLayers + 2;
    network.networkResponse = 0;

    // network.inputLayer.numRawInputs = numInputs;
    // network.inputLayer.rawInputs = rawInputArray;
    //
    // If initialising with this method, you must call NewInput()
    // with some inputs before you can use the network, to set these
    //
    network.inputLayer.numNeurons = numInputNeurons;
    network.inputLayer.neurons = (float*) malloc(numInputNeurons * sizeof(float));
    network.inputLayer.maxInput = maxInput;
    network.inputLayer.groupThresholds = (uint16_t*) malloc(numInputNeurons * sizeof(uint16_t));
    network.inputLayer.groupTotal = (uint16_t*) malloc(numInputNeurons * sizeof(uint16_t));

    network.outputLayer.numNeurons = numOutputNeurons;
    network.outputLayer.neurons = (float*) malloc(numOutputNeurons * sizeof(float));
    network.outputLayer.weightTable = outputLayerWeightTable;
    network.outputLayer.stringArray = outputArray;
    network.outputLayer.neuronBiasTable = outputLayerBiases;

    network.hiddenLayer.numNeurons = numHiddenNeurons;
    network.hiddenLayer.numLayers = numHiddenLayers;
    network.hiddenLayer.neuronTable = hiddenLayerNeuronTable;
    network.hiddenLayer.weightLayerTable = hiddenLayerNeuronWeightLayerTable;
    network.hiddenLayer.neuronBiasTable = hiddenLayerNeuronBiasTable;
}

void ardbann_newInputRaw(uint16_t rawInputArray[], uint16_t numInputs)
{
    network.inputLayer.rawInputs = rawInputArray;
    network.inputLayer.numRawInputs = numInputs;
    ardbann_calculateInputNeurons();
}

void ardbann_newInputStruct(networkSampleBuffer_t sampleBuffer, uint16_t numInputs)
{
    network.inputLayer.rawInputs = sampleBuffer.samples;
    network.inputLayer.numRawInputs = numInputs;
    ardbann_calculateInputNeurons();
}

void ardbann_calculateInputNeurons()
{
    uint8_t largestGroup = 0;

    for (uint16_t i = 0; i < network.inputLayer.numNeurons; i++)
    {
        network.inputLayer.groupThresholds[i] =
            ((network.inputLayer.maxInput + 1) / network.inputLayer.numNeurons) *
            (i + 1);
        network.inputLayer.groupTotal[i] = 0;
        // printf(network.inputLayer.groupThresholds[i]);
    }

    for (uint16_t i = 0; i < network.inputLayer.numRawInputs; i++)
    {
        for (uint16_t j = 0; j < network.inputLayer.numNeurons; j++)
        {
            if (network.inputLayer.rawInputs[i] <=
                network.inputLayer.groupThresholds[j])
            {
                // printf("%d + 1 in group %d, ",
                // network.inputLayer.groupTotal[j],
                //              j);
                network.inputLayer.groupTotal[j] += 1;
                break;
            }
        }
    }

    for (uint16_t i = 0; i < network.inputLayer.numNeurons; i++)
    {
        if (network.inputLayer.groupTotal[i] >
            network.inputLayer.groupTotal[largestGroup])
        {
            largestGroup = i;
        }
    }

    for (uint16_t i = 0; i < network.inputLayer.numNeurons; i++)
    {
        // printf("group total i: %u group total largest: %u ",
        //              network.inputLayer.groupTotal[i],
        //              network.inputLayer.groupTotal[largestGroup]);
        network.inputLayer.neurons[i] = network.inputLayer.groupTotal[i] /
                                        network.inputLayer.groupTotal[largestGroup];
        // printf("input neuron %d = %.3f, ", i,
        // network.inputLayer.neurons[i]);
    }
}

uint8_t ardbann_inputLayer()
{
    ardbann_sumAndSquash(network.inputLayer.neurons, network.hiddenLayer.neuronTable[0],
                 network.hiddenLayer.neuronBiasTable[0],
                 network.hiddenLayer.weightLayerTable[0],
                 network.inputLayer.numNeurons, network.hiddenLayer.numNeurons);
    // printf("Done Input -> 1st Hidden Layer");
    for (uint8_t i = 1; i < network.hiddenLayer.numLayers; i++)
    {
        ardbann_sumAndSquash(network.hiddenLayer.neuronTable[i - 1],
                     network.hiddenLayer.neuronTable[i],
                     network.hiddenLayer.neuronBiasTable[i],
                     network.hiddenLayer.weightLayerTable[i],
                     network.hiddenLayer.numNeurons,
                     network.hiddenLayer.numNeurons);
        // printf("Done Hidden Layer %d -> Hidden Layer %d\n", i - 1, i);
    }

    ardbann_sumAndSquash(
        network.hiddenLayer.neuronTable[network.hiddenLayer.numLayers - 1],
        network.outputLayer.neurons, network.outputLayer.neuronBiasTable,
        network.outputLayer.weightTable, network.hiddenLayer.numNeurons,
        network.outputLayer.numNeurons);

    /*printf("Done Hidden Layer %d -> Output Layer\n",
                network.hiddenLayer.numLayers);*/

    network.networkResponse = ardbann_outputLayer();
    return network.networkResponse;
}

void ardbann_sumAndSquash(float *Input, float *Output, float *Bias,
                           float **Weights, uint16_t numInputs,
                           uint16_t numOutputs)
{
    for (uint16_t i = 0; i < numOutputs; i++)
    {
        Output[i] = 0; // Bias[i];
        for (uint16_t j = 0; j < numInputs; j++)
        {
            Output[i] += Input[j] * Weights[i][j];
        }
        Output[i] = tanh(Output[i] * PI);

        // tanh is a quicker alternative to sigmoid
        // printf("i:%d This is the ardbann_SumAndSquash Output %.2f\n", i,
        // Output[i]);
    }
}

uint8_t ardbann_outputLayer()
{
    uint8_t mostLikelyOutput = 0;

    for (uint16_t i = 0; i < network.outputLayer.numNeurons; i++)
    {
        if (network.outputLayer.neurons[i] >
            network.outputLayer.neurons[mostLikelyOutput])
        {
            mostLikelyOutput = i;
        }
        // printf("i: %d neuron: %-3f likely: %d\n", i,
        // network.outputLayer.neurons[i], mostLikelyOutput);
    }
    return mostLikelyOutput;
}

void ardbann_printNetwork()
{
    printf("\nInput: [");
    for (uint16_t i = 0; i < (network.inputLayer.numRawInputs - 1); i++)
    {
        printf("%d, ", network.inputLayer.rawInputs[i]);
    }
    printf("%d]",network.inputLayer.rawInputs[network.inputLayer.numRawInputs - 1]);

    printf("\nInput Layer | Hidden Layer ");
    if (network.hiddenLayer.numLayers > 1)
    {
        printf("1 ");
        for (uint8_t i = 2; i <= network.hiddenLayer.numLayers; i++)
        {
            printf("| Hidden Layer %d ", i);
        }
    }
    printf("| Output Layer");

    bool nothingLeft = false;
    uint16_t i = 0;
    while (nothingLeft == false)
    {
        if ((i >= network.inputLayer.numNeurons) &&
            (i >= network.hiddenLayer.numNeurons) &&
            (i >= network.outputLayer.numNeurons))
        {
            nothingLeft = true;
        }
        else
        {
            if (i < network.inputLayer.numNeurons)
            {
                printf("%-12.3f| ", network.inputLayer.neurons[i]);
            }
            else
            {
                printf("            | ");
            }

            if (i < network.hiddenLayer.numNeurons)
            {
                if (network.hiddenLayer.numLayers == 1)
                {
                    printf("%-13.3f| ", network.hiddenLayer.neuronTable[0][i]);
                }
                else
                {
                    for (uint8_t j = 0; j < network.hiddenLayer.numLayers; j++)
                    {
                        printf("%-15.3f| ", network.hiddenLayer.neuronTable[j][i]);
                    }
                }
            }
            else
            {
                printf("             | ");
                if (network.hiddenLayer.numLayers > 1)
                {
                    printf("              | ");
                }
            }

            if (i < network.outputLayer.numNeurons)
            {
                printf("%.3f", network.outputLayer.neurons[i]);
            }
        }
        printf("\n");
        i++;
    }

    printf("I think this is output %d which is ", network.networkResponse);
    printf("%s \n", network.outputLayer.stringArray[network.networkResponse]);
}

void ardbann_trainDriverInTime(float learningRate, bool verbose,
                          uint8_t numTrainingSets, uint8_t inputPin,
                          uint16_t bufferSize, long numSeconds)
{
    char serialInput[10];
    uint16_t trainingData[network.outputLayer.numNeurons][numTrainingSets]
                         [bufferSize],
        randomOutput, randomTrainingSet;

    for (uint8_t i = 0; i < network.outputLayer.numNeurons; i++)
    {
        printf("\nAttach the sensor to material %u: ", i);
        printf("%s", network.outputLayer.stringArray[i]);
        scanf("%s", serialInput);
        while (serialInput[0] == 0)
        {
            scanf("%s", serialInput);
        }
        printf("\nReading...");
        for (uint8_t j = 0; j < numTrainingSets; j++)
        {
            printf("%u...", j + 1);
            for (uint16_t k = 0; k < bufferSize; k++)
            {
                trainingData[i][j][k] = analogRead(inputPin);
                printf("%u, ", trainingData[i][j][k]);
            }
            printf("\n");
            delay(50);
        }
    }

    if (verbose == true)
    {
        printf("\nOutput Errors: \n");
    }

    unsigned long startTime = millis();
    numSeconds *= 1000;

    while ((millis() - startTime) < numSeconds)
    {
        randomOutput = rand() % network.outputLayer.numNeurons;
        randomTrainingSet = rand() % numTrainingSets;
        ardbann_newInputRaw(trainingData[randomOutput][randomTrainingSet], bufferSize);
        ardbann_inputLayer();

        if (verbose == true)
        {
            ardbann_errorReporting(randomOutput);
            printf("%u | %u ", randomOutput, randomTrainingSet);
        }

        ardbann_train(randomOutput, learningRate);
    }
}

void ardbann_trainDriverInError(float learningRate, bool verbose,
                          uint8_t numTrainingSets, uint8_t inputPin,
                          uint16_t bufferSize, float desiredCost)
{
    char serialInput[10];
    uint16_t trainingData[network.outputLayer.numNeurons][numTrainingSets]
                         [bufferSize],
        randomOutput, randomTrainingSet;
    float currentCost[network.outputLayer.numNeurons];
    bool converged = false;

    for (uint8_t i = 0; i < network.outputLayer.numNeurons; i++)
    {
        printf("\nAttach the sensor to material %u: ", i);
        printf("%s", network.outputLayer.stringArray[i]);
        scanf("%s", serialInput);
        while (serialInput[0] == 0)
        {
            scanf("%s", serialInput);
        }
        printf("Reading...");
        for (uint8_t j = 0; j < numTrainingSets; j++)
        {
            printf("%u...", j + 1);
            for (uint8_t k = 0; k < bufferSize; k++)
            {
                trainingData[i][j][k] = analogRead(inputPin);
                printf("%u, ", trainingData[i][j][k]);
            }
            printf("\n");
            delay(50);
        }
    }

    if (verbose == true)
    {
        printf("\nOutput Errors: \n");
    }

    while (!converged)
    {
        randomOutput = rand() % network.outputLayer.numNeurons;
        randomTrainingSet = rand() % numTrainingSets;
        currentCost[randomOutput] = 0.0;
        ardbann_newInputRaw(trainingData[randomOutput][randomTrainingSet], bufferSize);
        ardbann_inputLayer();

        if (verbose == true)
        {
            ardbann_errorReporting(randomOutput);
            printf("%u | %u ", randomOutput, randomTrainingSet);
        }

        ardbann_train(randomOutput, learningRate);
        for (uint8_t i = 0; i < network.outputLayer.numNeurons; i++)
        {
            if (i == randomOutput)
            {
                currentCost[randomOutput] += pow(1 - network.outputLayer.neurons[i], 2);
            }
            else
            {
                currentCost[randomOutput] += pow(network.outputLayer.neurons[i], 2);
            }
        }
        currentCost[randomOutput] /= network.outputLayer.numNeurons;

        for (uint8_t i = 0; i < network.outputLayer.numNeurons; i++)
        {
            printf("%f", currentCost[i]);
            printf(", ");
            if (currentCost[i] > desiredCost)
            {
                break;
            }
            if (i == (network.outputLayer.numNeurons - 1))
            {
                converged = true;
            }
        }
        printf("%f", desiredCost);
    }
}

void ardbann_train(uint8_t correctOutput, float learningRate)
{
    float dOutputErrorToOutputSum[network.outputLayer.numNeurons];
    float dTotalErrorToHiddenNeuron = 0.0;
    float outputNeuronWeightChange[network.outputLayer.numNeurons]
                                  [network.hiddenLayer.numNeurons];

    for (uint16_t i = 0; i < network.outputLayer.numNeurons; i++)
    {
        if (i == correctOutput)
        {
            dOutputErrorToOutputSum[i] =
                (1 - network.outputLayer.neurons[i]) *
                ardbann_tanhDerivative(network.outputLayer.neurons[i]);
        }
        else
        {
            dOutputErrorToOutputSum[i] =
                -network.outputLayer.neurons[i] *
                ardbann_tanhDerivative(network.outputLayer.neurons[i]);
        }
        // printf("\ndOutputErrorToOutputSum[%d]: %.3f", i,
        // dOutputErrorToOutputSum[i]);
        for (uint16_t j = 0; j < network.hiddenLayer.numNeurons; j++)
        {
            outputNeuronWeightChange[i][j] =
                dOutputErrorToOutputSum[i] *
                network.hiddenLayer
                    .neuronTable[network.hiddenLayer.numLayers - 1][j] *
                learningRate;
            // printf("\n  outputNeuronWeightChange[%d][%d]: %.3f", i, j,
            //              outputNeuronWeightChange[i][j]);
        }
    }

    for (uint16_t i = 0; i < network.hiddenLayer.numNeurons; i++)
    {
        dTotalErrorToHiddenNeuron = 0.0;
        for (uint16_t j = 0; j < network.outputLayer.numNeurons; j++)
        {
            dTotalErrorToHiddenNeuron +=
                dOutputErrorToOutputSum[j] * network.outputLayer.weightTable[j][i];
            // printf("\nOld Output Weight[%d][%d]: %.3f", i, j,
            // network.outputLayer.weightTable[j][i]);
            network.outputLayer.weightTable[j][i] += outputNeuronWeightChange[j][i];
            // printf("\nNew Output Weight[%d][%d]: %.3f", i, j,
            // network.outputLayer.weightTable[j][i]);
        }
        for (uint16_t k = 0; k < network.inputLayer.numNeurons; k++)
        {
            // printf("\nOld Hidden Weight[%d][%d]: %.3f", i, k,
            // network.hiddenLayer.weightLayerTable[0][i][k]);
            network.hiddenLayer.weightLayerTable[0][i][k] +=
                dTotalErrorToHiddenNeuron *
                ardbann_tanhDerivative(network.hiddenLayer.neuronTable[0][i]) *
                network.inputLayer.neurons[k] * learningRate;
            // printf("\nNew Hidden Weight[%d][%d]: %.3f", i, k,
            // network.hiddenLayer.weightLayerTable[0][i][k]);
        }
    }
}

float ardbann_tanhDerivative(float inputValue)
{
    // if (inputValue < 0)
    //{
    //  return -1 * (1 - pow(tanh(inputValue), 2));
    //}
    // else
    //{
    return 1 - pow(tanh(inputValue * PI), 2);
    //}
}

void ardbann_printInputNeuronDetails(uint8_t neuronNum)
{
    if (neuronNum < network.inputLayer.numNeurons)
    {
        printf("\nInput Neuron %d: %.3f\n", neuronNum,
                      network.inputLayer.neurons[neuronNum]);
    }
    else
    {
        printf("\nERROR: You've asked for input neuron %d when only %d exist\n",
            neuronNum, network.inputLayer.numNeurons);
    }
}

void ardbann_printOutputNeuronDetails(uint8_t neuronNum)
{
    if (neuronNum < network.outputLayer.numNeurons)
    {

        printf("\nOutput Neuron %d:\n", neuronNum);

        for (uint16_t i = 0; i < network.hiddenLayer.numNeurons; i++)
        {
            printf(
                "%.3f-*->%.3f |",
                network.hiddenLayer.neuronTable[network.hiddenLayer.numLayers - 1][i],
                network.outputLayer.weightTable[neuronNum][i]);

            if (i == floor(network.hiddenLayer.numNeurons / 2))
            {
                printf(" = %.3f", network.outputLayer.neurons[neuronNum]);
            }
            printf("\n");
        }
    }
    else
    {
        printf(
            "\nERROR: You've asked for output neuron %d when only %d exist\n",
            neuronNum, network.outputLayer.numNeurons);
    }
}

void ardbann_printHiddenNeuronDetails(uint8_t layerNum, uint8_t neuronNum)
{
    if (neuronNum < network.hiddenLayer.numNeurons)
    {

        printf("\nHidden Neuron %d:\n", neuronNum);

        if (layerNum == 0)
        {

            for (uint16_t i = 0; i < network.inputLayer.numNeurons; i++)
            {
                printf("%.3f-*->%.3f |", network.inputLayer.neurons[i],
                              network.hiddenLayer.weightLayerTable[0][neuronNum][i]);

                if (i == floor(network.inputLayer.numNeurons / 2))
                {
                    printf(" = %.3f",
                                  network.hiddenLayer.neuronTable[0][neuronNum]);
                }
                printf("\n");
            }
        }
        else
        {

            for (uint16_t i = 0; i < network.hiddenLayer.numNeurons; i++)
            {
                printf(
                    "%.3f-*->%.3f |", network.hiddenLayer.neuronTable[layerNum - 1][i],
                    network.hiddenLayer.weightLayerTable[layerNum][neuronNum][i]);

                if (i == floor(network.hiddenLayer.numNeurons / 2))
                {
                    printf(" = %.3f",
                                  network.hiddenLayer.neuronTable[0][neuronNum]);
                }
                printf("\n");
            }
        }
    }
    else
    {
        printf(
            "\nERROR: You've asked for hidden neuron %d when only %d exist\n",
            neuronNum, network.hiddenLayer.numNeurons);
    }
}

void ardbann_errorReporting(uint8_t correctResponse)
{
    printf("\n");
    for (uint8_t i = 0; i < network.outputLayer.numNeurons; i++)
    {
        if (i == correctResponse)
        {
            printf("%-7.3f | ",
                          (1 - network.outputLayer.neurons[correctResponse]));
        }
        else
        {
            printf("%-7.3f | ", -network.outputLayer.neurons[i]);
        }
    }
}

void ardbann_benchmark(void)
{
    uint32_t testint = UINT32_MAX;
    float testfloat = FLT_MAX;
    struct timespec timeBefore;
    struct timespec timeAfter;
    struct timespec averageTime = 
    {
        .tv_nsec = 0,
    };

    for (uint8_t i = 0; i < 5; i++)
    {
        clock_gettime(CLOCK_REALTIME, &timeBefore);

        for (uint32_t i = 0; i < INT32_MAX; i++)
        {
            testint /= 2;
            testint += 5;
        }

        clock_gettime(CLOCK_REALTIME, &timeAfter);

        averageTime.tv_nsec += timeAfter.tv_nsec - timeBefore.tv_nsec;
        printf("Integer time was %ul nanoseconds, result %ul", timeAfter.tv_nsec - timeBefore.tv_nsec, testint);
    }

    averageTime.tv_nsec /= 5;
    printf("Average integer time was %ul nanoseconds", averageTime.tv_nsec);
    averageTime.tv_nsec = 0;

    for (uint8_t i = 0; i < 5; i++)
    {
        clock_gettime(CLOCK_REALTIME, &timeBefore);

        for (uint32_t i = 0; i < INT32_MAX; i++)
        {
            testfloat *= 0.5;
            testfloat += 5;
        }

        clock_gettime(CLOCK_REALTIME, &timeAfter);

        averageTime.tv_nsec += timeAfter.tv_nsec - timeBefore.tv_nsec;
        printf("Float time was %ul nanoseconds, result %.2f", timeAfter.tv_nsec - timeBefore.tv_nsec, testfloat);
    }

    averageTime.tv_nsec /= 5;
    printf("Average float time was %ul nanoseconds", averageTime.tv_nsec);
}

#ifndef ARDUINO
uint32_t millis(void)
{
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return (uint32_t) round(time.tv_nsec / 1000000);
}
#endif