/*
  Embann.cpp - EMbedded Backpropogating Artificial Neural Network.
  Created by Peter Frost, 27th August 2019
*/

#include "embann.h"


static network_t* network;
static trainingDataCollection_t trainingDataCollection = {
    .tail = NULL,
    .head = NULL,
    .numEntries = 0
};
static void embann_initInputLayer(uint16_t numInputNeurons);
static void embann_initHiddenLayer(uint16_t numHiddenNeurons,
                                   uint8_t numHiddenLayers,
                                   uint16_t numInputNeurons);
static void embann_initOutputLayer(uint16_t numOutputNeurons,
                                   uint16_t numHiddenNeurons);
#ifndef ARDUINO
static uint32_t millis(void);
#endif

/* Random float between -1 and 1 */
#define RAND_WEIGHT() ((float)rand() / (RAND_MAX / 2)) - 1
/* Throw an error if malloc failed */
#define CHECK_MALLOC(a) if (!a) {abort();}


int __attribute__((weak)) main(int argc, char const *argv[])
{
    embann_benchmark();
    embann_init(10, 10, 1, 10);
}


void embann_init(uint16_t numInputNeurons,
                 uint16_t numHiddenNeurons, 
                 uint8_t numHiddenLayers,
                 uint16_t numOutputNeurons)
{
    network = (network_t*) malloc(sizeof(network_t) + 
                                 (sizeof(hiddenLayer_t) * numHiddenLayers));
    CHECK_MALLOC(network);

    embann_initInputLayer(numInputNeurons);
    embann_initHiddenLayer(numHiddenNeurons,
                           numHiddenLayers,
                           numInputNeurons);
    embann_initOutputLayer(numOutputNeurons,
                           numHiddenNeurons);

    network->properties.numLayers = numHiddenLayers + 2;
    network->properties.numHiddenLayers = numHiddenLayers;
    network->properties.networkResponse = 0;
}

static void embann_initInputLayer(uint16_t numInputNeurons)
{
    inputLayer_t* inputLayer = (inputLayer_t*) malloc(sizeof(inputLayer_t) + 
                                                (sizeof(uNeuron_t*) * numInputNeurons));
    CHECK_MALLOC(inputLayer);

    printf("inputLayer: 0x%x, size: %d\n", (uint32_t) inputLayer, sizeof(inputLayer_t) + 
                                                (sizeof(uNeuron_t*) * numInputNeurons));

    inputLayer->numNeurons = numInputNeurons;
    for (uint8_t i = 0; i < numInputNeurons; i++)
    {
        uNeuron_t* pNeuron = (uNeuron_t*) malloc(sizeof(uNeuron_t));
        inputLayer->neuron[i] = pNeuron;
        inputLayer->neuron[i]->activation = 0.0F;
    }
    network->inputLayer = *inputLayer;

    printf("done input\n");
}

static void embann_initHiddenLayer(uint16_t numHiddenNeurons,
                                   uint8_t numHiddenLayers,
                                   uint16_t numInputNeurons)
{
    for (uint8_t i = 0; i < numHiddenLayers; i++)
    {
        hiddenLayer_t* hiddenLayer = (hiddenLayer_t*) malloc(sizeof(hiddenLayer_t) + 
                                                (sizeof(wNeuron_t*) * numHiddenNeurons));
        CHECK_MALLOC(hiddenLayer);

        printf("hiddenlayer: 0x%x, size: %d\n", (uint32_t) hiddenLayer, sizeof(hiddenLayer_t) + 
                                                (sizeof(wNeuron_t*) * numHiddenNeurons));

        for (uint16_t j = 0; j < numHiddenNeurons; j++)
        {    
            wNeuron_t* pNeuron = (wNeuron_t*) malloc(sizeof(wNeuron_t) + (sizeof(neuronParams_t*) * (i == 0 ? numInputNeurons : numHiddenNeurons)));
            hiddenLayer->neuron[j] = pNeuron;
            hiddenLayer->neuron[j]->activation = 0.0F;

            for (uint16_t k = 0; k < (i == 0 ? numInputNeurons : numHiddenNeurons); k++)
            {
                neuronParams_t* hiddenLayerParams = (neuronParams_t*) malloc(sizeof(neuronParams_t));
                CHECK_MALLOC(hiddenLayerParams);

                hiddenLayer->neuron[j]->params[k] = hiddenLayerParams;

                //printf("params array: 0x%x, bias 0x%x, weight 0x%x\n", (uint32_t) &hiddenLayer->neuron[j]->params[k],
                //                    (uint32_t)&hiddenLayer->neuron[j]->params[k]->bias,
                //                    (uint32_t)&hiddenLayer->neuron[j]->params[k]->weight);

                hiddenLayer->neuron[j]->params[k]->bias = RAND_WEIGHT();
                hiddenLayer->neuron[j]->params[k]->weight = RAND_WEIGHT();
            }
        }

        printf("done hidden\n");

        network->hiddenLayer[i] = *hiddenLayer;

        printf("hiddenlayer[i]: 0x%x\n", (uint32_t) &network->hiddenLayer[i]);
    }
}

static void embann_initOutputLayer(uint16_t numOutputNeurons,
                                   uint16_t numHiddenNeurons)
{
    outputLayer_t* outputLayer = (outputLayer_t*) malloc(sizeof(outputLayer_t) + 
                                                        (sizeof(wNeuron_t*) * numOutputNeurons));
    CHECK_MALLOC(outputLayer);

    printf("outputLayer: 0x%x, size: %d\n", (uint32_t) outputLayer, sizeof(outputLayer_t) + 
                                                (sizeof(wNeuron_t*) * numOutputNeurons));

    for (uint8_t i = 0; i < numOutputNeurons; i++)
    {
        wNeuron_t* pNeuron = (wNeuron_t*) malloc(sizeof(wNeuron_t) + (sizeof(neuronParams_t*) * numHiddenNeurons));
        outputLayer->neuron[i] = pNeuron;
        outputLayer->neuron[i]->activation = 0.0F;
        
        for (uint16_t j = 0; j < numHiddenNeurons; j++)
        {
            neuronParams_t* outputNeuronParams = (neuronParams_t*) malloc(sizeof(neuronParams_t));
            CHECK_MALLOC(outputNeuronParams);

            outputLayer->neuron[i]->params[j] = outputNeuronParams;
            
            outputLayer->neuron[i]->params[j]->bias = RAND_WEIGHT();
            outputLayer->neuron[i]->params[j]->weight = RAND_WEIGHT();
        }
    }
    outputLayer->numNeurons = numOutputNeurons;
    network->outputLayer = *outputLayer;

    printf("done output\n");
}

void embann_inputRaw(float data[])
{
    for (uint32_t i = 0; i < network->inputLayer.numNeurons; i++)
    {
        network->inputLayer.neuron[i]->activation = data[i];
    }
}

void embann_inputMinMaxScale(uint8_t data[], uint8_t min, uint8_t max)
{
    for (uint32_t i = 0; i < network->inputLayer.numNeurons; i++)
    {
        network->inputLayer.neuron[i]->activation = ((float)(data[i] - min)) / (max - min);
    }
}

void embann_inputStandardizeScale(uint8_t data[], float mean, float stdDev)
{
    for (uint32_t i = 0; i < network->inputLayer.numNeurons; i++)
    {
        network->inputLayer.neuron[i]->activation = ((float)data[i] - mean) / stdDev;
    }
}

float embann_getTrainingDataMean(void)
{
    float mean = 0.0F;
    uint32_t sum = 0;
    trainingData_t* pTrainingData = trainingDataCollection.head;
    
    while (pTrainingData)
    {
        for (uint32_t j = 0; j < pTrainingData->length; j++)
        {
            sum += pTrainingData->data[j];
        }
        mean += sum / pTrainingData->length;
        sum = 0;

        pTrainingData = pTrainingData->next;
    }

    mean /= trainingDataCollection.numEntries;

    return mean;
}

float embann_getTrainingDataStdDev(void)
{
    float stdDev = 0.0F;
    float sumofSquares = 0.0F;
    float mean = embann_getTrainingDataMean();
    trainingData_t* pTrainingData = trainingDataCollection.head;
    
    while (pTrainingData)
    {
        for (uint32_t j = 0; j < pTrainingData->length; j++)
        {
            sumofSquares += powf((float)pTrainingData->data[j] - mean, 2.0F);
        }
        sumofSquares /= pTrainingData->length;
        pTrainingData = pTrainingData->next;
    }

    stdDev = sqrtf(sumofSquares);

    return stdDev;
}

uint8_t embann_getTrainingDataMax(void)
{
    trainingData_t* pTrainingData = trainingDataCollection.head;
    uint8_t max;
    if (pTrainingData)
    {
        max = pTrainingData->data[0];
    }
    else
    {
        abort();
    }
    
    
    while (pTrainingData)
    {
        for (uint32_t j = 0; j < pTrainingData->length; j++)
        {
            if (pTrainingData->data[j] > max)
            {
                max = pTrainingData->data[j];
            }
        }
        pTrainingData = pTrainingData->next;
    }

    return max;
}

uint8_t embann_getTrainingDataMin(void)
{
    trainingData_t* pTrainingData = trainingDataCollection.head;
    uint8_t min;
    if (pTrainingData)
    {
        min = pTrainingData->data[0];
    }
    else
    {
        abort();
    }

    while (pTrainingData)
    {
        for (uint32_t j = 0; j < pTrainingData->length; j++)
        {
            if (pTrainingData->data[j] < min)
            {
                min = pTrainingData->data[j];
            }
        }
        pTrainingData = pTrainingData->next;
    }

    return min;
}


uint8_t embann_inputLayer()
{
    embann_sumAndSquash(network->inputLayer.neuron, 
                        network->hiddenLayer[0].neuron,
                        network->inputLayer.numNeurons, 
                        network->hiddenLayer[0].numNeurons);
    // printf("Done Input -> 1st Hidden Layer");
    for (uint8_t i = 1; i < network->properties.numHiddenLayers; i++)
    {
        embann_sumAndSquash(network->hiddenLayer[i - 1].neuron,
                            network->hiddenLayer[i].neuron,
                            network->hiddenLayer[i - 1].numNeurons,
                            network->hiddenLayer[i].numNeurons);
        // printf("Done Hidden Layer %d -> Hidden Layer %d\n", i - 1, i);
    }

    embann_sumAndSquash(
        network->hiddenLayer[network->properties.numHiddenLayers - 1].neuron,
        network->outputLayer.neuron, 
        network->hiddenLayer[network->properties.numHiddenLayers - 1].numNeurons,
        network->outputLayer.numNeurons);

    /*printf("Done Hidden Layer %d -> Output Layer\n",
                network->properties.numHiddenLayers);*/

    network->properties.networkResponse = embann_outputLayer();
    return network->properties.networkResponse;
}

void embann_sumAndSquash(wNeuron_t* Input[], wNeuron_t* Output[], uint16_t numInputs,
                           uint16_t numOutputs)
{
    for (uint16_t i = 0; i < numOutputs; i++)
    {
        Output[i]->activation = 0; // Bias[i];
        for (uint16_t j = 0; j < numInputs; j++)
        {
            Output[i]->activation += Input[j]->activation * Output[i]->params[j]->weight;
        }
        Output[i]->activation = tanh(Output[i]->activation * PI);

        // tanh is a quicker alternative to sigmoid
        // printf("i:%d This is the embann_SumAndSquash Output %.2f\n", i,
        // Output[i]);
    }
}

void embann_addTrainingData(uint8_t data[], uint32_t length, uint16_t correctResponse)
{
    trainingData_t* trainingDataNode;

    trainingDataNode = (trainingData_t*) malloc(sizeof(trainingData_t) + length);
    CHECK_MALLOC(trainingDataNode);

    trainingDataNode->prev = trainingDataCollection.tail;
    trainingDataNode->next = NULL;
    trainingDataNode->length = length;
    trainingDataNode->correctResponse = correctResponse;
    trainingDataNode->data[0] = data[0];

    if (trainingDataCollection.head == NULL)
    {
        trainingDataCollection.head = trainingDataNode;
        trainingDataCollection.tail = trainingDataNode;
    }
    else
    {
        trainingDataCollection.tail->next = trainingDataNode;
        trainingDataCollection.tail = trainingDataNode;
    }

    ++trainingDataCollection.numEntries;
}

void embann_copyTrainingData(uint8_t data[], uint32_t length, uint16_t correctResponse)
{
    trainingData_t* trainingDataNode;

    trainingDataNode = (trainingData_t*) malloc(sizeof(trainingData_t) + length);
    CHECK_MALLOC(trainingDataNode);

    trainingDataNode->prev = trainingDataCollection.tail;
    trainingDataNode->next = NULL;
    trainingDataNode->length = length;
    trainingDataNode->correctResponse = correctResponse;
    memcpy(trainingDataNode->data, data, length);

    if (trainingDataCollection.head == NULL)
    {
        trainingDataCollection.head = trainingDataNode;
        trainingDataCollection.tail = trainingDataNode;
    }
    else
    {
        trainingDataCollection.tail->next = trainingDataNode;
        trainingDataCollection.tail = trainingDataNode;
    }

    ++trainingDataCollection.numEntries;
}

void embann_shuffleTrainingData(void)
{

}

void embann_trainDriverInTime(float learningRate, long numSeconds, bool verbose)
{
    uint16_t randomOutput, randomTrainingSet;

    if (verbose == true)
    {
        printf("\nOutput Errors: \n");
    }

    unsigned long startTime = millis();
    numSeconds *= 1000;

    while ((millis() - startTime) < numSeconds)
    {
        randomOutput = rand() % network->outputLayer.numNeurons;
        randomTrainingSet = rand() % trainingDataCollection.numEntries;

        /*
            TODO, these are not 'right' but they will let the program run
        */
        embann_inputMinMaxScale(trainingDataCollection.head->data, 0, UINT8_MAX);
        embann_inputLayer();

        if (verbose == true)
        {
            embann_errorReporting(randomOutput);
            printf("%u | %u ", randomOutput, randomTrainingSet);
        }

        embann_train(randomOutput, learningRate);
    }
}

void embann_trainDriverInError(float learningRate, float desiredCost, bool verbose)
{
    uint16_t randomOutput, randomTrainingSet;
    float currentCost[network->outputLayer.numNeurons];
    bool converged = false;

    if (verbose == true)
    {
        printf("\nOutput Errors: \n");
    }

    while (!converged)
    {
        randomOutput = rand() % network->outputLayer.numNeurons;
        randomTrainingSet = rand() % trainingDataCollection.numEntries;
        currentCost[randomOutput] = 0.0;
        embann_inputMinMaxScale(trainingDataCollection.head->data, 0, UINT8_MAX);
        embann_inputLayer();

        if (verbose == true)
        {
            embann_errorReporting(randomOutput);
            printf("%u | %u ", randomOutput, randomTrainingSet);
        }

        embann_train(randomOutput, learningRate);
        for (uint8_t i = 0; i < network->outputLayer.numNeurons; i++)
        {
            if (i == randomOutput)
            {
                currentCost[randomOutput] += pow(1 - network->outputLayer.neuron[i]->activation, 2);
            }
            else
            {
                currentCost[randomOutput] += pow(network->outputLayer.neuron[i]->activation, 2);
            }
        }
        currentCost[randomOutput] /= network->outputLayer.numNeurons;

        for (uint8_t i = 0; i < network->outputLayer.numNeurons; i++)
        {
            printf("%f", currentCost[i]);
            printf(", ");
            if (currentCost[i] > desiredCost)
            {
                break;
            }
            if (i == (network->outputLayer.numNeurons - 1))
            {
                converged = true;
            }
        }
        printf("%f", desiredCost);
    }
}

void embann_train(uint8_t correctOutput, float learningRate)
{
    float dOutputErrorToOutputSum[network->outputLayer.numNeurons];
    float dTotalErrorToHiddenNeuron = 0.0;
    /* TODO, add support for multiple hidden layers */
    float outputNeuronWeightChange[network->outputLayer.numNeurons]
                                  [network->hiddenLayer[0].numNeurons];

    for (uint16_t i = 0; i < network->outputLayer.numNeurons; i++)
    {
        if (i == correctOutput)
        {
            dOutputErrorToOutputSum[i] =
                (1 - network->outputLayer.neuron[i]->activation) *
                embann_tanhDerivative(network->outputLayer.neuron[i]->activation);
        }
        else
        {
            dOutputErrorToOutputSum[i] =
                -network->outputLayer.neuron[i]->activation *
                embann_tanhDerivative(network->outputLayer.neuron[i]->activation);
        }
        // printf("\ndOutputErrorToOutputSum[%d]: %.3f", i,
        // dOutputErrorToOutputSum[i]);
        for (uint16_t j = 0; j < network->hiddenLayer[0].numNeurons; j++)
        {
            outputNeuronWeightChange[i][j] =
                dOutputErrorToOutputSum[i] *
                network->hiddenLayer[network->properties.numHiddenLayers - 1].neuron[j]->activation *
                learningRate;
            // printf("\n  outputNeuronWeightChange[%d][%d]: %.3f", i, j,
            //              outputNeuronWeightChange[i][j]);
        }
    }

    for (uint16_t i = 0; i < network->hiddenLayer[0].numNeurons; i++)
    {
        dTotalErrorToHiddenNeuron = 0.0;
        for (uint16_t j = 0; j < network->outputLayer.numNeurons; j++)
        {
            dTotalErrorToHiddenNeuron +=
                dOutputErrorToOutputSum[j] * network->outputLayer.neuron[j]->params[i]->weight;
            // printf("\nOld Output Weight[%d][%d]: %.3f", i, j,
            // network->outputLayer.neuron[j]->params[i]->weight);
            network->outputLayer.neuron[j]->params[i]->weight += outputNeuronWeightChange[j][i];
            // printf("\nNew Output Weight[%d][%d]: %.3f", i, j,
            // network->outputLayer.neuron[j]->params[i]->weight);
        }
        for (uint16_t k = 0; k < network->inputLayer.numNeurons; k++)
        {
            // printf("\nOld Hidden Weight[%d][%d]: %.3f", i, k,
            // network->network->hiddenLayer[0].neuron[i]->params[k]->weight);
            network->hiddenLayer[0].neuron[i]->params[k]->weight +=
                dTotalErrorToHiddenNeuron *
                embann_tanhDerivative(network->hiddenLayer[0].neuron[i]->activation) *
                network->inputLayer.neuron[k]->activation * learningRate;
            // printf("\nNew Hidden Weight[%d][%d]: %.3f", i, k,
            // network->network->hiddenLayer[0].neuron[i]->params[k]->weight);
        }
    }
}

float embann_tanhDerivative(float inputValue)
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

uint8_t embann_outputLayer()
{
    uint8_t mostLikelyOutput = 0;

    for (uint16_t i = 0; i < network->outputLayer.numNeurons; i++)
    {
        if (network->outputLayer.neuron[i] >
            network->outputLayer.neuron[mostLikelyOutput])
        {
            mostLikelyOutput = i;
        }
        // printf("i: %d neuron: %-3f likely: %d\n", i,
        // network->outputLayer.neurons[i], mostLikelyOutput);
    }
    return mostLikelyOutput;
}

void embann_printNetwork()
{
    printf("\nInput: [");
    for (uint16_t i = 0; i < (network->inputLayer.numNeurons - 1); i++)
    {
        printf("%0.3f, ", network->inputLayer.neuron[i]->activation);
    }
    printf("%0.3f]", network->inputLayer.neuron[network->inputLayer.numNeurons - 1]->activation);

    printf("\nInput Layer | Hidden Layer ");
    if (network->properties.numHiddenLayers > 1)
    {
        printf("1 ");
        for (uint8_t i = 2; i <= network->properties.numHiddenLayers; i++)
        {
            printf("| Hidden Layer %d ", i);
        }
    }
    printf("| Output Layer");

    bool nothingLeft = false;
    uint16_t i = 0;
    while (nothingLeft == false)
    { /* TODO, Make this compatible with multiple hidden layers */
        if ((i >= network->inputLayer.numNeurons) &&
            (i >= network->hiddenLayer[0].numNeurons) &&
            (i >= network->outputLayer.numNeurons))
        {
            nothingLeft = true;
        }
        else
        {
            if (i < network->inputLayer.numNeurons)
            {
                printf("%-12.3f| ", network->inputLayer.neuron[i]->activation);
            }
            else
            {
                printf("            | ");
            }

            if (i < network->hiddenLayer[0].numNeurons)
            {
                if (network->properties.numHiddenLayers == 1)
                {
                    printf("%-13.3f| ", network->hiddenLayer[0].neuron[i]->activation);
                }
                else
                {
                    for (uint8_t j = 0; j < network->properties.numHiddenLayers; j++)
                    {
                        printf("%-15.3f| ", network->hiddenLayer[j].neuron[i]->activation);
                    }
                }
            }
            else
            {
                printf("             | ");
                if (network->properties.numHiddenLayers > 1)
                {
                    printf("              | ");
                }
            }

            if (i < network->outputLayer.numNeurons)
            {
                printf("%.3f", network->outputLayer.neuron[i]->activation);
            }
        }
        printf("\n");
        i++;
    }

    printf("I think this is output %d ", network->properties.networkResponse);
}
void embann_printInputNeuronDetails(uint8_t neuronNum)
{
    if (neuronNum < network->inputLayer.numNeurons)
    {
        printf("\nInput Neuron %d: %.3f\n", neuronNum,
                      network->inputLayer.neuron[neuronNum]);
    }
    else
    {
        printf("\nERROR: You've asked for input neuron %d when only %d exist\n",
            neuronNum, network->inputLayer.numNeurons);
    }
}

void embann_printOutputNeuronDetails(uint8_t neuronNum)
{
    if (neuronNum < network->outputLayer.numNeurons)
    {

        printf("\nOutput Neuron %d:\n", neuronNum);

        for (uint16_t i = 0; i < network->hiddenLayer[0].numNeurons; i++)
        {
            printf(
                "%.3f-*->%.3f |",
                network->hiddenLayer[network->properties.numHiddenLayers - 1].neuron[i],
                network->outputLayer.neuron[neuronNum]->params[i]->weight);

            if (i == floor(network->hiddenLayer[0].numNeurons / 2))
            {
                printf(" = %.3f", network->outputLayer.neuron[neuronNum]);
            }
            printf("\n");
        }
    }
    else
    {
        printf(
            "\nERROR: You've asked for output neuron %d when only %d exist\n",
            neuronNum, network->outputLayer.numNeurons);
    }
}

void embann_printHiddenNeuronDetails(uint8_t layerNum, uint8_t neuronNum)
{
    if (neuronNum < network->hiddenLayer[0].numNeurons)
    {

        printf("\nHidden Neuron %d:\n", neuronNum);

        if (layerNum == 0)
        {

            for (uint16_t i = 0; i < network->inputLayer.numNeurons; i++)
            {
                printf("%.3f-*->%.3f |", network->inputLayer.neuron[i],
                              network->hiddenLayer[0].neuron[neuronNum]->params[i]->weight);

                if (i == floor(network->inputLayer.numNeurons / 2))
                {
                    printf(" = %.3f",
                                  network->hiddenLayer[0].neuron[neuronNum]);
                }
                printf("\n");
            }
        }
        else
        {

            for (uint16_t i = 0; i < network->hiddenLayer[0].numNeurons; i++)
            {
                printf(
                    "%.3f-*->%.3f |", network->hiddenLayer[layerNum - 1].neuron[i],
                    network->hiddenLayer[layerNum - 1].neuron[neuronNum]->params[i]->weight);

                if (i == floor(network->hiddenLayer[0].numNeurons / 2))
                {
                    printf(" = %.3f",
                                  network->hiddenLayer[0].neuron[neuronNum]);
                }
                printf("\n");
            }
        }
    }
    else
    {
        printf(
            "\nERROR: You've asked for hidden neuron %d when only %d exist\n",
            neuronNum, network->hiddenLayer[0].numNeurons);
    }
}

void embann_errorReporting(uint8_t correctResponse)
{
    printf("\n");
    for (uint8_t i = 0; i < network->outputLayer.numNeurons; i++)
    {
        if (i == correctResponse)
        {
            printf("%-7.3f | ",
                          (1 - network->outputLayer.neuron[correctResponse]->activation));
        }
        else
        {
            printf("%-7.3f | ", -network->outputLayer.neuron[i]->activation);
        }
    }
}

void embann_benchmark(void)
{
    uint32_t testInt = UINT32_MAX;
    float testFloat = FLT_MAX;
    double testDouble = DBL_MAX;
    struct timeval timeBefore;
    struct timeval timeAfter;
    struct timeval timeDiff;
    uint32_t averageTime = 0;

    for (uint8_t i = 0; i < 10; i++)
    {
        gettimeofday(&timeBefore, NULL);

        for (uint32_t i = 0; i < 100000; i++)
        {
            testInt /= 2;
            testInt += 5;
        }

        gettimeofday(&timeAfter, NULL);
        timersub(&timeAfter, &timeBefore, &timeDiff);

        averageTime += timeDiff.tv_usec;
        printf("Integer time was %ld microseconds, result %d\n", timeDiff.tv_usec, testInt);
    }

    averageTime /= 10;
    printf("Average integer time was %d microseconds\n", averageTime);
    averageTime = 0;

    for (uint8_t i = 0; i < 10; i++)
    {
        gettimeofday(&timeBefore, NULL);

        for (uint32_t i = 0; i < 100000; i++)
        {
            testFloat *= 0.5F;
            testFloat += 5;
        }

        gettimeofday(&timeAfter, NULL);
        timersub(&timeAfter, &timeBefore, &timeDiff);

        averageTime += timeDiff.tv_usec;
        printf("Float time was %ld microseconds, result %.2f\n", timeDiff.tv_usec, testFloat);
    }

    averageTime /=