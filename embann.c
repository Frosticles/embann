/*
  Embann.cpp - EMbedded Backpropogating Artificial Neural Network.
  Created by Peter Frost, 27th August 2019
*/

#include "embann.h"


static network_t* network;
static trainingDataCollection_t trainingDataCollection = {
    .tail = NULL,
    .head = NULL,
    .numEntries = 0U
};
static int embann_initInputLayer(uint16_t numInputNeurons);
static int embann_initHiddenLayer(uint16_t numHiddenNeurons,
                                   uint8_t numHiddenLayers,
                                   uint16_t numInputNeurons);
static int embann_initOutputLayer(uint16_t numOutputNeurons,
                                   uint16_t numHiddenNeurons);
#ifndef ARDUINO
static uint32_t millis(void);
#endif

#define TAG "Embann Core"

int WEAK_FUNCTION main(int argc, char const *argv[])
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    srandom(tv.tv_usec ^ tv.tv_sec);  /* Seed the PRNG */
    
    embann_benchmark();
    embann_init(10U, 10U, 1U, 10U);
}


int embann_init(uint16_t numInputNeurons,
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

    network->properties.numLayers = numHiddenLayers + 2U;
    network->properties.numHiddenLayers = numHiddenLayers;
    network->properties.networkResponse = 0U;

    return EOK;
}

static int embann_initInputLayer(uint16_t numInputNeurons)
{
    inputLayer_t* inputLayer = (inputLayer_t*) malloc(sizeof(inputLayer_t) + 
                                                (sizeof(uNeuron_t*) * numInputNeurons));
    CHECK_MALLOC(inputLayer);

    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wpointer-to-int-cast"
    // MISRA C 2012 14.4 - deliberate cast from pointer to integer
    // cppcheck-suppress misra-c2012-11.4
    EMBANN_LOGI(TAG, "inputLayer: 0x%x, size: %ld", (uint32_t) inputLayer, sizeof(inputLayer_t) + 
                                                (sizeof(uNeuron_t*) * numInputNeurons));
    #pragma GCC diagnostic pop

    inputLayer->numNeurons = numInputNeurons;
    for (uint8_t i = 0; i < numInputNeurons; i++)
    {
        uNeuron_t* pNeuron = (uNeuron_t*) malloc(sizeof(uNeuron_t));
        inputLayer->neuron[i] = pNeuron;
        inputLayer->neuron[i]->activation = 0.0F;
    }
    network->inputLayer = *inputLayer;

    EMBANN_LOGI(TAG, "done input");
    return EOK;
}

static int embann_initHiddenLayer(uint16_t numHiddenNeurons,
                                   uint8_t numHiddenLayers,
                                   uint16_t numInputNeurons)
{
    for (uint8_t i = 0; i < numHiddenLayers; i++)
    {
        hiddenLayer_t* hiddenLayer = (hiddenLayer_t*) malloc(sizeof(hiddenLayer_t) + 
                                                (sizeof(wNeuron_t*) * numHiddenNeurons));
        CHECK_MALLOC(hiddenLayer);

        #pragma GCC diagnostic push
        #pragma GCC diagnostic ignored "-Wpointer-to-int-cast"
        // MISRA C 2012 14.4 - deliberate cast from pointer to integer
        // cppcheck-suppress misra-c2012-11.4
        EMBANN_LOGI(TAG, "hiddenlayer: 0x%x, size: %ld", (uint32_t) hiddenLayer, sizeof(hiddenLayer_t) + 
                                                (sizeof(wNeuron_t*) * numHiddenNeurons));
        #pragma GCC diagnostic pop

        for (uint16_t j = 0; j < numHiddenNeurons; j++)
        {    
            wNeuron_t* pNeuron = (wNeuron_t*) malloc(sizeof(wNeuron_t) + (sizeof(neuronParams_t*) * (uint16_t)((i == 0U) ? numInputNeurons : numHiddenNeurons)));
            hiddenLayer->neuron[j] = pNeuron;
            hiddenLayer->neuron[j]->activation = 0.0F;

            for (uint16_t k = 0; k < (uint16_t)((i == 0U) ? numInputNeurons : numHiddenNeurons); k++)
            {
                neuronParams_t* hiddenLayerParams = (neuronParams_t*) malloc(sizeof(neuronParams_t));
                CHECK_MALLOC(hiddenLayerParams);

                hiddenLayer->neuron[j]->params[k] = hiddenLayerParams;

                //EMBANN_LOGI(TAG, "params array: 0x%x, bias 0x%x, weight 0x%x", (uint32_t) &hiddenLayer->neuron[j]->params[k],
                //                    (uint32_t)&hiddenLayer->neuron[j]->params[k]->bias,
                //                    (uint32_t)&hiddenLayer->neuron[j]->params[k]->weight);

                hiddenLayer->neuron[j]->params[k]->bias = RAND_WEIGHT();
                hiddenLayer->neuron[j]->params[k]->weight = RAND_WEIGHT();
            }
        }

        EMBANN_LOGI(TAG, "done hidden");

        network->hiddenLayer[i] = *hiddenLayer;

        #pragma GCC diagnostic push
        #pragma GCC diagnostic ignored "-Wpointer-to-int-cast"
        // MISRA C 2012 14.4 - deliberate cast from pointer to integer
        // cppcheck-suppress misra-c2012-11.4
        EMBANN_LOGI(TAG, "hiddenlayer[i]: 0x%x", (uint32_t) &network->hiddenLayer[i]);
        #pragma GCC diagnostic pop
    }
    return EOK;
}

static int embann_initOutputLayer(uint16_t numOutputNeurons,
                                   uint16_t numHiddenNeurons)
{
    outputLayer_t* outputLayer = (outputLayer_t*) malloc(sizeof(outputLayer_t) + 
                                                        (sizeof(wNeuron_t*) * numOutputNeurons));
    CHECK_MALLOC(outputLayer);

    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wpointer-to-int-cast"
    // MISRA C 2012 14.4 - deliberate cast from pointer to integer
    // cppcheck-suppress misra-c2012-11.4
    EMBANN_LOGI(TAG, "outputLayer: 0x%x, size: %ld", (uint32_t) outputLayer, sizeof(outputLayer_t) + 
                                                (sizeof(wNeuron_t*) * numOutputNeurons));
    #pragma GCC diagnostic pop

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

    EMBANN_LOGI(TAG, "done output");
    return EOK;
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
        network->inputLayer.neuron[i]->activation = (((float)data[i] - min)) / (max - min);
    }
}

void embann_inputStandardizeScale(uint8_t data[], float mean, float stdDev)
{
    for (uint32_t i = 0; i < network->inputLayer.numNeurons; i++)
    {
        network->inputLayer.neuron[i]->activation = ((float)data[i] - mean) / stdDev;
    }
}

int embann_getTrainingDataMean(float* mean)
{
    uint32_t sum = 0;
    trainingData_t* pTrainingData = trainingDataCollection.head;

    if (pTrainingData != NULL)
    {
        *mean = pTrainingData->data[0];
    }
    else
    {
        // Deviation from MISRA C2012 15.5 for reasonably simple error return values
        // cppcheck-suppress misra-c2012-15.5
        return ENOENT;
    }
    
    while (pTrainingData != NULL)
    {
        for (uint32_t j = 0; j < pTrainingData->length; j++)
        {
            sum += pTrainingData->data[j];
        }
        *mean += sum / pTrainingData->length;
        sum = 0;

        pTrainingData = pTrainingData->next;
    }

    *mean /= trainingDataCollection.numEntries;

    return EOK;
}

int embann_getTrainingDataStdDev(float* stdDev)
{
    float sumofSquares = 0.0F;
    float mean;
    trainingData_t* pTrainingData = trainingDataCollection.head;

    if (pTrainingData != NULL)
    {
        *stdDev = pTrainingData->data[0];
    }
    else
    {
        // Deviation from MISRA C2012 15.5 for reasonably simple error return values
        // cppcheck-suppress misra-c2012-15.5
        return ENOENT;
    }

    if (embann_getTrainingDataMean(&mean) != EOK)
    {
        // Deviation from MISRA C2012 15.5 for reasonably simple error return values
        // cppcheck-suppress misra-c2012-15.5
        return ENOENT;
    }
    
    while (pTrainingData != NULL)
    {
        for (uint32_t j = 0; j < pTrainingData->length; j++)
        {
            sumofSquares += powf((float)pTrainingData->data[j] - mean, 2.0F);
        }
        sumofSquares /= pTrainingData->length;
        pTrainingData = pTrainingData->next;
    }

    *stdDev = sqrtf(sumofSquares);

    return EOK;
}

int embann_getTrainingDataMax(uint8_t* max)
{
    trainingData_t* pTrainingData = trainingDataCollection.head;
    if (pTrainingData != NULL)
    {
        *max = pTrainingData->data[0];
    }
    else
    {
        // Deviation from MISRA C2012 15.5 for reasonably simple error return values
        // cppcheck-suppress misra-c2012-15.5
        return ENOENT;
    }
    
    
    while (pTrainingData != NULL)
    {
        for (uint32_t j = 0; j < pTrainingData->length; j++)
        {
            if (pTrainingData->data[j] > *max)
            {
                *max = pTrainingData->data[j];
            }
        }
        pTrainingData = pTrainingData->next;
    }

    return EOK;
}

int embann_getTrainingDataMin(uint8_t* min)
{
    trainingData_t* pTrainingData = trainingDataCollection.head;
    if (pTrainingData != NULL)
    {
        *min = pTrainingData->data[0];
    }
    else
    {
        // Deviation from MISRA C2012 15.5 for reasonably simple error return values
        // cppcheck-suppress misra-c2012-15.5
        return ENOENT;
    }

    while (pTrainingData != NULL)
    {
        for (uint32_t j = 0; j < pTrainingData->length; j++)
        {
            if (pTrainingData->data[j] < *min)
            {
                *min = pTrainingData->data[j];
            }
        }
        pTrainingData = pTrainingData->next;
    }

    return EOK;
}


uint8_t embann_inputLayer()
{
    embann_sumAndSquashInput(network->inputLayer.neuron, 
                        network->hiddenLayer[0].neuron,
                        network->inputLayer.numNeurons, 
                        network->hiddenLayer[0].numNeurons);
    // EMBANN_LOGI(TAG, "Done Input -> 1st Hidden Layer");
    for (uint8_t i = 1; i < network->properties.numHiddenLayers; i++)
    {
        embann_sumAndSquash(network->hiddenLayer[i - 1U].neuron,
                            network->hiddenLayer[i].neuron,
                            network->hiddenLayer[i - 1U].numNeurons,
                            network->hiddenLayer[i].numNeurons);
        // EMBANN_LOGI(TAG, "Done Hidden Layer %d -> Hidden Layer %d", i - 1, i);
    }

    embann_sumAndSquash(
        network->hiddenLayer[network->properties.numHiddenLayers - 1U].neuron,
        network->outputLayer.neuron, 
        network->hiddenLayer[network->properties.numHiddenLayers - 1U].numNeurons,
        network->outputLayer.numNeurons);

    /*EMBANN_LOGI(TAG, "Done Hidden Layer %d -> Output Layer", network->properties.numHiddenLayers);*/

    network->properties.networkResponse = embann_outputLayer();
    return network->properties.networkResponse;
}

int embann_sumAndSquash(wNeuron_t* Input[], wNeuron_t* Output[], uint16_t numInputs,
                           uint16_t numOutputs)
{
    for (uint16_t i = 0; i < numOutputs; i++)
    {
        Output[i]->activation = 0.0F; // Bias[i];
        for (uint16_t j = 0; j < numInputs; j++)
        {
            Output[i]->activation += Input[j]->activation * Output[i]->params[j]->weight;
        }
        Output[i]->activation = tanhf(Output[i]->activation * PI);

        // tanh is a quicker alternative to sigmoid
        // EMBANN_LOGI(TAG, "i:%d This is the embann_SumAndSquash Output %.2f", i, Output[i]);
    }
    return EOK;
}

int embann_sumAndSquashInput(uNeuron_t* Input[], wNeuron_t* Output[], uint16_t numInputs,
                           uint16_t numOutputs)
{
    for (uint16_t i = 0; i < numOutputs; i++)
    {
        Output[i]->activation = 0; // Bias[i];
        for (uint16_t j = 0; j < numInputs; j++)
        {
            Output[i]->activation += Input[j]->activation * Output[i]->params[j]->weight;
        }
        Output[i]->activation = tanhf(Output[i]->activation * PI);

        // tanh is a quicker alternative to sigmoid
        // EMBANN_LOGI(TAG, "i:%d This is the embann_SumAndSquash Output %.2f", i, Output[i]);
    }
    return EOK;
}

int embann_addTrainingData(uint8_t data[], uint32_t length, uint16_t correctResponse)
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
    return EOK;
}

int embann_copyTrainingData(uint8_t data[], uint32_t length, uint16_t correctResponse)
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
    return EOK;
}

void embann_shuffleTrainingData(void)
{

}

int embann_trainDriverInTime(float learningRate, uint32_t numSeconds, bool verbose)
{
    uint16_t randomOutput, randomTrainingSet;

    if (verbose == true)
    {
        EMBANN_LOGI(TAG, "\nOutput Errors: ");
    }

    uint32_t startTime = millis();

    while ((millis() - startTime) < (numSeconds * 1000UL))
    {
        randomOutput = random() % network->outputLayer.numNeurons;
        randomTrainingSet = random() % trainingDataCollection.numEntries;

        /*
            TODO, these are not 'right' but they will let the program run
        */
        embann_inputMinMaxScale(trainingDataCollection.head->data, 0U, UINT8_MAX);
        embann_inputLayer();

        if (verbose == true)
        {
            embann_errorReporting(randomOutput);
            EMBANN_LOGI(TAG, "%u | %u ", randomOutput, randomTrainingSet);
        }

        embann_train(randomOutput, learningRate);
    }
    return EOK;
}

int embann_trainDriverInError(float learningRate, float desiredCost, bool verbose)
{
    uint16_t randomOutput, randomTrainingSet;
    float currentCost[network->outputLayer.numNeurons];
    bool converged = false;

    if (verbose == true)
    {
        EMBANN_LOGI(TAG, "\nOutput Errors: ");
    }

    while (!converged)
    {
        randomOutput = random() % network->outputLayer.numNeurons;
        randomTrainingSet = random() % trainingDataCollection.numEntries;
        currentCost[randomOutput] = 0.0;
        embann_inputMinMaxScale(trainingDataCollection.head->data, 0U, UINT8_MAX);
        embann_inputLayer();

        if (verbose == true)
        {
            embann_errorReporting(randomOutput);
            EMBANN_LOGI(TAG, "%u | %u ", randomOutput, randomTrainingSet);
        }

        embann_train(randomOutput, learningRate);
        for (uint8_t i = 0; i < network->outputLayer.numNeurons; i++)
        {
            if (i == randomOutput)
            {
                currentCost[randomOutput] += pow(1 - network->outputLayer.neuron[i]->activation, 2.0F);
            }
            else
            {
                currentCost[randomOutput] += pow(network->outputLayer.neuron[i]->activation, 2.0F);
            }
        }
        currentCost[randomOutput] /= network->outputLayer.numNeurons;

        for (uint8_t i = 0; i < network->outputLayer.numNeurons; i++)
        {
            EMBANN_LOGI(TAG, "%f", currentCost[i]);
            EMBANN_LOGI(TAG, ", ");
            if (currentCost[i] > desiredCost)
            {
                break;
            }
            if (i == (network->outputLayer.numNeurons - 1U))
            {
                converged = true;
            }
        }
        EMBANN_LOGI(TAG, "%f", desiredCost);
    }
    return EOK;
}

int embann_train(uint8_t correctOutput, float learningRate)
{
    float dOutputErrorToOutputSum[network->outputLayer.numNeurons];
    float dTotalErrorToHiddenNeuron = 0.0F;
    /* TODO, add support for multiple hidden layers */
    float outputNeuronWeightChange[network->outputLayer.numNeurons]
                                  [network->hiddenLayer[0].numNeurons];
    float tanhDerivative;

    for (uint16_t i = 0; i < network->outputLayer.numNeurons; i++)
    {
        if (i == correctOutput)
        {
            embann_tanhDerivative(network->outputLayer.neuron[i]->activation, &tanhDerivative);
            dOutputErrorToOutputSum[i] =
                (1 - network->outputLayer.neuron[i]->activation) * tanhDerivative;
        }
        else
        {
            embann_tanhDerivative(network->outputLayer.neuron[i]->activation, &tanhDerivative);
            dOutputErrorToOutputSum[i] =
                -network->outputLayer.neuron[i]->activation * tanhDerivative;
        }
        // EMBANN_LOGI(TAG, "\ndOutputErrorToOutputSum[%d]: %.3f", i,
        // dOutputErrorToOutputSum[i]);
        for (uint16_t j = 0; j < network->hiddenLayer[0].numNeurons; j++)
        {
            outputNeuronWeightChange[i][j] =
                dOutputErrorToOutputSum[i] *
                network->hiddenLayer[network->properties.numHiddenLayers - 1U].neuron[j]->activation *
                learningRate;
            // EMBANN_LOGI(TAG, "\n  outputNeuronWeightChange[%d][%d]: %.3f", i, j,
            //              outputNeuronWeightChange[i][j]);
        }
    }

    for (uint16_t i = 0; i < network->hiddenLayer[0].numNeurons; i++)
    {
        dTotalErrorToHiddenNeuron = 0.0F;
        for (uint16_t j = 0; j < network->outputLayer.numNeurons; j++)
        {
            dTotalErrorToHiddenNeuron +=
                dOutputErrorToOutputSum[j] * network->outputLayer.neuron[j]->params[i]->weight;
            // EMBANN_LOGI(TAG, "\nOld Output Weight[%d][%d]: %.3f", i, j,
            // network->outputLayer.neuron[j]->params[i]->weight);
            network->outputLayer.neuron[j]->params[i]->weight += outputNeuronWeightChange[j][i];
            // EMBANN_LOGI(TAG, "\nNew Output Weight[%d][%d]: %.3f", i, j,
            // network->outputLayer.neuron[j]->params[i]->weight);
        }
        for (uint16_t k = 0; k < network->inputLayer.numNeurons; k++)
        {
            // EMBANN_LOGI(TAG, "\nOld Hidden Weight[%d][%d]: %.3f", i, k,
            // network->network->hiddenLayer[0].neuron[i]->params[k]->weight);
            embann_tanhDerivative(network->hiddenLayer[0].neuron[i]->activation, &tanhDerivative);
            network->hiddenLayer[0].neuron[i]->params[k]->weight +=
                dTotalErrorToHiddenNeuron * tanhDerivative *
                network->inputLayer.neuron[k]->activation * learningRate;
            // EMBANN_LOGI(TAG, "\nNew Hidden Weight[%d][%d]: %.3f", i, k,
            // network->network->hiddenLayer[0].neuron[i]->params[k]->weight);
        }
    }
    return EOK;
}

int embann_tanhDerivative(float inputValue, float* outputValue)
{
    *outputValue = 1.0F - powf(tanh(inputValue * PI), 2.0F);
    return EOK;
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
        // EMBANN_LOGI(TAG, "i: %d neuron: %-3f likely: %d", i,
        // network->outputLayer.neurons[i], mostLikelyOutput);
    }
    return mostLikelyOutput;
}

int embann_printNetwork(void)
{
    EMBANN_LOGI(TAG, "\nInput: [");
    for (uint16_t i = 0; i < (network->inputLayer.numNeurons - 1U); i++)
    {
        EMBANN_LOGI(TAG, "%0.3f, ", network->inputLayer.neuron[i]->activation);
    }
    EMBANN_LOGI(TAG, "%0.3f]", network->inputLayer.neuron[network->inputLayer.numNeurons - 1U]->activation);

    EMBANN_LOGI(TAG, "\nInput Layer | Hidden Layer ");
    if (network->properties.numHiddenLayers > 1U)
    {
        EMBANN_LOGI(TAG, "1 ");
        for (uint8_t i = 2; i <= network->properties.numHiddenLayers; i++)
        {
            EMBANN_LOGI(TAG, "| Hidden Layer %d ", i);
        }
    }
    EMBANN_LOGI(TAG, "| Output Layer");

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
                EMBANN_LOGI(TAG, "%-12.3f| ", network->inputLayer.neuron[i]->activation);
            }
            else
            {
                EMBANN_LOGI(TAG, "            | ");
            }

            if (i < network->hiddenLayer[0].numNeurons)
            {
                if (network->properties.numHiddenLayers == 1U)
                {
                    EMBANN_LOGI(TAG, "%-13.3f| ", network->hiddenLayer[0].neuron[i]->activation);
                }
                else
                {
                    for (uint8_t j = 0; j < network->properties.numHiddenLayers; j++)
                    {
                        EMBANN_LOGI(TAG, "%-15.3f| ", network->hiddenLayer[j].neuron[i]->activation);
                    }
                }
            }
            else
            {
                EMBANN_LOGI(TAG, "             | ");
                if (network->properties.numHiddenLayers > 1U)
                {
                    EMBANN_LOGI(TAG, "              | ");
                }
            }

            if (i < network->outputLayer.numNeurons)
            {
                EMBANN_LOGI(TAG, "%.3f", network->outputLayer.neuron[i]->activation);
            }
        }
        EMBANN_LOGI(TAG, "\n");
        i++;
    }

    EMBANN_LOGI(TAG, "I think this is output %d ", network->properties.networkResponse);
    return EOK;
}

int embann_printInputNeuronDetails(uint8_t neuronNum)
{
    if (neuronNum < network->inputLayer.numNeurons)
    {
        EMBANN_LOGI(TAG, "\nInput Neuron %d: %.3f", neuronNum,
                      network->inputLayer.neuron[neuronNum]->activation);
    }
    else
    {
        EMBANN_LOGI(TAG, "\nERROR: You've asked for input neuron %d when only %d exist",
            neuronNum, network->inputLayer.numNeurons);
    }
    return EOK;
}

int embann_printOutputNeuronDetails(uint8_t neuronNum)
{
    if (neuronNum < network->outputLayer.numNeurons)
    {

        EMBANN_LOGI(TAG, "\nOutput Neuron %d:", neuronNum);

        for (uint16_t i = 0; i < network->hiddenLayer[0].numNeurons; i++)
        {
            EMBANN_LOGI(TAG, 
                "%.3f-*->%.3f |",
                network->hiddenLayer[network->properties.numHiddenLayers - 1U].neuron[i]->activation,
                network->outputLayer.neuron[neuronNum]->params[i]->weight);

            if (i == floor(network->hiddenLayer[0].numNeurons / 2U))
            {
                EMBANN_LOGI(TAG, " = %.3f", network->outputLayer.neuron[neuronNum]->activation);
            }
            EMBANN_LOGI(TAG, "\n");
        }
    }
    else
    {
        EMBANN_LOGI(TAG, 
            "\nERROR: You've asked for output neuron %d when only %d exist",
            neuronNum, network->outputLayer.numNeurons);
    }
    return EOK;
}

int embann_printHiddenNeuronDetails(uint8_t layerNum, uint8_t neuronNum)
{
    if (neuronNum < network->hiddenLayer[0].numNeurons)
    {

        EMBANN_LOGI(TAG, "\nHidden Neuron %d:", neuronNum);

        if (layerNum == 0U)
        {

            for (uint16_t i = 0; i < network->inputLayer.numNeurons; i++)
            {
                EMBANN_LOGI(TAG, "%.3f-*->%.3f |", network->inputLayer.neuron[i]->activation,
                              network->hiddenLayer[0].neuron[neuronNum]->params[i]->weight);

                if (i == floor(network->inputLayer.numNeurons / 2U))
                {
                    EMBANN_LOGI(TAG, " = %.3f",
                                  network->hiddenLayer[0].neuron[neuronNum]->activation);
                }
                EMBANN_LOGI(TAG, "\n");
            }
        }
        else
        {

            for (uint16_t i = 0; i < network->hiddenLayer[0].numNeurons; i++)
            {
                EMBANN_LOGI(TAG, 
                    "%.3f-*->%.3f |", network->hiddenLayer[layerNum - 1U].neuron[i]->activation,
                    network->hiddenLayer[layerNum - 1U].neuron[neuronNum]->params[i]->weight);

                if (i == floor(network->hiddenLayer[0].numNeurons / 2U))
                {
                    EMBANN_LOGI(TAG, " = %.3f",
                                  network->hiddenLayer[0].neuron[neuronNum]->activation);
                }
                EMBANN_LOGI(TAG, "\n");
            }
        }
    }
    else
    {
        EMBANN_LOGI(TAG, 
            "\nERROR: You've asked for hidden neuron %d when only %d exist",
            neuronNum, network->hiddenLayer[0].numNeurons);
    }
    return EOK;
}

int embann_errorReporting(uint8_t correctResponse)
{
    EMBANN_LOGI(TAG, "\n");
    for (uint8_t i = 0; i < network->outputLayer.numNeurons; i++)
    {
        if (i == correctResponse)
        {
            EMBANN_LOGI(TAG, "%-7.3f | ",
                          (1 - network->outputLayer.neuron[correctResponse]->activation));
        }
        else
        {
            EMBANN_LOGI(TAG, "%-7.3f | ", -network->outputLayer.neuron[i]->activation);
        }
    }
    return EOK;
}

int embann_benchmark(void)
{    
    uint16_t numElements = (random() % 2) + 300;
    
    int32_t MAX_ALIGNMENT testInt[numElements];
    float MAX_ALIGNMENT testFloat[numElements];
    double MAX_ALIGNMENT testDouble[numElements];
    int32_t MAX_ALIGNMENT testIntWeight[numElements];
    float MAX_ALIGNMENT testFloatWeight[numElements];
    double MAX_ALIGNMENT testDoubleWeight[numElements];
    int32_t MAX_ALIGNMENT testIntBias[numElements];
    float MAX_ALIGNMENT testFloatBias[numElements];
    double MAX_ALIGNMENT testDoubleBias[numElements];
    struct timeval timeBefore;
    struct timeval timeAfter;
    struct timeval timeDiff;

    for (uint16_t i = 0; i < NUM_ARRAY_ELEMENTS(testInt); i++)
    {
        testInt[i] = INT32_MAX;
        testIntBias[i] = (random() % 21) - 10;
        testIntWeight[i] = (random() % 1000) + 1;  
    }
    
    for (uint16_t i = 0; i < NUM_ARRAY_ELEMENTS(testFloat); i++)
    {
        testFloat[i] = FLT_MAX;
        testFloatBias[i] = (float)(random() % 21) - 10;
        testFloatWeight[i] = RAND_WEIGHT();
    }

    for (uint16_t i = 0; i < NUM_ARRAY_ELEMENTS(testDouble); i++)
    {
        testDouble[i] = DBL_MAX;
        testDoubleBias[i] = (double)(random() % 21) - 10;
        testDoubleWeight[i] = RAND_WEIGHT();
    }

    gettimeofday(&timeBefore, NULL);
    //#pragma omp parallel for
    for (int32_t i = 0; i < 100000; i++)
    {
        for (uint16_t j = 0; j < NUM_ARRAY_ELEMENTS(testInt); j++)
        {
            testInt[j] /= testIntWeight[j];
            testInt[j] += testIntBias[j];
        }
    }
    gettimeofday(&timeAfter, NULL);
    timersub(&timeAfter, &timeBefore, &timeDiff);
    EMBANN_LOGI(TAG, "Integer time was %ld microseconds, result %d", timeDiff.tv_usec, testInt[0]);

    gettimeofday(&timeBefore, NULL);
    //#pragma omp parallel for
    for (int32_t i = 0; i < 100000; i++)
    {
        for (uint16_t j = 0; j < NUM_ARRAY_ELEMENTS(testFloat); j++)
        {
            testFloat[j] *= testFloatWeight[j];
            testFloat[j] += testFloatBias[j];
        }
    }
    gettimeofday(&timeAfter, NULL);
    timersub(&timeAfter, &timeBefore, &timeDiff);
    EMBANN_LOGI(TAG, "Float time was %ld microseconds, result %.2f", timeDiff.tv_usec, testFloat[0]);

    gettimeofday(&timeBefore, NULL);
    //#pragma omp parallel for
    for (int32_t i = 0; i < 100000; i++)
    {
        for (uint16_t j = 0; j < NUM_ARRAY_ELEMENTS(testDouble); j++)
        {
            testDouble[j] *= testDoubleWeight[j];
            testDouble[j] += testDoubleBias[j];
        }
    }
    gettimeofday(&timeAfter, NULL);
    timersub(&timeAfter, &timeBefore, &timeDiff);
    EMBANN_LOGI(TAG, "Double time was %ld microseconds, result %.2f", timeDiff.tv_usec, testDouble[0]);

    return EOK;
}

#ifndef ARDUINO
uint32_t millis(void)
{
    struct timeval time;
    gettimeofday(&time, NULL);
    return (uint32_t) round(time.tv_usec / 1000);
}
#endif