#include "embann.h"
#include "embann_log.h"

#define TAG "Embann Train"

extern network_t* pNetworkGlobal;

static int _trainOutput(accumulator_t* totalErrorInCurrentLayer, const numOutputs_t numOutputs, 
                        const numLayers_t lastHiddenLayer, activation_t learningRate, 
                        numOutputs_t correctOutput);
static int _trainHidden(accumulator_t* totalErrorInCurrentLayer, accumulator_t* totalErrorInNextLayer, 
                        const numLayers_t lastHiddenLayer, activation_t learningRate);




int embann_trainDriverInTime(activation_t learningRate, uint32_t numSeconds, bool verbose)
{
    numOutputs_t randomOutput;
    numTrainingDataEntries_t randomTrainingSet;

    if (verbose == true)
    {
        printf("\nOutput Errors: ");
    }

    uint32_t startTime = millis();

    while ((millis() - startTime) < (numSeconds * 1000UL))
    {
        randomOutput = random() % pNetworkGlobal->outputLayer->numNeurons;
        randomTrainingSet = random() % embann_getDataCollection()->numEntries;

        /*
            TODO, these are not 'right' but they will let the program run
        */
        embann_inputRaw(embann_getDataCollection()->head->data);
        EMBANN_ERROR_CHECK(embann_forwardPropagate());

        if (verbose == true)
        {
            printf("%u | %u ", randomOutput, randomTrainingSet);
            EMBANN_ERROR_CHECK(embann_errorReporting(randomOutput));
        }

        EMBANN_ERROR_CHECK(embann_train(randomOutput, learningRate));
    }
    return EOK;
}

int embann_trainDriverInError(activation_t learningRate, activation_t desiredCost, bool verbose)
{
    numOutputs_t randomOutput;
    numTrainingDataEntries_t randomTrainingSet;
#ifdef CONFIG_MEMORY_ALLOCATION_STATIC
    activation_t currentCost[CONFIG_NUM_OUTPUT_NEURONS];
#else
    activation_t currentCost[pNetworkGlobal->outputLayer->numNeurons];
#endif
    bool converged = false;

    if (verbose == true)
    {
        printf("\nOutput Errors: ");
    }

    while (!converged)
    {
        randomOutput = random() % pNetworkGlobal->outputLayer->numNeurons;
        randomTrainingSet = random() % embann_getDataCollection()->numEntries;
        currentCost[randomOutput] = 0;

        /*
            TODO, these are not 'right' but they will let the program run
        */
        embann_inputRaw(embann_getDataCollection()->head->data);
        EMBANN_ERROR_CHECK(embann_forwardPropagate());

        if (verbose == true)
        {
            printf("%u | %u | ", randomOutput, randomTrainingSet);
            EMBANN_ERROR_CHECK(embann_errorReporting(randomOutput));
        }

        EMBANN_ERROR_CHECK(embann_train(randomOutput, learningRate));
        for (numOutputs_t i = 0; i < pNetworkGlobal->outputLayer->numNeurons; i++)
        {
            if (i == randomOutput)
            {
                currentCost[randomOutput] += (1 - pNetworkGlobal->outputLayer->activation[i]);
            }
            else
            {
                currentCost[randomOutput] += pNetworkGlobal->outputLayer->activation[i];
            }
        }
        currentCost[randomOutput] /= pNetworkGlobal->outputLayer->numNeurons;

        converged = true;
        for (numOutputs_t i = 0; i < pNetworkGlobal->outputLayer->numNeurons; i++)
        {
            if (verbose == true)
            {
                printf("%" ACTIVATION_PRINT ", ", currentCost[i]);
            }
            if (currentCost[i] > desiredCost)
            {
                converged = false;
            }
        }
        if (verbose == true)
        {
            printf("%" ACTIVATION_PRINT "\n ", desiredCost);
        }
    }
    return EOK;
}





int embann_train(numOutputs_t correctOutput, activation_t learningRate)
{
    const numOutputs_t numOutputs = pNetworkGlobal->outputLayer->numNeurons;
    const numLayers_t lastHiddenLayer = pNetworkGlobal->properties.numHiddenLayers - 1U;
    // TODO, Python preprocessor to define required size for error arrays.
    accumulator_t totalErrorInCurrentLayer[CONFIG_NUM_HIDDEN_NEURONS];
    accumulator_t totalErrorInNextLayer[CONFIG_NUM_HIDDEN_NEURONS];

    if (correctOutput > numOutputs)
    {
        return ENOENT;
    }

    EMBANN_ERROR_CHECK(_trainOutput(totalErrorInCurrentLayer, numOutputs, lastHiddenLayer, learningRate, correctOutput));
    memcpy(totalErrorInNextLayer, totalErrorInCurrentLayer, sizeof(totalErrorInNextLayer));
    memset(totalErrorInCurrentLayer, 0, sizeof(totalErrorInCurrentLayer));
    EMBANN_ERROR_CHECK(_trainHidden(totalErrorInCurrentLayer, totalErrorInNextLayer, lastHiddenLayer, learningRate));
    return EOK;
}





static int _trainOutput(accumulator_t* totalErrorInCurrentLayer, const numOutputs_t numOutputs, 
                        const numLayers_t lastHiddenLayer, activation_t learningRate, 
                        numOutputs_t correctOutput)
{
    // TODO, add backpropagation for other activation functions (this is just ReLU)
    // TODO, add biasing
    const numHiddenNeurons_t numHiddenNeurons = pNetworkGlobal->hiddenLayer[lastHiddenLayer]->numNeurons;

#ifdef CONFIG_MEMORY_ALLOCATION_STATIC
    numOutputs_t correctOutputArray[CONFIG_NUM_OUTPUT_NEURONS];
#else
    numOutputs_t correctOutputArray[numOutputs];
#endif

    /* This should usually be more efficient iterating over the array with an if statement */
    memset(correctOutputArray, 0, numOutputs);
    correctOutputArray[correctOutput] = MAX_ACTIVATION;


    for (numOutputs_t i = 0; i < numOutputs; i++)
    {        
        totalErrorInCurrentLayer[i] = (pNetworkGlobal->outputLayer->activation[i] - correctOutputArray[i]);
    }

    for (numOutputs_t i = 0; i < numOutputs; i++)
    {        
        for (numHiddenNeurons_t j = 0; j < numHiddenNeurons; j++)
        {
            pNetworkGlobal->outputLayer->weight[i][j] -= 
                        learningRate * 
                        pNetworkGlobal->hiddenLayer[lastHiddenLayer]->activation[i] *
                        totalErrorInCurrentLayer[i];
        }
    }
    return EOK;
}






static int _trainHidden(accumulator_t* totalErrorInCurrentLayer, accumulator_t* totalErrorInNextLayer, 
                        const numLayers_t lastHiddenLayer, activation_t learningRate)
{
    // TODO, add backpropagation for other activation functions (this is just ReLU)
    // TODO, add biasing
    numHiddenNeurons_t numNeuronsInCurrentLayer = pNetworkGlobal->hiddenLayer[lastHiddenLayer]->numNeurons;
    numHiddenNeurons_t numNeuronsInNextLayer = pNetworkGlobal->outputLayer->numNeurons;

    for (numLayers_t i = lastHiddenLayer; i > 0; i--)
    {
        for (numHiddenNeurons_t j = 0; j < numNeuronsInCurrentLayer; j++)
        {
            for (numOutputs_t k = 0; k < numNeuronsInNextLayer; k++)
            {        
                totalErrorInCurrentLayer[j] += pNetworkGlobal->hiddenLayer[i]->weight[j][k] * 
                                                    totalErrorInNextLayer[k];
            }
        }

        for (numHiddenNeurons_t j = 0; j < numNeuronsInCurrentLayer; j++)
        {   
            for (numOutputs_t k = 0; k < numNeuronsInNextLayer; k++)
            {  
                pNetworkGlobal->hiddenLayer[i]->weight[j][k] -=
                            learningRate *
                            pNetworkGlobal->hiddenLayer[i - 1U]->activation[j] *
                            totalErrorInCurrentLayer[j];
            }
        }

        // TODO, Python preprocessor to define required size for error arrays.
        memcpy(totalErrorInNextLayer, totalErrorInCurrentLayer, numNeuronsInNextLayer * sizeof(accumulator_t));
        memset(totalErrorInCurrentLayer, 0, numNeuronsInCurrentLayer * sizeof(accumulator_t));

        numNeuronsInCurrentLayer = pNetworkGlobal->hiddenLayer[i - 1U]->numNeurons;
        numNeuronsInNextLayer = pNetworkGlobal->hiddenLayer[i]->numNeurons;
    }

    return EOK;
}






int embann_tanhDerivative(activation_t inputValue, weight_t* outputValue)
{
#ifdef ACTIVATION_IS_FLOAT
    *outputValue = 1.0F - powf(tanh(inputValue * PI), 2.0F);
#elif defined(ACTIVATION_IS_SIGNED) || defined(ACTIVATION_IS_UNSIGNED)
    *outputValue = 1;
#endif   
    return EOK;
}