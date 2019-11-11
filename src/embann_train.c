#include "embann.h"
#include "embann_log.h"

#define TAG "Embann Train"

extern network_t* pNetworkGlobal;

static int _trainOutput(numOutputs_t correctOutput, activation_t learningRate);


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
    activation_t currentCost[pNetworkGlobal->outputLayer->numNeurons];
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
    _trainOutput(correctOutput, learningRate);
    return EOK;
}





static int _trainOutput(numOutputs_t correctOutput, activation_t learningRate)
{
    const numOutputs_t numOutputs = pNetworkGlobal->outputLayer->numNeurons;
    const numLayers_t lastHiddenLayer = pNetworkGlobal->properties.numHiddenLayers - 1U;
    numOutputs_t correctOutputArray[numOutputs];

    if (correctOutput > numOutputs)
    {
        return ENOENT;
    }

    // This should usually be more efficient iterating over the array with an if statement
    memset(correctOutputArray, 0, numOutputs);
    correctOutputArray[correctOutput] = MAX_ACTIVATION;


    for (numOutputs_t i = 0; i < numOutputs; i++)
    {        
        pNetworkGlobal->outputLayer->activation[i] -= 
                    learningRate * 
                    pNetworkGlobal->hiddenLayer[lastHiddenLayer]->activation[i] *
                    (pNetworkGlobal->outputLayer->activation[i] - correctOutputArray[i]);
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