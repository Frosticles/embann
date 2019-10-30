// SPDX-License-Identifier: GPL-2.0-only
/*
    embann.c - EMbedded Backpropogating Artificial Neural Network.
    Copyright Peter Frost 2019
*/

#include "embann.h"
#include "embann_log.h"

#define TAG "Embann Core"



/* errno value specifically for internal embann errors */
static int embann_errno = EOK;
/* Pointer to the current network */
static network_t* pNetwork;
/* Structure of pointers to training data */
static trainingDataCollection_t trainingDataCollection = {
    .tail = NULL,
    .head = NULL,
    .numEntries = 0U
};






#ifdef TEST_BUILD
int main(int argc, char const *argv[])
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    srandom(tv.tv_usec ^ tv.tv_sec);  /* Seed the PRNG */
    
    EMBANN_ERROR_CHECK(embann_benchmark());
#ifdef CONFIG_MEMORY_ALLOCATION_STATIC
    EMBANN_ERROR_CHECK(embann_init(CONFIG_NUM_INPUT_NEURONS, 
                                    CONFIG_NUM_HIDDEN_NEURONS, 
                                    CONFIG_NUM_HIDDEN_LAYERS, 
                                    CONFIG_NUM_OUTPUT_NEURONS));
#else
    EMBANN_ERROR_CHECK(embann_init(15U, 10U, 5U, 3U));
#endif
    EMBANN_ERROR_CHECK(embann_forwardPropagate());

    uint8_t randomData[10];
    uint8_t retval;
    float fretval;
    for (uint8_t i = 0; i < NUM_ARRAY_ELEMENTS(randomData); i++)
    {
        randomData[i] = random();
    }

    EMBANN_ERROR_CHECK(embann_addTrainingData(randomData, NUM_ARRAY_ELEMENTS(randomData), 0));
    EMBANN_ERROR_CHECK(embann_copyTrainingData(randomData, NUM_ARRAY_ELEMENTS(randomData), 0));
    EMBANN_ERROR_CHECK(embann_getTrainingDataMax(&retval));
    EMBANN_ERROR_CHECK(embann_getTrainingDataMin(&retval));
    EMBANN_ERROR_CHECK(embann_getTrainingDataMean(&fretval));
    EMBANN_ERROR_CHECK(embann_getTrainingDataStdDev(&fretval));

#ifdef ACTIVATION_IS_FLOAT
    EMBANN_ERROR_CHECK(embann_trainDriverInTime(0.01, 1, true));
    EMBANN_ERROR_CHECK(embann_trainDriverInError(0.01, 0.1, true));
#elif defined(ACTIVATION_IS_SIGNED) || defined(ACTIVATION_IS_UNSIGNED)
    EMBANN_ERROR_CHECK(embann_trainDriverInError(1, 1, false));
    EMBANN_ERROR_CHECK(embann_trainDriverInTime(1, 1, false));
#endif

    EMBANN_ERROR_CHECK(embann_printNetwork());
    EMBANN_ERROR_CHECK(embann_printInputNeuronDetails(0));
    EMBANN_ERROR_CHECK(embann_printOutputNeuronDetails(0));
    EMBANN_ERROR_CHECK(embann_printHiddenNeuronDetails(0, 0));
    EMBANN_ERROR_CHECK(embann_errorReporting(0));
}
#endif




int embann_forwardPropagate(void)
{
    EMBANN_ERROR_CHECK(embann_sumAndSquashInput(
                            embann_getNetwork()->inputLayer->neuron, 
                            embann_getNetwork()->hiddenLayer[0]->neuron,
                            embann_getNetwork()->inputLayer->numNeurons, 
                            embann_getNetwork()->hiddenLayer[0]->numNeurons));

    EMBANN_LOGD(TAG, "Done Input -> 1st Hidden Layer");
    for (uint8_t i = 1; i < embann_getNetwork()->properties.numHiddenLayers; i++)
    {
        EMBANN_ERROR_CHECK(embann_sumAndSquash(
                            embann_getNetwork()->hiddenLayer[i - 1U]->neuron,
                            embann_getNetwork()->hiddenLayer[i]->neuron,
                            embann_getNetwork()->hiddenLayer[i - 1U]->numNeurons,
                            embann_getNetwork()->hiddenLayer[i]->numNeurons));

        EMBANN_LOGD(TAG, "Done Hidden Layer %d -> Hidden Layer %d", i - 1U, i);
    }

    

    EMBANN_ERROR_CHECK(embann_sumAndSquash(
        embann_getNetwork()->hiddenLayer[embann_getNetwork()->properties.numHiddenLayers - 1U]->neuron,
        embann_getNetwork()->outputLayer->neuron, 
        embann_getNetwork()->hiddenLayer[embann_getNetwork()->properties.numHiddenLayers - 1U]->numNeurons,
        embann_getNetwork()->outputLayer->numNeurons));

    EMBANN_LOGD(TAG, "Done Hidden Layer %d -> Output Layer", embann_getNetwork()->properties.numHiddenLayers);

    return EOK;
}






int embann_sumAndSquash(wNeuron_t* Input[], wNeuron_t* Output[], numInputs_t numInputs,
                            numOutputs_t numOutputs)
{
    
    // TODO, could overflow if numhidden >> numoutputs
    // TODO, add biasing
    accumulator_t accum = 0;
    for (numHiddenNeurons_t i = 0; i < numOutputs; i++)
    {
        accum = 0;

        for (numHiddenNeurons_t j = 0; j < numInputs; j++)
        {
            EMBANN_LOGV(TAG, "[%d] Input = 0x%x, Output = 0x%x", i, Input[j], Output[i]);
            EMBANN_LOGV(TAG, "[%d] [%d] In activation = %" ACTIVATION_PRINT " Out weight = %" WEIGHT_PRINT,
                                                    i, j, Input[j]->activation, Output[i]->params[j]->weight);

            accum += Input[j]->activation * Output[i]->params[j]->weight;
        }
    }

    for (numHiddenNeurons_t i = 0; i < numOutputs; i++)
    {
    #ifdef ACTIVATION_IS_FLOAT
        Output[i]->activation = tanhf(accum * PI);
#else
        accum = (accum > MAX_ACTIVATION) ? MAX_ACTIVATION : accum;
        Output[i]->activation = (accum < 0) ? 0 : accum;
#endif
        EMBANN_LOGD(TAG, "[%d] SumAndSquash Output %" ACTIVATION_PRINT, i, Output[i]->activation);
    }
    return EOK;
}






int embann_sumAndSquashInput(uNeuron_t* Input[], wNeuron_t* Output[], numInputs_t numInputs,
                                numOutputs_t numOutputs)
{
    accumulator_t accum = 0;
    
    for (numHiddenNeurons_t i = 0; i < numOutputs; i++)
    {
         accum = 0;

        for (numInputs_t j = 0; j < numInputs; j++)
        {
            EMBANN_LOGV(TAG, "[%d] Input = 0x%x, Output = 0x%x", i, Input[j], Output[i]);
            EMBANN_LOGV(TAG, "[%d] [%d] In activation = %" ACTIVATION_PRINT " Out weight = %" WEIGHT_PRINT,
                                                    i, j, Input[j]->activation, Output[i]->params[j]->weight);

            accum += Input[j]->activation * Output[i]->params[j]->weight;
        }
    }

    for (numHiddenNeurons_t i = 0; i < numOutputs; i++)
    {
#ifdef ACTIVATION_IS_FLOAT
        Output[i]->activation = tanhf(accum * PI);
#else
        accum = (accum > MAX_ACTIVATION) ? MAX_ACTIVATION : accum;
        Output[i]->activation = (accum < 0) ? 0 : accum;
#endif
    }
    return EOK;
}






int embann_calculateNetworkResponse(void)
{
    numOutputs_t mostLikelyOutput = 0;

    for (numOutputs_t i = 0; i < embann_getNetwork()->outputLayer->numNeurons; i++)
    {
        if (embann_getNetwork()->outputLayer->neuron[i]->activation >
            embann_getNetwork()->outputLayer->neuron[mostLikelyOutput]->activation)
        {
            mostLikelyOutput = i;
        }
        EMBANN_LOGV(TAG, "neuron[%d]: %" ACTIVATION_PRINT "likely: %d", i, embann_getNetwork()->outputLayer->neuron[i]->activation, mostLikelyOutput);
    }
    
    embann_getNetwork()->properties.networkResponse = mostLikelyOutput;
    return EOK;
}



network_t* embann_getNetwork(void)
{
    return pNetwork;
}



int embann_setNetwork(network_t* newNetwork)
{
    pNetwork = newNetwork;
    return EOK;
}




int* embann_getErrno(void)
{
    return &embann_errno;
}




trainingDataCollection_t* embann_getDataCollection(void)
{
    return &trainingDataCollection;
}




/*
    TODO try column-wise & row-wise memory access for better cache hit,
    loop interchange
*/

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
