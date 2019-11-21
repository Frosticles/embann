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
network_t* pNetworkGlobal;
/* Structure of pointers to training data */
static trainingDataCollection_t trainingDataCollection = {
    .tail = NULL,
    .head = NULL,
    .numEntries = 0U
};


static int embann_sumAndSquashHidden(hiddenLayer_t* input, hiddenLayer_t* output, numHiddenNeurons_t numInputs, numHiddenNeurons_t numOutputs);
static int embann_sumAndSquashOutput(hiddenLayer_t* input, outputLayer_t* output, numHiddenNeurons_t numInputs, numOutputs_t numOutputs);
static int embann_sumAndSquashInput(inputLayer_t* input, hiddenLayer_t* output, numInputs_t numInputs, numHiddenNeurons_t numOutputs);





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
    EMBANN_ERROR_CHECK(embann_printNetwork());
    EMBANN_ERROR_CHECK(embann_forwardPropagate());
    EMBANN_ERROR_CHECK(embann_printNetwork());

#ifdef CONFIG_MEMORY_ALLOCATION_STATIC
    activation_t randomData[CONFIG_NUM_INPUT_NEURONS];
#else
    activation_t randomData[pNetworkGlobal->inputLayer->numNeurons];
#endif
    activation_t retval;
    float fretval;
    for (uint8_t i = 0; i < NUM_ARRAY_ELEMENTS(randomData); i++)
    {
        randomData[i] = random();
    }

    EMBANN_ERROR_CHECK(embann_addTrainingData(randomData, sizeof(randomData), 0));
    EMBANN_ERROR_CHECK(embann_copyTrainingData(randomData, sizeof(randomData), 0));
    EMBANN_ERROR_CHECK(embann_getTrainingDataMax(&retval));
    EMBANN_ERROR_CHECK(embann_getTrainingDataMin(&retval));
    EMBANN_ERROR_CHECK(embann_getTrainingDataMean(&fretval));
    EMBANN_ERROR_CHECK(embann_getTrainingDataStdDev(&fretval));

#ifdef ACTIVATION_IS_FLOAT
    EMBANN_ERROR_CHECK(embann_trainDriverInTime(0.01, 1, true));
    EMBANN_ERROR_CHECK(embann_trainDriverInError(0.01, 0.1, true));
#elif defined(ACTIVATION_IS_SIGNED) || defined(ACTIVATION_IS_UNSIGNED)
    //EMBANN_ERROR_CHECK(embann_trainDriverInError(1, 1, false));
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
                            pNetworkGlobal->inputLayer, 
                            pNetworkGlobal->hiddenLayer[0],
                            pNetworkGlobal->inputLayer->numNeurons, 
                            pNetworkGlobal->hiddenLayer[0]->numNeurons));

    EMBANN_LOGD(TAG, "Done Input -> 1st Hidden Layer");
    for (uint8_t i = 1; i < pNetworkGlobal->properties.numHiddenLayers; i++)
    {
        EMBANN_ERROR_CHECK(embann_sumAndSquashHidden(
                            pNetworkGlobal->hiddenLayer[i - 1U],
                            pNetworkGlobal->hiddenLayer[i],
                            pNetworkGlobal->hiddenLayer[i - 1U]->numNeurons,
                            pNetworkGlobal->hiddenLayer[i]->numNeurons));

        EMBANN_LOGD(TAG, "Done Hidden Layer %d -> Hidden Layer %d", i - 1U, i);
    }

    

    EMBANN_ERROR_CHECK(embann_sumAndSquashOutput(
        pNetworkGlobal->hiddenLayer[pNetworkGlobal->properties.numHiddenLayers - 1U],
        pNetworkGlobal->outputLayer, 
        pNetworkGlobal->hiddenLayer[pNetworkGlobal->properties.numHiddenLayers - 1U]->numNeurons,
        pNetworkGlobal->outputLayer->numNeurons));

    EMBANN_LOGD(TAG, "Done Hidden Layer %d -> Output Layer", pNetworkGlobal->properties.numHiddenLayers);

    return EOK;
}





static int embann_sumAndSquashInput(inputLayer_t* input, hiddenLayer_t* output, numInputs_t numInputs, numHiddenNeurons_t numOutputs)
{
#ifdef CONFIG_MEMORY_ALLOCATION_STATIC
    accumulator_t accum[CONFIG_NUM_HIDDEN_NEURONS];
#else
    accumulator_t accum[numOutputs];
#endif
    
    for (numHiddenNeurons_t i = 0; i < numOutputs; i++)
    {
        accum[i] = 0;

        for (numInputs_t j = 0; j < numInputs; j++)
        {
            EMBANN_LOGV(TAG, "[%d] [%d] In activation = 0x%x, Out weight = 0x%x", 
                                                    i, j, &input->activation[i], &output->weight[i][j]);
            EMBANN_LOGV(TAG, "[%d] [%d] In activation = %" ACTIVATION_PRINT " Out weight = %" WEIGHT_PRINT,
                                                    i, j, input->activation[i], output->weight[i][j]);

            accum[i] += input->activation[i] * output->weight[i][j];
        }
    }

    for (numHiddenNeurons_t i = 0; i < numOutputs; i++)
    {
#ifdef ACTIVATION_IS_FLOAT
        output->activation[i] = tanhf(accum[i] * PI);
#else
        accum[i] = (accum[i] > MAX_ACTIVATION) ? MAX_ACTIVATION : accum[i];
        output->activation[i] = (accum[i] < 0) ? 0 : accum[i];
#endif
        EMBANN_LOGD(TAG, "[%d] SumAndSquash Output %" ACTIVATION_PRINT, i, output->activation[i]);
    }
    return EOK;
}





static int embann_sumAndSquashHidden(hiddenLayer_t* input, hiddenLayer_t* output, numHiddenNeurons_t numInputs, numHiddenNeurons_t numOutputs)
{
#ifdef CONFIG_MEMORY_ALLOCATION_STATIC
    accumulator_t accum[CONFIG_NUM_HIDDEN_NEURONS];
#else
    accumulator_t accum[numOutputs];
#endif
    
    // TODO, add biasing

    for (numHiddenNeurons_t i = 0; i < numOutputs; i++)
    {
        accum[i] = 0;

        for (numHiddenNeurons_t j = 0; j < numInputs; j++)
        {
            EMBANN_LOGV(TAG, "[%d] [%d] In activation = 0x%x, Out weight = 0x%x", 
                                                    i, j, &input->activation[i], &output->weight[i][j]);
            EMBANN_LOGV(TAG, "[%d] [%d] In activation = %" ACTIVATION_PRINT " Out weight = %" WEIGHT_PRINT,
                                                    i, j, input->activation[i], output->weight[i][j]);

            accum[i] += input->activation[i] * output->weight[i][j];
        }
    }

    for (numHiddenNeurons_t i = 0; i < numOutputs; i++)
    {
#ifdef ACTIVATION_IS_FLOAT
        output->activation[i] = tanhf(accum[i] * PI);
#else
        accum[i] = (accum[i] > MAX_ACTIVATION) ? MAX_ACTIVATION : accum[i];
        output->activation[i] = (accum[i] < 0) ? 0 : accum[i];
#endif
        EMBANN_LOGD(TAG, "[%d] SumAndSquash Output %" ACTIVATION_PRINT, i, output->activation[i]);
    }
    return EOK;
}




static int embann_sumAndSquashOutput(hiddenLayer_t* input, outputLayer_t* output, numHiddenNeurons_t numInputs, numOutputs_t numOutputs)
{
#ifdef CONFIG_MEMORY_ALLOCATION_STATIC
    accumulator_t accum[CONFIG_NUM_OUTPUT_NEURONS];
#else
    accumulator_t accum[numOutputs];
#endif
    
    // TODO, add biasing

    for (numOutputs_t i = 0; i < numOutputs; i++)
    {
        accum[i] = 0;

        for (numHiddenNeurons_t j = 0; j < numInputs; j++)
        {
            EMBANN_LOGV(TAG, "[%d] [%d] In activation = 0x%x, Out weight = 0x%x", 
                                                    i, j, &input->activation[i], &output->weight[i][j]);
            EMBANN_LOGV(TAG, "[%d] [%d] In activation = %" ACTIVATION_PRINT " Out weight = %" WEIGHT_PRINT,
                                                    i, j, input->activation[i], output->weight[i][j]);

            accum[i] += input->activation[i] * output->weight[i][j];
        }
    }

    for (numOutputs_t i = 0; i < numOutputs; i++)
    {
#ifdef ACTIVATION_IS_FLOAT
        output->activation[i] = tanhf(accum[i] * PI);
#else
        accum[i] = (accum[i] > MAX_ACTIVATION) ? MAX_ACTIVATION : accum[i];
        output->activation[i] = (accum[i] < 0) ? 0 : accum[i];
#endif
        EMBANN_LOGD(TAG, "[%d] SumAndSquash Output %" ACTIVATION_PRINT, i, output->activation[i]);
    }
    return EOK;
}






int embann_calculateNetworkResponse(void)
{
    numOutputs_t mostLikelyOutput = 0;

    for (numOutputs_t i = 0; i < pNetworkGlobal->outputLayer->numNeurons; i++)
    {
        if (pNetworkGlobal->outputLayer->activation[i] >
            pNetworkGlobal->outputLayer->activation[mostLikelyOutput])
        {
            mostLikelyOutput = i;
        }
        EMBANN_LOGV(TAG, "neuron[%d]: %" ACTIVATION_PRINT "likely: %d", i, pNetworkGlobal->outputLayer->activation[i], mostLikelyOutput);
    }
    
    pNetworkGlobal->properties.networkResponse = mostLikelyOutput;
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
    int32_t MAX_ALIGNMENT testInt[300];
    float MAX_ALIGNMENT testFloat[300];
    double MAX_ALIGNMENT testDouble[300];
    int32_t MAX_ALIGNMENT testIntWeight[300];
    float MAX_ALIGNMENT testFloatWeight[300];
    double MAX_ALIGNMENT testDoubleWeight[300];
    int32_t MAX_ALIGNMENT testIntBias[300];
    float MAX_ALIGNMENT testFloatBias[300];
    double MAX_ALIGNMENT testDoubleBias[300];
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
