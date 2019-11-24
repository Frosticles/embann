#include "embann.h"
#include "embann_log.h"

#define TAG "Embann Train"

extern network_t* pNetworkGlobal;

static int _trainOutput(accumulator_t* totalErrorInCurrentLayer, const numOutputs_t numOutputs, 
                        const numLayers_t lastHiddenLayer, activation_t learningRate, 
                        numOutputs_t correctOutput);
static int _trainHidden(accumulator_t* totalErrorInCurrentLayer, accumulator_t* totalErrorInNextLayer, 
                        const numLayers_t lastHiddenLayer, activation_t learningRate);
static int _trainInput(accumulator_t* totalErrorInCurrentLayer, accumulator_t* totalErrorInNextLayer, 
                        activation_t learningRate);
static int embann_train(numOutputs_t correctOutput, activation_t learningRate, 
                        accumulator_t* totalErrorInCurrentLayer, accumulator_t* totalErrorInNextLayer);




int embann_trainDriverInTime(activation_t learningRate, uint32_t numSeconds)
{
    numOutputs_t randomOutput;
    numTrainingDataEntries_t randomTrainingSet;
#ifdef CONFIG_MEMORY_ALLOCATION_STATIC
    // TODO, Python preprocessor to define required size for error arrays.
    accumulator_t totalErrorInCurrentLayer[CONFIG_NUM_INPUT_NEURONS];
    accumulator_t totalErrorInNextLayer[CONFIG_NUM_INPUT_NEURONS];
#else
    // TODO, for loop to find largest layer and size these correctly
    accumulator_t totalErrorInCurrentLayer[CONFIG_NUM_INPUT_NEURONS];
    accumulator_t totalErrorInNextLayer[CONFIG_NUM_INPUT_NEURONS];
#endif

    uint32_t startTime = millis();

    while ((millis() - startTime) < (numSeconds * 1000UL))
    {
        randomOutput = random() % pNetworkGlobal->outputLayer->numNeurons;
        randomTrainingSet = random() % embann_getDataCollection()->numEntries;

        // TODO, make this input method configurable
        // TODO, don't always just get the first training data
        embann_inputRaw(embann_getDataCollection()->head->data);
        EMBANN_ERROR_CHECK(embann_forwardPropagate());
        EMBANN_ERROR_CHECK(embann_train(randomOutput, learningRate, totalErrorInCurrentLayer, totalErrorInNextLayer));
    }
    return EOK;
}

int embann_trainDriverInError(activation_t learningRate, activation_t desiredCost)
{
    numOutputs_t randomOutput;
    numTrainingDataEntries_t randomTrainingSet;
    const numOutputs_t numOutputs = pNetworkGlobal->outputLayer->numNeurons;
    bool converged = false;
#ifdef CONFIG_MEMORY_ALLOCATION_STATIC
    // TODO, Python preprocessor to define required size for error arrays.
    accumulator_t totalErrorInCurrentLayer[CONFIG_NUM_INPUT_NEURONS];
    accumulator_t totalErrorInNextLayer[CONFIG_NUM_INPUT_NEURONS];
#else
    // TODO, for loop to find largest layer and size these correctly
    accumulator_t totalErrorInCurrentLayer[CONFIG_NUM_INPUT_NEURONS];
    accumulator_t totalErrorInNextLayer[CONFIG_NUM_INPUT_NEURONS];
#endif

#if (CONFIG_LOG_DEFAULT_LEVEL >= EMBANN_LOG_INFO)
    uint16_t count = 50000;
#endif

    while (!converged)
    {
        randomOutput = random() % numOutputs;
        randomTrainingSet = random() % embann_getDataCollection()->numEntries;
        converged = true;

        // TODO, make this input method configurable
        // TODO, don't always just get the first training data
        embann_inputRaw(embann_getDataCollection()->head->data);
        EMBANN_ERROR_CHECK(embann_forwardPropagate());

        memset(totalErrorInCurrentLayer, 0, sizeof(totalErrorInCurrentLayer));
        memset(totalErrorInNextLayer, 0, sizeof(totalErrorInNextLayer));
        totalErrorInCurrentLayer[randomOutput] = MAX_ACTIVATION;

        for (numOutputs_t i = 0; i < numOutputs; i++)
        {        
            totalErrorInCurrentLayer[i] = (pNetworkGlobal->outputLayer->activation[i] - totalErrorInCurrentLayer[i]);
            if (abs(totalErrorInCurrentLayer[i]) > desiredCost)
            {
                converged = false;
            }
        }

#if (CONFIG_LOG_DEFAULT_LEVEL >= EMBANN_LOG_INFO)
        if (count == 50000)
        {
            accumulator_t averageCost = 0;
            for (numOutputs_t i = 0; i < numOutputs; i++)
            {
                averageCost += totalErrorInCurrentLayer[i];
            }
            averageCost /= numOutputs;
            EMBANN_LOGI(TAG, "Average Cost: %d, e0 %d, e1 %d, e2 %d", averageCost, totalErrorInCurrentLayer[0],
                                                                        totalErrorInCurrentLayer[1],
                                                                        totalErrorInCurrentLayer[2]);
            EMBANN_ERROR_CHECK(embann_printNetwork());
            EMBANN_LOGI(TAG, "Output Neuron 0 Error = %" ACCUMULATOR_PRINT, totalErrorInCurrentLayer[0]);
            EMBANN_LOGI(TAG, "Output Weight [0][0] = %" WEIGHT_PRINT, pNetworkGlobal->outputLayer->weight[0][0]);
            EMBANN_LOGI(TAG, "Hidden Layer 0 Weight [0][0] = %" WEIGHT_PRINT, pNetworkGlobal->hiddenLayer[0]->weight[0][0]);
            count = 0;
        }
        else
        {
            count++;
        }
#endif

        for (numOutputs_t i = 0; i < numOutputs; i++)
        {        
            if (totalErrorInCurrentLayer[i] > 0)
            {
                totalErrorInCurrentLayer[i] = min(totalErrorInCurrentLayer[i], 1);
            }
            else
            {
                totalErrorInCurrentLayer[i] = max(totalErrorInCurrentLayer[i], -1);
            }
        }

        EMBANN_ERROR_CHECK(embann_train(randomOutput, learningRate, totalErrorInCurrentLayer, totalErrorInNextLayer));
    }
    return EOK;
}





static int embann_train(numOutputs_t correctOutput, activation_t learningRate, 
                        accumulator_t* totalErrorInCurrentLayer, accumulator_t* totalErrorInNextLayer)
{
    const numOutputs_t numOutputs = pNetworkGlobal->outputLayer->numNeurons;
    const numLayers_t lastHiddenLayer = pNetworkGlobal->properties.numHiddenLayers - 1U;

    if (correctOutput > numOutputs)
    {
        return ENOENT;
    }

    EMBANN_ERROR_CHECK(_trainOutput(totalErrorInCurrentLayer, numOutputs, lastHiddenLayer, learningRate, correctOutput));
    memcpy(totalErrorInNextLayer, totalErrorInCurrentLayer, CONFIG_NUM_INPUT_NEURONS * sizeof(accumulator_t));
    memset(totalErrorInCurrentLayer, 0, CONFIG_NUM_INPUT_NEURONS * sizeof(accumulator_t));
    EMBANN_ERROR_CHECK(_trainHidden(totalErrorInCurrentLayer, totalErrorInNextLayer, lastHiddenLayer, learningRate));
    EMBANN_ERROR_CHECK(_trainInput(totalErrorInCurrentLayer, totalErrorInNextLayer, learningRate));
    return EOK;
}





static int _trainOutput(accumulator_t* totalErrorInCurrentLayer, const numOutputs_t numOutputs, 
                        const numLayers_t lastHiddenLayer, activation_t learningRate, 
                        numOutputs_t correctOutput)
{
    // TODO, add backpropagation for other activation functions (this is just ReLU)
    // TODO, add biasing
    const numHiddenNeurons_t numHiddenNeurons = pNetworkGlobal->hiddenLayer[lastHiddenLayer]->numNeurons;

    EMBANN_LOGD(TAG, "Output Layer Error [0] = %" ACCUMULATOR_PRINT, totalErrorInCurrentLayer[0]);
    EMBANN_LOGD(TAG, "Old Output Weight [0][0] = %" WEIGHT_PRINT, pNetworkGlobal->outputLayer->weight[0][0]);

    for (numOutputs_t i = 0; i < numOutputs; i++)
    {        
        for (numHiddenNeurons_t j = 0; j < numHiddenNeurons; j++)
        {
            EMBANN_LOGV(TAG, "Old Output Weight [%d][%d] = %" WEIGHT_PRINT, i, j, pNetworkGlobal->outputLayer->weight[i][j]);

            pNetworkGlobal->outputLayer->weight[i][j] -= 
                        pNetworkGlobal->hiddenLayer[lastHiddenLayer]->activation[i] *
                        totalErrorInCurrentLayer[i];

            EMBANN_LOGV(TAG, "New Output Weight [%d][%d] = %" WEIGHT_PRINT, i, j, pNetworkGlobal->outputLayer->weight[i][j]);
        }
    }

    EMBANN_LOGD(TAG, "New Output Weight [0][0] = %" WEIGHT_PRINT, pNetworkGlobal->outputLayer->weight[0][0]);
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
            for (numHiddenNeurons_t k = 0; k < numNeuronsInNextLayer; k++)
            {        
                totalErrorInCurrentLayer[j] += pNetworkGlobal->hiddenLayer[i]->weight[j][k] * 
                                                    totalErrorInNextLayer[k];

                if (totalErrorInCurrentLayer[j] > 0)
                {
                    totalErrorInCurrentLayer[j] = min(totalErrorInCurrentLayer[j], 1);
                }
                else
                {
                    totalErrorInCurrentLayer[j] = max(totalErrorInCurrentLayer[j], -1);
                }
            }
        }

        EMBANN_LOGD(TAG, "Hidden Layer %d Error [0] = %" ACCUMULATOR_PRINT, i, totalErrorInCurrentLayer[0]);
        EMBANN_LOGD(TAG, "Old Hidden Layer %d Weight [0][0] = %" WEIGHT_PRINT, i, pNetworkGlobal->hiddenLayer[i]->weight[0][0]);

        for (numHiddenNeurons_t j = 0; j < numNeuronsInCurrentLayer; j++)
        {   
            for (numHiddenNeurons_t k = 0; k < numNeuronsInNextLayer; k++)
            {  
                EMBANN_LOGV(TAG, "Old Hidden Layer %d Weight [%d][%d] = %" WEIGHT_PRINT, i, j, k, pNetworkGlobal->hiddenLayer[i]->weight[j][k]);

                pNetworkGlobal->hiddenLayer[i]->weight[j][k] -=
                            (pNetworkGlobal->hiddenLayer[i - 1U]->activation[j] *
                            totalErrorInCurrentLayer[j]) /
                            (8);

                EMBANN_LOGV(TAG, "New Hidden Layer %d Weight [%d][%d] = %" WEIGHT_PRINT, i, j, k, pNetworkGlobal->hiddenLayer[i]->weight[j][k]);
            }
        }

        EMBANN_LOGD(TAG, "New Hidden Layer %d Weight [0][0] = %" WEIGHT_PRINT, i, pNetworkGlobal->hiddenLayer[i]->weight[0][0]);

        // TODO, Python preprocessor to define required size for error arrays.
        memcpy(totalErrorInNextLayer, totalErrorInCurrentLayer, CONFIG_NUM_INPUT_NEURONS * sizeof(accumulator_t));
        memset(totalErrorInCurrentLayer, 0, CONFIG_NUM_INPUT_NEURONS * sizeof(accumulator_t));

        numNeuronsInCurrentLayer = pNetworkGlobal->hiddenLayer[i - 1U]->numNeurons;
        numNeuronsInNextLayer = pNetworkGlobal->hiddenLayer[i]->numNeurons;
    }

    return EOK;
}






static int _trainInput(accumulator_t* totalErrorInCurrentLayer, accumulator_t* totalErrorInNextLayer, 
                        activation_t learningRate)
{
    // TODO, add backpropagation for other activation functions (this is just ReLU)
    // TODO, add biasing
    numHiddenNeurons_t numNeuronsInCurrentLayer = pNetworkGlobal->hiddenLayer[0]->numNeurons;
    numHiddenNeurons_t numNeuronsInNextLayer = pNetworkGlobal->inputLayer->numNeurons;

    for (numInputs_t i = 0; i < numNeuronsInCurrentLayer; i++)
    {
        for (numHiddenNeurons_t j = 0; j < numNeuronsInNextLayer; j++)
        {        
            totalErrorInCurrentLayer[j] += pNetworkGlobal->hiddenLayer[0]->weight[i][j] * 
                                                totalErrorInNextLayer[i];

            if (totalErrorInCurrentLayer[j] > 0)
            {
                totalErrorInCurrentLayer[j] = min(totalErrorInCurrentLayer[j], 1);
            }
            else
            {
                totalErrorInCurrentLayer[j] = max(totalErrorInCurrentLayer[j], -1);
            }
        }
    }

    EMBANN_LOGD(TAG, "Hidden Layer 0 Error [0] = %" ACCUMULATOR_PRINT, totalErrorInCurrentLayer[0]);
    EMBANN_LOGD(TAG, "Old Hidden Layer 0 Weight [0][0] = %" WEIGHT_PRINT, pNetworkGlobal->hiddenLayer[0]->weight[0][0]);

    for (numInputs_t i = 0; i < numNeuronsInCurrentLayer; i++)
    {   
        for (numHiddenNeurons_t j = 0; j < numNeuronsInNextLayer; j++)
        {  
            EMBANN_LOGV(TAG, "Old Hidden Layer 0 Weight [%d][%d] = %" WEIGHT_PRINT, i, j, pNetworkGlobal->hiddenLayer[0]->weight[i][j]);
            
            pNetworkGlobal->hiddenLayer[0]->weight[i][j] -=
                        (pNetworkGlobal->inputLayer->activation[j] *
                        totalErrorInCurrentLayer[j]) /
                        (8);

            EMBANN_LOGV(TAG, "New Hidden Layer 0 Weight [%d][%d] = %" WEIGHT_PRINT, i, j, pNetworkGlobal->hiddenLayer[0]->weight[i][j]);
        }
    }

    EMBANN_LOGD(TAG, "New Hidden Layer 0 Weight [0][0] = %" WEIGHT_PRINT, pNetworkGlobal->hiddenLayer[0]->weight[0][0]);

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