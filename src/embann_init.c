#include "embann.h"
#include "embann_log.h"
#ifdef CONFIG_MEMORY_ALLOCATION_STATIC
#include "embann_static.h"
#endif

#define TAG "Embann Init"


static void _printInputLayer(inputLayer_t* pInputLayer, numInputs_t numInputNeurons);
static void _printHiddenLayer(hiddenLayer_t* pHiddenLayer, numHiddenNeurons_t numHiddenNeurons);
static void _printHiddenNeuronParams(hiddenLayer_t* pHiddenLayer, numHiddenNeurons_t j, numHiddenNeurons_t k);
static void _printConnectedHiddenLayer(numLayers_t layerNum);
static void _printOutputLayer(outputLayer_t* pOutputLayer, numOutputs_t numOutputNeurons);










int embann_init(numInputs_t numInputNeurons,
                numHiddenNeurons_t numHiddenNeurons, 
                numLayers_t numHiddenLayers,
                numOutputs_t numOutputNeurons)
{
    network_t* pNetwork;
    if ((numInputNeurons == 0U) || (numHiddenNeurons == 0U) || 
        (numHiddenLayers == 0U) || (numOutputNeurons == 0U))
    {
        // Deviation from MISRA C2012 15.5 for reasonably simple error return values
        // cppcheck-suppress misra-c2012-15.5
        return EINVAL;
    }

#ifdef CONFIG_MEMORY_ALLOCATION_STATIC
    pNetwork = &staticNetwork;
#else
    pNetwork = (network_t*) malloc(sizeof(network_t) + 
                                 (sizeof(hiddenLayer_t) * numHiddenLayers));
    EMBANN_MALLOC_CHECK(pNetwork);
#endif

    EMBANN_ERROR_CHECK(embann_setNetwork(pNetwork));

    EMBANN_ERROR_CHECK(embann_initInputLayer(numInputNeurons));
    EMBANN_ERROR_CHECK(embann_initHiddenLayer(numHiddenNeurons,
                                                numHiddenLayers,
                                                numInputNeurons));
    EMBANN_ERROR_CHECK(embann_initOutputLayer(numOutputNeurons,
                                                numHiddenNeurons));

    embann_getNetwork()->properties.numLayers = numHiddenLayers + 2U;
    embann_getNetwork()->properties.numHiddenLayers = numHiddenLayers;
    embann_getNetwork()->properties.networkResponse = 0U;

    return EOK;
}





int embann_initInputLayer(numInputs_t numInputNeurons)
{
    inputLayer_t* pInputLayer;
#ifdef CONFIG_MEMORY_ALLOCATION_STATIC
    pInputLayer = staticNetwork.inputLayer;
#else
    uNeuron_t* pNeuron;
    pInputLayer = (inputLayer_t*) malloc(sizeof(inputLayer_t) + 
                                                (sizeof(uNeuron_t*) * numInputNeurons));
    EMBANN_MALLOC_CHECK(pInputLayer);
    pInputLayer->numNeurons = numInputNeurons;
#endif

    _printInputLayer(pInputLayer, numInputNeurons);

    for (numInputs_t i = 0; i < numInputNeurons; i++)
    {
#ifdef CONFIG_MEMORY_ALLOCATION_DYNAMIC
        pNeuron = (uNeuron_t*) malloc(sizeof(uNeuron_t));
        EMBANN_MALLOC_CHECK(pNeuron);
        pInputLayer->neuron[i] = pNeuron;
#endif
        pInputLayer->neuron[i]->activation = RAND_ACTIVATION();
        EMBANN_LOGI(TAG, "act [%d] = %d", i, pInputLayer->neuron[i]->activation);
    }

#ifdef CONFIG_MEMORY_ALLOCATION_DYNAMIC
    embann_getNetwork()->inputLayer = pInputLayer;
#endif

    for (numInputs_t k = 0; k < numInputNeurons; k++)
    {
        EMBANN_LOGI(TAG, "act [%d] = %d", k, embann_getNetwork()->inputLayer->neuron[k]->activation);
    }

    EMBANN_LOGI(TAG, "done input");
    return EOK;
}






int embann_initHiddenLayer(numHiddenNeurons_t numHiddenNeurons,
#if (defined(CONFIG_MEMORY_ALLOCATION_STATIC) && (CONFIG_NUM_HIDDEN_LAYERS > 1)) || defined(CONFIG_MEMORY_ALLOCATION_DYNAMIC)
                            numLayers_t numHiddenLayers,
#endif
                            numInputs_t numInputNeurons)
{
    EMBANN_ERROR_CHECK(embann_initInputToHiddenLayer(numHiddenNeurons, numInputNeurons));

#if (defined(CONFIG_MEMORY_ALLOCATION_STATIC) && (CONFIG_NUM_HIDDEN_LAYERS > 1)) || defined(CONFIG_MEMORY_ALLOCATION_DYNAMIC)
    if (numHiddenLayers > 1U)
    {
        EMBANN_ERROR_CHECK(embann_initHiddenToHiddenLayer(numHiddenNeurons, numHiddenLayers));
    }
#endif
    return EOK;
}






int embann_initInputToHiddenLayer(numHiddenNeurons_t numHiddenNeurons,
                                    numInputs_t numInputNeurons)
{
    hiddenLayer_t* pHiddenLayer;
#ifdef CONFIG_MEMORY_ALLOCATION_STATIC
    pHiddenLayer = staticNetwork.hiddenLayer[0];
#else
    wNeuron_t* pNeuron;
    neuronParams_t* pHiddenLayerParams;
    pHiddenLayer = (hiddenLayer_t*) malloc(sizeof(hiddenLayer_t) + (sizeof(wNeuron_t*) * numHiddenNeurons));
    EMBANN_MALLOC_CHECK(pHiddenLayer);
    pHiddenLayer->numNeurons = numHiddenNeurons;
#endif
    _printHiddenLayer(pHiddenLayer, numHiddenNeurons);


    for (numHiddenNeurons_t j = 0; j < numHiddenNeurons; j++)
    {
#ifdef CONFIG_MEMORY_ALLOCATION_DYNAMIC
        pNeuron = (wNeuron_t*) malloc(sizeof(wNeuron_t) + (sizeof(neuronParams_t*) * numInputNeurons));
        EMBANN_MALLOC_CHECK(pNeuron);
        pHiddenLayer->neuron[j] = pNeuron;
#endif
        pHiddenLayer->neuron[j]->activation = RAND_ACTIVATION();

        EMBANN_LOGI(TAG, "act [%d] = %d", j, pHiddenLayer->neuron[j]->activation);

        for (numInputs_t k = 0; k < numInputNeurons; k++)
        {
#ifdef CONFIG_MEMORY_ALLOCATION_DYNAMIC
            pHiddenLayerParams = (neuronParams_t*) malloc(sizeof(neuronParams_t));
            EMBANN_MALLOC_CHECK(pHiddenLayerParams);
            pHiddenLayer->neuron[j]->params[k] = pHiddenLayerParams;
#endif
            pHiddenLayer->neuron[j]->params[k]->bias = RAND_BIAS();
            pHiddenLayer->neuron[j]->params[k]->weight = RAND_WEIGHT();

            _printHiddenNeuronParams(pHiddenLayer, j, k);
        }
    }

#ifdef CONFIG_MEMORY_ALLOCATION_DYNAMIC
    embann_getNetwork()->hiddenLayer[0] = pHiddenLayer;
#endif
    for (uint16_t k = 0; k < numHiddenNeurons; k++)
    {
        EMBANN_LOGI(TAG, "act [%d] = %d", k, embann_getNetwork()->hiddenLayer[0]->neuron[k]->activation);
    }

    _printConnectedHiddenLayer(0);
    EMBANN_LOGI(TAG, "done hidden");
    return EOK;
}







#if (defined(CONFIG_MEMORY_ALLOCATION_STATIC) && (CONFIG_NUM_HIDDEN_LAYERS > 1)) || defined(CONFIG_MEMORY_ALLOCATION_DYNAMIC)
int embann_initHiddenToHiddenLayer(numHiddenNeurons_t numHiddenNeurons,
                                    numLayers_t numHiddenLayers)
{
    hiddenLayer_t* pHiddenLayer;
    wNeuron_t* pNeuron;
    neuronParams_t* pHiddenLayerParams;
    
    for (numLayers_t i = 1; i < (numHiddenLayers - 1U); i++)
    {
#ifdef CONFIG_MEMORY_ALLOCATION_STATIC
        pHiddenLayer = (hiddenLayer_t*) &staticHiddenLayer[i];
#else
        pHiddenLayer = (hiddenLayer_t*) malloc(sizeof(hiddenLayer_t) + 
                                                (sizeof(wNeuron_t*) * numHiddenNeurons));
        EMBANN_MALLOC_CHECK(pHiddenLayer);
#endif
        _printHiddenLayer(pHiddenLayer, numHiddenNeurons);
        pHiddenLayer->numNeurons = numHiddenNeurons;

        for (numHiddenNeurons_t j = 0; j < numHiddenNeurons; j++)
        {    
#ifdef CONFIG_MEMORY_ALLOCATION_DYNAMIC
            pNeuron = (wNeuron_t*) malloc(sizeof(wNeuron_t) + (sizeof(neuronParams_t*) * 
                                            (uint16_t)((i == 0U) ? numInputNeurons : numHiddenNeurons)));
            EMBANN_MALLOC_CHECK(pNeuron);
            pHiddenLayer->neuron[j] = pNeuron;
#endif
            pHiddenLayer->neuron[j]->activation = RAND_ACTIVATION();

            EMBANN_LOGI(TAG, "act [%d] = %d", j, pHiddenLayer->neuron[j]->activation);

            for (numHiddenNeurons_t k = 0; k < numHiddenNeurons; k++)
            {
#ifdef CONFIG_MEMORY_ALLOCATION_DYNAMIC
                pHiddenLayerParams = (neuronParams_t*) malloc(sizeof(neuronParams_t));
                EMBANN_MALLOC_CHECK(pHiddenLayerParams);
                pHiddenLayer->neuron[j]->params[k] = pHiddenLayerParams;
#endif
                _printHiddenNeuronParams(pHiddenLayer, j, k);

                pHiddenLayer->neuron[j]->params[k]->bias = RAND_BIAS();
                pHiddenLayer->neuron[j]->params[k]->weight = RAND_WEIGHT();
                EMBANN_LOGI(TAG, "act [%d] = %d", j, pHiddenLayer->neuron[j]->activation);
            }
        }

        EMBANN_LOGI(TAG, "done hidden");
#ifdef CONFIG_MEMORY_ALLOCATION_DYNAMIC
        embann_getNetwork()->hiddenLayer[i] = pHiddenLayer;
#endif
        _printConnectedHiddenLayer(i);

        for (uint16_t k = 0; k < (numHiddenNeurons - 1U); k++)
        {
            EMBANN_LOGI(TAG, "act [%d] = %d", k, embann_getNetwork()->hiddenLayer[i]->neuron[k]->activation);
        }
    }
    return EOK;
}
#endif







int embann_initOutputLayer(numOutputs_t numOutputNeurons,
                            numHiddenNeurons_t numHiddenNeurons)
{
    outputLayer_t* pOutputLayer;
#ifdef CONFIG_MEMORY_ALLOCATION_STATIC
    pOutputLayer = staticNetwork.outputLayer;
#else
    wNeuron_t* pNeuron;
    neuronParams_t* pOutputLayerParams;
    pOutputLayer = (outputLayer_t*) malloc(sizeof(outputLayer_t) + 
                                                        (sizeof(wNeuron_t*) * numOutputNeurons));
    EMBANN_MALLOC_CHECK(pOutputLayer);
    pOutputLayer->numNeurons = numOutputNeurons;
#endif

    _printOutputLayer(pOutputLayer, numOutputNeurons);

    for (numOutputs_t i = 0; i < numOutputNeurons; i++)
    {
#ifdef CONFIG_MEMORY_ALLOCATION_DYNAMIC
        pNeuron = (wNeuron_t*) malloc(sizeof(wNeuron_t) + (sizeof(neuronParams_t*) * numHiddenNeurons));
        EMBANN_MALLOC_CHECK(pNeuron);
        pOutputLayer->neuron[i] = pNeuron;
#endif

        pOutputLayer->neuron[i]->activation = RAND_ACTIVATION();
        EMBANN_LOGI(TAG, "act [%d] = %d", i, pOutputLayer->neuron[i]->activation);
        
        for (numHiddenNeurons_t j = 0; j < numHiddenNeurons; j++)
        {
#ifdef CONFIG_MEMORY_ALLOCATION_DYNAMIC
            pOutputLayerParams = (neuronParams_t*) malloc(sizeof(neuronParams_t));
            EMBANN_MALLOC_CHECK(pOutputLayerParams);
            pOutputLayer->neuron[i]->params[j] = pOutputLayerParams;
#endif
            pOutputLayer->neuron[i]->params[j]->bias = RAND_BIAS();
            pOutputLayer->neuron[i]->params[j]->weight = RAND_WEIGHT();
        }
    }
    
#ifdef CONFIG_MEMORY_ALLOCATION_DYNAMIC
    embann_getNetwork()->outputLayer = pOutputLayer;
#endif

    for (uint16_t k = 0; k < numOutputNeurons; k++)
    {
        EMBANN_LOGI(TAG, "act [%d] = %d", k, embann_getNetwork()->outputLayer->neuron[k]->activation);
    }

    EMBANN_LOGI(TAG, "done output");
    return EOK;
}





static void _printInputLayer(inputLayer_t* pInputLayer, numInputs_t numInputNeurons)
{
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpointer-to-int-cast"
    // MISRA C 2012 11.4 - deliberate cast from pointer to integer
    // cppcheck-suppress misra-c2012-11.4
    EMBANN_LOGI(TAG, "pInputLayer: 0x%x, size: %ld", (uint32_t) pInputLayer, sizeof(inputLayer_t) + 
                                                (sizeof(uNeuron_t*) * numInputNeurons));
#pragma GCC diagnostic pop
}





static void _printHiddenLayer(hiddenLayer_t* pHiddenLayer, numHiddenNeurons_t numHiddenNeurons)
{
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpointer-to-int-cast"
    // MISRA C 2012 11.4 - deliberate cast from pointer to integer
    // cppcheck-suppress misra-c2012-11.4
    EMBANN_LOGI(TAG, "hiddenlayer: 0x%x, size: %ld", (uint32_t) pHiddenLayer, sizeof(hiddenLayer_t) + 
                                            (sizeof(wNeuron_t*) * numHiddenNeurons));
#pragma GCC diagnostic pop
}





static void _printHiddenNeuronParams(hiddenLayer_t* pHiddenLayer, numHiddenNeurons_t j, numHiddenNeurons_t k)
{
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpointer-to-int-cast"
    // MISRA C 2012 11.4 - deliberate cast from pointer to integer
    // cppcheck-suppress misra-c2012-11.4
    EMBANN_LOGV(TAG, "params array: 0x%x, bias 0x%x, weight 0x%x", 
                        (uint32_t) &pHiddenLayer->neuron[j]->params[k],
                        (uint32_t) &pHiddenLayer->neuron[j]->params[k]->bias,
                        (uint32_t) &pHiddenLayer->neuron[j]->params[k]->weight);
#pragma GCC diagnostic pop
}




static void _printConnectedHiddenLayer(numLayers_t layerNum)
{
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpointer-to-int-cast"
    // MISRA C 2012 11.4 - deliberate cast from pointer to integer
    // cppcheck-suppress misra-c2012-11.4
    EMBANN_LOGI(TAG, "hiddenlayer[%d]: 0x%x", layerNum, (uint32_t) &embann_getNetwork()->hiddenLayer[layerNum]);
#pragma GCC diagnostic pop
}




static void _printOutputLayer(outputLayer_t* pOutputLayer, numOutputs_t numOutputNeurons)
{
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpointer-to-int-cast"
    // MISRA C 2012 11.4 - deliberate cast from pointer to integer
    // cppcheck-suppress misra-c2012-11.4
    EMBANN_LOGI(TAG, "pOutputLayer: 0x%x, size: %ld", (uint32_t) pOutputLayer, sizeof(outputLayer_t) + 
                                                (sizeof(wNeuron_t*) * numOutputNeurons));
#pragma GCC diagnostic pop
}
