#include "embann.h"
#include "embann_log.h"
#ifdef CONFIG_MEMORY_ALLOCATION_STATIC
#include "embann_static.h"
#endif

#define TAG "Embann Init"

extern network_t* pNetworkGlobal;


static void _printInputLayer(inputLayer_t* pInputLayer);
static void _printHiddenLayer(hiddenLayer_t* pHiddenLayer);
static void _printHiddenNeuronParams(hiddenLayer_t* pHiddenLayer, numHiddenNeurons_t j, numHiddenNeurons_t k);
static void _printConnectedHiddenLayer(numLayers_t layerNum);
static void _printOutputLayer(outputLayer_t* pOutputLayer);

static int embann_initInputToHiddenLayer(numHiddenNeurons_t numHiddenNeurons, numInputs_t numInputNeurons);
static int embann_initInputLayer(numInputs_t numInputNeurons);
static int embann_initOutputLayer(numOutputs_t numOutputNeurons, numHiddenNeurons_t numHiddenNeurons);
#if (defined(CONFIG_MEMORY_ALLOCATION_STATIC) && (CONFIG_NUM_HIDDEN_LAYERS > 1)) || defined(CONFIG_MEMORY_ALLOCATION_DYNAMIC)
static int embann_initHiddenToHiddenLayer(numHiddenNeurons_t numHiddenNeurons, numLayers_t numHiddenLayers);
#endif
static int embann_initHiddenLayer(numHiddenNeurons_t numHiddenNeurons,
#if (defined(CONFIG_MEMORY_ALLOCATION_STATIC) && (CONFIG_NUM_HIDDEN_LAYERS > 1)) || defined(CONFIG_MEMORY_ALLOCATION_DYNAMIC)
                                    numLayers_t numHiddenLayers,
#endif
                                    numInputs_t numInputNeurons);








int embann_init(numInputs_t numInputNeurons,
                numHiddenNeurons_t numHiddenNeurons, 
                numLayers_t numHiddenLayers,
                numOutputs_t numOutputNeurons)
{
    if ((numInputNeurons == 0U) || (numHiddenNeurons == 0U) || 
        (numHiddenLayers == 0U) || (numOutputNeurons == 0U))
    {
        // Deviation from MISRA C2012 15.5 for reasonably simple error return values
        // cppcheck-suppress misra-c2012-15.5
        return EINVAL;
    }

#ifdef CONFIG_MEMORY_ALLOCATION_STATIC
#if (CONFIG_NUM_INPUT_NEURONS == 0) || (CONFIG_NUM_HIDDEN_NEURONS == 0) || (CONFIG_NUM_OUTPUT_NEURONS == 0) || (CONFIG_NUM_HIDDEN_LAYERS == 0)
#error "Layer dimensions cannot be equal to 0"
#endif

    pNetworkGlobal = &staticNetwork;
#else
    network_t* pNetwork = (network_t*) malloc(sizeof(network_t) + 
                                 (sizeof(hiddenLayer_t) * numHiddenLayers));
    EMBANN_MALLOC_CHECK(pNetwork);
    pNetworkGlobal = pNetwork;
#endif

    EMBANN_ERROR_CHECK(embann_initInputLayer(numInputNeurons));
    EMBANN_ERROR_CHECK(embann_initHiddenLayer(numHiddenNeurons,
#if (defined(CONFIG_MEMORY_ALLOCATION_STATIC) && (CONFIG_NUM_HIDDEN_LAYERS > 1)) || defined(CONFIG_MEMORY_ALLOCATION_DYNAMIC)
                                                numHiddenLayers,
#endif
                                                numInputNeurons));
    EMBANN_ERROR_CHECK(embann_initOutputLayer(numOutputNeurons,
                                                numHiddenNeurons));

    pNetworkGlobal->properties.numLayers = numHiddenLayers + 2U;
    pNetworkGlobal->properties.numHiddenLayers = numHiddenLayers;
    pNetworkGlobal->properties.networkResponse = 0U;

    return EOK;
}





static int embann_initInputLayer(numInputs_t numInputNeurons)
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

    _printInputLayer(pInputLayer);

    for (numInputs_t i = 0; i < numInputNeurons; i++)
    {
#ifdef CONFIG_MEMORY_ALLOCATION_DYNAMIC
        pNeuron = (uNeuron_t*) malloc(sizeof(uNeuron_t));
        EMBANN_MALLOC_CHECK(pNeuron);
        pInputLayer->neuron[i] = pNeuron;
#endif
        pInputLayer->activation[i] = RAND_ACTIVATION();
        EMBANN_LOGD(TAG, "act [%d] = %" ACTIVATION_PRINT, i, pInputLayer->activation[i]);
    }

#ifdef CONFIG_MEMORY_ALLOCATION_DYNAMIC
    pNetworkGlobal->inputLayer = pInputLayer;
#endif

    for (numInputs_t k = 0; k < numInputNeurons; k++)
    {
        EMBANN_LOGI(TAG, "act [%d] = %" ACTIVATION_PRINT, k, pNetworkGlobal->inputLayer->activation[k]);
    }

    EMBANN_LOGI(TAG, "done input");
    return EOK;
}






static int embann_initHiddenLayer(numHiddenNeurons_t numHiddenNeurons,
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






static int embann_initInputToHiddenLayer(numHiddenNeurons_t numHiddenNeurons, numInputs_t numInputNeurons)
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
    _printHiddenLayer(pHiddenLayer);


    for (numHiddenNeurons_t j = 0; j < numHiddenNeurons; j++)
    {
#ifdef CONFIG_MEMORY_ALLOCATION_DYNAMIC
        pNeuron = (wNeuron_t*) malloc(sizeof(wNeuron_t) + (sizeof(neuronParams_t*) * numInputNeurons));
        EMBANN_MALLOC_CHECK(pNeuron);
        pHiddenLayer->neuron[j] = pNeuron;
#endif
        pHiddenLayer->activation[j] = RAND_ACTIVATION();

        EMBANN_LOGD(TAG, "act [%d] = %" ACTIVATION_PRINT, j, pHiddenLayer->activation[j]);

        for (numInputs_t k = 0; k < numInputNeurons; k++)
        {
#ifdef CONFIG_MEMORY_ALLOCATION_DYNAMIC
            pHiddenLayerParams = (neuronParams_t*) malloc(sizeof(neuronParams_t));
            EMBANN_MALLOC_CHECK(pHiddenLayerParams);
            pHiddenLayer->neuron[j]->params[k] = pHiddenLayerParams;
#endif
            pHiddenLayer->bias[j] = RAND_BIAS();
            pHiddenLayer->weight[j][k] = RAND_WEIGHT();

            _printHiddenNeuronParams(pHiddenLayer, j, k);
        }
    }

#ifdef CONFIG_MEMORY_ALLOCATION_DYNAMIC
    pNetworkGlobal->hiddenLayer[0] = pHiddenLayer;
#endif
    for (uint16_t k = 0; k < numHiddenNeurons; k++)
    {
        EMBANN_LOGI(TAG, "act [%d] = %" ACTIVATION_PRINT, k, pNetworkGlobal->hiddenLayer[0]->activation[k]);
    }

    _printConnectedHiddenLayer(0);
    EMBANN_LOGI(TAG, "done hidden");
    return EOK;
}







#if (defined(CONFIG_MEMORY_ALLOCATION_STATIC) && (CONFIG_NUM_HIDDEN_LAYERS > 1)) || defined(CONFIG_MEMORY_ALLOCATION_DYNAMIC)
static int embann_initHiddenToHiddenLayer(numHiddenNeurons_t numHiddenNeurons, numLayers_t numHiddenLayers)
{
    hiddenLayer_t* pHiddenLayer;
#ifdef CONFIG_MEMORY_ALLOCATION_DYNAMIC
    wNeuron_t* pNeuron;
    neuronParams_t* pHiddenLayerParams;
#endif
    
    for (numLayers_t i = 1; i < numHiddenLayers; i++)
    {
#ifdef CONFIG_MEMORY_ALLOCATION_STATIC
        pHiddenLayer = staticNetwork.hiddenLayer[i];
#else
        pHiddenLayer = (hiddenLayer_t*) malloc(sizeof(hiddenLayer_t) + 
                                                (sizeof(wNeuron_t*) * numHiddenNeurons));
        EMBANN_MALLOC_CHECK(pHiddenLayer);
#endif
        _printHiddenLayer(pHiddenLayer);
        pHiddenLayer->numNeurons = numHiddenNeurons;

        for (numHiddenNeurons_t j = 0; j < numHiddenNeurons; j++)
        {    
#ifdef CONFIG_MEMORY_ALLOCATION_DYNAMIC
            pNeuron = (wNeuron_t*) malloc(sizeof(wNeuron_t) + (sizeof(neuronParams_t*) * numHiddenNeurons));
            EMBANN_MALLOC_CHECK(pNeuron);
            pHiddenLayer->neuron[j] = pNeuron;
#endif
            pHiddenLayer->activation[j] = RAND_ACTIVATION();

            EMBANN_LOGD(TAG, "act [%d] = %" ACTIVATION_PRINT, j, pHiddenLayer->activation[j]);

            for (numHiddenNeurons_t k = 0; k < numHiddenNeurons; k++)
            {
#ifdef CONFIG_MEMORY_ALLOCATION_DYNAMIC
                pHiddenLayerParams = (neuronParams_t*) malloc(sizeof(neuronParams_t));
                EMBANN_MALLOC_CHECK(pHiddenLayerParams);
                pHiddenLayer->neuron[j]->params[k] = pHiddenLayerParams;
#endif
                _printHiddenNeuronParams(pHiddenLayer, j, k);

                pHiddenLayer->bias[j] = RAND_BIAS();
                pHiddenLayer->weight[j][k] = RAND_WEIGHT();
                EMBANN_LOGD(TAG, "act [%d] = %" ACTIVATION_PRINT, j, pHiddenLayer->activation[j]);
            }
        }

        EMBANN_LOGI(TAG, "done hidden");
#ifdef CONFIG_MEMORY_ALLOCATION_DYNAMIC
        pNetworkGlobal->hiddenLayer[i] = pHiddenLayer;
#endif
        _printConnectedHiddenLayer(i);

        for (uint16_t k = 0; k < (numHiddenNeurons - 1U); k++)
        {
            EMBANN_LOGI(TAG, "act [%d] = %" ACTIVATION_PRINT, k, pNetworkGlobal->hiddenLayer[i]->activation[k]);
        }
    }
    return EOK;
}
#endif







static int embann_initOutputLayer(numOutputs_t numOutputNeurons, numHiddenNeurons_t numHiddenNeurons)
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
#endif

    pOutputLayer->numNeurons = numOutputNeurons;

    _printOutputLayer(pOutputLayer);

    for (numOutputs_t i = 0; i < numOutputNeurons; i++)
    {
#ifdef CONFIG_MEMORY_ALLOCATION_DYNAMIC
        pNeuron = (wNeuron_t*) malloc(sizeof(wNeuron_t) + (sizeof(neuronParams_t*) * numHiddenNeurons));
        EMBANN_MALLOC_CHECK(pNeuron);
        pOutputLayer->neuron[i] = pNeuron;
#endif

        pOutputLayer->activation[i] = RAND_ACTIVATION();
        EMBANN_LOGD(TAG, "act [%d] = %" ACTIVATION_PRINT, i, pOutputLayer->activation[i]);
        
        for (numHiddenNeurons_t j = 0; j < numHiddenNeurons; j++)
        {
#ifdef CONFIG_MEMORY_ALLOCATION_DYNAMIC
            pOutputLayerParams = (neuronParams_t*) malloc(sizeof(neuronParams_t));
            EMBANN_MALLOC_CHECK(pOutputLayerParams);
            pOutputLayer->neuron[i]->params[j] = pOutputLayerParams;
#endif
            pOutputLayer->bias[i] = RAND_BIAS();
            pOutputLayer->weight[i][j] = RAND_WEIGHT();
        }
    }
    
#ifdef CONFIG_MEMORY_ALLOCATION_DYNAMIC
    pNetworkGlobal->outputLayer = pOutputLayer;
#endif

    for (uint16_t k = 0; k < numOutputNeurons; k++)
    {
        EMBANN_LOGI(TAG, "act [%d] = %" ACTIVATION_PRINT, k, pNetworkGlobal->outputLayer->activation[k]);
    }

    EMBANN_LOGI(TAG, "done output");
    return EOK;
}





static void _printInputLayer(inputLayer_t* pInputLayer)
{
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpointer-to-int-cast"
    // MISRA C 2012 11.4 - deliberate cast from pointer to integer
    // cppcheck-suppress misra-c2012-11.4
    EMBANN_LOGI(TAG, "pInputLayer: 0x%x", (uint32_t) pInputLayer);
#pragma GCC diagnostic pop
}





static void _printHiddenLayer(hiddenLayer_t* pHiddenLayer)
{
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpointer-to-int-cast"
    // MISRA C 2012 11.4 - deliberate cast from pointer to integer
    // cppcheck-suppress misra-c2012-11.4
    EMBANN_LOGI(TAG, "hiddenlayer: 0x%x", (uint32_t) pHiddenLayer);
#pragma GCC diagnostic pop
}





static void _printHiddenNeuronParams(hiddenLayer_t* pHiddenLayer, numHiddenNeurons_t j, numHiddenNeurons_t k)
{
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpointer-to-int-cast"
    // MISRA C 2012 11.4 - deliberate cast from pointer to integer
    // cppcheck-suppress misra-c2012-11.4
    EMBANN_LOGV(TAG, "params bias 0x%x, weight 0x%x", 
                        (uint32_t) &pHiddenLayer->bias[j],
                        (uint32_t) &pHiddenLayer->weight[j][k]);
#pragma GCC diagnostic pop
}




static void _printConnectedHiddenLayer(numLayers_t layerNum)
{
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpointer-to-int-cast"
    // MISRA C 2012 11.4 - deliberate cast from pointer to integer
    // cppcheck-suppress misra-c2012-11.4
    EMBANN_LOGI(TAG, "hiddenlayer[%d]: 0x%x", layerNum, (uint32_t) &pNetworkGlobal->hiddenLayer[layerNum]);
#pragma GCC diagnostic pop
}




static void _printOutputLayer(outputLayer_t* pOutputLayer)
{
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpointer-to-int-cast"
    // MISRA C 2012 11.4 - deliberate cast from pointer to integer
    // cppcheck-suppress misra-c2012-11.4
    EMBANN_LOGI(TAG, "pOutputLayer: 0x%x", (uint32_t) pOutputLayer);
#pragma GCC diagnostic pop
}
