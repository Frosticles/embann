#include "embann.h"
#include "embann_log.h"

#define TAG "Embann Data Management"

extern network_t* pNetworkGlobal;




int embann_inputRaw(activation_t data[])
{
    for (uint32_t i = 0; i < pNetworkGlobal->inputLayer->numNeurons; i++)
    {
        pNetworkGlobal->inputLayer->activation[i] = data[i];
        EMBANN_LOGD(TAG, "Input [%d] = %" ACTIVATION_PRINT, i, pNetworkGlobal->inputLayer->activation[i]);
    }
    return EOK;
}

int embann_inputMinMaxScale(activation_t data[], activation_t min, activation_t max)
{
    for (uint32_t i = 0; i < pNetworkGlobal->inputLayer->numNeurons; i++)
    {
        // TODO configurable types
        pNetworkGlobal->inputLayer->activation[i] = ((float)data[i] - min) / (max - min);
        EMBANN_LOGD(TAG, "Input [%d] = %" ACTIVATION_PRINT, i, pNetworkGlobal->inputLayer->activation[i]);
    }
    return EOK;
}

int embann_inputStandardizeScale(activation_t data[], float mean, float stdDev)
{
    for (uint32_t i = 0; i < pNetworkGlobal->inputLayer->numNeurons; i++)
    {
        // TODO configurable types
        pNetworkGlobal->inputLayer->activation[i] = ((float)data[i] - mean) / stdDev;
        EMBANN_LOGD(TAG, "Input [%d] = %" ACTIVATION_PRINT, i, pNetworkGlobal->inputLayer->activation[i]);
    }
    return EOK;
}

int embann_getTrainingDataMean(float* mean)
{
    uint32_t sum = 0;
    trainingData_t* pTrainingData = embann_getDataCollection()->head;

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

    if (pTrainingData->length == 0)
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

    *mean /= embann_getDataCollection()->numEntries;

    return EOK;
}

int embann_getTrainingDataStdDev(float* stdDev)
{
    float sumofSquares = 0.0F;
    float mean;
    trainingData_t* pTrainingData = embann_getDataCollection()->head;

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

    if (pTrainingData->length == 0)
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

int embann_getTrainingDataMax(activation_t* max)
{
    trainingData_t* pTrainingData = embann_getDataCollection()->head;
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

    if (pTrainingData->length == 0)
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

int embann_getTrainingDataMin(activation_t* min)
{
    trainingData_t* pTrainingData = embann_getDataCollection()->head;
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

    if (pTrainingData->length == 0)
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


int embann_addTrainingData(activation_t* data, uint32_t length, numOutputs_t correctResponse)
{
    trainingData_t* trainingDataNode;

    if (length == 0U)
    {
        // Deviation from MISRA C2012 15.5 for reasonably simple error return values
        // cppcheck-suppress misra-c2012-15.5
        return ENOENT;
    }

    trainingDataNode = (trainingData_t*) malloc(sizeof(trainingData_t));
    EMBANN_MALLOC_CHECK(trainingDataNode);

    trainingDataNode->prev = embann_getDataCollection()->tail;
    trainingDataNode->next = NULL;
    trainingDataNode->length = length;
    trainingDataNode->correctResponse = correctResponse;
    trainingDataNode->data = data;

    if (embann_getDataCollection()->head == NULL)
    {
        embann_getDataCollection()->head = trainingDataNode;
        embann_getDataCollection()->tail = trainingDataNode;
    }
    else
    {
        embann_getDataCollection()->tail->next = trainingDataNode;
        embann_getDataCollection()->tail = trainingDataNode;
    }

    ++embann_getDataCollection()->numEntries;
    return EOK;
}

int embann_copyTrainingData(activation_t data[], uint32_t length, numOutputs_t correctResponse)
{
    trainingData_t* trainingDataNode;

    if (length == 0U)
    {
        // Deviation from MISRA C2012 15.5 for reasonably simple error return values
        // cppcheck-suppress misra-c2012-15.5
        return ENOENT;
    }

    trainingDataNode = (trainingData_t*) malloc(sizeof(trainingData_t));
    EMBANN_MALLOC_CHECK(trainingDataNode);

    trainingDataNode->prev = embann_getDataCollection()->tail;
    trainingDataNode->next = NULL;
    trainingDataNode->length = length;
    trainingDataNode->correctResponse = correctResponse;
    trainingDataNode->data = (activation_t*) malloc(length);
    memcpy(trainingDataNode->data, data, length);

    if (embann_getDataCollection()->head == NULL)
    {
        embann_getDataCollection()->head = trainingDataNode;
        embann_getDataCollection()->tail = trainingDataNode;
    }
    else
    {
        embann_getDataCollection()->tail->next = trainingDataNode;
        embann_getDataCollection()->tail = trainingDataNode;
    }

    embann_getDataCollection()->numEntries++;
    return EOK;
}

int embann_shuffleTrainingData(void)
{
    return EOK;
}