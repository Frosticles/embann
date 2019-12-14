#include "embann.h"
#include "embann_log.h"

#define TAG "Embann Data Management"

extern network_t* pNetworkGlobal;
extern trainingDataCollection_t trainingDataCollection;

static trainingData_t trainingData[CONFIG_NUM_TRAINING_DATA_SETS];


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
        pNetworkGlobal->inputLayer->activation[i] = (data[i] - min) / (max - min);
        EMBANN_LOGD(TAG, "Input [%d] = %" ACTIVATION_PRINT, i, pNetworkGlobal->inputLayer->activation[i]);
    }
    return EOK;
}

int embann_inputStandardizeScale(activation_t data[], float mean, float stdDev)
{
    for (uint32_t i = 0; i < pNetworkGlobal->inputLayer->numNeurons; i++)
    {
        pNetworkGlobal->inputLayer->activation[i] = (data[i] - mean) / stdDev;
        EMBANN_LOGD(TAG, "Input [%d] = %" ACTIVATION_PRINT, i, pNetworkGlobal->inputLayer->activation[i]);
    }
    return EOK;
}

int embann_getTrainingDataMean(float* mean)
{
    float tempMean;
    uint32_t sum = 0;
    trainingData_t* pTrainingData = trainingDataCollection.head;

    if (pTrainingData != NULL)
    {
        tempMean = pTrainingData->data[0];
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

#ifdef CONFIG_MEMORY_ALLOCATION_STATIC
    const numTrainingDataSets_t numSets = trainingDataCollection.numSets;
    for (numTrainingDataSets_t i = 0; i < numSets; i++)
#else
    while (pTrainingData != NULL)
#endif
    {
        const numTrainingDataEntries_t numEntries = pTrainingData->length;
        
        for (numTrainingDataEntries_t j = 0; j < numEntries; j++)
        {
            sum += pTrainingData->data[j];
        }
        tempMean += (float)sum / numEntries;
        sum = 0;

#ifdef CONFIG_MEMORY_ALLOCATION_STATIC
        pTrainingData += sizeof(trainingData_t);
#else
        pTrainingData = pTrainingData->next;
#endif
    }

    tempMean /= trainingDataCollection.numSets;

    *mean = tempMean;

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
    

#ifdef CONFIG_MEMORY_ALLOCATION_STATIC
    const numTrainingDataSets_t numSets = trainingDataCollection.numSets;
    for (numTrainingDataSets_t i = 0; i < numSets; i++)
#else
    while (pTrainingData != NULL)
#endif
    {
        const numTrainingDataEntries_t numEntries = pTrainingData->length;
        
        for (numTrainingDataEntries_t j = 0; j < numEntries; j++)
        {
            sumofSquares += powf((float)pTrainingData->data[j] - mean, 2.0F);
        }
        sumofSquares /= numEntries;

#ifdef CONFIG_MEMORY_ALLOCATION_STATIC
        pTrainingData += sizeof(trainingData_t);
#else
        pTrainingData = pTrainingData->next;
#endif
    }

    *stdDev = sqrtf(sumofSquares);

    return EOK;
}

int embann_getTrainingDataMax(activation_t* max)
{
    trainingData_t* pTrainingData = trainingDataCollection.head;
    activation_t tempMax;

    if (pTrainingData != NULL)
    {
        tempMax = pTrainingData->data[0];
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
    
#ifdef CONFIG_MEMORY_ALLOCATION_STATIC
    const numTrainingDataSets_t numSets = trainingDataCollection.numSets;
    for (numTrainingDataSets_t i = 0; i < numSets; i++)
#else
    while (pTrainingData != NULL)
#endif
    {
        const numTrainingDataEntries_t numEntries = pTrainingData->length;
        
        for (numTrainingDataEntries_t j = 0; j < numEntries; j++)
        {
            if (pTrainingData->data[j] > tempMax)
            {
                tempMax = pTrainingData->data[j];
            }
        }

#ifdef CONFIG_MEMORY_ALLOCATION_STATIC
        pTrainingData += sizeof(trainingData_t);
#else
        pTrainingData = pTrainingData->next;
#endif
    }

    *max = tempMax;

    return EOK;
}

int embann_getTrainingDataMin(activation_t* min)
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

    if (pTrainingData->length == 0)
    {
        // Deviation from MISRA C2012 15.5 for reasonably simple error return values
        // cppcheck-suppress misra-c2012-15.5
        return ENOENT;
    }

#ifdef CONFIG_MEMORY_ALLOCATION_STATIC
    const numTrainingDataSets_t numSets = trainingDataCollection.numSets;
    for (numTrainingDataSets_t i = 0; i < numSets; i++)
#else
    while (pTrainingData != NULL)
#endif
    {
        const numTrainingDataEntries_t numEntries = pTrainingData->length;
        
        for (numTrainingDataEntries_t j = 0; j < numEntries; j++)
        {
            if (pTrainingData->data[j] < *min)
            {
                *min = pTrainingData->data[j];
            }
        }

#ifdef CONFIG_MEMORY_ALLOCATION_STATIC
        pTrainingData += sizeof(trainingData_t);
#else
        pTrainingData = pTrainingData->next;
#endif
    }

    return EOK;
}


#ifdef CONFIG_MEMORY_ALLOCATION_DYNAMIC
int embann_addTrainingData(activation_t* data, uint32_t numElements, numOutputs_t correctResponse)
{
    trainingData_t* trainingDataNode = (trainingData_t*) malloc(sizeof(trainingData_t));
    EMBANN_MALLOC_CHECK(trainingDataNode);

    if (numElements == 0U)
    {
        // Deviation from MISRA C2012 15.5 for reasonably simple error return values
        // cppcheck-suppress misra-c2012-15.5
        return ENOENT;
    }

    trainingDataNode->next = NULL;
    trainingDataNode->data = data;
    trainingDataNode->length = numElements;
    trainingDataNode->correctResponse = correctResponse;

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
    ++trainingDataCollection.numSets;
    return EOK;
}
#endif

int embann_copyTrainingData(activation_t data[], uint32_t numElements, numOutputs_t correctResponse)
{
#ifdef CONFIG_MEMORY_ALLOCATION_STATIC
    trainingData_t* trainingDataNode = &trainingData[trainingDataCollection.numSets];
#else
    trainingData_t* trainingDataNode = (trainingData_t*) malloc(sizeof(trainingData_t));
    EMBANN_MALLOC_CHECK(trainingDataNode);
#endif

    if (numElements == 0U)
    {
        // Deviation from MISRA C2012 15.5 for reasonably simple error return values
        // cppcheck-suppress misra-c2012-15.5
        return ENOENT;
    }

#ifdef CONFIG_MEMORY_ALLOCATION_DYNAMIC
    trainingDataNode->data = (activation_t*) malloc(numElements * sizeof(activation_t));
    trainingDataNode->next = NULL;
#endif
    trainingDataNode->length = numElements;
    trainingDataNode->correctResponse = correctResponse;
    memcpy(trainingDataNode->data, data, numElements * sizeof(activation_t));

    if (trainingDataCollection.head == NULL)
    {
        trainingDataCollection.head = trainingDataNode;
#ifdef CONFIG_MEMORY_ALLOCATION_DYNAMIC
        trainingDataCollection.tail = trainingDataNode;
    }
    else
    {
        trainingDataCollection.tail->next = trainingDataNode;
        trainingDataCollection.tail = trainingDataNode;
#endif
    }
    trainingDataCollection.numSets++;
    return EOK;
}

int embann_shuffleTrainingData(void)
{
    return EOK;
}





int embann_getRandomDataSet(trainingData_t** dataSet)
{
    if (trainingDataCollection.numSets == 0)
    {
        return ENOENT;
    }
    else
    {
#ifdef CONFIG_MEMORY_ALLOCATION_STATIC
        *dataSet = &trainingData[random() % trainingDataCollection.numSets];
        return EOK;
#endif
    }
}