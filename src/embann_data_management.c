#include "embann.h"
#include "embann_log.h"

#define TAG "Embann Data Management"




void embann_inputRaw(float data[])
{
    for (uint32_t i = 0; i < network->inputLayer->numNeurons; i++)
    {
        network->inputLayer->neuron[i]->activation = data[i];
    }
}

void embann_inputMinMaxScale(uint8_t data[], uint8_t min, uint8_t max)
{
    for (uint32_t i = 0; i < network->inputLayer->numNeurons; i++)
    {
        network->inputLayer->neuron[i]->activation = (((float)data[i] - min)) / (max - min);
    }
}

void embann_inputStandardizeScale(uint8_t data[], float mean, float stdDev)
{
    for (uint32_t i = 0; i < network->inputLayer->numNeurons; i++)
    {
        network->inputLayer->neuron[i]->activation = ((float)data[i] - mean) / stdDev;
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

int embann_shuffleTrainingData(void)
{
    return EOK;
}