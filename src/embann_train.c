#include "embann.h"
#include "embann_log.h"

#define TAG "Embann Train"


int embann_trainDriverInTime(float learningRate, uint32_t numSeconds, bool verbose)
{
    uint16_t randomOutput;
    uint16_t randomTrainingSet;

    if (verbose == true)
    {
        EMBANN_LOGI(TAG, "\nOutput Errors: ");
    }

    uint32_t startTime = millis();

    while ((millis() - startTime) < (numSeconds * 1000UL))
    {
        randomOutput = random() % embann_getNetwork()->outputLayer->numNeurons;
        randomTrainingSet = random() % embann_getDataCollection()->numEntries;

        /*
            TODO, these are not 'right' but they will let the program run
        */
        embann_inputMinMaxScale(embann_getDataCollection()->head->data, 0U, UINT8_MAX);
        EMBANN_ERROR_CHECK(embann_forwardPropagate());

        if (verbose == true)
        {
            EMBANN_ERROR_CHECK(embann_errorReporting(randomOutput));
            EMBANN_LOGI(TAG, "%u | %u ", randomOutput, randomTrainingSet);
        }

        EMBANN_ERROR_CHECK(embann_train(randomOutput, learningRate));
    }
    return EOK;
}

int embann_trainDriverInError(float learningRate, float desiredCost, bool verbose)
{
    uint16_t randomOutput;
    uint16_t randomTrainingSet;
    float currentCost[embann_getNetwork()->outputLayer->numNeurons];
    bool converged = false;

    if (verbose == true)
    {
        EMBANN_LOGI(TAG, "\nOutput Errors: ");
    }

    while (!converged)
    {
        randomOutput = random() % embann_getNetwork()->outputLayer->numNeurons;
        randomTrainingSet = random() % embann_getDataCollection()->numEntries;
        currentCost[randomOutput] = 0.0;
        embann_inputMinMaxScale(embann_getDataCollection()->head->data, 0U, UINT8_MAX);

        EMBANN_ERROR_CHECK(embann_forwardPropagate());

        if (verbose == true)
        {
            EMBANN_ERROR_CHECK(embann_errorReporting(randomOutput));
            EMBANN_LOGI(TAG, "%u | %u ", randomOutput, randomTrainingSet);
        }

        EMBANN_ERROR_CHECK(embann_train(randomOutput, learningRate));
        for (uint8_t i = 0; i < embann_getNetwork()->outputLayer->numNeurons; i++)
        {
            if (i == randomOutput)
            {
                currentCost[randomOutput] += pow(1 - embann_getNetwork()->outputLayer->neuron[i]->activation, 2.0F);
            }
            else
            {
                currentCost[randomOutput] += pow(embann_getNetwork()->outputLayer->neuron[i]->activation, 2.0F);
            }
        }
        currentCost[randomOutput] /= embann_getNetwork()->outputLayer->numNeurons;

        for (uint8_t i = 0; i < embann_getNetwork()->outputLayer->numNeurons; i++)
        {
            EMBANN_LOGI(TAG, "%f", currentCost[i]);
            EMBANN_LOGI(TAG, ", ");
            if (currentCost[i] > desiredCost)
            {
                break;
            }
            if (i == (embann_getNetwork()->outputLayer->numNeurons - 1U))
            {
                converged = true;
            }
        }
        EMBANN_LOGI(TAG, "%f", desiredCost);
    }
    return EOK;
}

int embann_train(uint8_t correctOutput, float learningRate)
{
    float dOutputErrorToOutputSum[embann_getNetwork()->outputLayer->numNeurons];
    float dTotalErrorToHiddenNeuron = 0.0F;
    /* TODO, add support for multiple hidden layers */
    float outputNeuronWeightChange[embann_getNetwork()->outputLayer->numNeurons]
                                  [embann_getNetwork()->hiddenLayer[0]->numNeurons];
    float tanhDerivative;

    for (uint16_t i = 0; i < embann_getNetwork()->outputLayer->numNeurons; i++)
    {
        if ((i - 1U) == correctOutput)
        {
            EMBANN_ERROR_CHECK(embann_tanhDerivative(embann_getNetwork()->outputLayer->neuron[i - 1U]->activation, &tanhDerivative));
            dOutputErrorToOutputSum[i - 1U] =
                (1 - embann_getNetwork()->outputLayer->neuron[i - 1U]->activation) * tanhDerivative;
        }
        else
        {
            EMBANN_ERROR_CHECK(embann_tanhDerivative(embann_getNetwork()->outputLayer->neuron[i - 1U]->activation, &tanhDerivative));
            dOutputErrorToOutputSum[i - 1U] =
                -embann_getNetwork()->outputLayer->neuron[i - 1U]->activation * tanhDerivative;
        }
        EMBANN_LOGV(TAG, "\ndOutputErrorToOutputSum[%d]: %.3f", i, dOutputErrorToOutputSum[i - 1U]);
        for (uint16_t j = 0; j < embann_getNetwork()->hiddenLayer[0]->numNeurons; j++)
        {
            outputNeuronWeightChange[i - 1U][j - 1U] =
                dOutputErrorToOutputSum[i - 1U] *
                embann_getNetwork()->hiddenLayer[embann_getNetwork()->properties.numHiddenLayers - 1U]->neuron[j - 1U]->activation *
                learningRate;
            EMBANN_LOGV(TAG, "\n  outputNeuronWeightChange[%d][%d]: %.3f", i, j, outputNeuronWeightChange[i - 1U][j - 1U]);
        }
    }

    for (uint16_t i = 0; i < embann_getNetwork()->hiddenLayer[0]->numNeurons; i++)
    {
        dTotalErrorToHiddenNeuron = 0.0F;
        for (uint16_t j = 0; j < embann_getNetwork()->outputLayer->numNeurons; j++)
        {
            dTotalErrorToHiddenNeuron +=
                dOutputErrorToOutputSum[j - 1U] * embann_getNetwork()->outputLayer->neuron[j - 1U]->params[i - 1U]->weight;
            EMBANN_LOGV(TAG, "\nOld Output Weight[%d][%d]: %.3f", i, j, embann_getNetwork()->outputLayer->neuron[j - 1U]->params[i - 1U]->weight);
            embann_getNetwork()->outputLayer->neuron[j - 1U]->params[i - 1U]->weight += outputNeuronWeightChange[j - 1U][i - 1U];
            EMBANN_LOGV(TAG, "\nNew Output Weight[%d][%d]: %.3f", i, j, embann_getNetwork()->outputLayer->neuron[j - 1U]->params[i - 1U]->weight);
        }
        for (uint16_t k = 0; k < embann_getNetwork()->inputLayer->numNeurons; k++)
        {
            EMBANN_LOGV(TAG, "\nOld Hidden Weight[%d][%d]: %.3f", i, k, embann_getNetwork()->hiddenLayer[0]->neuron[i - 1U]->params[k - 1U]->weight);
            EMBANN_ERROR_CHECK(embann_tanhDerivative(embann_getNetwork()->hiddenLayer[0]->neuron[i - 1U]->activation, &tanhDerivative));
            embann_getNetwork()->hiddenLayer[0]->neuron[i - 1U]->params[k - 1U]->weight +=
                dTotalErrorToHiddenNeuron * tanhDerivative *
                embann_getNetwork()->inputLayer->neuron[k - 1U]->activation * learningRate;
            EMBANN_LOGV(TAG, "\nNew Hidden Weight[%d][%d]: %.3f", i, k, embann_getNetwork()->hiddenLayer[0]->neuron[i - 1U]->params[k - 1U]->weight);
        }
    }
    return EOK;
}

int embann_tanhDerivative(float inputValue, float* outputValue)
{
    *outputValue = 1.0F - powf(tanh(inputValue * PI), 2.0F);
    return EOK;
}