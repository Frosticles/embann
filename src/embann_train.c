#include "embann.h"
#include "embann_log.h"

#define TAG "Embann Train"


int embann_trainDriverInTime(float learningRate, uint32_t numSeconds, bool verbose)
{
    uint16_t randomOutput, randomTrainingSet;

    if (verbose == true)
    {
        EMBANN_LOGI(TAG, "\nOutput Errors: ");
    }

    uint32_t startTime = millis();

    while ((millis() - startTime) < (numSeconds * 1000UL))
    {
        randomOutput = random() % network->outputLayer->numNeurons;
        randomTrainingSet = random() % trainingDataCollection.numEntries;

        /*
            TODO, these are not 'right' but they will let the program run
        */
        embann_inputMinMaxScale(trainingDataCollection.head->data, 0U, UINT8_MAX);
        EMBANN_ERROR_CHECK(embann_inputLayer(NULL));

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
    uint16_t randomOutput, randomTrainingSet;
    float currentCost[network->outputLayer->numNeurons];
    bool converged = false;

    if (verbose == true)
    {
        EMBANN_LOGI(TAG, "\nOutput Errors: ");
    }

    while (!converged)
    {
        randomOutput = random() % network->outputLayer->numNeurons;
        randomTrainingSet = random() % trainingDataCollection.numEntries;
        currentCost[randomOutput] = 0.0;
        embann_inputMinMaxScale(trainingDataCollection.head->data, 0U, UINT8_MAX);

        EMBANN_ERROR_CHECK(embann_inputLayer(NULL));

        if (verbose == true)
        {
            EMBANN_ERROR_CHECK(embann_errorReporting(randomOutput));
            EMBANN_LOGI(TAG, "%u | %u ", randomOutput, randomTrainingSet);
        }

        EMBANN_ERROR_CHECK(embann_train(randomOutput, learningRate));
        for (uint8_t i = 0; i < network->outputLayer->numNeurons; i++)
        {
            if (i == randomOutput)
            {
                currentCost[randomOutput] += pow(1 - network->outputLayer->neuron[i]->activation, 2.0F);
            }
            else
            {
                currentCost[randomOutput] += pow(network->outputLayer->neuron[i]->activation, 2.0F);
            }
        }
        currentCost[randomOutput] /= network->outputLayer->numNeurons;

        for (uint8_t i = 0; i < network->outputLayer->numNeurons; i++)
        {
            EMBANN_LOGI(TAG, "%f", currentCost[i]);
            EMBANN_LOGI(TAG, ", ");
            if (currentCost[i] > desiredCost)
            {
                break;
            }
            if (i == (network->outputLayer->numNeurons - 1U))
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
    float dOutputErrorToOutputSum[network->outputLayer->numNeurons];
    float dTotalErrorToHiddenNeuron = 0.0F;
    /* TODO, add support for multiple hidden layers */
    float outputNeuronWeightChange[network->outputLayer->numNeurons]
                                  [network->hiddenLayer[0].numNeurons];
    float tanhDerivative;

    for (uint16_t i = 0; i < network->outputLayer->numNeurons; i++)
    {
        if (i == correctOutput)
        {
            EMBANN_ERROR_CHECK(embann_tanhDerivative(network->outputLayer->neuron[i]->activation, &tanhDerivative));
            dOutputErrorToOutputSum[i] =
                (1 - network->outputLayer->neuron[i]->activation) * tanhDerivative;
        }
        else
        {
            EMBANN_ERROR_CHECK(embann_tanhDerivative(network->outputLayer->neuron[i]->activation, &tanhDerivative));
            dOutputErrorToOutputSum[i] =
                -network->outputLayer->neuron[i]->activation * tanhDerivative;
        }
        // EMBANN_LOGI(TAG, "\ndOutputErrorToOutputSum[%d]: %.3f", i,
        // dOutputErrorToOutputSum[i]);
        for (uint16_t j = 0; j < network->hiddenLayer[0].numNeurons; j++)
        {
            outputNeuronWeightChange[i][j] =
                dOutputErrorToOutputSum[i] *
                network->hiddenLayer[network->properties.numHiddenLayers - 1U].neuron[j]->activation *
                learningRate;
            // EMBANN_LOGI(TAG, "\n  outputNeuronWeightChange[%d][%d]: %.3f", i, j,
            //              outputNeuronWeightChange[i][j]);
        }
    }

    for (uint16_t i = 0; i < network->hiddenLayer[0].numNeurons; i++)
    {
        dTotalErrorToHiddenNeuron = 0.0F;
        for (uint16_t j = 0; j < network->outputLayer->numNeurons; j++)
        {
            dTotalErrorToHiddenNeuron +=
                dOutputErrorToOutputSum[j] * network->outputLayer->neuron[j]->params[i]->weight;
            // EMBANN_LOGI(TAG, "\nOld Output Weight[%d][%d]: %.3f", i, j,
            // network->outputLayer->neuron[j]->params[i]->weight);
            network->outputLayer->neuron[j]->params[i]->weight += outputNeuronWeightChange[j][i];
            // EMBANN_LOGI(TAG, "\nNew Output Weight[%d][%d]: %.3f", i, j,
            // network->outputLayer->neuron[j]->params[i]->weight);
        }
        for (uint16_t k = 0; k < network->inputLayer->numNeurons; k++)
        {
            // EMBANN_LOGI(TAG, "\nOld Hidden Weight[%d][%d]: %.3f", i, k,
            // network->network->hiddenLayer[0].neuron[i]->params[k]->weight);
            EMBANN_ERROR_CHECK(embann_tanhDerivative(network->hiddenLayer[0].neuron[i]->activation, &tanhDerivative));
            network->hiddenLayer[0].neuron[i]->params[k]->weight +=
                dTotalErrorToHiddenNeuron * tanhDerivative *
                network->inputLayer->neuron[k]->activation * learningRate;
            // EMBANN_LOGI(TAG, "\nNew Hidden Weight[%d][%d]: %.3f", i, k,
            // network->network->hiddenLayer[0].neuron[i]->params[k]->weight);
        }
    }
    return EOK;
}

int embann_tanhDerivative(float inputValue, float* outputValue)
{
    *outputValue = 1.0F - powf(tanh(inputValue * PI), 2.0F);
    return EOK;
}