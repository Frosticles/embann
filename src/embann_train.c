#include "embann.h"
#include "embann_log.h"

#define TAG "Embann Train"


int embann_trainDriverInTime(activation_t learningRate, uint32_t numSeconds, bool verbose)
{
    numOutputs_t randomOutput;
    numTrainingDataEntries_t randomTrainingSet;

    if (verbose == true)
    {
        printf("\nOutput Errors: ");
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
            printf("%u | %u ", randomOutput, randomTrainingSet);
            EMBANN_ERROR_CHECK(embann_errorReporting(randomOutput));
        }

        EMBANN_ERROR_CHECK(embann_train(randomOutput, learningRate));
    }
    return EOK;
}

int embann_trainDriverInError(activation_t learningRate, activation_t desiredCost, bool verbose)
{
    numOutputs_t randomOutput;
    numTrainingDataEntries_t randomTrainingSet;
    activation_t currentCost[embann_getNetwork()->outputLayer->numNeurons];
    bool converged = false;

    if (verbose == true)
    {
        printf("\nOutput Errors: ");
    }

    while (!converged)
    {
        randomOutput = random() % embann_getNetwork()->outputLayer->numNeurons;
        randomTrainingSet = random() % embann_getDataCollection()->numEntries;
        currentCost[randomOutput] = 0.0;

        /*
            TODO, these are not 'right' but they will let the program run
        */
        embann_inputMinMaxScale(embann_getDataCollection()->head->data, 0U, UINT8_MAX);
        EMBANN_ERROR_CHECK(embann_forwardPropagate());

        if (verbose == true)
        {
            printf("%u | %u | ", randomOutput, randomTrainingSet);
            EMBANN_ERROR_CHECK(embann_errorReporting(randomOutput));
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

        converged = true;
        for (uint8_t i = 0; i < embann_getNetwork()->outputLayer->numNeurons; i++)
        {
            if (verbose == true)
            {
                printf("%" ACTIVATION_PRINT ", ", currentCost[i]);
            }
            if (currentCost[i] > desiredCost)
            {
                converged = false;
            }
        }
        if (verbose == true)
        {
            printf("%" ACTIVATION_PRINT "\n ", desiredCost);
        }
    }
    return EOK;
}

int embann_train(numOutputs_t correctOutput, activation_t learningRate)
{
    activation_t dOutputErrorToOutputSum[embann_getNetwork()->outputLayer->numNeurons];
    weight_t dTotalErrorToHiddenNeuron = 0.0F;
    /* TODO, add support for multiple hidden layers */
    weight_t outputNeuronWeightChange[embann_getNetwork()->outputLayer->numNeurons]
                                  [embann_getNetwork()->hiddenLayer[0]->numNeurons];
    weight_t tanhDerivative = 0;

    for (uint16_t i = 0; i < embann_getNetwork()->outputLayer->numNeurons; i++)
    {
        if ((i) == correctOutput)
        {
            EMBANN_ERROR_CHECK(embann_tanhDerivative(embann_getNetwork()->outputLayer->neuron[i]->activation, 
                                                        &tanhDerivative));

            dOutputErrorToOutputSum[i] = (1 - embann_getNetwork()->outputLayer->neuron[i]->activation) * 
                                            tanhDerivative;
        }
        else
        {
            EMBANN_ERROR_CHECK(embann_tanhDerivative(embann_getNetwork()->outputLayer->neuron[i]->activation, 
                                                        &tanhDerivative));

            dOutputErrorToOutputSum[i] = -embann_getNetwork()->outputLayer->neuron[i]->activation * 
                                            tanhDerivative;
        }
        
        EMBANN_LOGV(TAG, "dOutputErrorToOutputSum[%d]: %" ACTIVATION_PRINT, i, dOutputErrorToOutputSum[i]);
        
        for (uint16_t j = 0; j < embann_getNetwork()->hiddenLayer[lastHiddenLayer - 1U]->numNeurons; j++)
        {
            outputNeuronWeightChange[i][j] = dOutputErrorToOutputSum[i] *
                                                embann_getNetwork()->hiddenLayer[lastHiddenLayer - 1U]->neuron[j]->activation *
                                                learningRate;

            EMBANN_LOGV(TAG, "outputNeuronWeightChange[%d][%d]: %" WEIGHT_PRINT, i, j, outputNeuronWeightChange[i][j]);
        }
    }

    for (uint16_t i = 0; i < embann_getNetwork()->hiddenLayer[0]->numNeurons; i++)
    {
        dTotalErrorToHiddenNeuron = 0.0F;
        for (uint16_t j = 0; j < embann_getNetwork()->outputLayer->numNeurons; j++)
        {
            dTotalErrorToHiddenNeuron += dOutputErrorToOutputSum[j] * 
                                            embann_getNetwork()->outputLayer->neuron[j]->params[i]->weight;

            EMBANN_LOGV(TAG, "Old Output Weight[%d][%d]: %" WEIGHT_PRINT, i, j, embann_getNetwork()->outputLayer->neuron[j]->params[i]->weight);
            
            embann_getNetwork()->outputLayer->neuron[j]->params[i]->weight += outputNeuronWeightChange[j][i];

            EMBANN_LOGV(TAG, "New Output Weight[%d][%d]: %" WEIGHT_PRINT, i, j, embann_getNetwork()->outputLayer->neuron[j]->params[i]->weight);
        }
        for (uint16_t k = 0; k < embann_getNetwork()->inputLayer->numNeurons; k++)
        {
            EMBANN_LOGV(TAG, "Old Hidden Weight[%d][%d]: %" WEIGHT_PRINT, i, k, embann_getNetwork()->hiddenLayer[0]->neuron[i]->params[k]->weight);
            
            EMBANN_ERROR_CHECK(embann_tanhDerivative(embann_getNetwork()->hiddenLayer[0]->neuron[i]->activation, 
                                                        &tanhDerivative));
            
            embann_getNetwork()->hiddenLayer[0]->neuron[i]->params[k]->weight += dTotalErrorToHiddenNeuron * 
                                                                                    tanhDerivative *
                                                                                    embann_getNetwork()->inputLayer->neuron[k]->activation * 
                                                                                    learningRate;

            EMBANN_LOGV(TAG, "New Hidden Weight[%d][%d]: %" WEIGHT_PRINT, i, k, embann_getNetwork()->hiddenLayer[0]->neuron[i]->params[k]->weight);
        }
    }
    return EOK;
}

int embann_tanhDerivative(activation_t inputValue, weight_t* outputValue)
{
#ifdef ACTIVATION_IS_FLOAT
    *outputValue = 1.0F - powf(tanh(inputValue * PI), 2.0F);
#elif defined(ACTIVATION_IS_SIGNED) || defined(ACTIVATION_IS_UNSIGNED)
    // TODO, scaled linear approximation
#endif   
    return EOK;
}