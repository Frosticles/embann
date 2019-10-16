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
#ifdef ACTIVATION_IS_FLOAT
                printf("%.3f, ", currentCost[i]);
#elif defined(ACTIVATION_IS_SIGNED) || defined(ACTIVATION_IS_UNSIGNED)
                printf("%d, ", currentCost[i]);
#endif
            }
            if (currentCost[i] > desiredCost)
            {
                converged = false;
            }
        }
        if (verbose == true)
        {
#ifdef ACTIVATION_IS_FLOAT
            printf("%.3f\n ", desiredCost);
#elif defined(ACTIVATION_IS_SIGNED) || defined(ACTIVATION_IS_UNSIGNED)
            printf("%d\n ", desiredCost);
#endif
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
        
#ifdef ACTIVATION_IS_FLOAT
        EMBANN_LOGV(TAG, "dOutputErrorToOutputSum[%d]: %.3f", i, dOutputErrorToOutputSum[i]);
#elif defined(ACTIVATION_IS_SIGNED) || defined(ACTIVATION_IS_UNSIGNED)
        EMBANN_LOGV(TAG, "dOutputErrorToOutputSum[%d]: %d", i, dOutputErrorToOutputSum[i]);
#endif
        
        for (uint16_t j = 0; j < embann_getNetwork()->hiddenLayer[embann_getNetwork()->properties.numHiddenLayers - 1U]->numNeurons; j++)
        {
            outputNeuronWeightChange[i][j] = dOutputErrorToOutputSum[i] *
                                                embann_getNetwork()->hiddenLayer[embann_getNetwork()->properties.numHiddenLayers - 1U]->neuron[j]->activation *
                                                learningRate;
#ifdef ACTIVATION_IS_FLOAT
            EMBANN_LOGV(TAG, "outputNeuronWeightChange[%d][%d]: %.3f", i, j, outputNeuronWeightChange[i][j]);
#elif defined(ACTIVATION_IS_SIGNED) || defined(ACTIVATION_IS_UNSIGNED)
            EMBANN_LOGV(TAG, "outputNeuronWeightChange[%d][%d]: %d", i, j, outputNeuronWeightChange[i][j]);
#endif
        }
    }

    for (uint16_t i = 0; i < embann_getNetwork()->hiddenLayer[0]->numNeurons; i++)
    {
        dTotalErrorToHiddenNeuron = 0.0F;
        for (uint16_t j = 0; j < embann_getNetwork()->outputLayer->numNeurons; j++)
        {
            dTotalErrorToHiddenNeuron += dOutputErrorToOutputSum[j] * 
                                            embann_getNetwork()->outputLayer->neuron[j]->params[i]->weight;

#ifdef WEIGHT_IS_FLOAT
            EMBANN_LOGV(TAG, "Old Output Weight[%d][%d]: %.3f", i, j, embann_getNetwork()->outputLayer->neuron[j]->params[i]->weight);
#elif defined(WEIGHT_IS_SIGNED) || defined(WEIGHT_IS_UNSIGNED)
            EMBANN_LOGV(TAG, "Old Output Weight[%d][%d]: %d", i, j, embann_getNetwork()->outputLayer->neuron[j]->params[i]->weight);
#endif
            
            embann_getNetwork()->outputLayer->neuron[j]->params[i]->weight += outputNeuronWeightChange[j][i];

#ifdef WEIGHT_IS_FLOAT
            EMBANN_LOGV(TAG, "New Output Weight[%d][%d]: %.3f", i, j, embann_getNetwork()->outputLayer->neuron[j]->params[i]->weight);
#elif defined(WEIGHT_IS_SIGNED) || defined(WEIGHT_IS_UNSIGNED)
            EMBANN_LOGV(TAG, "New Output Weight[%d][%d]: %d", i, j, embann_getNetwork()->outputLayer->neuron[j]->params[i]->weight);
#endif
        }
        for (uint16_t k = 0; k < embann_getNetwork()->inputLayer->numNeurons; k++)
        {
#ifdef WEIGHT_IS_FLOAT
            EMBANN_LOGV(TAG, "Old Hidden Weight[%d][%d]: %.3f", i, k, embann_getNetwork()->hiddenLayer[0]->neuron[i]->params[k]->weight);
#elif defined(WEIGHT_IS_SIGNED) || defined(WEIGHT_IS_UNSIGNED)
            EMBANN_LOGV(TAG, "Old Hidden Weight[%d][%d]: %d", i, k, embann_getNetwork()->hiddenLayer[0]->neuron[i]->params[k]->weight);
#endif   
            
            EMBANN_ERROR_CHECK(embann_tanhDerivative(embann_getNetwork()->hiddenLayer[0]->neuron[i]->activation, 
                                                        &tanhDerivative));
            
            embann_getNetwork()->hiddenLayer[0]->neuron[i]->params[k]->weight += dTotalErrorToHiddenNeuron * 
                                                                                    tanhDerivative *
                                                                                    embann_getNetwork()->inputLayer->neuron[k]->activation * 
                                                                                    learningRate;

#ifdef WEIGHT_IS_FLOAT
            EMBANN_LOGV(TAG, "New Hidden Weight[%d][%d]: %.3f", i, k, embann_getNetwork()->hiddenLayer[0]->neuron[i]->params[k]->weight);
#elif defined(WEIGHT_IS_SIGNED) || defined(WEIGHT_IS_UNSIGNED)
            EMBANN_LOGV(TAG, "New Hidden Weight[%d][%d]: %d", i, k, embann_getNetwork()->hiddenLayer[0]->neuron[i]->params[k]->weight);
#endif   
        
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