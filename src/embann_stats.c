#include "embann.h"
#include "embann_log.h"

#define TAG "Embann Stats"


int embann_printNetwork(void)
{
    printf("\nInput Layer | Hidden Layer 1 ");
    for (uint8_t j = 2; j <= embann_getNetwork()->properties.numHiddenLayers; j++)
    {
        printf("| Hidden Layer %d ", j);
    }
    printf("| Output Layer\n");

    uint16_t maxNumNeurons = embann_getNetwork()->inputLayer->numNeurons;
    
    for (uint8_t i = 0; i < embann_getNetwork()->properties.numHiddenLayers; i++)
    {
        maxNumNeurons = (embann_getNetwork()->hiddenLayer[i]->numNeurons > maxNumNeurons) ? 
                            embann_getNetwork()->hiddenLayer[i]->numNeurons : maxNumNeurons;
    }
    maxNumNeurons = (embann_getNetwork()->outputLayer->numNeurons > maxNumNeurons) ? 
                            embann_getNetwork()->outputLayer->numNeurons : maxNumNeurons;

    for (uint16_t i = 0; i < maxNumNeurons; i++)
    {
        if (i < embann_getNetwork()->inputLayer->numNeurons)
        {
#ifdef ACTIVATION_IS_FLOAT
            printf("%-12.3f| ", embann_getNetwork()->inputLayer->neuron[i]->activation);
#elif defined(ACTIVATION_IS_SIGNED) || defined(ACTIVATION_IS_UNSIGNED)
            printf("%-12d| ", embann_getNetwork()->inputLayer->neuron[i]->activation);
#endif
        }
        else
        {
            printf("            | ");
        }

        //printf("%d", embann_getNetwork()->hiddenLayer[0]->numNeurons);
        for (uint8_t j = 0; j < embann_getNetwork()->properties.numHiddenLayers; j++)
        {
            if (i < embann_getNetwork()->hiddenLayer[j]->numNeurons)
            {
#ifdef ACTIVATION_IS_FLOAT
                printf("%-15.3f| ", embann_getNetwork()->hiddenLayer[j]->neuron[i]->activation);
#elif defined(ACTIVATION_IS_SIGNED) || defined(ACTIVATION_IS_UNSIGNED)
                printf("%-15d| ", embann_getNetwork()->hiddenLayer[j]->neuron[i]->activation);
#endif
            }
            else
            {
                printf("               | ");
            }
        }

        if (i < embann_getNetwork()->outputLayer->numNeurons)
        {
#ifdef ACTIVATION_IS_FLOAT
            printf("%.3f", embann_getNetwork()->outputLayer->neuron[i]->activation);
#elif defined(ACTIVATION_IS_SIGNED) || defined(ACTIVATION_IS_UNSIGNED)
            printf("%d", embann_getNetwork()->outputLayer->neuron[i]->activation);
#endif
        }
        printf("\n");
    }

    printf("I think this is output %d \n", embann_getNetwork()->properties.networkResponse);
    return EOK;
}

int embann_printInputNeuronDetails(numInputs_t neuronNum)
{
    if (neuronNum < embann_getNetwork()->inputLayer->numNeurons)
    {
#ifdef ACTIVATION_IS_FLOAT
        printf("\nInput Neuron %d: %.3f\n", neuronNum,
                      embann_getNetwork()->inputLayer->neuron[neuronNum]->activation);
#elif defined(ACTIVATION_IS_SIGNED) || defined(ACTIVATION_IS_UNSIGNED)
        printf("\nInput Neuron %d: %d\n", neuronNum,
                      embann_getNetwork()->inputLayer->neuron[neuronNum]->activation);
#endif
    }
    else
    {
        printf("\nERROR: You've asked for input neuron %d when only %d exist\n",
            neuronNum, embann_getNetwork()->inputLayer->numNeurons);
    }
    return EOK;
}

int embann_printOutputNeuronDetails(numOutputs_t neuronNum)
{
    if (neuronNum < embann_getNetwork()->outputLayer->numNeurons)
    {

        printf("\nOutput Neuron %d:\n", neuronNum);

        for (uint16_t i = 0; i < embann_getNetwork()->hiddenLayer[0]->numNeurons; i++)
        {
#ifdef ACTIVATION_IS_FLOAT
#ifdef WEIGHT_IS_FLOAT
            printf("%.3f-*->%.3f |",
                embann_getNetwork()->hiddenLayer[embann_getNetwork()->properties.numHiddenLayers - 1U]->neuron[i]->activation,
                embann_getNetwork()->outputLayer->neuron[neuronNum]->params[i]->weight);
#elif defined(WEIGHT_IS_SIGNED) || defined(WEIGHT_IS_UNSIGNED)
            printf("%.3f-*->%d |",
                embann_getNetwork()->hiddenLayer[embann_getNetwork()->properties.numHiddenLayers - 1U]->neuron[i]->activation,
                embann_getNetwork()->outputLayer->neuron[neuronNum]->params[i]->weight);
#endif //WEIGHT_IS_FLOAT
#elif defined(ACTIVATION_IS_SIGNED) || defined(ACTIVATION_IS_UNSIGNED)
#ifdef WEIGHT_IS_FLOAT
            printf("%d-*->%.3f |",
                embann_getNetwork()->hiddenLayer[embann_getNetwork()->properties.numHiddenLayers - 1U]->neuron[i]->activation,
                embann_getNetwork()->outputLayer->neuron[neuronNum]->params[i]->weight);
#elif defined(WEIGHT_IS_SIGNED) || defined(WEIGHT_IS_UNSIGNED)
            printf("%d-*->%d |",
                embann_getNetwork()->hiddenLayer[embann_getNetwork()->properties.numHiddenLayers - 1U]->neuron[i]->activation,
                embann_getNetwork()->outputLayer->neuron[neuronNum]->params[i]->weight);
#endif //WEIGHT_IS_FLOAT
#endif
            if (i == floor(embann_getNetwork()->hiddenLayer[0]->numNeurons / 2U))
            {
#ifdef ACTIVATION_IS_FLOAT
                printf(" = %.3f", embann_getNetwork()->outputLayer->neuron[neuronNum]->activation);
#elif defined(ACTIVATION_IS_SIGNED) || defined(ACTIVATION_IS_UNSIGNED)
                printf(" = %d", embann_getNetwork()->outputLayer->neuron[neuronNum]->activation);
#endif
            }
            printf("\n");
        }
    }
    else
    {
        printf(
            "\nERROR: You've asked for output neuron %d when only %d exist",
            neuronNum, embann_getNetwork()->outputLayer->numNeurons);
    }
    return EOK;
}

int embann_printHiddenNeuronDetails(numLayers_t layerNum, numHiddenNeurons_t neuronNum)
{
    if (neuronNum < embann_getNetwork()->hiddenLayer[layerNum]->numNeurons)
    {
        printf("\nHidden Neuron %d:\n", neuronNum);

        if (layerNum == 0U)
        {
            for (uint16_t i = 0; i < embann_getNetwork()->inputLayer->numNeurons; i++)
            {
#ifdef ACTIVATION_IS_FLOAT
#ifdef WEIGHT_IS_FLOAT
                printf("%.3f-*->%.3f |", 
                        embann_getNetwork()->inputLayer->neuron[i]->activation,
                        embann_getNetwork()->hiddenLayer[0]->neuron[neuronNum]->params[i]->weight);
#elif defined(WEIGHT_IS_SIGNED) || defined(WEIGHT_IS_UNSIGNED)
                printf("%.3f-*->%d |", 
                        embann_getNetwork()->inputLayer->neuron[i]->activation,
                        embann_getNetwork()->hiddenLayer[0]->neuron[neuronNum]->params[i]->weight);
#endif //WEIGHT_IS_FLOAT
#elif defined(ACTIVATION_IS_SIGNED) || defined(ACTIVATION_IS_UNSIGNED)
#ifdef WEIGHT_IS_FLOAT
                printf("%d-*->%.3f |", 
                        embann_getNetwork()->inputLayer->neuron[i]->activation,
                        embann_getNetwork()->hiddenLayer[0]->neuron[neuronNum]->params[i]->weight);
#elif defined(WEIGHT_IS_SIGNED) || defined(WEIGHT_IS_UNSIGNED)
                printf("%d-*->%d |", 
                        embann_getNetwork()->inputLayer->neuron[i]->activation,
                        embann_getNetwork()->hiddenLayer[0]->neuron[neuronNum]->params[i]->weight);
#endif //WEIGHT_IS_FLOAT
#endif
                if (i == floor(embann_getNetwork()->inputLayer->numNeurons / 2U))
                {       
#ifdef ACTIVATION_IS_FLOAT
                    printf(" = %.3f", embann_getNetwork()->hiddenLayer[0]->neuron[neuronNum]->activation);
#elif defined(ACTIVATION_IS_SIGNED) || defined(ACTIVATION_IS_UNSIGNED)
                    printf(" = %d", embann_getNetwork()->hiddenLayer[0]->neuron[neuronNum]->activation);
#endif
                }
                printf("\n");
            }
        }
        else
        {
            for (uint16_t i = 0; i < embann_getNetwork()->hiddenLayer[layerNum]->numNeurons; i++)
            {
#ifdef ACTIVATION_IS_FLOAT
#ifdef WEIGHT_IS_FLOAT
                printf("%.3f-*->%.3f |", 
                    embann_getNetwork()->hiddenLayer[layerNum - 1U]->neuron[i]->activation,
                    embann_getNetwork()->hiddenLayer[layerNum - 1U]->neuron[neuronNum]->params[i]->weight);
#elif defined(WEIGHT_IS_SIGNED) || defined(WEIGHT_IS_UNSIGNED)
                printf("%.3f-*->%d |", 
                    embann_getNetwork()->hiddenLayer[layerNum - 1U]->neuron[i]->activation,
                    embann_getNetwork()->hiddenLayer[layerNum - 1U]->neuron[neuronNum]->params[i]->weight);
#endif //WEIGHT_IS_FLOAT
#elif defined(ACTIVATION_IS_SIGNED) || defined(ACTIVATION_IS_UNSIGNED)
#ifdef WEIGHT_IS_FLOAT
                printf("%d-*->%.3f |", 
                    embann_getNetwork()->hiddenLayer[layerNum - 1U]->neuron[i]->activation,
                    embann_getNetwork()->hiddenLayer[layerNum - 1U]->neuron[neuronNum]->params[i]->weight);
#elif defined(WEIGHT_IS_SIGNED) || defined(WEIGHT_IS_UNSIGNED)
                printf("%d-*->%d |", 
                    embann_getNetwork()->hiddenLayer[layerNum - 1U]->neuron[i]->activation,
                    embann_getNetwork()->hiddenLayer[layerNum - 1U]->neuron[neuronNum]->params[i]->weight);
#endif //WEIGHT_IS_FLOAT
#endif

                if (i == floor(embann_getNetwork()->hiddenLayer[layerNum]->numNeurons / 2U))
                {
#ifdef ACTIVATION_IS_FLOAT
                    printf(" = %.3f", embann_getNetwork()->hiddenLayer[layerNum]->neuron[neuronNum]->activation);
#elif defined(ACTIVATION_IS_SIGNED) || defined(ACTIVATION_IS_UNSIGNED)
                    printf(" = %df", embann_getNetwork()->hiddenLayer[layerNum]->neuron[neuronNum]->activation);
#endif
                }
                printf("\n");
            }
        }
    }
    else
    {
        printf("\nERROR: You've asked for hidden neuron %d when only %d exist",
            neuronNum, embann_getNetwork()->hiddenLayer[0]->numNeurons);
    }
    return EOK;
}

int embann_errorReporting(numOutputs_t correctResponse)
{
    printf("\nErrors: ");
    for (uint8_t i = 0; i <= embann_getNetwork()->outputLayer->numNeurons - 1; i++)
    {
        if (i == correctResponse)
        {
#ifdef ACTIVATION_IS_FLOAT
            printf("%-7.3f | ", (1 - embann_getNetwork()->outputLayer->neuron[correctResponse]->activation));
#elif defined(ACTIVATION_IS_SIGNED) || defined(ACTIVATION_IS_UNSIGNED)
            printf("%-7d | ", (1 - embann_getNetwork()->outputLayer->neuron[correctResponse]->activation));
#endif
        }
        else
        {
#ifdef ACTIVATION_IS_FLOAT
            printf("%-7.3f | ", -embann_getNetwork()->outputLayer->neuron[i]->activation);
#elif defined(ACTIVATION_IS_SIGNED) || defined(ACTIVATION_IS_UNSIGNED)
            printf("%-7d | ", -embann_getNetwork()->outputLayer->neuron[i]->activation);
#endif
        }
    }

    if (embann_getNetwork()->outputLayer->numNeurons == correctResponse)
    {
#ifdef ACTIVATION_IS_FLOAT
        printf("%-7.3f\n", (1 - embann_getNetwork()->outputLayer->neuron[correctResponse]->activation));
#elif defined(ACTIVATION_IS_SIGNED) || defined(ACTIVATION_IS_UNSIGNED)
        printf("%-7d\n", (1 - embann_getNetwork()->outputLayer->neuron[correctResponse]->activation));
#endif
    }
    else
    {
#ifdef ACTIVATION_IS_FLOAT
        printf("%-7.3f\n", -embann_getNetwork()->outputLayer->neuron[embann_getNetwork()->outputLayer->numNeurons - 1U]->activation);
#elif defined(ACTIVATION_IS_SIGNED) || defined(ACTIVATION_IS_UNSIGNED)
        printf("%-7d\n", -embann_getNetwork()->outputLayer->neuron[embann_getNetwork()->outputLayer->numNeurons - 1U]->activation);
#endif
    }
    return EOK;
}