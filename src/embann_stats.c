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
            printf("%-12" ACTIVATION_PRINT "| ", embann_getNetwork()->inputLayer->neuron[i]->activation);
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
                printf("%-15" ACTIVATION_PRINT "| ", embann_getNetwork()->hiddenLayer[j]->neuron[i]->activation);
            }
            else
            {
                printf("               | ");
            }
        }

        if (i < embann_getNetwork()->outputLayer->numNeurons)
        {
            printf("%" ACTIVATION_PRINT, embann_getNetwork()->outputLayer->neuron[i]->activation);
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
        printf("\nInput Neuron %d: %" ACTIVATION_PRINT "\n", neuronNum,
                      embann_getNetwork()->inputLayer->neuron[neuronNum]->activation);
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
            printf("%" ACTIVATION_PRINT "-*->%" WEIGHT_PRINT " |", 
                embann_getNetwork()->hiddenLayer[embann_getNetwork()->properties.numHiddenLayers - 1U]->neuron[i]->activation,
                embann_getNetwork()->outputLayer->neuron[neuronNum]->params[i]->weight);

            if (i == floor(embann_getNetwork()->hiddenLayer[0]->numNeurons / 2U))
            {
                printf(" = %" ACTIVATION_PRINT, embann_getNetwork()->outputLayer->neuron[neuronNum]->activation);
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
                printf("%" ACTIVATION_PRINT "-*->%" WEIGHT_PRINT " |", 
                        embann_getNetwork()->inputLayer->neuron[i]->activation,
                        embann_getNetwork()->hiddenLayer[0]->neuron[neuronNum]->params[i]->weight);

                if (i == floor(embann_getNetwork()->inputLayer->numNeurons / 2U))
                {       
                    printf(" = %" ACTIVATION_PRINT, embann_getNetwork()->hiddenLayer[0]->neuron[neuronNum]->activation);
                }
                printf("\n");
            }
        }
        else
        {
            for (uint16_t i = 0; i < embann_getNetwork()->hiddenLayer[layerNum]->numNeurons; i++)
            {
                printf("%" ACTIVATION_PRINT "-*->%" WEIGHT_PRINT " |", 
                    embann_getNetwork()->hiddenLayer[layerNum - 1U]->neuron[i]->activation,
                    embann_getNetwork()->hiddenLayer[layerNum - 1U]->neuron[neuronNum]->params[i]->weight);


                if (i == floor(embann_getNetwork()->hiddenLayer[layerNum]->numNeurons / 2U))
                {
                    printf(" = %" ACTIVATION_PRINT, embann_getNetwork()->hiddenLayer[layerNum]->neuron[neuronNum]->activation);
                }
                printf("\n");
            }
        }
    }
    else
    {
        printf("\nERROR: You've asked for hidden neuron %d when only %d exist",
            neuronNum, embann_getNetwork()->hiddenLayer[layerNum]->numNeurons);
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
            printf("%-7" ACTIVATION_PRINT " | ", (1 - embann_getNetwork()->outputLayer->neuron[correctResponse]->activation));
        }
        else
        {
            printf("%-7" ACTIVATION_PRINT " | ", -embann_getNetwork()->outputLayer->neuron[i]->activation);
        }
    }

    if (embann_getNetwork()->outputLayer->numNeurons == correctResponse)
    {
        printf("%-7" ACTIVATION_PRINT "\n", (1 - embann_getNetwork()->outputLayer->neuron[correctResponse]->activation));
    }
    else
    {
        printf("%-7" ACTIVATION_PRINT "\n", -embann_getNetwork()->outputLayer->neuron[embann_getNetwork()->outputLayer->numNeurons - 1U]->activation);
    }
    return EOK;
}