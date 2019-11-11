#include "embann.h"
#include "embann_log.h"

#define TAG "Embann Stats"

extern network_t* pNetworkGlobal;



int embann_printNetwork(void)
{
    printf("\nInput Layer | Hidden Layer 1 ");
    for (uint8_t j = 2; j <= pNetworkGlobal->properties.numHiddenLayers; j++)
    {
        printf("| Hidden Layer %d ", j);
    }
    printf("| Output Layer\n");

    uint16_t maxNumNeurons = pNetworkGlobal->inputLayer->numNeurons;
    
    for (uint8_t i = 0; i < pNetworkGlobal->properties.numHiddenLayers; i++)
    {
        maxNumNeurons = (pNetworkGlobal->hiddenLayer[i]->numNeurons > maxNumNeurons) ? 
                            pNetworkGlobal->hiddenLayer[i]->numNeurons : maxNumNeurons;
    }
    maxNumNeurons = (pNetworkGlobal->outputLayer->numNeurons > maxNumNeurons) ? 
                            pNetworkGlobal->outputLayer->numNeurons : maxNumNeurons;

    for (uint16_t i = 0; i < maxNumNeurons; i++)
    {
        if (i < pNetworkGlobal->inputLayer->numNeurons)
        {
            printf("%-12" ACTIVATION_PRINT "| ", pNetworkGlobal->inputLayer->activation[i]);
        }
        else
        {
            printf("            | ");
        }

        for (uint8_t j = 0; j < pNetworkGlobal->properties.numHiddenLayers; j++)
        {
            if (i < pNetworkGlobal->hiddenLayer[j]->numNeurons)
            {
                printf("%-15" ACTIVATION_PRINT "| ", pNetworkGlobal->hiddenLayer[j]->activation[i]);
            }
            else
            {
                printf("               | ");
            }
        }

        if (i < pNetworkGlobal->outputLayer->numNeurons)
        {
            printf("%" ACTIVATION_PRINT, pNetworkGlobal->outputLayer->activation[i]);
        }
        printf("\n");
    }

    printf("I think this is output %d \n", pNetworkGlobal->properties.networkResponse);
    return EOK;
}






int embann_printInputNeuronDetails(numInputs_t neuronNum)
{
    if (neuronNum < pNetworkGlobal->inputLayer->numNeurons)
    {
        printf("\nInput Neuron %d: %" ACTIVATION_PRINT "\n", neuronNum,
                      pNetworkGlobal->inputLayer->activation[neuronNum]);
    }
    else
    {
        printf("\nERROR: You've asked for input neuron %d when only %d exist\n",
            neuronNum, pNetworkGlobal->inputLayer->numNeurons);
    }
    return EOK;
}






int embann_printOutputNeuronDetails(numOutputs_t neuronNum)
{
    if (neuronNum < pNetworkGlobal->outputLayer->numNeurons)
    {

        printf("\nOutput Neuron %d:\n", neuronNum);

        for (uint16_t i = 0; i < pNetworkGlobal->hiddenLayer[0]->numNeurons; i++)
        {
            printf("%" ACTIVATION_PRINT "-*->%" WEIGHT_PRINT " |", 
                pNetworkGlobal->hiddenLayer[pNetworkGlobal->properties.numHiddenLayers - 1U]->activation[i],
                pNetworkGlobal->outputLayer->weight[neuronNum][i]);

            if (i == floor(pNetworkGlobal->hiddenLayer[0]->numNeurons / 2U))
            {
                printf(" = %" ACTIVATION_PRINT, pNetworkGlobal->outputLayer->activation[neuronNum]);
            }
            printf("\n");
        }
    }
    else
    {
        printf(
            "\nERROR: You've asked for output neuron %d when only %d exist",
            neuronNum, pNetworkGlobal->outputLayer->numNeurons);
    }
    return EOK;
}






int embann_printHiddenNeuronDetails(numLayers_t layerNum, numHiddenNeurons_t neuronNum)
{
    if (neuronNum < pNetworkGlobal->hiddenLayer[layerNum]->numNeurons)
    {
        printf("\nHidden Neuron %d:\n", neuronNum);

        if (layerNum == 0U)
        {
            for (uint16_t i = 0; i < pNetworkGlobal->inputLayer->numNeurons; i++)
            {
                printf("%" ACTIVATION_PRINT "-*->%" WEIGHT_PRINT " |", 
                        pNetworkGlobal->inputLayer->activation[i],
                        pNetworkGlobal->hiddenLayer[0]->weight[neuronNum][i]);

                if (i == floor(pNetworkGlobal->inputLayer->numNeurons / 2U))
                {       
                    printf(" = %" ACTIVATION_PRINT, pNetworkGlobal->hiddenLayer[0]->activation[neuronNum]);
                }
                printf("\n");
            }
        }
        else
        {
            for (uint16_t i = 0; i < pNetworkGlobal->hiddenLayer[layerNum]->numNeurons; i++)
            {
                printf("%" ACTIVATION_PRINT "-*->%" WEIGHT_PRINT " |", 
                    pNetworkGlobal->hiddenLayer[layerNum - 1U]->activation[i],
                    pNetworkGlobal->hiddenLayer[layerNum - 1U]->weight[neuronNum][i]);


                if (i == floor(pNetworkGlobal->hiddenLayer[layerNum]->numNeurons / 2U))
                {
                    printf(" = %" ACTIVATION_PRINT, pNetworkGlobal->hiddenLayer[layerNum]->activation[neuronNum]);
                }
                printf("\n");
            }
        }
    }
    else
    {
        printf("\nERROR: You've asked for hidden neuron %d when only %d exist",
            neuronNum, pNetworkGlobal->hiddenLayer[layerNum]->numNeurons);
    }
    return EOK;
}






int embann_errorReporting(numOutputs_t correctResponse)
{
    printf("\nErrors: ");
    for (uint8_t i = 0; i <= pNetworkGlobal->outputLayer->numNeurons - 1; i++)
    {
        if (i == correctResponse)
        {
            printf("%-7" ACTIVATION_PRINT " | ", (1 - pNetworkGlobal->outputLayer->activation[correctResponse]));
        }
        else
        {
            printf("%-7" ACTIVATION_PRINT " | ", -pNetworkGlobal->outputLayer->activation[i]);
        }
    }

    if (pNetworkGlobal->outputLayer->numNeurons == correctResponse)
    {
        printf("%-7" ACTIVATION_PRINT "\n", (1 - pNetworkGlobal->outputLayer->activation[correctResponse]));
    }
    else
    {
        printf("%-7" ACTIVATION_PRINT "\n", -pNetworkGlobal->outputLayer->activation[pNetworkGlobal->outputLayer->numNeurons - 1U]);
    }
    return EOK;
}
