#include "embann.h"
#include "embann_log.h"

#define TAG "Embann Stats"


int embann_printNetwork(void)
{
    EMBANN_LOGI(TAG, "\nInput: [");
    for (uint16_t i = 0; i < (network->inputLayer->numNeurons - 1U); i++)
    {
        EMBANN_LOGI(TAG, "%0.3f, ", network->inputLayer->neuron[i]->activation);
    }
    EMBANN_LOGI(TAG, "%0.3f]", network->inputLayer->neuron[network->inputLayer->numNeurons - 1U]->activation);

    EMBANN_LOGI(TAG, "\nInput Layer | Hidden Layer ");
    if (network->properties.numHiddenLayers > 1U)
    {
        EMBANN_LOGI(TAG, "1 ");
        for (uint8_t j = 2; j <= network->properties.numHiddenLayers; j++)
        {
            EMBANN_LOGI(TAG, "| Hidden Layer %d ", j);
        }
    }
    EMBANN_LOGI(TAG, "| Output Layer");

    bool nothingLeft = false;
    uint16_t k = 0;
    while (nothingLeft == false)
    { /* TODO, Make this compatible with multiple hidden layers */
        if ((k >= network->inputLayer->numNeurons) &&
            (k >= network->hiddenLayer[0].numNeurons) &&
            (k >= network->outputLayer->numNeurons))
        {
            nothingLeft = true;
        }
        else
        {
            if (k < network->inputLayer->numNeurons)
            {
                EMBANN_LOGI(TAG, "%-12.3f| ", network->inputLayer->neuron[k]->activation);
            }
            else
            {
                EMBANN_LOGI(TAG, "            | ");
            }

            if (k < network->hiddenLayer[0].numNeurons)
            {
                if (network->properties.numHiddenLayers == 1U)
                {
                    EMBANN_LOGI(TAG, "%-13.3f| ", network->hiddenLayer[0].neuron[k]->activation);
                }
                else
                {
                    for (uint8_t l = 0; l < network->properties.numHiddenLayers; l++)
                    {
                        EMBANN_LOGI(TAG, "%-15.3f| ", network->hiddenLayer[l].neuron[k]->activation);
                    }
                }
            }
            else
            {
                EMBANN_LOGI(TAG, "             | ");
                if (network->properties.numHiddenLayers > 1U)
                {
                    EMBANN_LOGI(TAG, "              | ");
                }
            }

            if (k < network->outputLayer->numNeurons)
            {
                EMBANN_LOGI(TAG, "%.3f", network->outputLayer->neuron[k]->activation);
            }
        }
        EMBANN_LOGI(TAG, "\n");
        k++;
    }

    EMBANN_LOGI(TAG, "I think this is output %d ", network->properties.networkResponse);
    return EOK;
}

int embann_printInputNeuronDetails(uint8_t neuronNum)
{
    if (neuronNum < network->inputLayer->numNeurons)
    {
        EMBANN_LOGI(TAG, "\nInput Neuron %d: %.3f", neuronNum,
                      network->inputLayer->neuron[neuronNum]->activation);
    }
    else
    {
        EMBANN_LOGI(TAG, "\nERROR: You've asked for input neuron %d when only %d exist",
            neuronNum, network->inputLayer->numNeurons);
    }
    return EOK;
}

int embann_printOutputNeuronDetails(uint8_t neuronNum)
{
    if (neuronNum < network->outputLayer->numNeurons)
    {

        EMBANN_LOGI(TAG, "\nOutput Neuron %d:", neuronNum);

        for (uint16_t i = 0; i < network->hiddenLayer[0].numNeurons; i++)
        {
            EMBANN_LOGI(TAG, 
                "%.3f-*->%.3f |",
                network->hiddenLayer[network->properties.numHiddenLayers - 1U].neuron[i]->activation,
                network->outputLayer->neuron[neuronNum]->params[i]->weight);

            if (i == floor(network->hiddenLayer[0].numNeurons / 2U))
            {
                EMBANN_LOGI(TAG, " = %.3f", network->outputLayer->neuron[neuronNum]->activation);
            }
            EMBANN_LOGI(TAG, "\n");
        }
    }
    else
    {
        EMBANN_LOGI(TAG, 
            "\nERROR: You've asked for output neuron %d when only %d exist",
            neuronNum, network->outputLayer->numNeurons);
    }
    return EOK;
}

int embann_printHiddenNeuronDetails(uint8_t layerNum, uint8_t neuronNum)
{
    if (neuronNum < network->hiddenLayer[0].numNeurons)
    {

        EMBANN_LOGI(TAG, "\nHidden Neuron %d:", neuronNum);

        if (layerNum == 0U)
        {

            for (uint16_t i = 0; i < network->inputLayer->numNeurons; i++)
            {
                EMBANN_LOGI(TAG, "%.3f-*->%.3f |", network->inputLayer->neuron[i]->activation,
                              network->hiddenLayer[0].neuron[neuronNum]->params[i]->weight);

                if (i == floor(network->inputLayer->numNeurons / 2U))
                {
                    EMBANN_LOGI(TAG, " = %.3f",
                                  network->hiddenLayer[0].neuron[neuronNum]->activation);
                }
                EMBANN_LOGI(TAG, "\n");
            }
        }
        else
        {

            for (uint16_t i = 0; i < network->hiddenLayer[0].numNeurons; i++)
            {
                EMBANN_LOGI(TAG, 
                    "%.3f-*->%.3f |", network->hiddenLayer[layerNum - 1U].neuron[i]->activation,
                    network->hiddenLayer[layerNum - 1U].neuron[neuronNum]->params[i]->weight);

                if (i == floor(network->hiddenLayer[0].numNeurons / 2U))
                {
                    EMBANN_LOGI(TAG, " = %.3f",
                                  network->hiddenLayer[0].neuron[neuronNum]->activation);
                }
                EMBANN_LOGI(TAG, "\n");
            }
        }
    }
    else
    {
        EMBANN_LOGI(TAG, 
            "\nERROR: You've asked for hidden neuron %d when only %d exist",
            neuronNum, network->hiddenLayer[0].numNeurons);
    }
    return EOK;
}

int embann_errorReporting(uint8_t correctResponse)
{
    EMBANN_LOGI(TAG, "\n");
    for (uint8_t i = 0; i < network->outputLayer->numNeurons; i++)
    {
        if (i == correctResponse)
        {
            EMBANN_LOGI(TAG, "%-7.3f | ",
                          (1 - network->outputLayer->neuron[correctResponse]->activation));
        }
        else
        {
            EMBANN_LOGI(TAG, "%-7.3f | ", -network->outputLayer->neuron[i]->activation);
        }
    }
    return EOK;
}