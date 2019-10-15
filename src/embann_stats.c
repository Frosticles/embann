#include "embann.h"
#include "embann_log.h"

#define TAG "Embann Stats"


int embann_printNetwork(void)
{
    EMBANN_LOGI(TAG, "\nInput: [");
    for (uint16_t i = 0; i < (embann_getNetwork()->inputLayer->numNeurons - 1U); i++)
    {
        EMBANN_LOGI(TAG, "%0.3f, ", embann_getNetwork()->inputLayer->neuron[i]->activation);
    }
    EMBANN_LOGI(TAG, "%0.3f]", embann_getNetwork()->inputLayer->neuron[embann_getNetwork()->inputLayer->numNeurons - 1U]->activation);

    EMBANN_LOGI(TAG, "\nInput Layer | Hidden Layer ");
    if (embann_getNetwork()->properties.numHiddenLayers > 1U)
    {
        EMBANN_LOGI(TAG, "1 ");
        for (uint8_t j = 2; j <= embann_getNetwork()->properties.numHiddenLayers; j++)
        {
            EMBANN_LOGI(TAG, "| Hidden Layer %d ", j);
        }
    }
    EMBANN_LOGI(TAG, "| Output Layer");

    bool nothingLeft = false;
    uint16_t k = 0;
    while (nothingLeft == false)
    { /* TODO, Make this compatible with multiple hidden layers */
        if ((k >= embann_getNetwork()->inputLayer->numNeurons) &&
            (k >= embann_getNetwork()->hiddenLayer[0]->numNeurons) &&
            (k >= embann_getNetwork()->outputLayer->numNeurons))
        {
            nothingLeft = true;
        }
        else
        {
            if (k < embann_getNetwork()->inputLayer->numNeurons)
            {
                EMBANN_LOGI(TAG, "%-12.3f| ", embann_getNetwork()->inputLayer->neuron[k]->activation);
            }
            else
            {
                EMBANN_LOGI(TAG, "            | ");
            }

            if (k < embann_getNetwork()->hiddenLayer[0]->numNeurons)
            {
                if (embann_getNetwork()->properties.numHiddenLayers == 1U)
                {
                    EMBANN_LOGI(TAG, "%-13.3f| ", embann_getNetwork()->hiddenLayer[0]->neuron[k]->activation);
                }
                else
                {
                    for (uint8_t l = 0; l < embann_getNetwork()->properties.numHiddenLayers; l++)
                    {
                        EMBANN_LOGI(TAG, "%-15.3f| ", embann_getNetwork()->hiddenLayer[l]->neuron[k]->activation);
                    }
                }
            }
            else
            {
                EMBANN_LOGI(TAG, "             | ");
                if (embann_getNetwork()->properties.numHiddenLayers > 1U)
                {
                    EMBANN_LOGI(TAG, "              | ");
                }
            }

            if (k < embann_getNetwork()->outputLayer->numNeurons)
            {
                EMBANN_LOGI(TAG, "%.3f", embann_getNetwork()->outputLayer->neuron[k]->activation);
            }
        }
        EMBANN_LOGI(TAG, "\n");
        k++;
    }

    EMBANN_LOGI(TAG, "I think this is output %d ", embann_getNetwork()->properties.networkResponse);
    return EOK;
}

int embann_printInputNeuronDetails(uint8_t neuronNum)
{
    if (neuronNum < embann_getNetwork()->inputLayer->numNeurons)
    {
        EMBANN_LOGI(TAG, "\nInput Neuron %d: %.3f", neuronNum,
                      embann_getNetwork()->inputLayer->neuron[neuronNum]->activation);
    }
    else
    {
        EMBANN_LOGI(TAG, "\nERROR: You've asked for input neuron %d when only %d exist",
            neuronNum, embann_getNetwork()->inputLayer->numNeurons);
    }
    return EOK;
}

int embann_printOutputNeuronDetails(uint8_t neuronNum)
{
    if (neuronNum < embann_getNetwork()->outputLayer->numNeurons)
    {

        EMBANN_LOGI(TAG, "\nOutput Neuron %d:", neuronNum);

        for (uint16_t i = 0; i < embann_getNetwork()->hiddenLayer[0]->numNeurons; i++)
        {
            EMBANN_LOGI(TAG, 
                "%.3f-*->%.3f |",
                embann_getNetwork()->hiddenLayer[embann_getNetwork()->properties.numHiddenLayers - 1U]->neuron[i]->activation,
                embann_getNetwork()->outputLayer->neuron[neuronNum]->params[i]->weight);

            if (i == floor(embann_getNetwork()->hiddenLayer[0]->numNeurons / 2U))
            {
                EMBANN_LOGI(TAG, " = %.3f", embann_getNetwork()->outputLayer->neuron[neuronNum]->activation);
            }
            EMBANN_LOGI(TAG, "\n");
        }
    }
    else
    {
        EMBANN_LOGI(TAG, 
            "\nERROR: You've asked for output neuron %d when only %d exist",
            neuronNum, embann_getNetwork()->outputLayer->numNeurons);
    }
    return EOK;
}

int embann_printHiddenNeuronDetails(uint8_t layerNum, uint8_t neuronNum)
{
    if (neuronNum < embann_getNetwork()->hiddenLayer[0]->numNeurons)
    {

        EMBANN_LOGI(TAG, "\nHidden Neuron %d:", neuronNum);

        if (layerNum == 0U)
        {

            for (uint16_t i = 0; i < embann_getNetwork()->inputLayer->numNeurons; i++)
            {
                EMBANN_LOGI(TAG, "%.3f-*->%.3f |", embann_getNetwork()->inputLayer->neuron[i]->activation,
                              embann_getNetwork()->hiddenLayer[0]->neuron[neuronNum]->params[i]->weight);

                if (i == floor(embann_getNetwork()->inputLayer->numNeurons / 2U))
                {
                    EMBANN_LOGI(TAG, " = %.3f",
                                  embann_getNetwork()->hiddenLayer[0]->neuron[neuronNum]->activation);
                }
                EMBANN_LOGI(TAG, "\n");
            }
        }
        else
        {

            for (uint16_t i = 0; i < embann_getNetwork()->hiddenLayer[0]->numNeurons; i++)
            {
                EMBANN_LOGI(TAG, 
                    "%.3f-*->%.3f |", embann_getNetwork()->hiddenLayer[layerNum - 1U]->neuron[i]->activation,
                    embann_getNetwork()->hiddenLayer[layerNum - 1U]->neuron[neuronNum]->params[i]->weight);

                if (i == floor(embann_getNetwork()->hiddenLayer[0]->numNeurons / 2U))
                {
                    EMBANN_LOGI(TAG, " = %.3f",
                                  embann_getNetwork()->hiddenLayer[0]->neuron[neuronNum]->activation);
                }
                EMBANN_LOGI(TAG, "\n");
            }
        }
    }
    else
    {
        EMBANN_LOGI(TAG, 
            "\nERROR: You've asked for hidden neuron %d when only %d exist",
            neuronNum, embann_getNetwork()->hiddenLayer[0]->numNeurons);
    }
    return EOK;
}

int embann_errorReporting(uint8_t correctResponse)
{
    EMBANN_LOGI(TAG, "\n");
    for (uint8_t i = 0; i < embann_getNetwork()->outputLayer->numNeurons; i++)
    {
        if (i == correctResponse)
        {
            EMBANN_LOGI(TAG, "%-7.3f | ",
                          (1 - embann_getNetwork()->outputLayer->neuron[correctResponse]->activation));
        }
        else
        {
            EMBANN_LOGI(TAG, "%-7.3f | ", -embann_getNetwork()->outputLayer->neuron[i]->activation);
        }
    }
    return EOK;
}