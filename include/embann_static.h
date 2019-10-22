#include "embann_data_types.h"
#include "embann_config.h"

static uNeuron_t inputLayer_1;
static uNeuron_t inputLayer_2;
static inputLayer_t staticInputLayer = {
    .numNeurons = CONFIG_NUM_INPUT_NEURONS,
    .neuron = {
        &inputLayer_1,
        &inputLayer_2
    }
};





static neuronParams_t hiddenLayer_0_1_0;
static neuronParams_t hiddenLayer_0_1_1;
static neuronParams_t hiddenLayer_0_2_0;
static neuronParams_t hiddenLayer_0_2_1;
static wNeuron_t hiddenLayer_0_1 = {
    .activation = 0,
    .params = {&hiddenLayer_0_1_0, &hiddenLayer_0_1_1}
};
static wNeuron_t hiddenLayer_0_2 = {
    .activation = 0,
    .params = {&hiddenLayer_0_2_0, &hiddenLayer_0_2_1}
};
static hiddenLayer_t staticHiddenLayer_0 = {
    .numNeurons = CONFIG_NUM_HIDDEN_NEURONS,
    .neuron = {&hiddenLayer_0_1, &hiddenLayer_0_2}
};


static neuronParams_t hiddenLayer_1_1_0;
static neuronParams_t hiddenLayer_1_1_1;
static neuronParams_t hiddenLayer_1_2_0;
static neuronParams_t hiddenLayer_1_2_1;
static wNeuron_t hiddenLayer_1_1 = {
    .activation = 0,
    .params = {&hiddenLayer_1_1_0, &hiddenLayer_1_1_1}
};
static wNeuron_t hiddenLayer_1_2 = {
    .activation = 0,
    .params = {&hiddenLayer_1_2_0, &hiddenLayer_1_2_1}
};
static hiddenLayer_t staticHiddenLayer_1 = {
    .numNeurons = CONFIG_NUM_HIDDEN_NEURONS,
    .neuron = {&hiddenLayer_1_1, &hiddenLayer_1_2}
};






static neuronParams_t outputLayer_1_0;
static neuronParams_t outputLayer_1_1;
static neuronParams_t outputLayer_2_0;
static neuronParams_t outputLayer_2_1;
static wNeuron_t outputLayer_1 = {
    .activation = 0,
    .params = {&outputLayer_1_0, &outputLayer_1_1}
};
static wNeuron_t outputLayer_2 = {
    .activation = 0,
    .params = {&outputLayer_2_0, &outputLayer_2_1}
};
static outputLayer_t staticOutputLayer = {
    .numNeurons = CONFIG_NUM_OUTPUT_NEURONS,
    .neuron = {&outputLayer_1, &outputLayer_2}
};


static network_t staticNetwork = {
    .inputLayer = &staticInputLayer,
    .hiddenLayer = {
        &staticHiddenLayer_0,
        &staticHiddenLayer_1
    },
    .outputLayer = &staticOutputLayer
};
