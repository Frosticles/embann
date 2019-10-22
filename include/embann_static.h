#include "embann_data_types.h"
#include "embann_config.h"

static uNeuron_t inputLayer_0;
static uNeuron_t inputLayer_1;
static inputLayer_t staticInputLayer = {
    .numNeurons = CONFIG_NUM_INPUT_NEURONS,
    .neuron = {
        &inputLayer_0,
        &inputLayer_1
    }
};



static neuronParams_t hiddenLayer_0_0_0;
static neuronParams_t hiddenLayer_0_0_1;
static neuronParams_t hiddenLayer_0_1_0;
static neuronParams_t hiddenLayer_0_1_1;
static wNeuron_t hiddenLayer_0_0 = {
    .activation = 0,
    .params = {
        &hiddenLayer_0_0_0,
        &hiddenLayer_0_0_1
    }
};
static wNeuron_t hiddenLayer_0_1 = {
    .activation = 0,
    .params = {
        &hiddenLayer_0_1_0,
        &hiddenLayer_0_1_1
    }
};
static hiddenLayer_t staticHiddenLayer_0 = {
    .numNeurons = CONFIG_NUM_HIDDEN_NEURONS,
    .neuron = {
        &hiddenLayer_0_0,
        &hiddenLayer_0_1
    }
};



static neuronParams_t outputLayer_0_0;
static neuronParams_t outputLayer_0_1;
static neuronParams_t outputLayer_1_0;
static neuronParams_t outputLayer_1_1;
static wNeuron_t outputLayer_0 = {
    .activation = 0,
    .params = {
        &outputLayer_0_0,
        &outputLayer_0_1
    }
};
static wNeuron_t outputLayer_1 = {
    .activation = 0,
    .params = {
        &outputLayer_1_0,
        &outputLayer_1_1
    }
};
static outputLayer_t staticOutputLayer = {
    .numNeurons = CONFIG_NUM_OUTPUT_NEURONS,
    .neuron = {
        &outputLayer_0,
        &outputLayer_1
    }
};



static network_t staticNetwork = {
    .inputLayer = &staticInputLayer,
    .hiddenLayer = {
        &staticHiddenLayer_0
    },
    .outputLayer = &staticOutputLayer
};
