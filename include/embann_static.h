/* File auto-generated by generate-static-var.py */
#include "embann_data_types.h"
#include "embann_config.h"

/*
 * Input Layer
 */
static activation_t inputNeurons[CONFIG_NUM_INPUT_NEURONS];
static inputLayer_t staticInputLayer = {
    .numNeurons = CONFIG_NUM_INPUT_NEURONS,
    .activation = inputNeurons
};




/*
 * Hidden Layer 0
 */
static activation_t hiddenNeuronsActivations_0[CONFIG_NUM_HIDDEN_NEURONS];
static bias_t hiddenNeuronBias_0[CONFIG_NUM_HIDDEN_NEURONS];

static weight_t hiddenNeuronWeights_0_0[CONFIG_NUM_INPUT_NEURONS];
static weight_t hiddenNeuronWeights_0_1[CONFIG_NUM_INPUT_NEURONS];
static weight_t hiddenNeuronWeights_0_2[CONFIG_NUM_INPUT_NEURONS];
static weight_t hiddenNeuronWeights_0_3[CONFIG_NUM_INPUT_NEURONS];
static weight_t hiddenNeuronWeights_0_4[CONFIG_NUM_INPUT_NEURONS];
static weight_t hiddenNeuronWeights_0_5[CONFIG_NUM_INPUT_NEURONS];
static weight_t hiddenNeuronWeights_0_6[CONFIG_NUM_INPUT_NEURONS];
static weight_t hiddenNeuronWeights_0_7[CONFIG_NUM_INPUT_NEURONS];
static weight_t hiddenNeuronWeights_0_8[CONFIG_NUM_INPUT_NEURONS];
static weight_t hiddenNeuronWeights_0_9[CONFIG_NUM_INPUT_NEURONS];
static weight_t* hiddenNeuronWeights_0[CONFIG_NUM_HIDDEN_NEURONS] =
{
    hiddenNeuronWeights_0_0,
    hiddenNeuronWeights_0_1,
    hiddenNeuronWeights_0_2,
    hiddenNeuronWeights_0_3,
    hiddenNeuronWeights_0_4,
    hiddenNeuronWeights_0_5,
    hiddenNeuronWeights_0_6,
    hiddenNeuronWeights_0_7,
    hiddenNeuronWeights_0_8,
    hiddenNeuronWeights_0_9
};

static hiddenLayer_t staticHiddenLayer_0 =
{
    .numNeurons = CONFIG_NUM_HIDDEN_NEURONS,
    .activation = hiddenNeuronsActivations_0,
    .bias = hiddenNeuronBias_0,
    .weight = hiddenNeuronWeights_0,
};


/*
 * Hidden Layer 1
 */
static activation_t hiddenNeuronsActivations_1[CONFIG_NUM_HIDDEN_NEURONS];
static bias_t hiddenNeuronBias_1[CONFIG_NUM_HIDDEN_NEURONS];

static weight_t hiddenNeuronWeights_1_0[CONFIG_NUM_HIDDEN_NEURONS];
static weight_t hiddenNeuronWeights_1_1[CONFIG_NUM_HIDDEN_NEURONS];
static weight_t hiddenNeuronWeights_1_2[CONFIG_NUM_HIDDEN_NEURONS];
static weight_t hiddenNeuronWeights_1_3[CONFIG_NUM_HIDDEN_NEURONS];
static weight_t hiddenNeuronWeights_1_4[CONFIG_NUM_HIDDEN_NEURONS];
static weight_t hiddenNeuronWeights_1_5[CONFIG_NUM_HIDDEN_NEURONS];
static weight_t hiddenNeuronWeights_1_6[CONFIG_NUM_HIDDEN_NEURONS];
static weight_t hiddenNeuronWeights_1_7[CONFIG_NUM_HIDDEN_NEURONS];
static weight_t hiddenNeuronWeights_1_8[CONFIG_NUM_HIDDEN_NEURONS];
static weight_t hiddenNeuronWeights_1_9[CONFIG_NUM_HIDDEN_NEURONS];
static weight_t* hiddenNeuronWeights_1[CONFIG_NUM_HIDDEN_NEURONS] =
{
    hiddenNeuronWeights_1_0,
    hiddenNeuronWeights_1_1,
    hiddenNeuronWeights_1_2,
    hiddenNeuronWeights_1_3,
    hiddenNeuronWeights_1_4,
    hiddenNeuronWeights_1_5,
    hiddenNeuronWeights_1_6,
    hiddenNeuronWeights_1_7,
    hiddenNeuronWeights_1_8,
    hiddenNeuronWeights_1_9
};

static hiddenLayer_t staticHiddenLayer_1 =
{
    .numNeurons = CONFIG_NUM_HIDDEN_NEURONS,
    .activation = hiddenNeuronsActivations_1,
    .bias = hiddenNeuronBias_1,
    .weight = hiddenNeuronWeights_1,
};


/*
 * Hidden Layer 2
 */
static activation_t hiddenNeuronsActivations_2[CONFIG_NUM_HIDDEN_NEURONS];
static bias_t hiddenNeuronBias_2[CONFIG_NUM_HIDDEN_NEURONS];

static weight_t hiddenNeuronWeights_2_0[CONFIG_NUM_HIDDEN_NEURONS];
static weight_t hiddenNeuronWeights_2_1[CONFIG_NUM_HIDDEN_NEURONS];
static weight_t hiddenNeuronWeights_2_2[CONFIG_NUM_HIDDEN_NEURONS];
static weight_t hiddenNeuronWeights_2_3[CONFIG_NUM_HIDDEN_NEURONS];
static weight_t hiddenNeuronWeights_2_4[CONFIG_NUM_HIDDEN_NEURONS];
static weight_t hiddenNeuronWeights_2_5[CONFIG_NUM_HIDDEN_NEURONS];
static weight_t hiddenNeuronWeights_2_6[CONFIG_NUM_HIDDEN_NEURONS];
static weight_t hiddenNeuronWeights_2_7[CONFIG_NUM_HIDDEN_NEURONS];
static weight_t hiddenNeuronWeights_2_8[CONFIG_NUM_HIDDEN_NEURONS];
static weight_t hiddenNeuronWeights_2_9[CONFIG_NUM_HIDDEN_NEURONS];
static weight_t* hiddenNeuronWeights_2[CONFIG_NUM_HIDDEN_NEURONS] =
{
    hiddenNeuronWeights_2_0,
    hiddenNeuronWeights_2_1,
    hiddenNeuronWeights_2_2,
    hiddenNeuronWeights_2_3,
    hiddenNeuronWeights_2_4,
    hiddenNeuronWeights_2_5,
    hiddenNeuronWeights_2_6,
    hiddenNeuronWeights_2_7,
    hiddenNeuronWeights_2_8,
    hiddenNeuronWeights_2_9
};

static hiddenLayer_t staticHiddenLayer_2 =
{
    .numNeurons = CONFIG_NUM_HIDDEN_NEURONS,
    .activation = hiddenNeuronsActivations_2,
    .bias = hiddenNeuronBias_2,
    .weight = hiddenNeuronWeights_2,
};


/*
 * Hidden Layer 3
 */
static activation_t hiddenNeuronsActivations_3[CONFIG_NUM_HIDDEN_NEURONS];
static bias_t hiddenNeuronBias_3[CONFIG_NUM_HIDDEN_NEURONS];

static weight_t hiddenNeuronWeights_3_0[CONFIG_NUM_HIDDEN_NEURONS];
static weight_t hiddenNeuronWeights_3_1[CONFIG_NUM_HIDDEN_NEURONS];
static weight_t hiddenNeuronWeights_3_2[CONFIG_NUM_HIDDEN_NEURONS];
static weight_t hiddenNeuronWeights_3_3[CONFIG_NUM_HIDDEN_NEURONS];
static weight_t hiddenNeuronWeights_3_4[CONFIG_NUM_HIDDEN_NEURONS];
static weight_t hiddenNeuronWeights_3_5[CONFIG_NUM_HIDDEN_NEURONS];
static weight_t hiddenNeuronWeights_3_6[CONFIG_NUM_HIDDEN_NEURONS];
static weight_t hiddenNeuronWeights_3_7[CONFIG_NUM_HIDDEN_NEURONS];
static weight_t hiddenNeuronWeights_3_8[CONFIG_NUM_HIDDEN_NEURONS];
static weight_t hiddenNeuronWeights_3_9[CONFIG_NUM_HIDDEN_NEURONS];
static weight_t* hiddenNeuronWeights_3[CONFIG_NUM_HIDDEN_NEURONS] =
{
    hiddenNeuronWeights_3_0,
    hiddenNeuronWeights_3_1,
    hiddenNeuronWeights_3_2,
    hiddenNeuronWeights_3_3,
    hiddenNeuronWeights_3_4,
    hiddenNeuronWeights_3_5,
    hiddenNeuronWeights_3_6,
    hiddenNeuronWeights_3_7,
    hiddenNeuronWeights_3_8,
    hiddenNeuronWeights_3_9
};

static hiddenLayer_t staticHiddenLayer_3 =
{
    .numNeurons = CONFIG_NUM_HIDDEN_NEURONS,
    .activation = hiddenNeuronsActivations_3,
    .bias = hiddenNeuronBias_3,
    .weight = hiddenNeuronWeights_3,
};


/*
 * Hidden Layer 4
 */
static activation_t hiddenNeuronsActivations_4[CONFIG_NUM_HIDDEN_NEURONS];
static bias_t hiddenNeuronBias_4[CONFIG_NUM_HIDDEN_NEURONS];

static weight_t hiddenNeuronWeights_4_0[CONFIG_NUM_HIDDEN_NEURONS];
static weight_t hiddenNeuronWeights_4_1[CONFIG_NUM_HIDDEN_NEURONS];
static weight_t hiddenNeuronWeights_4_2[CONFIG_NUM_HIDDEN_NEURONS];
static weight_t hiddenNeuronWeights_4_3[CONFIG_NUM_HIDDEN_NEURONS];
static weight_t hiddenNeuronWeights_4_4[CONFIG_NUM_HIDDEN_NEURONS];
static weight_t hiddenNeuronWeights_4_5[CONFIG_NUM_HIDDEN_NEURONS];
static weight_t hiddenNeuronWeights_4_6[CONFIG_NUM_HIDDEN_NEURONS];
static weight_t hiddenNeuronWeights_4_7[CONFIG_NUM_HIDDEN_NEURONS];
static weight_t hiddenNeuronWeights_4_8[CONFIG_NUM_HIDDEN_NEURONS];
static weight_t hiddenNeuronWeights_4_9[CONFIG_NUM_HIDDEN_NEURONS];
static weight_t* hiddenNeuronWeights_4[CONFIG_NUM_HIDDEN_NEURONS] =
{
    hiddenNeuronWeights_4_0,
    hiddenNeuronWeights_4_1,
    hiddenNeuronWeights_4_2,
    hiddenNeuronWeights_4_3,
    hiddenNeuronWeights_4_4,
    hiddenNeuronWeights_4_5,
    hiddenNeuronWeights_4_6,
    hiddenNeuronWeights_4_7,
    hiddenNeuronWeights_4_8,
    hiddenNeuronWeights_4_9
};

static hiddenLayer_t staticHiddenLayer_4 =
{
    .numNeurons = CONFIG_NUM_HIDDEN_NEURONS,
    .activation = hiddenNeuronsActivations_4,
    .bias = hiddenNeuronBias_4,
    .weight = hiddenNeuronWeights_4,
};


static hiddenLayer_t* staticHiddenLayers[CONFIG_NUM_HIDDEN_LAYERS] =
{
    &staticHiddenLayer_0,
    &staticHiddenLayer_1,
    &staticHiddenLayer_2,
    &staticHiddenLayer_3,
    &staticHiddenLayer_4
};




/*
 * Output Layer
 */
static activation_t outputNeuronsActivations[CONFIG_NUM_OUTPUT_NEURONS];
static bias_t outputNeuronBias[CONFIG_NUM_OUTPUT_NEURONS];

static weight_t outputNeuronWeights_0[CONFIG_NUM_HIDDEN_NEURONS];
static weight_t outputNeuronWeights_1[CONFIG_NUM_HIDDEN_NEURONS];
static weight_t outputNeuronWeights_2[CONFIG_NUM_HIDDEN_NEURONS];
static weight_t* outputNeuronWeights[CONFIG_NUM_OUTPUT_NEURONS] =
{
    outputNeuronWeights_0,
    outputNeuronWeights_1,
    outputNeuronWeights_2
};

static outputLayer_t staticOutputLayer =
{
    .numNeurons = CONFIG_NUM_OUTPUT_NEURONS,
    .activation = outputNeuronsActivations,
    .bias = outputNeuronBias,
    .weight = outputNeuronWeights
};




static network_t staticNetwork = {
    .inputLayer = &staticInputLayer,
    .hiddenLayer = staticHiddenLayers,
    .outputLayer = &staticOutputLayer
};
