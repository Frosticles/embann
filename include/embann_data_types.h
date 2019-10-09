// SPDX-License-Identifier: GPL-2.0-only
/*
    Embann_data_types.h - EMbedded Backpropogating Artificial Neural Network.
    Copyright Peter Frost 2019
*/

#ifndef Embann_data_types_h
#define Embann_data_types_h


/*
    Types for use with Intel VPDPBUSD Instruction: 
    https://software.intel.com/en-us/articles/lower-numerical-precision-deep-learning-inference-and-training

        typedef uint8_t activation_t;
        typedef int32_t bias_t;
        typedef int8_t weight_t;

        s32Accum += ((u8Act[0] * s8Wei[0]) + (u8Act[1] * s8Wei[1]) + (u8Act[2] * s8Wei[2]) + (u8Act[3] * s8Wei[3])) + u32Bias[0]
        ... Repeated 15 more times ...

    Types for use with Xtensa MAC16 Instruction:
    https://iis-people.ee.ethz.ch/~gmichi/asocd/exercises/ex_05.pdf

        typedef int16_t activation_t;
        typedef int32_t bias_t;
        typedef int16_t weight_t;

        s32Accum += ((s16Act * s16Wei) + u32Bias
*/

#ifdef CONFIG_ACTIVATION_DATA_TYPE_INT8
typedef int8_t activation_t;
#endif
#ifdef CONFIG_ACTIVATION_DATA_TYPE_INT16
typedef int16_t activation_t;
#endif
#ifdef CONFIG_ACTIVATION_DATA_TYPE_INT32
typedef int32_t activation_t;
#endif
#ifdef CONFIG_ACTIVATION_DATA_TYPE_INT64
typedef int64_t activation_t;
#endif
#ifdef CONFIG_ACTIVATION_DATA_TYPE_UINT8
typedef uint8_t activation_t;
#endif
#ifdef CONFIG_ACTIVATION_DATA_TYPE_UINT16
typedef uint16_t activation_t;
#endif
#ifdef CONFIG_ACTIVATION_DATA_TYPE_UINT32
typedef uint32_t activation_t;
#endif
#ifdef CONFIG_ACTIVATION_DATA_TYPE_UINT64
typedef uint64_t activation_t;
#endif
#ifdef CONFIG_ACTIVATION_DATA_TYPE_FLOAT
typedef float activation_t;
#endif
#ifdef CONFIG_ACTIVATION_DATA_TYPE_DOUBLE
typedef double activation_t;
#endif

#ifdef CONFIG_BIAS_DATA_TYPE_INT8
typedef int8_t bias_t;
#endif
#ifdef CONFIG_BIAS_DATA_TYPE_INT16
typedef int16_t bias_t;
#endif
#ifdef CONFIG_BIAS_DATA_TYPE_INT32
typedef int32_t bias_t;
#endif
#ifdef CONFIG_BIAS_DATA_TYPE_INT64
typedef int64_t bias_t;
#endif
#ifdef CONFIG_BIAS_DATA_TYPE_UINT8
typedef uint8_t bias_t;
#endif
#ifdef CONFIG_BIAS_DATA_TYPE_UINT16
typedef uint16_t bias_t;
#endif
#ifdef CONFIG_BIAS_DATA_TYPE_UINT32
typedef uint32_t bias_t;
#endif
#ifdef CONFIG_BIAS_DATA_TYPE_UINT64
typedef uint64_t bias_t;
#endif
#ifdef CONFIG_BIAS_DATA_TYPE_FLOAT
typedef float bias_t;
#endif
#ifdef CONFIG_BIAS_DATA_TYPE_DOUBLE
typedef double bias_t;
#endif

#ifdef CONFIG_WEIGHT_DATA_TYPE_INT8
typedef int8_t weight_t;
#endif
#ifdef CONFIG_WEIGHT_DATA_TYPE_INT16
typedef int16_t weight_t;
#endif
#ifdef CONFIG_WEIGHT_DATA_TYPE_INT32
typedef int32_t weight_t;
#endif
#ifdef CONFIG_WEIGHT_DATA_TYPE_INT64
typedef int64_t weight_t;
#endif
#ifdef CONFIG_WEIGHT_DATA_TYPE_UINT8
typedef uint8_t weight_t;
#endif
#ifdef CONFIG_WEIGHT_DATA_TYPE_UINT16
typedef uint16_t weight_t;
#endif
#ifdef CONFIG_WEIGHT_DATA_TYPE_UINT32
typedef uint32_t weight_t;
#endif
#ifdef CONFIG_WEIGHT_DATA_TYPE_UINT64
typedef uint64_t weight_t;
#endif
#ifdef CONFIG_WEIGHT_DATA_TYPE_FLOAT
typedef float weight_t;
#endif
#ifdef CONFIG_WEIGHT_DATA_TYPE_DOUBLE
typedef double weight_t;
#endif

#endif //Embann_data_types_h