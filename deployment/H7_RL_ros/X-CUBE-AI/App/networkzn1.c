/**
  ******************************************************************************
  * @file    networkzn1.c
  * @author  AST Embedded Analytics Research Platform
  * @date    2026-10-29T20:45:30+0800
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */


#include "networkzn1.h"
#include "networkzn1_data.h"

#include "ai_platform.h"
#include "ai_platform_interface.h"
#include "ai_math_helpers.h"

#include "core_common.h"
#include "core_convert.h"

#include "layers.h"



#undef AI_NET_OBJ_INSTANCE
#define AI_NET_OBJ_INSTANCE g_networkzn1
 
#undef AI_NETWORKZN1_MODEL_SIGNATURE
#define AI_NETWORKZN1_MODEL_SIGNATURE     "0x02658f986406f3ffa583de4ca25c0690"

#ifndef AI_TOOLS_REVISION_ID
#define AI_TOOLS_REVISION_ID     ""
#endif

#undef AI_TOOLS_DATE_TIME
#define AI_TOOLS_DATE_TIME   "2026-10-29T20:45:30+0800"

#undef AI_TOOLS_COMPILE_TIME
#define AI_TOOLS_COMPILE_TIME    __DATE__ " " __TIME__

#undef AI_NETWORKZN1_N_BATCHES
#define AI_NETWORKZN1_N_BATCHES         (1)

static ai_ptr g_networkzn1_activations_map[1] = AI_C_ARRAY_INIT;
static ai_ptr g_networkzn1_weights_map[1] = AI_C_ARRAY_INIT;



/**  Array declarations section  **********************************************/
/* Array#0 */
AI_ARRAY_OBJ_DECLARE(
  observation_history_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 160, AI_STATIC)

/* Array#1 */
AI_ARRAY_OBJ_DECLARE(
  _encoder_encoder_0_Gemm_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 128, AI_STATIC)

/* Array#2 */
AI_ARRAY_OBJ_DECLARE(
  _encoder_encoder_1_Elu_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 128, AI_STATIC)

/* Array#3 */
AI_ARRAY_OBJ_DECLARE(
  _encoder_encoder_2_Gemm_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#4 */
AI_ARRAY_OBJ_DECLARE(
  _encoder_encoder_1_1_Elu_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#5 */
AI_ARRAY_OBJ_DECLARE(
  _encoder_encoder_4_Gemm_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 3, AI_STATIC)

/* Array#6 */
AI_ARRAY_OBJ_DECLARE(
  velocity_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 3, AI_STATIC)

/* Array#7 */
AI_ARRAY_OBJ_DECLARE(
  _Slice_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)

/* Array#8 */
AI_ARRAY_OBJ_DECLARE(
  _Concat_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 35, AI_STATIC)

/* Array#9 */
AI_ARRAY_OBJ_DECLARE(
  _mlp_mlp_0_Gemm_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 256, AI_STATIC)

/* Array#10 */
AI_ARRAY_OBJ_DECLARE(
  _mlp_mlp_1_Elu_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 256, AI_STATIC)

/* Array#11 */
AI_ARRAY_OBJ_DECLARE(
  _mlp_mlp_2_Gemm_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 128, AI_STATIC)

/* Array#12 */
AI_ARRAY_OBJ_DECLARE(
  _mlp_mlp_1_1_Elu_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 128, AI_STATIC)

/* Array#13 */
AI_ARRAY_OBJ_DECLARE(
  _mlp_mlp_4_Gemm_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#14 */
AI_ARRAY_OBJ_DECLARE(
  _mlp_mlp_1_2_Elu_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#15 */
AI_ARRAY_OBJ_DECLARE(
  actions_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 6, AI_STATIC)

/* Array#16 */
AI_ARRAY_OBJ_DECLARE(
  _Constant_4_output_0_2D_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1, AI_STATIC)

/* Array#17 */
AI_ARRAY_OBJ_DECLARE(
  _encoder_encoder_0_Gemm_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 20480, AI_STATIC)

/* Array#18 */
AI_ARRAY_OBJ_DECLARE(
  _encoder_encoder_0_Gemm_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 128, AI_STATIC)

/* Array#19 */
AI_ARRAY_OBJ_DECLARE(
  _encoder_encoder_2_Gemm_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8192, AI_STATIC)

/* Array#20 */
AI_ARRAY_OBJ_DECLARE(
  _encoder_encoder_2_Gemm_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#21 */
AI_ARRAY_OBJ_DECLARE(
  _encoder_encoder_4_Gemm_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 192, AI_STATIC)

/* Array#22 */
AI_ARRAY_OBJ_DECLARE(
  _encoder_encoder_4_Gemm_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 3, AI_STATIC)

/* Array#23 */
AI_ARRAY_OBJ_DECLARE(
  _mlp_mlp_0_Gemm_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8960, AI_STATIC)

/* Array#24 */
AI_ARRAY_OBJ_DECLARE(
  _mlp_mlp_0_Gemm_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 256, AI_STATIC)

/* Array#25 */
AI_ARRAY_OBJ_DECLARE(
  _mlp_mlp_2_Gemm_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32768, AI_STATIC)

/* Array#26 */
AI_ARRAY_OBJ_DECLARE(
  _mlp_mlp_2_Gemm_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 128, AI_STATIC)

/* Array#27 */
AI_ARRAY_OBJ_DECLARE(
  _mlp_mlp_4_Gemm_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8192, AI_STATIC)

/* Array#28 */
AI_ARRAY_OBJ_DECLARE(
  _mlp_mlp_4_Gemm_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#29 */
AI_ARRAY_OBJ_DECLARE(
  actions_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 384, AI_STATIC)

/* Array#30 */
AI_ARRAY_OBJ_DECLARE(
  actions_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 6, AI_STATIC)

/**  Tensor declarations section  *********************************************/
/* Tensor #0 */
AI_TENSOR_OBJ_DECLARE(
  _Concat_output_0_output, AI_STATIC,
  0, 0x0,
  AI_SHAPE_INIT(4, 1, 35, 1, 1), AI_STRIDE_INIT(4, 4, 4, 140, 140),
  1, &_Concat_output_0_output_array, NULL)

/* Tensor #1 */
AI_TENSOR_OBJ_DECLARE(
  _Constant_4_output_0_2D, AI_STATIC,
  1, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &_Constant_4_output_0_2D_array, NULL)

/* Tensor #2 */
AI_TENSOR_OBJ_DECLARE(
  _Slice_output_0_output, AI_STATIC,
  2, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &_Slice_output_0_output_array, NULL)

/* Tensor #3 */
AI_TENSOR_OBJ_DECLARE(
  _encoder_encoder_0_Gemm_output_0_bias, AI_STATIC,
  3, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &_encoder_encoder_0_Gemm_output_0_bias_array, NULL)

/* Tensor #4 */
AI_TENSOR_OBJ_DECLARE(
  _encoder_encoder_0_Gemm_output_0_output, AI_STATIC,
  4, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &_encoder_encoder_0_Gemm_output_0_output_array, NULL)

/* Tensor #5 */
AI_TENSOR_OBJ_DECLARE(
  _encoder_encoder_0_Gemm_output_0_weights, AI_STATIC,
  5, 0x0,
  AI_SHAPE_INIT(4, 160, 128, 1, 1), AI_STRIDE_INIT(4, 4, 640, 81920, 81920),
  1, &_encoder_encoder_0_Gemm_output_0_weights_array, NULL)

/* Tensor #6 */
AI_TENSOR_OBJ_DECLARE(
  _encoder_encoder_1_1_Elu_output_0_output, AI_STATIC,
  6, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_encoder_encoder_1_1_Elu_output_0_output_array, NULL)

/* Tensor #7 */
AI_TENSOR_OBJ_DECLARE(
  _encoder_encoder_1_Elu_output_0_output, AI_STATIC,
  7, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &_encoder_encoder_1_Elu_output_0_output_array, NULL)

/* Tensor #8 */
AI_TENSOR_OBJ_DECLARE(
  _encoder_encoder_2_Gemm_output_0_bias, AI_STATIC,
  8, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_encoder_encoder_2_Gemm_output_0_bias_array, NULL)

/* Tensor #9 */
AI_TENSOR_OBJ_DECLARE(
  _encoder_encoder_2_Gemm_output_0_output, AI_STATIC,
  9, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_encoder_encoder_2_Gemm_output_0_output_array, NULL)

/* Tensor #10 */
AI_TENSOR_OBJ_DECLARE(
  _encoder_encoder_2_Gemm_output_0_weights, AI_STATIC,
  10, 0x0,
  AI_SHAPE_INIT(4, 128, 64, 1, 1), AI_STRIDE_INIT(4, 4, 512, 32768, 32768),
  1, &_encoder_encoder_2_Gemm_output_0_weights_array, NULL)

/* Tensor #11 */
AI_TENSOR_OBJ_DECLARE(
  _encoder_encoder_4_Gemm_output_0_bias, AI_STATIC,
  11, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 1, 1), AI_STRIDE_INIT(4, 4, 4, 12, 12),
  1, &_encoder_encoder_4_Gemm_output_0_bias_array, NULL)

/* Tensor #12 */
AI_TENSOR_OBJ_DECLARE(
  _encoder_encoder_4_Gemm_output_0_output, AI_STATIC,
  12, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 1, 1), AI_STRIDE_INIT(4, 4, 4, 12, 12),
  1, &_encoder_encoder_4_Gemm_output_0_output_array, NULL)

/* Tensor #13 */
AI_TENSOR_OBJ_DECLARE(
  _encoder_encoder_4_Gemm_output_0_weights, AI_STATIC,
  13, 0x0,
  AI_SHAPE_INIT(4, 64, 3, 1, 1), AI_STRIDE_INIT(4, 4, 256, 768, 768),
  1, &_encoder_encoder_4_Gemm_output_0_weights_array, NULL)

/* Tensor #14 */
AI_TENSOR_OBJ_DECLARE(
  _mlp_mlp_0_Gemm_output_0_bias, AI_STATIC,
  14, 0x0,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 4, 4, 1024, 1024),
  1, &_mlp_mlp_0_Gemm_output_0_bias_array, NULL)

/* Tensor #15 */
AI_TENSOR_OBJ_DECLARE(
  _mlp_mlp_0_Gemm_output_0_output, AI_STATIC,
  15, 0x0,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 4, 4, 1024, 1024),
  1, &_mlp_mlp_0_Gemm_output_0_output_array, NULL)

/* Tensor #16 */
AI_TENSOR_OBJ_DECLARE(
  _mlp_mlp_0_Gemm_output_0_weights, AI_STATIC,
  16, 0x0,
  AI_SHAPE_INIT(4, 35, 256, 1, 1), AI_STRIDE_INIT(4, 4, 140, 35840, 35840),
  1, &_mlp_mlp_0_Gemm_output_0_weights_array, NULL)

/* Tensor #17 */
AI_TENSOR_OBJ_DECLARE(
  _mlp_mlp_1_1_Elu_output_0_output, AI_STATIC,
  17, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &_mlp_mlp_1_1_Elu_output_0_output_array, NULL)

/* Tensor #18 */
AI_TENSOR_OBJ_DECLARE(
  _mlp_mlp_1_2_Elu_output_0_output, AI_STATIC,
  18, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_mlp_mlp_1_2_Elu_output_0_output_array, NULL)

/* Tensor #19 */
AI_TENSOR_OBJ_DECLARE(
  _mlp_mlp_1_Elu_output_0_output, AI_STATIC,
  19, 0x0,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 4, 4, 1024, 1024),
  1, &_mlp_mlp_1_Elu_output_0_output_array, NULL)

/* Tensor #20 */
AI_TENSOR_OBJ_DECLARE(
  _mlp_mlp_2_Gemm_output_0_bias, AI_STATIC,
  20, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &_mlp_mlp_2_Gemm_output_0_bias_array, NULL)

/* Tensor #21 */
AI_TENSOR_OBJ_DECLARE(
  _mlp_mlp_2_Gemm_output_0_output, AI_STATIC,
  21, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &_mlp_mlp_2_Gemm_output_0_output_array, NULL)

/* Tensor #22 */
AI_TENSOR_OBJ_DECLARE(
  _mlp_mlp_2_Gemm_output_0_weights, AI_STATIC,
  22, 0x0,
  AI_SHAPE_INIT(4, 256, 128, 1, 1), AI_STRIDE_INIT(4, 4, 1024, 131072, 131072),
  1, &_mlp_mlp_2_Gemm_output_0_weights_array, NULL)

/* Tensor #23 */
AI_TENSOR_OBJ_DECLARE(
  _mlp_mlp_4_Gemm_output_0_bias, AI_STATIC,
  23, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_mlp_mlp_4_Gemm_output_0_bias_array, NULL)

/* Tensor #24 */
AI_TENSOR_OBJ_DECLARE(
  _mlp_mlp_4_Gemm_output_0_output, AI_STATIC,
  24, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_mlp_mlp_4_Gemm_output_0_output_array, NULL)

/* Tensor #25 */
AI_TENSOR_OBJ_DECLARE(
  _mlp_mlp_4_Gemm_output_0_weights, AI_STATIC,
  25, 0x0,
  AI_SHAPE_INIT(4, 128, 64, 1, 1), AI_STRIDE_INIT(4, 4, 512, 32768, 32768),
  1, &_mlp_mlp_4_Gemm_output_0_weights_array, NULL)

/* Tensor #26 */
AI_TENSOR_OBJ_DECLARE(
  actions_bias, AI_STATIC,
  26, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 1, 1), AI_STRIDE_INIT(4, 4, 4, 24, 24),
  1, &actions_bias_array, NULL)

/* Tensor #27 */
AI_TENSOR_OBJ_DECLARE(
  actions_output, AI_STATIC,
  27, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 1, 1), AI_STRIDE_INIT(4, 4, 4, 24, 24),
  1, &actions_output_array, NULL)

/* Tensor #28 */
AI_TENSOR_OBJ_DECLARE(
  actions_weights, AI_STATIC,
  28, 0x0,
  AI_SHAPE_INIT(4, 64, 6, 1, 1), AI_STRIDE_INIT(4, 4, 256, 1536, 1536),
  1, &actions_weights_array, NULL)

/* Tensor #29 */
AI_TENSOR_OBJ_DECLARE(
  observation_history_output, AI_STATIC,
  29, 0x0,
  AI_SHAPE_INIT(4, 1, 160, 1, 1), AI_STRIDE_INIT(4, 4, 4, 640, 640),
  1, &observation_history_output_array, NULL)

/* Tensor #30 */
AI_TENSOR_OBJ_DECLARE(
  velocity_output, AI_STATIC,
  30, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 1, 1), AI_STRIDE_INIT(4, 4, 4, 12, 12),
  1, &velocity_output_array, NULL)



/**  Layer declarations section  **********************************************/


AI_TENSOR_CHAIN_OBJ_DECLARE(
  actions_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_mlp_mlp_1_2_Elu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &actions_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &actions_weights, &actions_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  actions_layer, 18,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense,
  &actions_chain,
  NULL, &actions_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_float _mlp_mlp_1_2_Elu_output_0_nl_params_data[] = { 1.0 };
AI_ARRAY_OBJ_DECLARE(
    _mlp_mlp_1_2_Elu_output_0_nl_params, AI_ARRAY_FORMAT_FLOAT,
    _mlp_mlp_1_2_Elu_output_0_nl_params_data, _mlp_mlp_1_2_Elu_output_0_nl_params_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  _mlp_mlp_1_2_Elu_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_mlp_mlp_4_Gemm_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_mlp_mlp_1_2_Elu_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _mlp_mlp_1_2_Elu_output_0_layer, 17,
  NL_TYPE, 0x0, NULL,
  nl, forward_elu,
  &_mlp_mlp_1_2_Elu_output_0_chain,
  NULL, &actions_layer, AI_STATIC, 
  .nl_params = &_mlp_mlp_1_2_Elu_output_0_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _mlp_mlp_4_Gemm_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_mlp_mlp_1_1_Elu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_mlp_mlp_4_Gemm_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_mlp_mlp_4_Gemm_output_0_weights, &_mlp_mlp_4_Gemm_output_0_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _mlp_mlp_4_Gemm_output_0_layer, 16,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense,
  &_mlp_mlp_4_Gemm_output_0_chain,
  NULL, &_mlp_mlp_1_2_Elu_output_0_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_float _mlp_mlp_1_1_Elu_output_0_nl_params_data[] = { 1.0 };
AI_ARRAY_OBJ_DECLARE(
    _mlp_mlp_1_1_Elu_output_0_nl_params, AI_ARRAY_FORMAT_FLOAT,
    _mlp_mlp_1_1_Elu_output_0_nl_params_data, _mlp_mlp_1_1_Elu_output_0_nl_params_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  _mlp_mlp_1_1_Elu_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_mlp_mlp_2_Gemm_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_mlp_mlp_1_1_Elu_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _mlp_mlp_1_1_Elu_output_0_layer, 15,
  NL_TYPE, 0x0, NULL,
  nl, forward_elu,
  &_mlp_mlp_1_1_Elu_output_0_chain,
  NULL, &_mlp_mlp_4_Gemm_output_0_layer, AI_STATIC, 
  .nl_params = &_mlp_mlp_1_1_Elu_output_0_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _mlp_mlp_2_Gemm_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_mlp_mlp_1_Elu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_mlp_mlp_2_Gemm_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_mlp_mlp_2_Gemm_output_0_weights, &_mlp_mlp_2_Gemm_output_0_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _mlp_mlp_2_Gemm_output_0_layer, 14,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense,
  &_mlp_mlp_2_Gemm_output_0_chain,
  NULL, &_mlp_mlp_1_1_Elu_output_0_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_float _mlp_mlp_1_Elu_output_0_nl_params_data[] = { 1.0 };
AI_ARRAY_OBJ_DECLARE(
    _mlp_mlp_1_Elu_output_0_nl_params, AI_ARRAY_FORMAT_FLOAT,
    _mlp_mlp_1_Elu_output_0_nl_params_data, _mlp_mlp_1_Elu_output_0_nl_params_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  _mlp_mlp_1_Elu_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_mlp_mlp_0_Gemm_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_mlp_mlp_1_Elu_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _mlp_mlp_1_Elu_output_0_layer, 13,
  NL_TYPE, 0x0, NULL,
  nl, forward_elu,
  &_mlp_mlp_1_Elu_output_0_chain,
  NULL, &_mlp_mlp_2_Gemm_output_0_layer, AI_STATIC, 
  .nl_params = &_mlp_mlp_1_Elu_output_0_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _mlp_mlp_0_Gemm_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Concat_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_mlp_mlp_0_Gemm_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_mlp_mlp_0_Gemm_output_0_weights, &_mlp_mlp_0_Gemm_output_0_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _mlp_mlp_0_Gemm_output_0_layer, 12,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense,
  &_mlp_mlp_0_Gemm_output_0_chain,
  NULL, &_mlp_mlp_1_Elu_output_0_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _Concat_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_Slice_output_0_output, &_encoder_encoder_4_Gemm_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Concat_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _Concat_output_0_layer, 11,
  CONCAT_TYPE, 0x0, NULL,
  concat, forward_concat,
  &_Concat_output_0_chain,
  NULL, &_mlp_mlp_0_Gemm_output_0_layer, AI_STATIC, 
  .axis = AI_SHAPE_CHANNEL, 
)


AI_STATIC_CONST ai_u8 _Slice_output_0_axes_data[] = { 2 };
AI_ARRAY_OBJ_DECLARE(
    _Slice_output_0_axes, AI_ARRAY_FORMAT_U8,
    _Slice_output_0_axes_data, _Slice_output_0_axes_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 _Slice_output_0_starts_data[] = { 128 };
AI_ARRAY_OBJ_DECLARE(
    _Slice_output_0_starts, AI_ARRAY_FORMAT_S16,
    _Slice_output_0_starts_data, _Slice_output_0_starts_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 _Slice_output_0_ends_data[] = { 160 };
AI_ARRAY_OBJ_DECLARE(
    _Slice_output_0_ends, AI_ARRAY_FORMAT_S16,
    _Slice_output_0_ends_data, _Slice_output_0_ends_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  _Slice_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &observation_history_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Slice_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _Slice_output_0_layer, 5,
  SLICE_TYPE, 0x0, NULL,
  slice, forward_slice,
  &_Slice_output_0_chain,
  NULL, &_Concat_output_0_layer, AI_STATIC, 
  .axes = &_Slice_output_0_axes, 
  .starts = &_Slice_output_0_starts, 
  .ends = &_Slice_output_0_ends, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  velocity_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_encoder_encoder_4_Gemm_output_0_output, &_Constant_4_output_0_2D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &velocity_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  velocity_layer, 20,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &velocity_chain,
  NULL, &_Slice_output_0_layer, AI_STATIC, 
  .operation = ai_div_f32, 
  .buffer_operation = ai_div_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _encoder_encoder_4_Gemm_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_encoder_encoder_1_1_Elu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_encoder_encoder_4_Gemm_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_encoder_encoder_4_Gemm_output_0_weights, &_encoder_encoder_4_Gemm_output_0_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _encoder_encoder_4_Gemm_output_0_layer, 10,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense,
  &_encoder_encoder_4_Gemm_output_0_chain,
  NULL, &velocity_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_float _encoder_encoder_1_1_Elu_output_0_nl_params_data[] = { 1.0 };
AI_ARRAY_OBJ_DECLARE(
    _encoder_encoder_1_1_Elu_output_0_nl_params, AI_ARRAY_FORMAT_FLOAT,
    _encoder_encoder_1_1_Elu_output_0_nl_params_data, _encoder_encoder_1_1_Elu_output_0_nl_params_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  _encoder_encoder_1_1_Elu_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_encoder_encoder_2_Gemm_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_encoder_encoder_1_1_Elu_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _encoder_encoder_1_1_Elu_output_0_layer, 9,
  NL_TYPE, 0x0, NULL,
  nl, forward_elu,
  &_encoder_encoder_1_1_Elu_output_0_chain,
  NULL, &_encoder_encoder_4_Gemm_output_0_layer, AI_STATIC, 
  .nl_params = &_encoder_encoder_1_1_Elu_output_0_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _encoder_encoder_2_Gemm_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_encoder_encoder_1_Elu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_encoder_encoder_2_Gemm_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_encoder_encoder_2_Gemm_output_0_weights, &_encoder_encoder_2_Gemm_output_0_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _encoder_encoder_2_Gemm_output_0_layer, 8,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense,
  &_encoder_encoder_2_Gemm_output_0_chain,
  NULL, &_encoder_encoder_1_1_Elu_output_0_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_float _encoder_encoder_1_Elu_output_0_nl_params_data[] = { 1.0 };
AI_ARRAY_OBJ_DECLARE(
    _encoder_encoder_1_Elu_output_0_nl_params, AI_ARRAY_FORMAT_FLOAT,
    _encoder_encoder_1_Elu_output_0_nl_params_data, _encoder_encoder_1_Elu_output_0_nl_params_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  _encoder_encoder_1_Elu_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_encoder_encoder_0_Gemm_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_encoder_encoder_1_Elu_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _encoder_encoder_1_Elu_output_0_layer, 7,
  NL_TYPE, 0x0, NULL,
  nl, forward_elu,
  &_encoder_encoder_1_Elu_output_0_chain,
  NULL, &_encoder_encoder_2_Gemm_output_0_layer, AI_STATIC, 
  .nl_params = &_encoder_encoder_1_Elu_output_0_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _encoder_encoder_0_Gemm_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &observation_history_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_encoder_encoder_0_Gemm_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_encoder_encoder_0_Gemm_output_0_weights, &_encoder_encoder_0_Gemm_output_0_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _encoder_encoder_0_Gemm_output_0_layer, 6,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense,
  &_encoder_encoder_0_Gemm_output_0_chain,
  NULL, &_encoder_encoder_1_Elu_output_0_layer, AI_STATIC, 
)


#if (AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 319272, 1, 1),
    319272, NULL, NULL),
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 1548, 1, 1),
    1548, NULL, NULL),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORKZN1_IN_NUM, &observation_history_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORKZN1_OUT_NUM, &actions_output, &velocity_output),
  &_encoder_encoder_0_Gemm_output_0_layer, 0x6d393f1f, NULL)

#else

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 319272, 1, 1),
      319272, NULL, NULL)
  ),
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 1548, 1, 1),
      1548, NULL, NULL)
  ),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORKZN1_IN_NUM, &observation_history_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORKZN1_OUT_NUM, &actions_output, &velocity_output),
  &_encoder_encoder_0_Gemm_output_0_layer, 0x6d393f1f, NULL)

#endif	/*(AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)*/



/******************************************************************************/
AI_DECLARE_STATIC
ai_bool networkzn1_configure_activations(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  if (ai_platform_get_activations_map(g_networkzn1_activations_map, 1, params)) {
    /* Updating activations (byte) offsets */
    
    observation_history_output_array.data = AI_PTR(g_networkzn1_activations_map[0] + 140);
    observation_history_output_array.data_start = AI_PTR(g_networkzn1_activations_map[0] + 140);
    _encoder_encoder_0_Gemm_output_0_output_array.data = AI_PTR(g_networkzn1_activations_map[0] + 780);
    _encoder_encoder_0_Gemm_output_0_output_array.data_start = AI_PTR(g_networkzn1_activations_map[0] + 780);
    _encoder_encoder_1_Elu_output_0_output_array.data = AI_PTR(g_networkzn1_activations_map[0] + 780);
    _encoder_encoder_1_Elu_output_0_output_array.data_start = AI_PTR(g_networkzn1_activations_map[0] + 780);
    _encoder_encoder_2_Gemm_output_0_output_array.data = AI_PTR(g_networkzn1_activations_map[0] + 1292);
    _encoder_encoder_2_Gemm_output_0_output_array.data_start = AI_PTR(g_networkzn1_activations_map[0] + 1292);
    _encoder_encoder_1_1_Elu_output_0_output_array.data = AI_PTR(g_networkzn1_activations_map[0] + 1292);
    _encoder_encoder_1_1_Elu_output_0_output_array.data_start = AI_PTR(g_networkzn1_activations_map[0] + 1292);
    _encoder_encoder_4_Gemm_output_0_output_array.data = AI_PTR(g_networkzn1_activations_map[0] + 1280);
    _encoder_encoder_4_Gemm_output_0_output_array.data_start = AI_PTR(g_networkzn1_activations_map[0] + 1280);
    velocity_output_array.data = AI_PTR(g_networkzn1_activations_map[0] + 1536);
    velocity_output_array.data_start = AI_PTR(g_networkzn1_activations_map[0] + 1536);
    _Slice_output_0_output_array.data = AI_PTR(g_networkzn1_activations_map[0] + 1408);
    _Slice_output_0_output_array.data_start = AI_PTR(g_networkzn1_activations_map[0] + 1408);
    _Concat_output_0_output_array.data = AI_PTR(g_networkzn1_activations_map[0] + 140);
    _Concat_output_0_output_array.data_start = AI_PTR(g_networkzn1_activations_map[0] + 140);
    _mlp_mlp_0_Gemm_output_0_output_array.data = AI_PTR(g_networkzn1_activations_map[0] + 512);
    _mlp_mlp_0_Gemm_output_0_output_array.data_start = AI_PTR(g_networkzn1_activations_map[0] + 512);
    _mlp_mlp_1_Elu_output_0_output_array.data = AI_PTR(g_networkzn1_activations_map[0] + 512);
    _mlp_mlp_1_Elu_output_0_output_array.data_start = AI_PTR(g_networkzn1_activations_map[0] + 512);
    _mlp_mlp_2_Gemm_output_0_output_array.data = AI_PTR(g_networkzn1_activations_map[0] + 0);
    _mlp_mlp_2_Gemm_output_0_output_array.data_start = AI_PTR(g_networkzn1_activations_map[0] + 0);
    _mlp_mlp_1_1_Elu_output_0_output_array.data = AI_PTR(g_networkzn1_activations_map[0] + 512);
    _mlp_mlp_1_1_Elu_output_0_output_array.data_start = AI_PTR(g_networkzn1_activations_map[0] + 512);
    _mlp_mlp_4_Gemm_output_0_output_array.data = AI_PTR(g_networkzn1_activations_map[0] + 0);
    _mlp_mlp_4_Gemm_output_0_output_array.data_start = AI_PTR(g_networkzn1_activations_map[0] + 0);
    _mlp_mlp_1_2_Elu_output_0_output_array.data = AI_PTR(g_networkzn1_activations_map[0] + 256);
    _mlp_mlp_1_2_Elu_output_0_output_array.data_start = AI_PTR(g_networkzn1_activations_map[0] + 256);
    actions_output_array.data = AI_PTR(g_networkzn1_activations_map[0] + 0);
    actions_output_array.data_start = AI_PTR(g_networkzn1_activations_map[0] + 0);
    return true;
  }
  AI_ERROR_TRAP(net_ctx, INIT_FAILED, NETWORK_ACTIVATIONS);
  return false;
}




/******************************************************************************/
AI_DECLARE_STATIC
ai_bool networkzn1_configure_weights(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  if (ai_platform_get_weights_map(g_networkzn1_weights_map, 1, params)) {
    /* Updating weights (byte) offsets */
    
    _Constant_4_output_0_2D_array.format |= AI_FMT_FLAG_CONST;
    _Constant_4_output_0_2D_array.data = AI_PTR(g_networkzn1_weights_map[0] + 0);
    _Constant_4_output_0_2D_array.data_start = AI_PTR(g_networkzn1_weights_map[0] + 0);
    _encoder_encoder_0_Gemm_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _encoder_encoder_0_Gemm_output_0_weights_array.data = AI_PTR(g_networkzn1_weights_map[0] + 4);
    _encoder_encoder_0_Gemm_output_0_weights_array.data_start = AI_PTR(g_networkzn1_weights_map[0] + 4);
    _encoder_encoder_0_Gemm_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _encoder_encoder_0_Gemm_output_0_bias_array.data = AI_PTR(g_networkzn1_weights_map[0] + 81924);
    _encoder_encoder_0_Gemm_output_0_bias_array.data_start = AI_PTR(g_networkzn1_weights_map[0] + 81924);
    _encoder_encoder_2_Gemm_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _encoder_encoder_2_Gemm_output_0_weights_array.data = AI_PTR(g_networkzn1_weights_map[0] + 82436);
    _encoder_encoder_2_Gemm_output_0_weights_array.data_start = AI_PTR(g_networkzn1_weights_map[0] + 82436);
    _encoder_encoder_2_Gemm_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _encoder_encoder_2_Gemm_output_0_bias_array.data = AI_PTR(g_networkzn1_weights_map[0] + 115204);
    _encoder_encoder_2_Gemm_output_0_bias_array.data_start = AI_PTR(g_networkzn1_weights_map[0] + 115204);
    _encoder_encoder_4_Gemm_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _encoder_encoder_4_Gemm_output_0_weights_array.data = AI_PTR(g_networkzn1_weights_map[0] + 115460);
    _encoder_encoder_4_Gemm_output_0_weights_array.data_start = AI_PTR(g_networkzn1_weights_map[0] + 115460);
    _encoder_encoder_4_Gemm_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _encoder_encoder_4_Gemm_output_0_bias_array.data = AI_PTR(g_networkzn1_weights_map[0] + 116228);
    _encoder_encoder_4_Gemm_output_0_bias_array.data_start = AI_PTR(g_networkzn1_weights_map[0] + 116228);
    _mlp_mlp_0_Gemm_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _mlp_mlp_0_Gemm_output_0_weights_array.data = AI_PTR(g_networkzn1_weights_map[0] + 116240);
    _mlp_mlp_0_Gemm_output_0_weights_array.data_start = AI_PTR(g_networkzn1_weights_map[0] + 116240);
    _mlp_mlp_0_Gemm_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _mlp_mlp_0_Gemm_output_0_bias_array.data = AI_PTR(g_networkzn1_weights_map[0] + 152080);
    _mlp_mlp_0_Gemm_output_0_bias_array.data_start = AI_PTR(g_networkzn1_weights_map[0] + 152080);
    _mlp_mlp_2_Gemm_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _mlp_mlp_2_Gemm_output_0_weights_array.data = AI_PTR(g_networkzn1_weights_map[0] + 153104);
    _mlp_mlp_2_Gemm_output_0_weights_array.data_start = AI_PTR(g_networkzn1_weights_map[0] + 153104);
    _mlp_mlp_2_Gemm_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _mlp_mlp_2_Gemm_output_0_bias_array.data = AI_PTR(g_networkzn1_weights_map[0] + 284176);
    _mlp_mlp_2_Gemm_output_0_bias_array.data_start = AI_PTR(g_networkzn1_weights_map[0] + 284176);
    _mlp_mlp_4_Gemm_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _mlp_mlp_4_Gemm_output_0_weights_array.data = AI_PTR(g_networkzn1_weights_map[0] + 284688);
    _mlp_mlp_4_Gemm_output_0_weights_array.data_start = AI_PTR(g_networkzn1_weights_map[0] + 284688);
    _mlp_mlp_4_Gemm_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _mlp_mlp_4_Gemm_output_0_bias_array.data = AI_PTR(g_networkzn1_weights_map[0] + 317456);
    _mlp_mlp_4_Gemm_output_0_bias_array.data_start = AI_PTR(g_networkzn1_weights_map[0] + 317456);
    actions_weights_array.format |= AI_FMT_FLAG_CONST;
    actions_weights_array.data = AI_PTR(g_networkzn1_weights_map[0] + 317712);
    actions_weights_array.data_start = AI_PTR(g_networkzn1_weights_map[0] + 317712);
    actions_bias_array.format |= AI_FMT_FLAG_CONST;
    actions_bias_array.data = AI_PTR(g_networkzn1_weights_map[0] + 319248);
    actions_bias_array.data_start = AI_PTR(g_networkzn1_weights_map[0] + 319248);
    return true;
  }
  AI_ERROR_TRAP(net_ctx, INIT_FAILED, NETWORK_WEIGHTS);
  return false;
}


/**  PUBLIC APIs SECTION  *****************************************************/



AI_DEPRECATED
AI_API_ENTRY
ai_bool ai_networkzn1_get_info(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if (report && net_ctx)
  {
    ai_network_report r = {
      .model_name        = AI_NETWORKZN1_MODEL_NAME,
      .model_signature   = AI_NETWORKZN1_MODEL_SIGNATURE,
      .model_datetime    = AI_TOOLS_DATE_TIME,
      
      .compile_datetime  = AI_TOOLS_COMPILE_TIME,
      
      .runtime_revision  = ai_platform_runtime_get_revision(),
      .runtime_version   = ai_platform_runtime_get_version(),

      .tool_revision     = AI_TOOLS_REVISION_ID,
      .tool_version      = {AI_TOOLS_VERSION_MAJOR, AI_TOOLS_VERSION_MINOR,
                            AI_TOOLS_VERSION_MICRO, 0x0},
      .tool_api_version  = AI_STRUCT_INIT,

      .api_version            = ai_platform_api_get_version(),
      .interface_api_version  = ai_platform_interface_api_get_version(),
      
      .n_macc            = 86872,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .params            = AI_STRUCT_INIT,
      .activations       = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0x6d393f1f,
    };

    if (!ai_platform_api_get_network_report(network, &r)) return false;

    *report = r;
    return true;
  }
  return false;
}



AI_API_ENTRY
ai_bool ai_networkzn1_get_report(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if (report && net_ctx)
  {
    ai_network_report r = {
      .model_name        = AI_NETWORKZN1_MODEL_NAME,
      .model_signature   = AI_NETWORKZN1_MODEL_SIGNATURE,
      .model_datetime    = AI_TOOLS_DATE_TIME,
      
      .compile_datetime  = AI_TOOLS_COMPILE_TIME,
      
      .runtime_revision  = ai_platform_runtime_get_revision(),
      .runtime_version   = ai_platform_runtime_get_version(),

      .tool_revision     = AI_TOOLS_REVISION_ID,
      .tool_version      = {AI_TOOLS_VERSION_MAJOR, AI_TOOLS_VERSION_MINOR,
                            AI_TOOLS_VERSION_MICRO, 0x0},
      .tool_api_version  = AI_STRUCT_INIT,

      .api_version            = ai_platform_api_get_version(),
      .interface_api_version  = ai_platform_interface_api_get_version(),
      
      .n_macc            = 86872,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .map_signature     = AI_MAGIC_SIGNATURE,
      .map_weights       = AI_STRUCT_INIT,
      .map_activations   = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0x6d393f1f,
    };

    if (!ai_platform_api_get_network_report(network, &r)) return false;

    *report = r;
    return true;
  }
  return false;
}


AI_API_ENTRY
ai_error ai_networkzn1_get_error(ai_handle network)
{
  return ai_platform_network_get_error(network);
}


AI_API_ENTRY
ai_error ai_networkzn1_create(
  ai_handle* network, const ai_buffer* network_config)
{
  return ai_platform_network_create(
    network, network_config, 
    AI_CONTEXT_OBJ(&AI_NET_OBJ_INSTANCE),
    AI_TOOLS_API_VERSION_MAJOR, AI_TOOLS_API_VERSION_MINOR, AI_TOOLS_API_VERSION_MICRO);
}


AI_API_ENTRY
ai_error ai_networkzn1_create_and_init(
  ai_handle* network, const ai_handle activations[], const ai_handle weights[])
{
  ai_error err;
  ai_network_params params;

  err = ai_networkzn1_create(network, AI_NETWORKZN1_DATA_CONFIG);
  if (err.type != AI_ERROR_NONE) {
    return err;
  }
  
  if (ai_networkzn1_data_params_get(&params) != true) {
    err = ai_networkzn1_get_error(*network);
    return err;
  }
#if defined(AI_NETWORKZN1_DATA_ACTIVATIONS_COUNT)
  /* set the addresses of the activations buffers */
  for (ai_u16 idx=0; activations && idx<params.map_activations.size; idx++) {
    AI_BUFFER_ARRAY_ITEM_SET_ADDRESS(&params.map_activations, idx, activations[idx]);
  }
#endif
#if defined(AI_NETWORKZN1_DATA_WEIGHTS_COUNT)
  /* set the addresses of the weight buffers */
  for (ai_u16 idx=0; weights && idx<params.map_weights.size; idx++) {
    AI_BUFFER_ARRAY_ITEM_SET_ADDRESS(&params.map_weights, idx, weights[idx]);
  }
#endif
  if (ai_networkzn1_init(*network, &params) != true) {
    err = ai_networkzn1_get_error(*network);
  }
  return err;
}


AI_API_ENTRY
ai_buffer* ai_networkzn1_inputs_get(ai_handle network, ai_u16 *n_buffer)
{
  if (network == AI_HANDLE_NULL) {
    network = (ai_handle)&AI_NET_OBJ_INSTANCE;
    AI_NETWORK_OBJ(network)->magic = AI_MAGIC_CONTEXT_TOKEN;
  }
  return ai_platform_inputs_get(network, n_buffer);
}


AI_API_ENTRY
ai_buffer* ai_networkzn1_outputs_get(ai_handle network, ai_u16 *n_buffer)
{
  if (network == AI_HANDLE_NULL) {
    network = (ai_handle)&AI_NET_OBJ_INSTANCE;
    AI_NETWORK_OBJ(network)->magic = AI_MAGIC_CONTEXT_TOKEN;
  }
  return ai_platform_outputs_get(network, n_buffer);
}


AI_API_ENTRY
ai_handle ai_networkzn1_destroy(ai_handle network)
{
  return ai_platform_network_destroy(network);
}


AI_API_ENTRY
ai_bool ai_networkzn1_init(
  ai_handle network, const ai_network_params* params)
{
  ai_network* net_ctx = AI_NETWORK_OBJ(ai_platform_network_init(network, params));
  ai_bool ok = true;

  if (!net_ctx) return false;
  ok &= networkzn1_configure_weights(net_ctx, params);
  ok &= networkzn1_configure_activations(net_ctx, params);

  ok &= ai_platform_network_post_init(network);

  return ok;
}


AI_API_ENTRY
ai_i32 ai_networkzn1_run(
  ai_handle network, const ai_buffer* input, ai_buffer* output)
{
  return ai_platform_network_process(network, input, output);
}


AI_API_ENTRY
ai_i32 ai_networkzn1_forward(ai_handle network, const ai_buffer* input)
{
  return ai_platform_network_process(network, input, NULL);
}



#undef AI_NETWORKZN1_MODEL_SIGNATURE
#undef AI_NET_OBJ_INSTANCE
#undef AI_TOOLS_DATE_TIME
#undef AI_TOOLS_COMPILE_TIME

