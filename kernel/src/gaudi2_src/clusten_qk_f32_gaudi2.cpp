/**********************************************************************
Copyright (c) 2022 Habana Labs.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

*   Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
*   Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
********************************************************************/

#include <vector>
#include <cstring>
#include <iostream>
#include "clusten_qk_f32_gaudi2.hpp"

extern unsigned char _binary___clusten_qk_fwd_f32_gaudi2_o_start;
extern unsigned char _binary___clusten_qk_fwd_f32_gaudi2_o_end;
extern unsigned char _binary___clusten_qk_bwd_f32_gaudi2_o_start;
extern unsigned char _binary___clusten_qk_bwd_f32_gaudi2_o_end;

 tpc_lib_api::GlueCodeReturn CLUSTENQKF32Gaudi2::GetKernelName(
             char kernelName [tpc_lib_api::MAX_NODE_NAME])
 {

    if(m_mode == fwd)
        strcpy(kernelName,"clusten_qk_fwd_f32_gaudi2");
    else if(m_mode == bwd)
        strcpy(kernelName,"clusten_qk_bwd_f32_gaudi2");
    else
        return tpc_lib_api::GLUE_NODE_NOT_FOUND;
     return tpc_lib_api::GLUE_SUCCESS;
 }


tpc_lib_api::GlueCodeReturn CLUSTENQKF32Gaudi2::GetGcDefinitions(
            tpc_lib_api::HabanaKernelParams* in_defs,
            tpc_lib_api::HabanaKernelInstantiation* out_defs)
{
    
    tpc_lib_api::GlueCodeReturn retVal;
    /*************************************************************************************
    *   Stage I - validate input
    **************************************************************************************/
    //validate correct amount of input tensors
    if (m_mode == fwd)
    {
        if (in_defs->inputTensorNr != 3)
        {
            in_defs->inputTensorNr  = 3;
            return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_COUNT;
        }
        if (in_defs->outputTensorNr !=1)
        {
            in_defs->outputTensorNr  = 1;
            return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_COUNT;
        }

        if (in_defs->inputTensors[0].geometry.dataType != tpc_lib_api::DATA_F32 ||
            in_defs->inputTensors[1].geometry.dataType != tpc_lib_api::DATA_F32 ||
            in_defs->inputTensors[2].geometry.dataType != tpc_lib_api::DATA_I32 ||
            in_defs->outputTensors[0].geometry.dataType != tpc_lib_api::DATA_F32)
        {
            in_defs->inputTensors[0].geometry.dataType = tpc_lib_api::DATA_F32;
            in_defs->inputTensors[1].geometry.dataType = tpc_lib_api::DATA_F32;
            in_defs->inputTensors[2].geometry.dataType = tpc_lib_api::DATA_I32;
            in_defs->outputTensors[0].geometry.dataType = tpc_lib_api::DATA_F32;
            return tpc_lib_api::GLUE_INCOMPATIBLE_DATA_TYPE;
        }
    }
    else if (m_mode == bwd)
    {
        if (in_defs->inputTensorNr != 4)
        {
            in_defs->inputTensorNr  = 4;
            return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_COUNT;
        }
        if (in_defs->outputTensorNr !=2)
        {
            in_defs->outputTensorNr = 2;
            return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_COUNT;
        }

        if (in_defs->inputTensors[0].geometry.dataType != tpc_lib_api::DATA_F32 ||
            in_defs->inputTensors[1].geometry.dataType != tpc_lib_api::DATA_F32 ||
            in_defs->inputTensors[2].geometry.dataType != tpc_lib_api::DATA_F32 ||
            in_defs->inputTensors[3].geometry.dataType != tpc_lib_api::DATA_I32 ||
            in_defs->outputTensors[0].geometry.dataType != tpc_lib_api::DATA_F32 ||
            in_defs->outputTensors[1].geometry.dataType != tpc_lib_api::DATA_F32)
        {
            in_defs->inputTensors[0].geometry.dataType = tpc_lib_api::DATA_F32;
            in_defs->inputTensors[1].geometry.dataType = tpc_lib_api::DATA_F32;
            in_defs->inputTensors[2].geometry.dataType = tpc_lib_api::DATA_F32;
            in_defs->inputTensors[3].geometry.dataType = tpc_lib_api::DATA_I32;
            in_defs->outputTensors[0].geometry.dataType = tpc_lib_api::DATA_F32;
            in_defs->outputTensors[1].geometry.dataType = tpc_lib_api::DATA_F32;
            return tpc_lib_api::GLUE_INCOMPATIBLE_DATA_TYPE;
        }
        in_defs->outputTensors[1].pData = (float*)calloc(in_defs->outputTensors[1].geometry.dims[0] * in_defs->outputTensors[1].geometry.dims[1] * in_defs->outputTensors[1].geometry.dims[2] * in_defs->outputTensors[1].geometry.dims[3], sizeof(float));
    }
    else
    {
        return tpc_lib_api::GLUE_NODE_NOT_FOUND;
    }
    
    

    
    

    // Tensor 0 should be input feature map.
    // The semantics of the input tensors and their order is a convention
    // between TPC kernel writer and the write of the layer at the
    // framework level.
    /*************************************************************************************
    *    Stage II -  Define index space geometry. In this example the index space matches
    *    the dimensions of the output tensor, up to dim 0.
    **************************************************************************************/
    out_defs->indexSpaceRank = 3;

    if(m_mode == fwd)
    {
        uint64_t* querySizes = in_defs->inputTensors[0].geometry.maxSizes;
        uint64_t* nbhdIndexSizes = in_defs->inputTensors[2].geometry.maxSizes;
        out_defs->indexSpaceGeometry[0] = nbhdIndexSizes[0]; // nbhd
        out_defs->indexSpaceGeometry[1] = querySizes[1]; // seq
        out_defs->indexSpaceGeometry[2] = querySizes[2] * querySizes[3]; // batch * head
    }
    else
    {
        uint64_t* querySizes = in_defs->inputTensors[1].geometry.maxSizes;
        out_defs->indexSpaceGeometry[0] = querySizes[0]; // channel 
        out_defs->indexSpaceGeometry[1] = querySizes[1]; // seq
        out_defs->indexSpaceGeometry[2] = querySizes[2] * querySizes[3]; // batch * head
    }

    /*************************************************************************************
    *    Stage III -  Define index space mapping
    **************************************************************************************/
    for (unsigned i = 0; i < in_defs->inputTensorNr; i++)
    {
        for (unsigned j = 0; j < out_defs->indexSpaceRank; j++)
        {
            out_defs->inputTensorAccessPattern[i].mapping[j].indexSpaceDim = 0;
            out_defs->inputTensorAccessPattern[i].mapping[j].a = 0;
            out_defs->inputTensorAccessPattern[i].mapping[j].start_b = 0;
            out_defs->inputTensorAccessPattern[i].mapping[j].end_b   = 0;
        }
    }
    for (unsigned i = 0; i < in_defs->inputTensorNr; i++)
    {
        for (unsigned j = 0; j < out_defs->indexSpaceRank; j++)
        {
            out_defs->outputTensorAccessPattern[i].mapping[j].indexSpaceDim = 0;
            out_defs->outputTensorAccessPattern[i].mapping[j].a = 0;
            out_defs->outputTensorAccessPattern[i].mapping[j].start_b = 0;
            out_defs->outputTensorAccessPattern[i].mapping[j].end_b   = 0;
        }
    }

    /*************************************************************************************
    *    Stage IV -  define scalar parameters/Set Auxiliary Tensor
    **************************************************************************************/
    out_defs->kernel.paramsNr =0;

    /*************************************************************************************
    *    Stage V -  Load ISA into the descriptor.
    **************************************************************************************/
    unsigned IsaSize = (&_binary___clusten_qk_fwd_f32_gaudi2_o_end - &_binary___clusten_qk_fwd_f32_gaudi2_o_start);
    unsigned char *binary_kernel = &_binary___clusten_qk_fwd_f32_gaudi2_o_start;
    switch (m_mode){
        case fwd:
            IsaSize = (&_binary___clusten_qk_fwd_f32_gaudi2_o_end - &_binary___clusten_qk_fwd_f32_gaudi2_o_start);
            binary_kernel = &_binary___clusten_qk_fwd_f32_gaudi2_o_start;
            break;
        case bwd:
            IsaSize = (&_binary___clusten_qk_bwd_f32_gaudi2_o_end - &_binary___clusten_qk_bwd_f32_gaudi2_o_start);
            binary_kernel = &_binary___clusten_qk_bwd_f32_gaudi2_o_start;
            break;
        default:
            break;

    }
    unsigned givenBinarySize = out_defs->kernel.elfSize;
    out_defs->kernel.elfSize = IsaSize;

    if (givenBinarySize >= IsaSize)
    {
        // copy binary out
        memcpy (out_defs->kernel.kernelElf,
                binary_kernel,
                IsaSize);
    }
    else
    {
       retVal = tpc_lib_api::GLUE_INSUFFICIENT_ELF_BUFFER;
       return retVal;
    }

    return tpc_lib_api::GLUE_SUCCESS;
}

