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

void main(
    const tensor d_attn,         // b x h x n x m
    const tensor query,          // b x h x n x c
    const tensor key,            // b x h x c x n (reordered by cluster)
    const tensor nbhd_idx,       // b x n x m
    tensor d_query,              // b x h x n x c
    tensor d_key)                // b x h x c x n
{
    const int dim = get_dim_size(query, 0);
    const int length = get_dim_size(query, 1);
    const int heads = get_dim_size(query, 2);
    const int batch_size = get_dim_size(query, 3);
    const int nbhd_size = get_dim_size(nbhd_idx, 0);

    const int d_key_numel = batch_size * heads * dim * length;

    const int channel = 0;
    const int seq = 1;
    const int batch_head = 2;

    const int5 index_space_start = get_index_space_offset();
    const int5 index_space_end   = get_index_space_size() + index_space_start;

    /* CWHB */
    const int channel_step  = 1;
    const int channel_start = index_space_start[channel] * channel_step;
    const int channel_end   = index_space_end[channel] * channel_step;

    const int seq_step  = 1;
    const int seq_start = index_space_start[seq] * seq_step;
    const int seq_end   = index_space_end[seq] * seq_step;

    const int batch_head_step  = 1;
    const int batch_head_start = index_space_start[batch_head] * batch_head_step;
    const int batch_head_end   = index_space_end[batch_head] * batch_head_step;

    #pragma loop_taken
    for (int z = batch_head_start; z < batch_head_end; z += batch_head_step)
    {
        #pragma loop_taken
        for (int c = channel_start; c < channel_end; c += channel_step)
        {
            #pragma loop_taken
            for (int i = seq_start; i < seq_end; i += seq_step)
            {
                const int b = z / heads;
                const int h = z - b * heads;

                long int index;
                float dq_update = 0.0;
                float d_attn_tmp;

                int5 q_coords = {c, i, h, b, 0};
                float* q_addr = (float*)gen_addr(q_coords, query);
                float q_val = s_f32_ld_l(q_addr);

                #pragma unroll
                for (unsigned int ni=0; ni < nbhd_size; ++ni) {
                    int5 nbi_coords = {ni, i, b, 0, 0};
                    int* nbi_addr = (int*)gen_addr(nbi_coords, nbhd_idx);
                    long int nbi = s_i32_ld_l(nbi_addr); 

                    // calculate d_query = key * d_att
                    // calculate d_key = query * d_att
                    int5 da_coords = {ni, i, h, b, 0};
                    float* da_addr = (float*)gen_addr(da_coords, d_attn);
                    d_attn_tmp = s_f32_ld_l(da_addr);
                    int5 k_coords = {c, nbi, h, b, 0};
                    float* k_addr = (float*)gen_addr(k_coords, key);
                    dq_update += s_f32_ld_l(k_addr) * d_attn_tmp;

                    float dk_add = q_val * d_attn_tmp;

                    float* dk_addr = (float*)gen_addr(k_coords, d_key);
                    s_f32_st_l(dk_addr, dk_add + s_f32_ld_l(dk_addr));
                }
                float* dq_addr = (float*)gen_addr(q_coords, d_query);
                s_f32_st_l(dq_addr, dq_update);
            }
        }
    }
}