## URIAL with VLLM 

```bash
conda create -n vllm python=3.10
conda activate vllm
pip install vllm
pip install datasets

# https://docs.vllm.ai/en/latest/models/engine_args.html

python -m vllm.entrypoints.api_server \
        --download-dir /net/nfs/s2-research/llama2/ \
        --host 127.0.0.1 --port 2333 \
        --model meta-llama/Llama-2-70b-hf --tensor-parallel-size 2 --dtype bfloat16 

python src/vllm_infer/vllm_urial.py \
    --data_name alpaca_eval \
    --model_name Llama-2-70b-hf \
    --urial_name inst_help \
    --top_p 1 --temperature 0 --repetition_penalty 1.1


python src/vllm_infer/vllm_urial.py \
    --data_name alpaca_eval \
    --model_name Llama-2-70b-hf \
    --urial_name inst_1k \
    --top_p 0.9 --temperature 0.7 --repetition_penalty 1.1
```


## Using Mixtral 8x7b for AlpacaEval 


```bash 
python -m vllm.entrypoints.api_server \
        --download-dir /net/nfs/s2-research/llama2/ \
        --host 127.0.0.1 --port 2333 \
        --model mistralai/Mixtral-8x7B-v0.1 --tensor-parallel-size 8

python src/vllm_infer/vllm_urial.py \
    --data_name alpaca_eval \
    --vllm_url http://localhost:2333/generate \
    --model_name Mixtral-8x7B-v0.1 \
    --urial_name inst_help \
    --top_p 1 --temperature 0
```