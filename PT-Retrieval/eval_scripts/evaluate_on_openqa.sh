model_cfg=bert-base-uncased
filename=ptv2-dpr-multidata-128-40-8e-3-64
checkpoint=dpr_biencoder.39.1045
psl=64

dataset=$1
topk=$2

if [ "$dataset" = "curatedtrec" ]; then
    match="regex"
else
    match="string"
fi

python3 dense_retriever.py \
	--pretrained_model_cfg $model_cfg \
	--model_file checkpoints/$filename/$checkpoint \
	--ctx_file data/wikipedia_split/psgs_w100.tsv \
	--qa_file data/retriever/qas/$dataset-test.csv \
	--encoded_ctx_file "encoded_files/encoded-wiki-$filename-*.pkl" \
	--n-docs $topk \
	--batch_size 128 \
	--sequence_length 256 \
	--prefix --pre_seq_len $psl --match $match

# add "--match regex" for CuratedTREC 