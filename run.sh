python -m bert.bert_original > out/bert.bert_original.log

nsys profile -o 0919 --force-overwrite true python -m bert.bert_select_n

nsys profile -o 0919 --force-overwrite true python -m gpt2.gpt2_slice
nsys profile -o 1011 --force-overwrite true python -m gpt2.gpt2_select_n

# output/xxxx_activation.log 里记录了每个层的激活值