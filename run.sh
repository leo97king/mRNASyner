cd CDS_opt
python cds_opt.py --model_mode test --input_file ./data/ --out_file ./result/

cd ../UTR_gen
python cds2utr_test.py --in_dir ../CDS_opt/result/ --out_dir ./result/

cd ../RibonanzaNet-Deg
python predict.py