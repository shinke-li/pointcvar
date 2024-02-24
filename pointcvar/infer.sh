if [ ! -d "output" ]; then
mkdir "output"
fi

for model in 'dgcnn'; do 

for cor in 'mn40_dgcnn_add_chamfer'; do 

for sev in  5; do

CUDA_VISIBLE_DEVICES=2 python main.py --entry test --model-path /media/mldadmin/home/s122mdg33_07/lujc/check_code/CVaR/pointcvar/runs/dgcnn_dgcnn_run_1/model_best_test.pth --exp-config configs/corruption/${model}.yaml --severity ${sev} --corruption ${cor} --output ./output/${model}_none_${cor}_${sev}.txt --extra-config configs/infer/cvar.yaml

done
done
done