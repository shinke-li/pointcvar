if [ ! -d "output" ]; then
mkdir "output"
fi

for model in 'backdoor_pointnet_pl' 'backdoor_pointnet_cl' 'backdoor_dgcnn_pl'; do 

for dataset in 'modelnet40' 'shapenet'; do

for cor in ${dataset}'_backdoor'; do 

for sev in 5; do

CUDA_VISIBLE_DEVICES=3 python main.py --entry test --model-path ./runs/${dataset}_${model}.pth --exp-config configs/corruption/${dataset}_${model}.yaml --severity ${sev} --corruption ${cor} --output ./output/${dataset}_${model}_clean_${cor}_${sev}.txt 

done
done
done
done