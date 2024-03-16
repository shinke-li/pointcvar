if [ ! -d "output" ]; then
mkdir "output"
fi

gpu_id=3

for model in 'backdoor_pointnet_pl' 'backdoor_pointnet_cl' 'backdoor_dgcnn_pl'; do 

for dataset in 'modelnet40' 'shapenet'; do

for cor in ${dataset}'_backdoor'; do 

for sev in 5; do

CUDA_VISIBLE_DEVICES=${gpu_id} python main.py --entry test --model-path ./runs/${dataset}_${model}.pth --exp-config configs/corruption/${dataset}_${model}.yaml --severity ${sev} --corruption ${cor} --output ./output/${dataset}_${model}_clean_${cor}_${sev}.txt 

done
done
done
done



for model in 'backdoor_pointnet_pl' 'backdoor_pointnet_cl' 'backdoor_dgcnn_pl'; do 

for dataset in 'modelnet40' 'shapenet'; do

for cor in ${dataset}'_backdoor'; do 

for robust in 'ror' 'srs' 'sor' 'cvar' 'iter_cvar'; do

for sev in 5; do

CUDA_VISIBLE_DEVICES=${gpu_id} python main.py --entry test --model-path ./runs/${dataset}_${model}.pth --exp-config configs/corruption/${dataset}_${model}.yaml --severity ${sev} --corruption ${cor} --output ./output/${dataset}_${model}_${robust}_${cor}_${sev}.txt --extra-config configs/infer/${robust}.yaml

done
done
done
done
done