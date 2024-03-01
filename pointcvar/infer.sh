if [ ! -d "output" ]; then
mkdir "output"
fi

for model in 'pointnet' 'dgcnn'; do 

for dataset in 'modelnet40' 'shapenet'; do

for cor in ${dataset}'_add_global' ${dataset}'_add_local' ${dataset}'_'${model}'_add_chamfer' ${dataset}'_'${model}'_add_cluster' ${dataset}'_'${model}'_add_hausdorff' ${dataset}'_'${model}'_add_object' ; do 

for sev in 5; do

CUDA_VISIBLE_DEVICES=3 python main.py --entry test --model-path ./runs/${dataset}_${model}.pth --exp-config configs/corruption/${dataset}_${model}.yaml --severity ${sev} --corruption ${cor} --output ./output/${dataset}_${model}_clean_${cor}_${sev}.txt 

done
done
done
done