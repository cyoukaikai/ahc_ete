#!/bin/bash

log_date=`date +"%y-%m-%d"` #-%H-%M-%S

#cd ../ensemble_tracking_exports/results/experiments/tracking_results
experiment_flag=AHC_ETE_test
root_dir=./AHC_ETE_benchmark
for result_dir in $(ls -d $root_dir/MOT16*); do
	echo "================================================================="
	echo "result_dir = "$result_dir #, "$root_dir"/"$result_dir"
	python -m motmetrics.apps.eval_motchallenge ./MOT16_train_part_results  "$result_dir" |& tee $experiment_flag-MOT16_"$log_date"
#$root_dir/experts-progress-MOT16_20-04-14_"$result_dir"
	echo "================================================================="
done


<<COMMENT
DATASET=("MOT16" )   #  "MOT17" "MOT20" "MOT16" 
for ele in ${!DATASET[@]}; do
	dataset=${DATASET[ele]}
	echo "================================================================="
	echo "dataset = "$dataset
	
	summary_file="$experiment_flag"_"$log_date"_"$dataset"_"summary"
	rm $summary_file
	for sub_dir in $(ls -d $root_dir/$dataset/*); do  # | grep -v "MOT16"
		echo "$sub_dir"
		script_file="$experiment_flag"_"$log_date"_`echo $sub_dir | cut -d'/' -f4`
		echo "script_file="$script_file
		python -m motmetrics.apps.eval_motchallenge ./$dataset/train  "$sub_dir" |& tee $script_file
		
		tac $script_file | head -1  | tail -1 >> "$summary_file"
		#echo "================================================================="
	done

done


# python -m motmetrics.apps.eval_motchallenge ./MOT15/train  ./DeepSort_tracking_results/MOT15_train_results
#python -m motmetrics.apps.eval_motchallenge ./MOT17/train  ./AHC_ETE_results

# DATASET=("MOT15" "MOT17" "MOT20" )   # "MOT16" 
DATASET=("MOT17" "MOT20" )   # "MOT16" 

for ele in ${!DATASET[@]}; do
	dataset=${DATASET[ele]}
	echo "================================================================="
	echo "dataset = "$dataset
	python -m motmetrics.apps.eval_motchallenge ./$dataset/train  ./DeepSort_tracking_results/"$dataset"_train_results
	echo "================================================================="
done



#python -m motmetrics.apps.eval_motchallenge ./MOT17/train  ./AHC_ETE_results

# DATASET=("MOT15" "MOT17" "MOT20" )   # "MOT16" 
DATASET=("MOT17" "MOT20" )   # "MOT16" 

for ele in ${!DATASET[@]}; do
	dataset=${DATASET[ele]}
	echo "================================================================="
	echo "dataset = "$dataset
	python -m motmetrics.apps.eval_motchallenge ./$dataset/train  ./DeepSort_tracking_results/"$dataset"_train_results
	echo "================================================================="
done


#cd ../ensemble_tracking_exports/results/experiments/tracking_results
root_dir=./progress_experts/MOT16
for result_dir in $(ls $root_dir); do
	echo "================================================================="
	echo "result_dir = "$result_dir #, "$root_dir"/"$result_dir"
	python -m motmetrics.apps.eval_motchallenge ./MOT16/train  "$root_dir"/"$result_dir" |& tee $root_dir/experts-progress-MOT16_20-04-14_"$result_dir"
	echo "================================================================="
done

COMMENT
