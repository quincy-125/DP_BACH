IFS=$'\n'
for i in `grep -v "Num_patches" /research/bsi/projects/PI/tertiary/Hart_Steven_m087494/s211408.DigitalPathology/Quincy/confirm_tf_record_final.xls`
do
	label=`echo "$i"|cut -f2 `
        cat=`echo "$i"|cut -f5`
        i1=`echo $i|cut -f1|sed -e 's/.tfrecords//g'`
        tf=`echo $i|cut -f1`
        echo $label
        echo $cat
        echo $i1
	mkdir -p /research/bsi/projects/PI/tertiary/Hart_Steven_m087494/s211408.DigitalPathology/Quincy/Data/CLAM/tfrecord_level_normal1/$cat/$label
        cp /research/bsi/projects/PI/tertiary/Hart_Steven_m087494/s211408.DigitalPathology/Quincy/Data/CLAM/tfrecord_level_normal1/$tf /research/bsi/projects/PI/tertiary/Hart_Steven_m087494/s211408.DigitalPathology/Quincy/Data/CLAM/tfrecord_level_normal1/$cat/$label/$i1.$label.$cat.tfrecords
	#exit
done
