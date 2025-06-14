# my oneliners

# count races
for R in $(echo white black asian indian latino eastern); do echo -n "---${R}: "; cat dataset_attributes.txt | grep Valid | grep ${R}  | wc -l; done
for R in $(echo white black asian indian latino eastern); do echo -n "---${R}: "; cat dataset_attributes.txt | grep Test | grep ${R}  | wc -l; done

for R in $(echo white black asian indian latino eastern); do echo -n "---${R}: "; cat work_on_train.txt |  grep ${R} | grep " 1 "  | wc -l; done
for R in $(echo white black asian indian latino eastern); do echo -n "---${R}: "; cat work_on_train.txt |  grep ${R} | grep " 0 "  | wc -l; done
