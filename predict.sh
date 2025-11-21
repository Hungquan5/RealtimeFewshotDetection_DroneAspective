# Activate conda inside script
source /opt/conda/etc/profile.d/conda.sh
conda activate yoloe

cd /code
python preprocessing.py

python predict.py