
DATA_PATH="/home/graham/Documents/hardware-aware-transformers/data/binary/iwslt14_de_en"

# Create a new validation set with the first 500 sentences
head -n 500 "${DATA_PATH}/valid.de" > "${DATA_PATH}/valid_small.de"
head -n 500 "${DATA_PATH}/valid.en" > "${DATA_PATH}/valid_small.en"

echo "Created smaller validation files: valid_small.de and valid_small.en"