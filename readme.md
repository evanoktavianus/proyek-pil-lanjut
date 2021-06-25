## **Kode Proyek Akhir**

Repository ini adalah kode untuk tugas akhir proyek mandiri a.n. Evan Oktavianus, DIK Fasilkom UI, NPM 2006624324.

**Requirements:**

python 3.7

pytorch 

cudatoolkit=10.1   

dill

jsonlines

pandas

sklearn

allennlp 0.9 with gradient accumulation and fp16

**File utama yang tersedia meliputi:**
1. prepare_training_data.py
Kode untuk memproses sample training, validasi, dan test set dari SCIDOCS
5. create_training_py
Kode untuk membuat training triplets dan tokenisasi untuk masing2 query paper dengan positive dan negative paper
2. train-scibert.py
Kode untuk training model SPECTER dengan menggunakan dasar SCIBERT
1. test-sidocs.py
Kode untuk testing model SPECTER menggunakan benchmark dataset SCIDOCS


**Langkah untuk mereproduksi:
1. persiapkan training data menggunakan prepare_training_data.py
   
2. proses trianing data menggunakan create_training_files.py dalam environment allennlp:
   
    python create_training_files.py --data-dir data --metadata data/metadata.json --outdir data/preprocessed/

3. lakukan training menggunakan train-scibert.py dalam environment lightning pytorch

    python train-scibert.py --save_dir ./save-scibert --gpus 1 --train_file preprocessed-30000/data-train.p --dev_file preprocessed-30000/data-val.p --test_file preprocessed-30000/data-test.p --batch_size 4 --num_workers 4 --num_epochs 2 --grad_accum 8 

4. lakukan testing menggunakan model yang telah ditrained untuk mendapatkan embeddings:
    
    #python test.py --save_dir save-scidocs --gpus 1 --test_only $true --test_checkpoint save-scibert-complete/version_0/checkpoints/ep-epoch=0_avg_val_loss-avg_val_loss=0.366.ckpt

5. lakukan testing menggunakan benchmark SCIDOCS menggunakan test-scidocs.py

