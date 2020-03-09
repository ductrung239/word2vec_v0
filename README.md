# Word2Vec

# Mục tiêu của bài này để hướng dẫn sử dụng thư viên Gensim để training mô hình word2vec:

##	1. Download dữ liệu từ các nguồn khác nhau trên Internet hoặc sử dụng bộ dataset thu nhỏ để train
##  2. Clean và tiền xử lý dữ liệu để đưa vào training
##  3. Training sử dụng mô hình word2vec bằng cách sử dụng thư viên Gensim
##	4. Kiểm tra mô hình

# Bước 1

- Bạn có thể download dữ liệu từ nhiều nguồn khác nhau trên Internet như Wikipedia tiếng Việt (Wiki có hỗ trợ để dump dữ liệu và tải về máy), các trang báo mạng tiếng Việt như dantri.vn… Ở đây mình có viết một hàm để tài dữ liệu từ một url nào đó, ví dụ từ link 1 bài báo trên dân trí “https://dantri.com.vn/xa-hoi/thu-tuong-tam-hoan-cong-tac-nuoc-ngoai-de-tap-trung-chong-dich-covid-19-20200308125947359.htm"” và ghi vào một file text, clean luôn nếu bạn cần:

```
from word_embedding.utils import download_html
url_path = "https://dantri.com.vn/xa-hoi/thu-tuong-tam-hoan-cong-tac-nuoc-ngoai-de-tap-trung-chong-dich-covid-19-20200308125947359.htm"
output_path = "data/word_embedding/real/html/html_data.txt"
download_html(url_path, output_path, should_clean=True)
```

Tham khảo thêm: https://github.com/deepai-solutions/core_nlp/blob/master/word_embedding/utils.py

- Hoặc có thể sử dụng bộ dữ liệu nhỏ mà mình đã xử lý sẵn.

# Bước 2

- Clean dữ liệu nếu có dính html tag, css, script. Sau đó tách từ để mỗi từ (word) có nghĩa và mang đúng nghĩa. Mình xin giới thiệu hàm để tự động tìm các files trong một thư mục, làm sạch các tag, script, tách từ rồi ghi vào một file cùng tên nhưng đặt trong thư mục mới.

```
input_dir = 'data/word_embedding/real/html'
output_dir = 'data/word_embedding/real/training'
from tokenization.crf_tokenizer import CrfTokenizer
from word_embedding.utils import clean_files_from_dir
crf_config_root_path = "tokenization/"
crf_model_path = "models/pretrained_tokenizer.crfsuite"
tokenizer = CrfTokenizer(config_root_path=crf_config_root_path, model_path=crf_model_path)
clean_files_from_dir(input_dir, output_dir, should_tokenize=True, tokenizer=tokenizer
```
- Hoặc sử dụng bộ dữ liệu mà mình đã tiền xử lý sẵn

# Bước 3
- Sau khi đã có được dữ liệu tiền xử lý. Chúng ta sử dụng thư viện Gensim để train mô hình. Ở đây mình sử dụng số chiều là 300. Giá trị sg = 1 là Skipgram và sg = 0 là CBOW.

```
## Skip Gram model

import gensim
from gensim.models import Word2Vec, KeyedVectors  
model = gensim.models.Word2Vec(data, min_count = 1, size =300, window = 5, sg = 1) 
model.wv.save_word2vec_format('skipgram-vi-model.bin', binary=True)
```
```
## CBOW model

import gensim
from gensim.models import Word2Vec, KeyedVectors  
model = gensim.models.Word2Vec(data, min_count = 1, size =300,window = 5, sg = 0) 
model.wv.save_word2vec_format('CBOW-vi-model.bin', binary=True)
```
Tham khảo thêm: https://radimrehurek.com/gensim/models/word2vec.html

# Bước 4

- Kiểm tra mô hình
```
model = gensim.models.KeyedVectors.load_word2vec_format("tên model mà bạn đã train")
model.most_similar(positive=['tình_yêu'],topn=20)
```
