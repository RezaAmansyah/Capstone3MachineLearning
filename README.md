# Capstone3MachineLearning


### **Context and Problem Statement**

sebuah perusahaan asuransi kendaraan di Amerika sedang merancang strategi untuk perkembangan bisnis dengan cara meningkatkan *revenue* perusahaan dan mempertahankan customer yang sudah ada. hal ini di karenakan untuk mendapatkan customer baru khususnya dalam bidang asuransi di butuhkan setidaknya CAC (customer acqusition cost) yang cukup tinggi 593$ percustomer dalam 1 tahun  ([sumber](https://www.venasolutions.com/blog/average-cac-by-industry)). berarti biaya untuk mendapatkan customer baru per orang dalam satu bulan berada pada rentang 49 $. 


dikarenakan biaya untuk mendapatkan customer baru cukup tinggi pada industri asuransi maka, perusahaan asuransi kendaraan di Amerika Serikat Menggunakan sebuah metrik pengukuran untuk menilai seberapa berharga customer perusahaan dalam jangka panjang yakni CLV (Customer Lifetime Value). Diharapkan dengan mengggunakan metrik pengukuran CLV perusahaan asuransi dapat mengembangkan revenue perusahaan dengan mepertahankan customer yang sudah ada. 

Berdasarkan permasalahan di atas perusahaan membutuhkan seorang Data Scientist untuk memprediksi nilai CLV (dalam dollar $)menggunakan machine learning  agar perusahaan bisa mengetahui nilai CLV dari masing masing customer, sehingga perusahaan bisa merancang strategi yang tepat berdasarkan nilai CLV. 

### **Goals** 

Permasalahan tersebut akhirnya akan membawa kepada sebuah pertanyaan ***Tujuan apa yang ingin di capai dengan menggunakan Machine Learning?***

1. Dengan Menggunakan Machine Learning dalam memprediksi nilai CLV, maka perusaahan bisa melakukan segmentasi customer lebih mudah. sebagai contoh : Customer A di prediksi memiliki nilai CLV 6000$ maka perusahaan akan lebih mudah mengelompokan customer kedalam segmentasi ***low CLV*** / ***medium CLV*** / ***High CLV***, sehingga perusahaan bisa mengoptimalkan strategi dalam mempertahankan customer berdasarkan nilai CLV dari customer. 
2. Dengan Menggunakan Machine Learning dalam memprediksi nilai CLV, maka perusahaan bisa mengetahui faktor apa yang mempengaruhui nilai CLV seorang customer, sehingga perusahaan dapat membuat strategi pemasaran yang tepat dalam upaya meningkatkan revenue dan mempertahankan customer berdasarkan faktor faktor penting. 


### **Analytical Approach**

berdasarkan **problem statement** dan **goals** yang ingin di capai di butuhkan cara dan pendekatan dengan machine learning untuk menemukan prediksi nilai CLV dari masing masing customer. sehingga perusahaan dapat mengambil langkah yang tepat untuk mengoptimalkan strategi perusahaan dalam meningkatkan revenue dan mempertahankan customer. 

dengan menggunakan metode **regressi** untuk membangun model machine learning akan membantu memprediksi nilai **Customer Lifetime Value** dari setiap Customer.


### **Metric Evaluation**

metrik yang akan di gunakan untuk mengukur performa dari model adalah : 

**1. MAPE (mean absolute percentage error)**

MAPE didefinisikan sebagai rata-rata kesalahan absolut antara nilai aktual dan nilai prediksi, dinyatakan dalam persentase dari nilai aktual. 

$$MAPE = \frac{1}{n} \sum\limits_{i=1}^n |\frac{y_i - \hat{y}_i}{y_i}|$$

cara kerja MAPE : 
- menghitung error (selisih) dari nilai actual - prediksi. kemudian di absolutkan 
- menghitung persentasi dari error . error yang sudah absolut di bagi dengan nilai aktual untuk mendapatkan error dalam bentuk persentase
- Hitung rata-rata dari semua kesalahan persentase untuk mendapatkan MAPE.

Keunggulan MAPE : 
- karena dalam bentuk persentasi akan lebih mudah di interpretasi dan di jelaskan untuk orang yang non-teknis. 
- lebih mudah membandingkan untuk evaluasi metrik antar model 

Kekurangan MAPE : 
- jika nilai aktual mendekati nol maka mape bisa menjadi sangat besar karena pembagian dengan nol. 
- MAPE cenderung memberikan bobot yang lebih besar pada kesalahan yang terjadi pada nilai aktual yang kecil, karena kesalahan persentasenya menjadi besar
- sensitif terhadap outlier

**2. MAE (mean absolute error)**

MAE didefinisikan sebagai rata-rata dari semua nilai absolut selisih antara nilai aktual dan nilai prediksi. 

$$MAE = \frac{1}{n} \sum\limits_{i=1}^n |y_i - \hat{y}_i|$$

cara kerja MAE : 
- menghitung error (selisih) dari nilai aktual - nilai prediksi kemudian di absolutkan 
- menghitung rata rata dari semua error untuk mendapatkan MAE 

keunggulan MAE : 
- MAE mudah dihitung dan diinterpretasikan, karena menggambarkan kesalahan rata-rata dalam unit yang sama dengan data asli
- karena menghitung nilai absolut maka MAE lebih robust terhadap outlier di bandingkan metrik lain seperti MSE. 

kekurangan MAE : 
- MAE hanya memperhitungkan besarnya kesalahan tanpa memperhatikan apakah prediksi terlalu tinggi atau terlalu rendah.

**3. RMSE (root mean squared error)**

RMSE didefinisikan sebagai akar kuadrat dari rata-rata kuadrat perbedaan antara nilai aktual dan nilai prediksi

$$RMSE = \sqrt{\frac{1}{n} \sum\limits_{i=1}^n (y_i - \hat{y}_i)^2}$$

cara kerja RMSE : 
- menghitung selisih nilai aktual - prediksi. kemudian selisih akan di kuadratkan untuk mendapatkan error kuadrat 
- menghitung rata rata dari jumlah seluruh error kuadrat untuk mendapatkan MSE
- mengambil akar kuadrat dari MSE agar menjadi RMSE 

keunggulan RMSE : 
- Karena RMSE mengkuadratkan selisih antara nilai aktual dan nilai prediksi, kesalahan yang lebih besar memiliki pengaruh yang lebih signifikan terhadap nilai RMSE dibandingkan kesalahan yang lebih kecil. hal ini berarti jika terdapat nilai prediksi yang memiliki perbedaan sangat besar terhadap nilai aktual maka RMSE akan lebih tinggi.  
- RMSE diukur dalam unit yang sama dengan variabel yang diprediksi. Ini memudahkan interpretasi hasil karena kita dapat langsung memahami besaran kesalahan dalam konteks yang sama dengan data yang digunakan. jika memprediksi ribuan Dollar maka RMSE juga akan berada dalam ribuan dollar. 

kekurangan RMSE : 
- Sensitivitas terhadap kesalahan besar dapat menjadi kelemahan jika data memiliki outlier yang tidak representatif

Alasan Menggunakan MAPE, MAE, dan RMSE : 
- RMSE :  Karena diukur dalam unit yang sama dengan variabel yang diprediksi yaitu CLV(customer lifetime value$), RMSE memberikan gambaran yang mudah dipahami tentang skala kesalahan dalam bentuk CLV $. RMSE memberikan penalti yang besar terhadap outlier di residual (error) sehingga cocok untuk mendapatkan gambaran model terbaik dalam memprediksi nilai CLV. 
- MAPE : dalam bisnis MAPE lebih mudah di pahami dan di interpretasikan oleh orang non teknis karena kesalahan yang di hitung di tampilkan dalam bentuk persen. dalam hal ini adalah kesalahan antara nilai aktual dan prediksi terhadap CLV dan di sajikan dalam persen. 
- MAE : MAE dihitung dengan mengambil rata-rata dari nilai absolut selisih antara nilai prediksi dan nilai aktual. MAE dapat dengan mudah diinterpretasikan sebagai kesalahan rata-rata yang dihasilkan oleh model. Unit MAE sama dengan unit Variabel Target (CLV) sehingga mudah dalam menjelaskannya. 

Metrik yang menjadi prioritas adalah : 
1. untuk komparasi model metrik yang di gunakan adalah RMSE. 
2. untuk penjelasan dalam bisnis metrik yang di gunakan adalah MAPE karena lebih mudah di interpretasikan.


### **Conclussion**

- Metrik Utama yang di gunakan adalah RMSE dan MAPE.
    - RMSE di gunakan untuk menilai performa model agar mendapatkan model terbaik. karena cara kerja RMSE yang memberikan penalti besar terhadap residual(error) dalam prediksi sehingga kita bisa mendapatkan performa model secara maksimal. 
    - MAPE di gunakan untuk menginterpretasikan perfroma model, karena bentuk MAPE dalam persentasi maka akan lebih mudah menjelaskan performa model kepada orang non teknis. 

- Berdasarkan Hyperparameter Tuning model terbaik adalah : 
    - Gradient Boosting Regressor
    - Parameter terbaiknya adalah :  n_estimators: 100, min_samples_split: 18, min_samples_leaf: 3, max_depth: 3, learning_rate: 0.1
    - Nilai RMSE pada data test : 988
    - Nilai MAPE pada secara kesulurhan data test : 4.8% 
    - Nilai MAPE per segmentasi :
        - LOW CLV : 3.2 %
        - MEDIUM CLV : 2.7 %
        - HIGH CLV : 8.9 % 
        - model cukup baik dan stabil dalam memprediksi nilai CLV untuk segmentasi LOW CLV dan MEDIUM CLV. dan performanya berkurang saat memprediksi segmentasi HIGH CLV. 

- Berdasarkan Feature Importance, Feature yang paling penting adalah: 
    - **Number Of Policies** : Jumlah Polis Asuransi yang di miliki customer. 
    - **Monthly Premium Auto** : Jumlah Premi Asuransi yang dibayarkan oleh customer setiap bulannya
    - Feature lainnya tidak memiliki pengaruh yang besar.
   

- Interpretasi model Gradint Boosting Regressor dengan SHAP : 
    - Berdasarkan **Number Of Policies** : semakin banyak jumlah polis yang dimiliki oleh customer maka semakin tinggi nilai CLV yang di prediksi oleh model. 
    - Berdasarkan **Monthly Premium Auto** : semakin besar biaya premi asuransi yang di bayar oleh customer maka semakin tinggi CLV seorang customer berdasarkan prediksi dari model machine learning. 
    - **note** : terdapat customer yang memiliki polis asuransi hanya satu tapi di prediksi memiliki nilai CLV yang tinggi karena jumlah premi asuransi yang di bayarkan perbulan relatif tinggi yaitu > 200$ 


- Limitasi Model : 

    Model ini hanya valid untuk memprediksi target : 
    - **customer lifetime value** : 1898$ - 16589$
    
    dan model ini valid di gunakan jika menggunakan feature : 
    - **Number Of Policies** : 1 - 9
    - **Monthly Premium Auto** : 61$ - 297$ 
    - **Income** : 0 - 99934$
    - **Total Claim Amount** : 0.4$ - 2759$
    - **Coverage** : Basic, Extended, Premium
    - **Renew Offer Type** : Offer1, Offer2, Offer3, Offer4
    - **Vehicle Class** : Four-Door-Car, Two-Door-Car, SUV, Sports Car, Luxury Car, Luxury SUV
    - **Employment Status** : Employeed, Unemployeed, Medical-Leave(cuti alasan kesehatan), Disabled, Retired
    - **Education** : High School or Below, College, Bachelor, Master, Doctor
    - **Marital Status** : Married, Single, Divorced

    Berdasarkan informasi di atas dapat disimpulkan bahwa model tidak akan valid di gunakan jika terdapat nilai atau kategori yang tidak sesuai. 

### **Recomendation**



Langkah yang bisa di lakukan oleh perusahaan agar dapat meningkatkan revenue dan menjaga loyalitas customer adalah :
- membagi segmentasi customer agar strategi pemasaran yang di lakukan bisa efektif dan tepat sasaran.  
- melakukan strategi pemasaran dengan menggunakan ***upselling*** dan ***crossselling*** kepada customer yang tergolong LOW CLV berdasarkan hasil prediksi. dengan tujuan meningkatkan jumlah polis per customer. seperti menawarkan **Bundling Produk** asuransi
- meningkatkan loyalitas customer yang tergolong dan di prediksi sebagai MEDIUM CLV dengan memberikan *personalized Offers* yakni produk asuransi yang sesuai kebutuhan customer. 
- memberikan penawaran produk berupa ***Exclucive Offers*** kepada customer yang di golongkan HIGH CLV. penawaran tersebut bisa berupa produk exclusive dengan berbabagi benefit seperti pengecekan kendaraan secara berskala. Dan memberikan Program Loyalitas Customer agar mendorong Cusomer untuk tetap menggunakan layanan asuransi perusahaan kita. 
- Membuat Program loyalitas customer agar customer terus menggunakan layanan asuransi perusahaan. 

Beberapa langkah yang bisa di gunakan untuk pengembangan projek dan model adalah : 

- membuat model lain atau melakukan strategi lain dalam mengembangkan model agar bisa memprediksi CLV yang lebih tinggi dari 17.000$ 
- menambah fitur fitur baru seperti kolom tingakatan kepuasan customer (dalam skala 1-5) agar kita bisa mengetahui kepuasan masing masing customer persegmentasi.
- menggunakan model dan hyperparameter tuning lainnya yang belom di gunakan seperti Linear Regression. 
- menggunakan metrik Evaluasi lainnya dalam menilai performa model seperti RMSPE. 
