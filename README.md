Emotion Detection Project

Bu proje, video ve ses dosyalarından duygu tanıma işlemi gerçekleştirmektedir. Proje, yüz ifadelerini ve ses özelliklerini kullanarak duygu tespiti yapar.

1. Literatür ve Makale Araştırmaları
------------------------------------

- "A Review on Emotion Detection Techniques Using Facial and Vocal Expressions" - Journal of Behavioral and Brain Science
- "Speech Emotion Recognition with deep learning" - https://www.sciencedirect.com/science/article/pii/S1877050920318512

2. DataSet, Veri Özellikleri ve Özellikleri
-------------------------------------------
Projemizde ses duygu surumu tespiti için kullanılan veri seti, RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) veri setidir. Bu veri seti, çeşitli duygu ifadelerini içeren ses kayıtlarından oluşur.
Yüz ifadeleri duygu durumu için ise FER-2013 llanılmıştır.

Veri setinin özellikleri:

- Ses dosyaları: .wav formatında, 48 kHz örnekleme frekansında duygu içeren konuşmalar.

3. Veri Ön İşleme, Özellik Mühendisliği
---------------------------------------
Veri ön işleme adımları:
- Ses dosyalarının yüklendi ve örnekleme frekansının yeniden ayarlandı.
- MFCC (Mel-frequency cepstral coefficients) ve Mel Spectrogram özelliklerinin çıkarıldı.
- Veri setindeki dosyaların etiketlenmesi ve kategorik forma dönüştürülmesi yapıldı.

Özellik mühendisliği teknikleri:
- Ses dosyalarında zenginleştirme teknikleri (pitch shifting ve time stretching)yapıldı.
- Yüz ifadelerinin gri tonlamalı hale getirilmesi ve normalizasyonu yapıldı.

4. Modelleme, Test ve Doğrulama
-------------------------------
Proje iki model içermektedir:

1. Görüntü tabanlı duygu tanıma modeli: CNN (Convolutional Neural Network) kullanılarak oluşturulmuştur.
2. Ses tabanlı duygu tanıma modeli: GRU (Gated Recurrent Unit) katmanları içeren kompleks bir model kullanılarak oluşturulmuştur.

Model eğitimi ve doğrulama adımları:
- Veri setinin eğitim ve test setlerine bölündü.
- Modellerin eğitimi ve performanslarının izlendi (ModelCheckpoint ve EarlyStopping kullanılarak).
- Eğitim süreci sonunda en iyi modeller kaydedildi ve test setinde değerlendirildi.

5. Dağıtım İçin: Kaynaklar, Ortam, API, Kitaplık ve Teknoloji Yığınları
----------------------------------------------------------------------
Projenin dağıtımı için gerekli olan ortam ve bağımlılıklar:
- Python 3.7+
- OpenCV: Görüntü işleme için
- NumPy: Sayısal işlemler için
- Keras ve TensorFlow: Derin öğrenme modelleri için
- Librosa: Ses işleme için
- Pillow: Görüntü işleme için
- Tkinter: GUI uygulaması için

API ve kütüphaneler:
- OpenCV: `cv2.CascadeClassifier` kullanarak yüz tespiti
- Librosa: `librosa.load` ve `librosa.feature.mfcc` kullanarak ses özellikleri çıkarma
- TensorFlow ve Keras: Modellerin oluşturulması, eğitilmesi ve tahmin işlemleri için


### Projeyi GitHub'dan İndirmesi ve Kurulumu ----

1. **GitHub'dan Projeyi Klonlama:**
   - Terminali açın.
   - Aşağıdaki komutu kullanarak projeyi klonlayın:
     
     "git clone https://github.com/mertkarababa1/MertKarababa.git"
     

2. **Proje Klasörüne Geçiş:**
   - Terminalde, klonladığınız proje klasörüne geçiş yapın:

     "cd MertKarababa"

3. **Python Sanal Ortamı Oluşturma ve Etkinleştirme:**
   - Sanal ortam oluşturmak için aşağıdaki komutu kullanın:
     
     "python -m venv env"
     
   - Sanal ortamı etkinleştirin:
     - Windows:
      
       ".\env\Scripts\activate"
       
     - MacOS/Linux:
       
       "source env/bin/activate"
       

4. **Gerekli Kütüphaneleri Yükleme:**
   - Sanal ortam etkinleştirildikten sonra, gerekli kütüphaneleri manuel olarak yükleyin. Aşağıdaki komutları terminale sırasıyla yazın:
     
     pip install opencv-python
     pip install numpy
     pip install keras
     pip install tensorflow
     pip install librosa
     pip install Pillow
     

5. **Veri Seti Dosyalarının Yerleştirilmesi:**
   - voice_model.py dosyası için RAVDESS (https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio) veri setini indirin ve projenin dosya dizimindeki "/data" klasörünün altına yerleştirin.
   - cam_model.py dosyası için FER-2013 (https://www.kaggle.com/datasets/msambare/fer2013) veri setini indirin ve projenin dosya dizimindeki "/data" klasörünün altına yerleştirin.


6. **Model Dosyalarının Yerleştirilmesi:**
   - `cam_model.h5` ve `best_complex_model1.keras` dosyalarının `models` klasörüne yerleştirildiğinden emin olun. Eğer bu klasör yoksa, oluşturun ve dosyaları bu klasöre kopyalayın.

7. **Projeyi Çalıştırma:**
   - Projeyi çalıştırmak için aşağıdaki komutu kullanın:
    
     "python main.py"

     not:Lütfen projede kullanılmış tüm dosya yollarını kendizine göre düzenleyin. main.py dosyasını çalıştırmak için model dosyalarının yollarının doğru verildiğinden emin olun.
     

