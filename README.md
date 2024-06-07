Sayin Hocam projenin dosya boyutu cok yuksek oldugu icin yukleme yapamadim, size projemin Github linkini iletiyorum. (https://github.com/mertkarababa1/MertKarababa)
Projenin nasil calistirilacagina dair detayli bilgi asagidadir.

1. Proje 

Bu proje, video ve ses dosyalarindan duygu tanima islemi gerceklestirmektedir. Proje, yuz ifadelerini ve ses ozelliklerini kullanarak duygu durumu tespiti yapar.

## 1. Literatur ve Makale Arastirmalari

- "A Review on Emotion Detection Techniques Using Facial and Vocal Expressions" - Journal of Behavioral and Brain Science
- "Speech Emotion Recognition with deep learning" - https://www.sciencedirect.com/science/article/pii/S1877050920318512

## 2. DataSet, Veri Ozellikleri ve Ozellikleri

Projemizde ses duygu surumu tespiti icin kullanilan veri seti, RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) veri setidir. Bu veri seti, cesitli duygu ifadelerini iceren ses kayitlarindan olusur. Yuz ifadeleri duygu durumu icin ise FER-2013 kullanilmistir.

Veri setinin ozellikleri:

- Ses dosyalari: .wav formatinda, 48 kHz ornekleme frekansinda duygu iceren konusmalar.

## 3. Veri On Isleme, Ozellik Muhendisligi

Veri on isleme adimlari:
- Ses dosyalarinin yuklenmesi ve ornekleme frekansinin yeniden ayarlanmasi.
- MFCC (Mel-frequency cepstral coefficients) ve Mel Spectrogram ozelliklerinin cikarilmasi.
- Veri setindeki dosyalarin etiketlenmesi ve kategorik forma donusturulmesi.

Ozellik muhendisligi teknikleri:
- Ses dosyalarinda zenginlestirme teknikleri (pitch shifting ve time stretching).
- Yuz ifadelerinin gri tonlamali hale getirilmesi ve normalizasyonu.

## 4. Modelleme, Test ve Dogrulama

Proje iki model icermektedir:
1. Goruntu tabanli duygu tanima modeli: CNN (Convolutional Neural Network) kullanilarak olusturulmustur.
2. Ses tabanli duygu tanima modeli: GRU (Gated Recurrent Unit) katmanlari iceren kompleks bir model kullanilarak olusturulmustur.

Model egitimi ve dogrulama adimlari:
- Veri setinin egitim ve test setlerine bolunmesi.
- Modellerin egitimi ve performanslarinin izlenmesi (ModelCheckpoint ve EarlyStopping kullanilarak).
- Egitim sureci sonunda en iyi modellerin kaydedilmesi ve test setinde degerlendirilmesi.

## 5. Dagitim Icin: Kaynaklar, Ortam, API, Kitaplik ve Teknoloji Yiginlari

Projenin dagitimi icin gerekli olan ortam ve bagimliliklar:
- Python 3.7+
- OpenCV: Goruntu isleme icin
- NumPy: Sayisal islemler icin
- Keras ve TensorFlow: Derin ogrenme modelleri icin
- Librosa: Ses isleme icin
- Pillow: Goruntu isleme icin
- Tkinter: GUI uygulamasi icin

API ve kutuphaneler:
- OpenCV: `cv2.CascadeClassifier` kullanarak yuz tespiti
- Librosa: `librosa.load` ve `librosa.feature.mfcc` kullanarak ses ozellikleri cikarma
- TensorFlow ve Keras: Modellerin olusturulmasi, egitilmesi ve tahmin islemleri icin

### Projeyi GitHub'dan Indirmesi ve Kurulumu ----

1. **GitHub'dan Projeyi Klonlama:**
   - Terminali acin.
   - Asagidaki komutu kullanarak projeyi klonlayin:
     
     "git clone https://github.com/mertkarababa1/MertKarababa.git"
     

2. **Proje Klasorune Gecis:**
   - Terminalde, klonladiginiz proje klasorune gecis yapin:
     
     "cd MertKarababa"
    

3. **Python Sanal Ortami Olusturma ve Etkinlestirme:**
   - Sanal ortami olusturmak icin asagidaki komutu kullanin:
    
     "python -m venv env"
    
   - Sanal ortami etkinlestirin:
     - Windows:
       
      ".\env\Scripts\activate"
       
     - MacOS/Linux:
      
       "source env/bin/activate"
       

4. **Gerekli Kutuphaneleri Yukleme:**
   - Sanal ortam etkinlestirildikten sonra, gerekli kutuphaneleri manuel olarak yukleyin. Asagidaki komutlari terminale sirayla yazin:
     
     pip install opencv-python
     pip install numpy
     pip install keras
     pip install tensorflow
     pip install librosa
     pip install Pillow
    

5. **Veri Seti Dosyalarinin Yerleştirilmesi:**
   - `voice_model.py` dosyasi icin RAVDESS (https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio) veri setini indirin ve projenin dosya dizimindeki `/data` klasorunun altina yerlestirin.

   - `cam_model.py` dosyasi icin FER-2013 (https://www.kaggle.com/datasets/msambare/fer2013) veri setini indirin ve projenin dosya dizimindeki `/data` klasorunun altina yerlestirin.

6. **Model Dosyalarinin Yerleştirilmesi:**
   - `cam_model.h5` ve `best_complex_model1.keras` dosyalarinin `models` klasorune yerlestirildiginden emin olun. Eger bu klasor yoksa, olusturun ve dosyalari bu klasore kopyalayin.

7. **Projeyi Calistirma:**

   - Projeyi calistirmak icin asagidaki komutu kullanin ya da run edin:
main.py dosyasi uzerinden proje calisacaktir.
     "python main.py"
    

Not: Lutfen, projede kullanilmis tum dosya yollarini kendinize gore duzenleyin !.. `main.py` dosyasini calistirmak icin model dosyalarinin yollarinin dogru verildiginden emin olun.
