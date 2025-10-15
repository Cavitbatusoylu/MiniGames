# 🎮 Oyunlar Tech - Bilgisayar Görüsü ile Kontrol Edilen Oyunlar Koleksiyonu

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![OpenCV](https://img.shields.io/static/v1?style=for-the-badge&message=OpenCV&color=5C3EE8&logo=OpenCV&logoColor=FFFFFF&label=)
![MediaPipe](https://img.shields.io/badge/MediaPipe-4285F4?style=for-the-badge&logo=google&logoColor=white)
![Pygame](https://img.shields.io/badge/PyGame-008080?style=for-the-badge&logo=PyGame&logoColor=white)
![NumPy](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white)

Bu proje, bilgisayar görüsü teknolojilerini kullanarak kontrol edilen çeşitli oyunların koleksiyonunu içerir. El hareketleri, yüz takibi ve renk takibi gibi farklı kontrol yöntemleriyle oynanabilen eğlenceli oyunlar sunar.

## 📋 İçindekiler

- [🎯 Oyunlar](#-oyunlar)
- [🛠️ Teknoloji Stack](#️-teknoloji-stack)
- [📦 Kurulum](#-kurulum)
- [🚀 Kullanım](#-kullanım)
- [🎮 Oyun Detayları](#-oyun-detayları)
- [📸 Ekran Görüntüleri](#-ekran-görüntüleri)
- [🤝 Katkıda Bulunma](#-katkıda-bulunma)
- [📄 Lisans](#-lisans)

## 🎯 Oyunlar

### 1. **Mimikle Kontrol Edilen Mini Shooter** 🚀
- **Kontrol:** Yüz hareketleri ile kontrol
- **Özellikler:** Uzay gemisi kontrolü, düşman avlama, parçacık efektleri
- **Teknoloji:** MediaPipe Face Detection, Pygame
- **Dosya:** `Mimikle Kontrol Edilen Mini Shooter.py`

### 2. **El Hareketiyle Flappy Bird** 🐦
- **Kontrol:** El hareketleri ile kontrol
- **Özellikler:** Klasik Flappy Bird oyunu, el takibi
- **Teknoloji:** MediaPipe Hand Tracking, Pygame
- **Dosya:** `El Hareketiyle Flappy Bird.py`

### 3. **Renk Takibi ile Breakout** 🎯
- **Kontrol:** Renk takibi ile paddle kontrolü
- **Özellikler:** Breakout tarzı oyun, renk tabanlı kontrol
- **Teknoloji:** OpenCV Color Tracking, Pygame
- **Dosya:** `Renk_Takibi_ile_Paddle_Kontrol_(Breakout_tarzı)[1].py`

### 4. **Yüz Takibiyle Pac-Man** 👻
- **Kontrol:** Yüz hareketleri ile kontrol
- **Özellikler:** Klasik Pac-Man oyunu, yüz takibi
- **Teknoloji:** MediaPipe Face Detection, Pygame
- **Dosya:** `Yüz Takibiyle Pac-Man.py`

### 5. **Gesture-Controlled Arcade Game** 🎮
- **Kontrol:** El hareketleri ile kontrol
- **Özellikler:** Uzay gemisi kontrolü, jest tanıma
- **Teknoloji:** OpenCV, MediaPipe, Pygame
- **Klasör:** `cv-game-main/`

### 6. **Endless Game Automation** 🏃‍♂️
- **Kontrol:** Vücut pozisyonu ile kontrol
- **Özellikler:** Sonsuz koşu oyunları otomasyonu
- **Teknoloji:** MediaPipe Pose Detection
- **Klasör:** `Endless-Game-automation-using-mediapipe-main/`

### 7. **Hand Chrome Dino** 🦕
- **Kontrol:** El hareketleri ile Chrome Dino kontrolü
- **Özellikler:** Chrome Dino oyunu kontrolü
- **Teknoloji:** MediaPipe Hand Tracking, PyAutoGUI
- **Klasör:** `HandChromeDino-main/`

### 8. **Pacman MediaPipe** 👻
- **Kontrol:** El hareketleri ile kontrol
- **Özellikler:** Gelişmiş Pac-Man oyunu, jest tanıma
- **Teknoloji:** MediaPipe Gesture Recognition, OpenCV
- **Klasör:** `Pacman-MediaPipe-main/`

## 🛠️ Teknoloji Stack

- **Python 3.x** - Ana programlama dili
- **OpenCV** - Bilgisayar görüsü ve görüntü işleme
- **MediaPipe** - El, yüz ve pozisyon takibi
- **Pygame** - Oyun geliştirme framework'ü
- **NumPy** - Sayısal hesaplamalar
- **PyAutoGUI** - Otomatik kontrol (Chrome Dino için)

## 📦 Kurulum

### Gereksinimler
```bash
pip install opencv-python
pip install mediapipe
pip install pygame
pip install numpy
pip install pyautogui
```

### Hızlı Kurulum
```bash
# Ana dizine gidin
cd Oyunlar_Tech

# Her oyun için gerekli paketleri yükleyin
pip install -r requirements.txt  # (varsa)
```

## 🚀 Kullanım

### Genel Kullanım Adımları:

1. **Kamerayı Hazırlayın**
   - Webcam'inizin çalıştığından emin olun
   - İyi aydınlatma sağlayın
   - Kameraya net görünün

2. **Oyunu Başlatın**
   ```bash
   python "Oyun_Adı.py"
   ```

3. **Kontrol Yöntemlerini Öğrenin**
   - Her oyun farklı kontrol yöntemleri kullanır
   - Oyun içi talimatları takip edin

### Kontrol Yöntemleri:

#### 🖐️ El Hareketi Kontrolü
- **El Takibi:** MediaPipe ile el pozisyonlarını takip eder
- **Jest Tanıma:** Parmak hareketlerini tanır
- **Hareket Kontrolü:** El hareketlerini oyun kontrollerine çevirir

#### 👤 Yüz Takibi Kontrolü
- **Yüz Algılama:** MediaPipe ile yüz özelliklerini takip eder
- **Baş Hareketi:** Baş hareketlerini oyun kontrollerine çevirir
- **Mimik Kontrolü:** Yüz ifadelerini kullanır

#### 🎨 Renk Takibi Kontrolü
- **Renk Algılama:** OpenCV ile belirli renkleri takip eder
- **Nesne Takibi:** Renkli nesneleri takip eder
- **Pozisyon Kontrolü:** Renk pozisyonunu oyun kontrollerine çevirir

## 🎮 Oyun Detayları

### Mimikle Kontrol Edilen Mini Shooter
- **Kontrol:** Yüz hareketleri ile uzay gemisini kontrol edin
- **Amaç:** Düşmanları yok edin ve puan toplayın
- **Özellikler:** Parçacık efektleri, çoklu düşman türleri

### El Hareketiyle Flappy Bird
- **Kontrol:** El hareketleri ile kuşu kontrol edin
- **Amaç:** Boruları geçin ve yüksek skor yapın
- **Özellikler:** Tam ekran modu, gerçek zamanlı el takibi

### Renk Takibi ile Breakout
- **Kontrol:** Renkli nesne ile paddle'ı kontrol edin
- **Amaç:** Tüm tuğları kırın
- **Özellikler:** Çoklu tuğ türleri, güç-up'lar

### Yüz Takibiyle Pac-Man
- **Kontrol:** Yüz hareketleri ile Pac-Man'i kontrol edin
- **Amaç:** Tüm yemleri toplayın
- **Özellikler:** Düşman AI, labirent tasarımı

## 📸 Ekran Görüntüleri

*Oyunların ekran görüntüleri buraya eklenecek*

## 🎯 Özellikler

- ✅ **Çoklu Kontrol Yöntemleri:** El, yüz ve renk takibi
- ✅ **Gerçek Zamanlı İşleme:** Düşük gecikme ile kontrol
- ✅ **Modern Oyun Mekanikleri:** Parçacık efektleri, AI düşmanlar
- ✅ **Kolay Kurulum:** Tek komutla çalıştırma
- ✅ **Eğitici İçerik:** Bilgisayar görüsü öğrenme kaynağı

## 🔧 Sorun Giderme

### Yaygın Sorunlar:

1. **Kamera Erişim Hatası**
   ```bash
   # Kameranın başka bir uygulama tarafından kullanılmadığından emin olun
   ```

2. **MediaPipe Kurulum Hatası**
   ```bash
   pip install --upgrade mediapipe
   ```

3. **Pygame Ses Hatası**
   ```bash
   # Ses sürücülerinizi kontrol edin
   ```

### Performans Optimizasyonu:

- **Düşük FPS:** Kameranın çözünürlüğünü düşürün
- **Yüksek CPU Kullanımı:** MediaPipe ayarlarını optimize edin
- **Gecikme:** Threading kullanımını kontrol edin

## 🤝 Katkıda Bulunma

1. Bu repository'yi fork edin
2. Feature branch oluşturun (`git checkout -b feature/AmazingFeature`)
3. Değişikliklerinizi commit edin (`git commit -m 'Add some AmazingFeature'`)
4. Branch'inizi push edin (`git push origin feature/AmazingFeature`)
5. Pull Request oluşturun

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için `LICENSE` dosyasına bakın.

## 👨‍💻 Geliştirici

**Cavit** - Bilgisayar Görüsü ve Oyun Geliştirme Tutkunu

## 🙏 Teşekkürler

- [MediaPipe](https://mediapipe.dev/) - Google'ın makine öğrenmesi framework'ü
- [OpenCV](https://opencv.org/) - Bilgisayar görüsü kütüphanesi
- [Pygame](https://www.pygame.org/) - Python oyun geliştirme kütüphanesi
- Tüm açık kaynak katkıda bulunanlar

---

⭐ Bu projeyi beğendiyseniz yıldız vermeyi unutmayın!

🎮 **İyi Oyunlar!** 🎮
