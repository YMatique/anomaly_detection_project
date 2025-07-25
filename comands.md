# Treinamento
python main.py --mode train

# Detecção com webcam
python main.py --mode detect

# Detecção com vídeo específico
python main.py --mode detect --source "video.mp4"

# Threshold personalizado
python main.py --mode detect --threshold 0.03

# Modelo específico
python main.py --mode detect --model "meu_modelo.h5"

# Ajuda
python main.py --help


# Testar dectação em tempo real
python main.py --mode detect

# Testar com vídeo específico
python main.py --mode detect --source "caminho/do/video.mp4"

# Ajustar threshold se necessário

# Threshold mais sensível (detecta mais anomalias)
python main.py --mode detect --threshold 0.03

# Threshold menos sensível (detecta menos anomalias)
python main.py --mode detect --threshold 0.08


# Screenshots
results/screenshots/
├── fixed_20250725_143022_001.png
├── fixed_20250725_143045_002.png
└── ...


# COMANDOS

# Detecção básica
python main.py --mode detect

# Testar sensibilidade
python main.py --mode detect --threshold 0.03  # Mais sensível
python main.py --mode detect --threshold 0.07  # Menos sensível

# Com vídeo de teste
python main.py --mode detect --source "video_teste.mp4"