# ==================== INSTALAÇÃO TENSORFLOW ====================

# OPÇÃO 1: Instalação Automática (RECOMENDADA)
# Execute este comando no terminal:
pip install tensorflow

# Se der erro, tente:
pip install tensorflow-cpu

# OPÇÃO 2: Versões específicas que funcionam
pip install tensorflow==2.12.0
# OU
pip install tensorflow==2.10.0

# OPÇÃO 3: Para sistemas mais antigos
pip install tensorflow==2.8.0

# ==================== VERIFICAR PYTHON ====================
# Primeiro, verifique sua versão do Python:
python --version

# TensorFlow funciona com:
# ✅ Python 3.8, 3.9, 3.10, 3.11
# ❌ Python 3.12+ (não suportado ainda)
# ❌ Python 3.7 ou menor

# ==================== SCRIPT INSTALAÇÃO COMPLETA ====================
# install_everything.py

"""
Script para instalar TUDO automaticamente
"""
import subprocess
import sys
import platform

def run_command(command):
    """Executa comando e retorna resultado"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def check_python_version():
    """Verifica versão do Python"""
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major != 3:
        print("❌ TensorFlow requer Python 3.x")
        return False
    
    if version.minor < 8:
        print("❌ TensorFlow requer Python 3.8+")
        return False
    
    if version.minor > 11:
        print("⚠️ Python 3.12+ pode ter problemas com TensorFlow")
        print("💡 Recomendado: Python 3.8-3.11")
    
    return True

def install_tensorflow():
    """Tenta instalar TensorFlow"""
    print("\n🔧 Instalando TensorFlow...")
    
    # Lista de versões para tentar
    versions = [
        "tensorflow",
        "tensorflow-cpu", 
        "tensorflow==2.12.0",
        "tensorflow==2.10.0",
        "tensorflow==2.8.0"
    ]
    
    for version in versions:
        print(f"📦 Tentando: {version}")
        success, stdout, stderr = run_command(f"pip install {version}")
        
        if success:
            print(f"✅ {version} instalado!")
            
            # Testar import
            try:
                import tensorflow as tf
                print(f"✅ TensorFlow {tf.__version__} funcionando!")
                return True
            except ImportError:
                print(f"❌ {version} instalado mas não funciona")
                continue
        else:
            print(f"❌ Erro ao instalar {version}")
            if stderr:
                print(f"   Erro: {stderr[:200]}...")
    
    return False

def install_other_packages():
    """Instala outras dependências"""
    packages = [
        "opencv-python",
        "numpy", 
        "matplotlib",
        "scikit-learn",
        "pillow"
    ]
    
    print("\n📦 Instalando outras dependências...")
    
    for package in packages:
        print(f"📦 Instalando {package}...")
        success, _, _ = run_command(f"pip install {package}")
        
        if success:
            print(f"✅ {package} OK")
        else:
            print(f"❌ {package} falhou")

def test_imports():
    """Testa todos os imports necessários"""
    print("\n🧪 Testando imports...")
    
    packages = [
        ("tensorflow", "tf"),
        ("cv2", "opencv-python"),
        ("numpy", "numpy"),
        ("matplotlib", "matplotlib"),
        ("sklearn", "scikit-learn")
    ]
    
    results = {}
    
    for import_name, package_name in packages:
        try:
            if import_name == "tensorflow":
                import tensorflow as tf
                print(f"✅ TensorFlow {tf.__version__}")
                results[package_name] = True
            elif import_name == "cv2":
                import cv2
                print(f"✅ OpenCV {cv2.__version__}")
                results[package_name] = True
            elif import_name == "numpy":
                import numpy as np
                print(f"✅ NumPy {np.__version__}")
                results[package_name] = True
            elif import_name == "matplotlib":
                import matplotlib
                print(f"✅ Matplotlib {matplotlib.__version__}")
                results[package_name] = True
            elif import_name == "sklearn":
                import sklearn
                print(f"✅ Scikit-learn {sklearn.__version__}")
                results[package_name] = True
        except ImportError:
            print(f"❌ {package_name} não encontrado")
            results[package_name] = False
    
    return results

def main():
    print("🏠 INSTALADOR AUTOMÁTICO")
    print("Sistema de Detecção de Invasões")
    print("=" * 50)
    
    # Informações do sistema
    print(f"\n💻 Sistema: {platform.system()} {platform.release()}")
    print(f"🐍 Python: {sys.version}")
    
    # Verificar Python
    if not check_python_version():
        print("\n❌ Versão do Python incompatível!")
        print("💡 Instale Python 3.8-3.11 e tente novamente")
        return
    
    # Atualizar pip
    print("\n📦 Atualizando pip...")
    run_command("python -m pip install --upgrade pip")
    
    # Instalar TensorFlow
    tf_success = install_tensorflow()
    
    if not tf_success:
        print("\n❌ FALHA AO INSTALAR TENSORFLOW!")
        print("\n💡 SOLUÇÕES ALTERNATIVAS:")
        print("1. Use Google Colab (gratuito, TensorFlow pré-instalado)")
        print("2. Instale Anaconda e use: conda install tensorflow")
        print("3. Use um ambiente virtual Python")
        return
    
    # Instalar outras dependências
    install_other_packages()
    
    # Testar tudo
    results = test_imports()
    
    # Resumo final
    print("\n" + "=" * 50)
    print("📋 RESUMO FINAL")
    print("=" * 50)
    
    success_count = sum(results.values())
    total_count = len(results)
    
    if success_count == total_count:
        print("🎉 TUDO INSTALADO COM SUCESSO!")
        print("✅ Você pode executar o sistema agora!")
        print("\n🚀 Para começar:")
        print("   python main.py --mode train")
    else:
        print(f"⚠️ {total_count - success_count} dependências falharam")
        print("\n🔧 Pacotes que falharam:")
        for package, success in results.items():
            if not success:
                print(f"   ❌ {package}")

if __name__ == "__main__":
    main()

# ==================== COMANDOS MANUAIS ====================

# Se o script automático não funcionar, tente estes comandos manualmente:

# 1. Atualizar pip
python -m pip install --upgrade pip

# 2. Instalar TensorFlow (tente na ordem):
pip install tensorflow
# OU
pip install tensorflow-cpu
# OU  
pip install tensorflow==2.12.0

# 3. Instalar outras dependências
pip install opencv-python numpy matplotlib scikit-learn pillow

# 4. Verificar instalação
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"
python -c "import cv2; print('OpenCV:', cv2.__version__)"

# ==================== ALTERNATIVAS ====================

# OPÇÃO A: Usar Google Colab
# 1. Vá para colab.research.google.com
# 2. TensorFlow já vem instalado
# 3. Upload seu código e execute lá

# OPÇÃO B: Usar Anaconda
# 1. Instale Anaconda
# 2. conda create -n invasao python=3.9
# 3. conda activate invasao  
# 4. conda install tensorflow opencv

# OPÇÃO C: Ambiente Virtual
# 1. python -m venv venv_invasao
# 2. venv_invasao\Scripts\activate  (Windows)
# 3. pip install tensorflow opencv-python numpy

# ==================== VERSÃO SEM TENSORFLOW ====================
# Se nada funcionar, posso criar uma versão usando só OpenCV
# para processamento básico de imagem (sem deep learning)

"""
Version without TensorFlow (OpenCV only)
Para casos extremos onde TensorFlow não instala
"""

import cv2
import numpy as np
from collections import deque
import time
import os

class SimpleMotionDetector:
    """Detector simples baseado em diferença de frames"""
    
    def __init__(self, threshold=30):
        self.threshold = threshold
        self.background = None
        self.frame_buffer = deque(maxlen=10)
        
    def detect_motion(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        if self.background is None:
            self.background = gray
            return False, 0
        
        # Diferença com background
        frame_delta = cv2.absdiff(self.background, gray)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        
        # Dilatação para preencher buracos
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Calcular área total de movimento
        total_area = sum(cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 500)
        
        # Atualizar background lentamente
        self.background = cv2.addWeighted(self.background, 0.95, gray, 0.05, 0)
        
        is_anomaly = total_area > self.threshold * 1000
        return is_anomaly, total_area

def simple_detection_opencv():
    """Detecção simples usando apenas OpenCV"""
    print("🔍 Detector Simples (Apenas OpenCV)")
    print("Detecta movimentos grandes/anômalos")
    print("Pressione 'q' para sair")
    
    detector = SimpleMotionDetector(threshold=50)
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        is_anomaly, motion_area = detector.detect_motion(frame)
        
        # Visualização
        color = (0, 0, 255) if is_anomaly else (0, 255, 0)
        status = "MOVIMENTO ANÔMALO!" if is_anomaly else "Normal"
        
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, f"Área: {motion_area:.0f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Detector Simples (OpenCV)', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Para usar o detector simples:
# simple_detection_opencv()