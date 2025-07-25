"""
Detecção em tempo real
"""
def real_time_detection(model_path: str = "models/best_model.h5", 
                       threshold: float = 0.05,
                       source: int = 0):
    """Executa detecção em tempo real"""
    
    # Verificar se modelo existe
    if not os.path.exists(model_path):
        print(f"Modelo não encontrado em {model_path}")
        print("Execute o treinamento primeiro com: python train.py")
        return
    
    # Inicializar detector
    print("Carregando modelo...")
    detector = AnomalyDetector(model_path, threshold)
    
    # Inicializar câmera
    print("Inicializando câmera...")
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print("Erro ao abrir câmera")
        return
    
    print("Iniciando detecção em tempo real...")
    print("Pressione 'q' para sair, 's' para capturar screenshot")
    
    screenshot_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Erro ao capturar frame")
                break
            
            # Detectar anomalia
            is_anomaly, error, proc_time = detector.detect_anomaly(frame)
            
            # Preparar visualização
            display_frame = frame.copy()
            
            # Cor e texto baseado na detecção
            if is_anomaly:
                color = (0, 0, 255)  # Vermelho
                status = "INVASAO DETECTADA!"
                thickness = 3
            else:
                color = (0, 255, 0)  # Verde
                status = "NORMAL"
                thickness = 2
            
            # Adicionar informações no frame
            cv2.putText(display_frame, status, (10, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, thickness)
            cv2.putText(display_frame, f"Erro: {error:.4f}", (10, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Threshold: {threshold:.4f}", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # FPS
            if proc_time > 0:
                fps = 1.0 / proc_time
                cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 160), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Mostrar frame
            cv2.imshow('Detector de Invasões - ConvLSTM Autoencoder', display_frame)
            
            # Controles
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Salvar screenshot
                screenshot_name = f"screenshot_{screenshot_count:03d}.png"
                cv2.imwrite(screenshot_name, display_frame)
                print(f"Screenshot salvo: {screenshot_name}")
                screenshot_count += 1
                
    except KeyboardInterrupt:
        print("\nDetecção interrompida pelo usuário")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        # Estatísticas finais
        stats = detector.get_statistics()
        print("\n=== ESTATÍSTICAS FINAIS ===")
        for key, value in stats.items():
            print(f"{key}: {value:.4f}")