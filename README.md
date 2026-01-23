# Proyecto: ReID multicámara (Perimetrales)

Resumen rápido
- Código para mantener IDs globales de personas entre múltiples cámaras usando apariencias (OSNet) con fallback HS+ORB.

Requisitos
- Python 3.10+ (se probó con 3.12 en el entorno del repositorio).
- Instalar dependencias:

```bash
python -m pip install -r requirements.txt
```

Notas sobre GPU
- Para usar GPU asegúrate de instalar la versión de `torch` compilada con tu CUDA (por ejemplo `+cu130`).
- Verifica `torch.cuda.is_available()`.

Pesos OSNet
- El código descargará automáticamente los pesos imagenet / Market/Duke cuando sea necesario y los almacenará en `~/.cache/torch/checkpoints/`.
- También puedes colocar manualmente los checkpoints en `models/osnet/`:
  - `models/osnet/osnet_x1_0_market1501.pt`
  - `models/osnet/osnet_x1_0_dukemtmcreid.pt`

Archivos clave
- `src/analityc/core/perimetrales_multicam.py`: Procesador multicámara (ahora soporta extractor OSNet si está disponible).
- `src/analityc/core/botsort_wrapper.py`: Wrapper BoTSORT simplificado que usa OSNet si existe, con fallback HS+ORB.

Prueba rápida
- Ejecuta el script de test sintético (o usa la REPL) para validar que una misma persona en dos posiciones/cámaras conserva `global_id`.

Ejemplo (línea de comandos):

```bash
python -c "from src.analityc.core.perimetrales_multicam import PerimetralesMultiCam; print('OK')"
```

Si quieres que ejecute pruebas reales con tus secuencias de video, indícame las rutas o dame acceso a un par de frames.