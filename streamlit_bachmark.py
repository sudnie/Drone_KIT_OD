import streamlit as st
import torch
import numpy as np
import cv2
import time
import tempfile
import os
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO
import shap

# ========================== 🛠️ 페이지 전역 및 테마 설정 ==========================
st.set_page_config(
    page_title="YOLOv11 드론 정밀 진단 및 실시간 추적 대시보드",
    page_icon="🎯",
    layout="wide"
)

# 한글 폰트가 설치되어 있지 않은 클라우드 환경에서도 그래프가 깨지지 않도록 영문 레이블 혼용 및 Matplotlib 기본 설정 조율
plt.rcParams['figure.facecolor'] = '#f7f7f7'
plt.rcParams['axes.facecolor'] = '#f7f7f7'

# ========================== 🔒 모델 경로 설정 (로컬 및 클라우드 하이브리드 자동 추적) ==========================
# Streamlit Cloud(Linux 배포) 환경과 로컬 Windows 환경을 모두 충족하기 위한 상대/절대 하이브리드 탐색
PT_MODEL_PATH = "models/best_deron.pt" if os.path.exists("models/best_deron.pt") else r"C:\Users\a4376\Documents\VisionAI\projet\best_deron.pt"
OV_MODEL_DIR  = "models/best_deron_int8_openvino_model" if os.path.exists("models/best_deron_int8_openvino_model") else r"C:\Users\a4376\Documents\VisionAI\projet\best_deron_int8_openvino_model"

# ========================== 🧠 캐시 모델 로드 메커니즘 (성능 극대화) ==========================
@st.cache_resource
def load_models(pt_path, ov_path):
    try:
        pt_model = YOLO(pt_path, task='detect')
        ov_model = YOLO(ov_path, task='detect')
        return pt_model, ov_model, None
    except Exception as e:
        return None, None, str(e)

# 모델 로딩 수행
pt_model, ov_model, error_msg = load_models(PT_MODEL_PATH, OV_MODEL_DIR)

if error_msg:
    st.error(f"❌ 가중치 파일 로딩에 실패했습니다. 아래 경로를 확인해 주십시오:\n1. PyTorch 가중치: `{PT_MODEL_PATH}`\n2. OpenVINO 가중치 폴더: `{OV_MODEL_DIR}`\n\n상세 정보: {error_msg}")
    st.stop()

# ========================== 🏠 Streamlit 메인 화면 인터페이스 ==========================
st.title("🎯 YOLOv11 드론 탑지 멀티 하드웨어 성능 검증 및 SHAP 설명가능성 시스템")
st.markdown("""
본 대시보드는 백엔드에서 **YOLOv11 오리지널 모델(PyTorch)**과 **OpenVINO INT8 하드웨어 최적화 모델**의 정합성을 실시간 검증합니다.
아래 세 가지 기능 탭을 통해 **초고속 CPU 추론 대조**, **디바이스 성능 벤치마킹 및 SHAP 게임이론 기여도 분석**, 혹은 **비디오 드론 동적 실시간 추적**을 구동할 수 있습니다.
""")

# ========================== ⚙️ 사이드바 컨트롤 패널 ==========================
st.sidebar.header("⚙️ 진단 제어 및 최적화 설정")

# 1. 🔌 하드웨어 가속 스위치 (기본값: 비활성화)
st.sidebar.subheader("🔌 하드웨어 가속")
enable_gpu = st.sidebar.toggle(
    "CUDA (GPU) 테스트 활성화", 
    value=False, 
    help="서버나 인프라 환경에 NVIDIA GPU 및 CUDA 런타임이 구현되어 있을 경우 활성화하십시오. 비활성화 시 순수 CPU 비교로 구동됩니다."
)

gpu_available = torch.cuda.is_available()
gpu_device = 'cuda:0' if (gpu_available and enable_gpu) else 'cpu'

if enable_gpu:
    if gpu_available:
        st.sidebar.success("🟢 GPU 가속 모드가 성공적으로 활성화되었습니다.")
    else:
        st.sidebar.warning("⚠️ 시스템 내 CUDA GPU 디바이스를 찾을 수 없습니다. 자동으로 CPU 런타임으로 우회합니다.")
else:
    st.sidebar.info("⚪ CPU 전용 하드웨어 벤치마크 모드로 작동 중입니다.")

# 2. 성능 벤치마크 강도 조절
st.sidebar.markdown("---")
st.sidebar.subheader("⏱️ 벤치마크 파라미터")
warmup_runs = st.sidebar.slider("웜업 루프 반복 횟수 (Warmup)", min_value=5, max_value=20, value=10, help="모델 컴파일 및 메모리 할당 지연을 제거하는 사전 실행 연산입니다.")
benchmark_runs = st.sidebar.slider("벤치마크 측정 횟수 (Benchmark)", min_value=10, max_value=100, value=50, help="실제 연산 횟수로, 높은 값일수록 정교한 평균 연산 성능이 도출됩니다.")

# 3. SHAP 수학 기여도 파라미터 조절
st.sidebar.markdown("---")
st.sidebar.subheader("🧩 SHAP 시각화 정밀도")
max_evals = st.sidebar.slider("SHAP 탐색 횟수 (Max Evals)", min_value=100, max_value=1500, value=800, step=50, help="순열 모델에서 격자 영향도를 역추적하는 한계 수치입니다. (513회 이상 권장)")
grid_size = st.sidebar.selectbox("기여도 연산용 그리드 분할수", options=[12, 16, 20], index=1, help="이미지를 분할할 세그먼트 밀도입니다. 16x16 이 가장 안전하고 정밀하게 활성화됩니다.")
batch_size = st.sidebar.slider("SHAP 배치 처리 크기 (Batch Size)", min_value=1, max_value=16, value=4, step=1)

st.sidebar.markdown("---")
st.sidebar.caption("💡 본 웹앱은 Streamlit Community Cloud (Headless Linux 컨테이너 환경) 배포 가이드를 완벽하게 지원합니다.")

# ========================== 📂 핵심 기능 탑재용 멀티 탭 구조 ==========================
tab_fast_cpu, tab_image, tab_video = st.tabs([
    "⚡ CPU 초고속 연산 (PyTorch vs OpenVINO)", 
    "🛸 이미지 심층 분석 (벤치마크 & SHAP)", 
    "🎥 동영상 실시간 추적 (동적 렌더링)"
])

# ---------------------------------------------------------------------------
#                TAB 1: ⚡ CPU 초고속 연산 (PyTorch vs OpenVINO)
# ---------------------------------------------------------------------------
with tab_fast_cpu:
    st.subheader("초고속 CPU 단말 추론 및 탐지 정확도 대조")
    st.markdown("이미지 한 장을 올려 PyTorch CPU 추론과 OpenVINO CPU 추론의 원본 검출 박스 및 프레임 성능 차이를 수 초 안에 계산하여 정량적 데이터로 제시합니다.")
    
    uploaded_fast = st.file_uploader("📂 이미지 탐지 및 속도 분석용 드론 사진 선택...", type=["jpg", "jpeg", "png", "bmp", "webp"], key="fast_uploader")
    
    if uploaded_fast is not None:
        file_bytes = np.asarray(bytearray(uploaded_fast.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, 1)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        st.image(img_rgb, caption="업로드한 원본 프레임", use_container_width=True)
        
        if st.button("⚡ CPU 초고속 추론 테스트 시작", type="primary", key="btn_run_fast"):
            # 세분화된 진행률 레이아웃 구현
            fast_progress = st.progress(0)
            fast_status = st.empty()
            
            fast_warmup = 2
            fast_bench = 5
            
            cpu_configs = [
                {"name": "PyTorch (CPU)", "model": pt_model},
                {"name": "OpenVINO (CPU)", "model": ov_model}
            ]
            
            fast_results = {}
            fast_frames = {}
            
            # 단계 1: PyTorch CPU 연산
            fast_status.text("⏳ [단계 1 / 4] PyTorch (CPU) 연산 준비 및 웜업(2회) 시작...")
            fast_progress.progress(15)
            for _ in range(fast_warmup):
                _ = pt_model(img_bgr, device="cpu", verbose=False)
                
            fast_status.text("⏳ [단계 2 / 4] PyTorch (CPU) 5회 정밀 지연 시간 측정 중...")
            fast_progress.progress(40)
            latencies_pt = []
            for _ in range(fast_bench):
                t0 = time.perf_counter()
                _ = pt_model(img_bgr, device="cpu", verbose=False)
                t1 = time.perf_counter()
                latencies_pt.append((t1 - t0) * 1000)
                
            avg_lat_pt = np.mean(latencies_pt)
            res_pt = pt_model(img_bgr, device="cpu", verbose=False)[0]
            max_conf_pt = float(torch.max(res_pt.boxes.conf)) if len(res_pt.boxes) > 0 else 0.0
            
            fast_results["PyTorch (CPU)"] = {"latency": avg_lat_pt, "fps": 1000/avg_lat_pt, "conf": max_conf_pt, "count": len(res_pt.boxes)}
            fast_frames["PyTorch (CPU)"] = cv2.cvtColor(res_pt.plot(), cv2.COLOR_BGR2RGB)
            
            # 단계 2: OpenVINO CPU 연산
            fast_status.text("⏳ [단계 3 / 4] OpenVINO (CPU) 최적화 웜업(2회) 구동 중...")
            fast_progress.progress(65)
            for _ in range(fast_warmup):
                _ = ov_model(img_bgr, device="cpu", verbose=False)
                
            fast_status.text("⏳ [단계 4 / 4] OpenVINO (CPU) 5회 실시간 고속 연산 및 대조 중...")
            fast_progress.progress(90)
            latencies_ov = []
            for _ in range(fast_bench):
                t0 = time.perf_counter()
                _ = ov_model(img_bgr, device="cpu", verbose=False)
                t1 = time.perf_counter()
                latencies_ov.append((t1 - t0) * 1000)
                
            avg_lat_ov = np.mean(latencies_ov)
            res_ov = ov_model(img_bgr, device="cpu", verbose=False)[0]
            max_conf_ov = float(np.max(res_ov.boxes.conf.cpu().numpy())) if len(res_ov.boxes) > 0 else 0.0
            
            fast_results["OpenVINO (CPU)"] = {"latency": avg_lat_ov, "fps": 1000/avg_lat_ov, "conf": max_conf_ov, "count": len(res_ov.boxes)}
            fast_frames["OpenVINO (CPU)"] = cv2.cvtColor(res_ov.plot(), cv2.COLOR_BGR2RGB)
            
            # 진행 상태 바 소멸
            fast_progress.progress(100)
            time.sleep(0.3)
            fast_progress.empty()
            fast_status.empty()
            
            st.success("🎉 CPU 초고속 추론 및 대조 분석이 완료되었습니다!")
            
            # 정량 수학적 비교 분석 계산
            speedup_ratio = avg_lat_pt / avg_lat_ov
            latency_diff = avg_lat_pt - avg_lat_ov
            percent_improved = ((avg_lat_pt - avg_lat_ov) / avg_lat_pt) * 100
            
            # 메트릭 카드 시각화
            st.markdown("### 📊 하드웨어 최적화 가속 성능 지표 (CPU Performance Matrix)")
            col_calc1, col_calc2, col_calc3 = st.columns(3)
            with col_calc1:
                st.metric(
                    label="🚀 CPU 가속 비율 (Relative Speedup)", 
                    value=f"{speedup_ratio:.2f} 배 향상", 
                    delta=f"OpenVINO 지연 시간 대폭 개선"
                )
            with col_calc2:
                st.metric(
                    label="⏱️ 단일 프레임 단축 시간 (Latency Delta)", 
                    value=f"{latency_diff:.2f} ms", 
                    delta=f"기존 PyTorch CPU 대비 {percent_improved:.1f}% 속도 향상",
                    delta_color="normal"
                )
            with col_calc3:
                st.metric(
                    label="🧩 실시간 검출 개수 대조",
                    value=f"OpenVINO: {fast_results['OpenVINO (CPU)']['count']}개",
                    delta=f"PyTorch CPU: {fast_results['PyTorch (CPU)']['count']}개"
                )
            
            st.markdown("---")
            
            # 이미지 렌더링 출력 비교
            st.markdown("### 🔲 검출 경계상자 출력 정밀도 비교 (Detected Bounding Boxes)")
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                st.image(
                    fast_frames["PyTorch (CPU)"], 
                    caption=f"PyTorch (CPU) | 지연시간: {avg_lat_pt:.1f}ms | 연산속도: {fast_results['PyTorch (CPU)']['fps']:.1f} FPS | 신뢰도: {max_conf_pt:.2f}", 
                    use_container_width=True
                )
            with col_f2:
                st.image(
                    fast_frames["OpenVINO (CPU)"], 
                    caption=f"OpenVINO (CPU) | 지연시간: {avg_lat_ov:.1f}ms | 연산속도: {fast_results['OpenVINO (CPU)']['fps']:.1f} FPS | 신뢰도: {max_conf_ov:.2f}", 
                    use_container_width=True
                )

# ---------------------------------------------------------------------------
#                      TAB 2: 🛸 이미지 심층 분석 (벤치마크 & SHAP)
# ---------------------------------------------------------------------------
with tab_image:
    st.subheader("심층 성능 벤치마크 및 인공지능설명가능성(XAI/SHAP) 분석")
    st.markdown("다각도 벤치마킹을 실행하고 게임이론(Shapley Value)을 픽셀 수준으로 계산하여 모델이 드론의 어느 영역(날개, 프레임 등)을 보며 분류를 수행하는지 입증합니다.")
    
    uploaded_image = st.file_uploader("📂 정밀 진단용 고해상도 드론 이미지 등록...", type=["jpg", "jpeg", "png", "bmp", "webp"], key="img_uploader")

    if uploaded_image is not None:
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        img_original_bgr = cv2.imdecode(file_bytes, 1)
        img_original_rgb = cv2.cvtColor(img_original_bgr, cv2.COLOR_BGR2RGB)
        
        img_640_rgb = cv2.resize(img_original_rgb, (640, 640))
        img_640_bgr = cv2.cvtColor(img_640_rgb, cv2.COLOR_RGB2BGR)
        img_input = img_640_rgb.astype(np.float32) / 255.0
        
        st.image(img_original_rgb, caption="업로드된 분석 정밀 원본", use_container_width=True)
        
        if st.button("🚀 정밀 종합 진단 모델 실행 (대량 연산)", type="primary", key="btn_run_img"):
            # 초정밀 단계 분할형 로더 구현 (12단계 분리)
            step_progress = st.progress(0)
            step_status = st.empty()
            
            configs = []
            if enable_gpu:
                configs.append({"name": "PyTorch (GPU)", "model": pt_model, "device": gpu_device})
            configs.extend([
                {"name": "PyTorch (CPU)", "model": pt_model, "device": "cpu"},
                {"name": "OpenVINO (CPU)", "model": ov_model, "device": "cpu"}
            ])
            
            benchmark_results = {}
            annotated_frames = {}
            
            # --- 파트 A: 성능 벤치마킹 단계 세분화 ---
            for c_idx, cfg in enumerate(configs):
                cfg_name = cfg["name"]
                model_inst = cfg["model"]
                dev = cfg["device"]
                
                # 웜업 단계
                step_status.text(f"⏳ [단계 {c_idx*2 + 1} / {len(configs)*2 + 6}] {cfg_name} 가속 구동 및 {warmup_runs}회 웜업 연산 중...")
                step_progress.progress(int((c_idx*2 + 1) * (40 / (len(configs)*2 + 6))))
                for _ in range(warmup_runs):
                    _ = model_inst(img_640_bgr, device=dev, verbose=False)
                
                # 정밀 벤치마킹 연산 단계
                step_status.text(f"⏳ [단계 {c_idx*2 + 2} / {len(configs)*2 + 6}] {cfg_name} 하드웨어 {benchmark_runs}회 연속 기가플롭스 처리율 연산 중...")
                step_progress.progress(int((c_idx*2 + 2) * (40 / (len(configs)*2 + 6))))
                
                lats = []
                for _ in range(benchmark_runs):
                    t0 = time.perf_counter()
                    _ = model_inst(img_640_bgr, device=dev, verbose=False)
                    t1 = time.perf_counter()
                    lats.append((t1 - t0) * 1000)
                    
                avg_l = np.mean(lats)
                fps_val = 1000 / avg_l
                
                res_temp = model_inst(img_640_bgr, device=dev, verbose=False)[0]
                box_count = len(res_temp.boxes)
                c_max = float(torch.max(res_temp.boxes.conf)) if box_count > 0 and isinstance(res_temp.boxes.conf, torch.Tensor) else (float(np.max(res_temp.boxes.conf)) if box_count > 0 else 0.0)
                
                benchmark_results[cfg_name] = {"latency": avg_l, "fps": fps_val, "conf": c_max, "count": box_count}
                annotated_frames[cfg_name] = cv2.cvtColor(res_temp.plot(), cv2.COLOR_BGR2RGB)

            # --- 파트 B: SHAP 정밀 마스킹 모델 구성 단계 ---
            base_step = len(configs)*2
            step_status.text(f"⏳ [단계 {base_step + 1} / {base_step + 6}] SHAP 인페인팅 래퍼 수립 및 가상 배경 정렬 중...")
            step_progress.progress(45)
            
            bg_reference = np.ones_like(img_input) * 0.5 
            
            def custom_masker(mask):
                B = mask.shape[0]
                masked_imgs = []
                mask_2d = mask.reshape(-1, grid_size, grid_size)
                for i in range(B):
                    mask_upscaled = cv2.resize(mask_2d[i].astype(np.uint8), (640, 640), interpolation=cv2.INTER_NEAREST)
                    mask_3d = np.expand_dims(mask_upscaled, axis=-1)
                    out_img = img_input * mask_3d + bg_reference * (1 - mask_3d)
                    masked_imgs.append(out_img)
                return np.array(masked_imgs)

            def predict_pt(img_numpy_batch):
                outputs = []
                for img_np in img_numpy_batch:
                    img_bgr_in = (img_np * 255).astype(np.uint8)
                    results = pt_model(img_bgr_in, device=gpu_device, verbose=False)
                    boxes = results[0].boxes
                    uav_conf = float(torch.max(boxes.conf)) if len(boxes) > 0 else 0.0
                    outputs.append([1.0 - uav_conf, uav_conf])
                return np.array(outputs)

            def predict_ov(img_numpy_batch):
                outputs = []
                for img_np in img_numpy_batch:
                    img_bgr_in = (img_np * 255).astype(np.uint8)
                    results = ov_model(img_bgr_in, device='cpu', verbose=False)
                    boxes = results[0].boxes
                    uav_conf = float(np.max(boxes.conf.cpu().numpy())) if len(boxes) > 0 else 0.0
                    outputs.append([1.0 - uav_conf, uav_conf])
                return np.array(outputs)

            num_features = grid_size * grid_size
            pt_f = lambda m: predict_pt(custom_masker(m))
            ov_f = lambda m: predict_ov(custom_masker(m))

            # PyTorch SHAP 기여도 연산
            step_status.text(f"⏳ [단계 {base_step + 2} / {base_step + 6}] PyTorch 모델을 위한 {max_evals}회 분할 SHAP 순열 분포 수렴 중...")
            step_progress.progress(60)
            explainer_pt = shap.Explainer(pt_f, shap.maskers.Independent(np.zeros((1, num_features))))
            shap_values_pt = explainer_pt(np.ones((1, num_features)), max_evals=max_evals, batch_size=batch_size)
            
            # OpenVINO SHAP 기여도 연산
            step_status.text(f"⏳ [단계 {base_step + 3} / {base_step + 6}] OpenVINO 최적화 맵을 위한 {max_evals}회 분할 SHAP 순열 연산 작동 중...")
            step_progress.progress(80)
            explainer_ov = shap.Explainer(ov_f, shap.maskers.Independent(np.zeros((1, num_features))))
            shap_values_ov = explainer_ov(np.ones((1, num_features)), max_evals=max_evals, batch_size=batch_size)
            
            # 매트릭스 재구성 및 크기 확장
            step_status.text(f"⏳ [단계 {base_step + 4} / {base_step + 6}] 2차원 공간 샤플리 데이터 정렬 및 이중 선형 보간 처리 중...")
            step_progress.progress(90)
            pt_values = shap_values_pt.values[0, :, 1].reshape(grid_size, grid_size)
            ov_values = shap_values_ov.values[0, :, 1].reshape(grid_size, grid_size)
            pt_heatmap = cv2.resize(pt_values, (640, 640), interpolation=cv2.INTER_LINEAR)
            ov_heatmap = cv2.resize(ov_values, (640, 640), interpolation=cv2.INTER_LINEAR)
            
            # 최종 렌더링 준비
            step_status.text(f"⏳ [단계 {base_step + 5} / {base_step + 6}] 하드웨어 가속 다이어그램 및 색인 테이블 생성 중...")
            step_progress.progress(98)
            time.sleep(0.3)
            
            step_progress.progress(100)
            step_status.empty()
            step_progress.empty()
            
            st.success("🎉 정밀 종합 진단 모델 및 XAI 기여도 분석 리포트가 완성되었습니다!")
            
            # 정량 분석 데이터 산출
            pt_cpu_lat = benchmark_results["PyTorch (CPU)"]["latency"]
            ov_cpu_lat = benchmark_results["OpenVINO (CPU)"]["latency"]
            cpu_speedup = pt_cpu_lat / ov_cpu_lat
            cpu_reduced_ms = pt_cpu_lat - ov_cpu_lat
            cpu_reduced_pct = (cpu_reduced_ms / pt_cpu_lat) * 100
            
            # KPI 출력
            st.markdown("### 📊 다중 디바이스 런타임 성능 대조 지표 (Runtime Metrics)")
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.metric(
                    label="⚡ CPU 최적화 가속 성능", 
                    value=f"{cpu_speedup:.2f} 배 개선", 
                    delta=f"OpenVINO 최적화 분기 동작 완료"
                )
            with col_m2:
                st.metric(
                    label="⏱️ 단일 프레임 지연 단축", 
                    value=f"{cpu_reduced_ms:.2f} ms", 
                    delta=f"연산 부하 {cpu_reduced_pct:.1f}% 감소",
                    delta_color="normal"
                )
            with col_m3:
                if enable_gpu and "PyTorch (GPU)" in benchmark_results:
                    gpu_lat = benchmark_results["PyTorch (GPU)"]["latency"]
                    gpu_ratio = pt_cpu_lat / gpu_lat
                    st.metric(
                        label="🚀 GPU 가속 대조 비율", 
                        value=f"{gpu_ratio:.1f} 배", 
                        delta=f"GPU 단일 프레임 처리 지연: {gpu_lat:.1f} ms",
                        delta_color="normal"
                    )
                else:
                    st.metric(
                        label="🔌 GPU 성능 진단 슬롯", 
                        value="비활성화", 
                        delta="사이드바 가속기를 켜 주십시오."
                    )
                    
            st.markdown("---")
            
            # 바운딩박스 출력군
            st.markdown("### 🔲 검출 경계상자 출력 결과 비교 (BBox Outputs)")
            col_b = st.columns(len(configs))
            for b_idx, name_b in enumerate(benchmark_results.keys()):
                with col_b[b_idx]:
                    st.image(
                        annotated_frames[name_b], 
                        caption=f"{name_b} | 지연시간: {benchmark_results[name_b]['latency']:.1f}ms | 객체수: {benchmark_results[name_b]['count']}개 | 신뢰도: {benchmark_results[name_b]['conf']:.2f}", 
                        use_container_width=True
                    )
                
            st.markdown("---")
            
            # SHAP 열지도 대조군
            st.markdown("### 🔮 특징 가중치 활성화 기여도 비교 (SHAP Interpretability)")
            st.info("💡 SHAP 해석법: 【빨간색 영역】은 드론의 실루엣이나 특징점으로서 탐지 확률을 높여준 핵심 영역입니다. 반대로 【파란색 영역】은 배경의 노이즈 등으로 판단을 방해한 부분입니다.")
            
            fig_shap, axes_shap = plt.subplots(1, 2, figsize=(14, 6))
            v_max = max(np.max(np.abs(pt_heatmap)), np.max(np.abs(ov_heatmap)))
            
            axes_shap[0].imshow(img_640_rgb)
            im0 = axes_shap[0].imshow(pt_heatmap, cmap='seismic', alpha=0.5, vmin=-v_max, vmax=v_max)
            axes_shap[0].set_title("SHAP Explanation - PyTorch (.pt)", fontsize=12, fontweight='bold')
            axes_shap[0].axis('off')
            
            axes_shap[1].imshow(img_640_rgb)
            im1 = axes_shap[1].imshow(ov_heatmap, cmap='seismic', alpha=0.5, vmin=-v_max, vmax=v_max)
            axes_shap[1].set_title("SHAP Explanation - OpenVINO (INT8)", fontsize=12, fontweight='bold')
            axes_shap[1].axis('off')
            
            cbar_ax = fig_shap.add_axes([0.25, 0.05, 0.5, 0.03])
            cbar = fig_shap.colorbar(im1, cax=cbar_ax, orientation='horizontal')
            cbar.set_label("Shapley Value (Red: Promotes Drone Identification | Blue: Background Interference)", fontsize=10, fontweight='bold')
            
            col_s1, col_s2 = st.columns([4, 1])
            with col_s1:
                st.pyplot(fig_shap)
                
            st.markdown("---")
            
            # 벤치마킹 성능 바차트
            st.markdown("### 📊 정밀 벤치마크 처리 성능 비교 차트 (Latency vs Throughput)")
            fig_perf, ax_perf = plt.subplots(figsize=(10, 4))
            labels = list(benchmark_results.keys())
            perf_latencies = [benchmark_results[l]["latency"] for l in labels]
            perf_fps = [benchmark_results[l]["fps"] for l in labels]
            
            x_indices = np.arange(len(labels))
            bar_width = 0.3
            
            color_lat = '#ff7f0e'
            ax_perf.set_ylabel('Inference Latency (ms)', color=color_lat, fontweight='bold')
            r1 = ax_perf.bar(x_indices - bar_width/2, perf_latencies, bar_width, label='Latency (ms)', color=color_lat, alpha=0.8)
            ax_perf.tick_params(axis='y', labelcolor=color_lat)
            
            ax_fps = ax_perf.twinx()
            color_fps = '#1f77b4'
            ax_fps.set_ylabel('Throughput (FPS)', color=color_fps, fontweight='bold')
            r2 = ax_fps.bar(x_indices + bar_width/2, perf_fps, bar_width, label='FPS', color=color_fps, alpha=0.8)
            ax_fps.tick_params(axis='y', labelcolor=color_fps)
            
            ax_perf.bar_label(r1, fmt='%.1f ms', padding=3, fontsize=9)
            ax_fps.bar_label(r2, fmt='%.1f FPS', padding=3, fontsize=9)
            
            ax_perf.set_xticks(x_indices)
            ax_perf.set_xticklabels(labels, fontweight='bold')
            ax_perf.grid(True, linestyle='--', alpha=0.5)
            
            st.pyplot(fig_perf)

# ---------------------------------------------------------------------------
#                      TAB 3: 🎥 동영상 실시간 추적 (동적 렌더링)
# ---------------------------------------------------------------------------
with tab_video:
    st.subheader("실시간 드론 비행 동적 추적 및 다운로드 시스템")
    st.markdown("동영상을 등록하고 하드웨어 백엔드를 지정하면 YOLOv11 모델이 한 장씩 추적을 수행하며 프레임상에 표식을 렌더링해 다운로드용 파일로 제작합니다.")
    
    uploaded_video = st.file_uploader("📂 추적 모의시험용 드론 비행 비디오 파일 등록...", type=["mp4", "avi", "mov", "mkv"], key="vid_uploader")
    
    if uploaded_video is not None:
        st.markdown("#### ⚙️ 비디오 실시간 런타임 제어")
        
        # GPU 활성 상태에 맞춰 무선 선택 가속 항목 변경
        engine_options = ["PyTorch (CPU)", "OpenVINO (CPU)"]
        if enable_gpu and gpu_available:
            engine_options.insert(0, "PyTorch (GPU)")
            
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            engine_option = st.radio(
                "🚀 드론 추적 실행 백엔드 선택:",
                options=engine_options,
                help="GPU 하드웨어가 활성화되어 있다면 PyTorch GPU 버전을, 그 외 임베디드 저사양 및 데스크톱 환경에서는 OpenVINO CPU 버전 가속화를 권장합니다."
            )
        with col_v2:
            conf_threshold = st.slider("객체 인식 탐지 신뢰도 임계값 (Confidence Threshold)", min_value=0.05, max_value=0.95, value=0.25, step=0.05)
            
        selected_model = pt_model
        selected_device = 'cpu'
        
        if engine_option == "PyTorch (GPU)":
            selected_model = pt_model
            selected_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        elif engine_option == "PyTorch (CPU)":
            selected_model = pt_model
            selected_device = 'cpu'
        elif engine_option == "OpenVINO (CPU)":
            selected_model = ov_model
            selected_device = 'cpu'
            
        if st.button("🎥 실시간 비디오 드론 추적 연산 시작", type="primary", key="btn_run_video"):
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_video.read())
            tfile.close()
            
            cap = cv2.VideoCapture(tfile.name)
            
            if not cap.isOpened():
                st.error("❌ 비디오 파일을 읽을 수 없습니다. (H.264 코덱 및 MP4 확장자를 권장합니다.)")
            else:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                st.info(f"📊 비디오 프로필 추출 성공: 해상도 {width}x{height} | 고유 프레임률 {fps:.1f} FPS | 전체 프레임 수 {total_frames} 프레임")
                
                # 영상 렌더링 프레임 뷰어 및 바 분할
                video_placeholder = st.empty()
                progress_bar_v = st.progress(0)
                status_text_v = st.empty()
                
                out_tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                out_tfile.close()
                
                # 디스플레이에 보장하는 mp4v 비디오 껍데기 포맷팅
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out_writer = cv2.VideoWriter(out_tfile.name, fourcc, fps, (width, height))
                
                frame_count = 0
                start_track_time = time.time()
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                        
                    frame_count += 1
                    
                    results = selected_model(frame, device=selected_device, conf=conf_threshold, verbose=False)
                    annotated_frame = results[0].plot()
                    
                    out_writer.write(annotated_frame)
                    
                    frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    # 프레임별 미디어를 브라우저 플레이어에 지속 주입
                    video_placeholder.image(frame_rgb, caption=f"추적 및 시각화 프레임 동적 렌더링 중: [{frame_count}/{total_frames}]", use_container_width=True)
                    
                    # 렌더링 세분화 진척율 출력
                    progress_percent = int((frame_count / total_frames) * 100)
                    progress_bar_v.progress(progress_percent)
                    status_text_v.text(f"🚀 드론 프레임 특징 피쳐 실시간 추출 중... 현재 처리 프레임: {frame_count}/{total_frames} ({progress_percent}%)")
                
                cap.release()
                out_writer.release()
                
                total_duration = time.time() - start_track_time
                avg_fps_run = frame_count / total_duration
                
                progress_bar_v.empty()
                status_text_v.empty()
                video_placeholder.empty()
                
                st.success(f"🎉 드론 동적 추적 렌더링이 완료되었습니다! 분석 소요 시간: {total_duration:.1f}초 | 하드웨어 처리율: {avg_fps_run:.1f} FPS")
                
                with open(out_tfile.name, 'rb') as f:
                    st.download_button(
                        label="📥 추적이 완성된 최종 렌더링 결과 비디오 다운로드 (.mp4)",
                        data=f,
                        file_name="drone_tracked_output.mp4",
                        mime="video/mp4"
                    )
