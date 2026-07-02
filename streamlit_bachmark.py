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

# ========================== 🛠️ 페이지 전역 설정 ==========================
st.set_page_config(
    page_title="YOLOv11 드론 정밀 분석 및 설명가능한 AI(XAI) 시스템",
    page_icon="🎯",
    layout="wide"
)

# Linux 헤드리스 서버 환경에서 시각화 차트 배경 조율
plt.rcParams['figure.facecolor'] = '#f7f7f7'
plt.rcParams['axes.facecolor'] = '#f7f7f7'

# ========================== 🔒 모델 가중치 경로 설정 ==========================
# 로컬 Windows 환경 및 클라우드 Linux 환경 하이브리드 자동 추적
PT_MODEL_PATH = "models/best_deron.pt" if os.path.exists("models/best_deron.pt") else r"C:\Users\a4376\Documents\VisionAI\projet\best_deron.pt"
OV_MODEL_DIR  = "models/best_deron_int8_openvino_model" if os.path.exists("models/best_deron_int8_openvino_model") else r"C:\Users\a4376\Documents\VisionAI\projet\best_deron_int8_openvino_model"

# ========================== 🧠 캐시 모델 로드 메커니즘 ==========================
@st.cache_resource
def load_models(pt_path, ov_path):
    try:
        pt_model = YOLO(pt_path, task='detect')
        ov_model = YOLO(ov_path, task='detect')
        return pt_model, ov_model, None
    except Exception as e:
        return None, None, str(e)

# 모델 로드 실행
pt_model, ov_model, error_msg = load_models(PT_MODEL_PATH, OV_MODEL_DIR)

if error_msg:
    st.error(f"❌ 모델 초기화 실패! 지정된 경로에 가중치 파일이 존재하는지 확인하십시오:\n1. PyTorch: `{PT_MODEL_PATH}`\n2. OpenVINO: `{OV_MODEL_DIR}`\n\n상세 오류: {error_msg}")
    st.stop()

# ========================== 🏠 Streamlit 메인 화면 인터페이스 ==========================
st.title("🎯 YOLOv11 드론 탐지 멀티 엔진 성능 검증 및 SHAP 기여도 분석 시스템")
st.markdown("""
본 시스템은 백엔드에서 **YOLOv11 원본 모델(PyTorch)**과 **OpenVINO INT8 최적화 모델**을 성공적으로 로딩하였습니다.
아래 기능 탭을 선택하여 초고속 CPU 추론 대조, 디바이스 성능 벤치마킹 및 SHAP 게임이론 기반 분석, 또는 실시간 비디오 추적을 구동할 수 있습니다.
""")

# ========================== ⚙️ 사이드바 설정 제어 패널 ==========================
st.sidebar.header("⚙️ 진단 제어 및 최적화 설정")

# 1. 🔌 하드웨어 가속 설정 (CUDA 가속 스위치, 기본값 비활성화)
st.sidebar.subheader("🔌 하드웨어 가속")
enable_gpu = st.sidebar.toggle(
    "CUDA (GPU) 테스트 활성화", 
    value=False, 
    help="로컬 또는 서버 환경에 NVIDIA GPU 및 CUDA 가속 환경이 구축되어 있을 경우 활성화하십시오. 비활성화 시 순수 CPU 비교로 작동합니다."
)

gpu_available = torch.cuda.is_available()
gpu_device = 'cuda:0' if (gpu_available and enable_gpu) else 'cpu'

if enable_gpu:
    if gpu_available:
        st.sidebar.success("🟢 CUDA 가속 모드가 성공적으로 활성화되었습니다.")
    else:
        st.sidebar.warning("⚠️ 가용한 CUDA 환경을 탐지할 수 없습니다. CPU 실행 모드로 자동 우회합니다.")
else:
    st.sidebar.info("⚪ CPU 전용 하드웨어 벤치마크 모드로 작동 중입니다.")

# 2. 성능 벤치마크 반복 주기 제어
st.sidebar.markdown("---")
st.sidebar.subheader("⏱️ 벤치마크 설정")
warmup_runs = st.sidebar.slider("웜업 루프 반복 횟수 (Warmup)", min_value=5, max_value=20, value=10, help="모델 컴파일 및 하드웨어 초기 기동 지연 노이즈를 배제하기 위한 사전 연산입니다.")
benchmark_runs = st.sidebar.slider("성능 테스트 횟수 (Benchmark)", min_value=10, max_value=100, value=50, help="평균 프레임 및 처리 속도(ms) 산출을 위한 실제 테스트 횟수입니다.")

# 3. SHAP 수학적 가시화 정밀도 설정
st.sidebar.markdown("---")
st.sidebar.subheader("🧩 SHAP 시각화 정밀도")
max_evals = st.sidebar.slider("SHAP 탐색 횟수 (Max Evals)", min_value=100, max_value=1500, value=800, step=50, help="값이 클수록 픽셀 기여도가 정교해지지만, 계산 지연이 늘어납니다.")
grid_size = st.sidebar.selectbox("분할 그리드(Grid) 크기", options=[12, 16, 20], index=1)
batch_size = st.sidebar.slider("SHAP 배치 크기 (Batch Size)", min_value=1, max_value=16, value=4, step=1)

st.sidebar.markdown("---")
st.sidebar.caption("💡 본 시스템은 Streamlit 컨테이너 배포 표준을 준수하며 클라우드 가상화 인프라 환경에서 원활히 작동합니다.")

# ========================== 📂 멀티 기능 탭 내비게이션 구조 ==========================
tab_fast_cpu, tab_image, tab_video = st.tabs([
    "⚡ CPU 초고속 연산 (PyTorch vs OpenVINO)", 
    "🛸 이미지 심층 분석 (성능 검증 & SHAP)", 
    "🎥 동영상 실시간 추적 (동적 렌더링)"
])

# ---------------------------------------------------------------------------
#                TAB 1: ⚡ CPU 초고속 연산 (PyTorch vs OpenVINO)
# ---------------------------------------------------------------------------
with tab_fast_cpu:
    st.subheader("CPU 단말 초고속 추론 및 검출 성능 대조")
    st.markdown("이미지를 업로드하면 신속하게 최소 루프(웜업 2회, 추론 5회)만을 구동하여 PyTorch CPU와 OpenVINO CPU의 검출 경계상자 및 연산 지연 속도를 실시간 비교 분석합니다.")
    
    uploaded_fast = st.file_uploader("📂 비교 연산에 사용할 드론 이미지 선택...", type=["jpg", "jpeg", "png", "bmp", "webp"], key="fast_uploader")
    
    if uploaded_fast is not None:
        file_bytes = np.asarray(bytearray(uploaded_fast.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, 1)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # 2026 표준 스펙 적용: use_container_width=True 대신 width='stretch' 사용
        st.image(img_rgb, caption="업로드된 원본 이미지", width='stretch')
        
        if st.button("⚡ CPU 초고속 연산 시작", type="primary", key="btn_run_fast"):
            fast_progress = st.progress(0)
            fast_status = st.empty()
            
            fast_warmup = 2
            fast_bench = 5
            
            fast_results = {}
            fast_frames = {}
            
            fast_status.text("⏳ [단계 1 / 4] PyTorch (CPU) 로드 및 웜업 연산(2회) 진행 중...")
            fast_progress.progress(20)
            for _ in range(fast_warmup):
                _ = pt_model(img_bgr, device="cpu", verbose=False)
                
            fast_status.text("⏳ [단계 2 / 4] PyTorch (CPU) 5회 정밀 단일 프레임 지연 측정 중...")
            fast_progress.progress(45)
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
            
            fast_status.text("⏳ [단계 3 / 4] OpenVINO (CPU) 이종 가속기 활성화 및 웜업(2회) 진행 중...")
            fast_progress.progress(70)
            for _ in range(fast_warmup):
                _ = ov_model(img_bgr, device="cpu", verbose=False)
                
            fast_status.text("⏳ [단계 4 / 4] OpenVINO (CPU) INT8 최적화 프레임 연산 및 대조 중...")
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
            
            fast_progress.progress(100)
            time.sleep(0.3)
            fast_progress.empty()
            fast_status.empty()
            
            st.success("🎉 CPU 초고속 연산 및 대조 분석이 성공적으로 마무리되었습니다!")
            
            # 수학적 벤치마킹 비교 수치 도출
            speedup_ratio = avg_lat_pt / avg_lat_ov
            latency_diff = avg_lat_pt - avg_lat_ov
            percent_improved = ((avg_lat_pt - avg_lat_ov) / avg_lat_pt) * 100
            
            st.markdown("### 📊 하드웨어 최적화 가속 지표 비교 (CPU Performance Contrast)")
            col_calc1, col_calc2, col_calc3 = st.columns(3)
            with col_calc1:
                st.metric(
                    label="🚀 CPU 가속 비율 (Relative Speedup)", 
                    value=f"{speedup_ratio:.2f} 배 개선됨", 
                    delta="OpenVINO 최적화 분기 동작 완료"
                )
            with col_calc2:
                st.metric(
                    label="⏱️ 단일 프레임 지연 감소폭 (Latency Reduction)", 
                    value=f"{latency_diff:.2f} ms 단축", 
                    delta=f"연산 소요 지연 {percent_improved:.1f}% 개선",
                    delta_color="normal"
                )
            with col_calc3:
                st.metric(
                    label="🧩 객체 검출 정확도 대조",
                    value=f"OpenVINO: {max_conf_ov:.2f}",
                    delta=f"PyTorch CPU: {max_conf_pt:.2f} (검출 수: {fast_results['OpenVINO (CPU)']['count']} vs {fast_results['PyTorch (CPU)']['count']})"
                )
            
            st.markdown("---")
            
            # 비교 렌더링 프레임 시각화
            st.markdown("### 🔲 검출 경계상자 출력 결과 비교 (Detected Bounding Boxes)")
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                st.image(fast_frames["PyTorch (CPU)"], caption=f"PyTorch (CPU) | 지연: {avg_lat_pt:.1f}ms | 처리율: {fast_results['PyTorch (CPU)']['fps']:.1f} FPS", width='stretch')
            with col_f2:
                st.image(fast_frames["OpenVINO (CPU)"], caption=f"OpenVINO (CPU) | 지연: {avg_lat_ov:.1f}ms | 처리율: {fast_results['OpenVINO (CPU)']['fps']:.1f} FPS", width='stretch')

# ---------------------------------------------------------------------------
#                      TAB 2: 🛸 이미지 심층 분석 (벤치마크 & SHAP)
# ---------------------------------------------------------------------------
with tab_image:
    st.subheader("정밀 벤치마크 및 설명가능한 인공지능 (XAI/SHAP) 분석")
    st.markdown("다각도 성능 테스트를 실행하고, 게임이론(Shapley Value)을 픽셀 레벨로 분석하여 인공지능이 드론의 기체, 날개, 배경부 중 어느 지점을 기준으로 객체를 판정했는지 가시화합니다.")
    
    uploaded_image = st.file_uploader("📂 정밀 분석 및 SHAP 시각화용 드론 이미지 등록...", type=["jpg", "jpeg", "png", "bmp", "webp"], key="img_uploader")

    if uploaded_image is not None:
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        img_original_bgr = cv2.imdecode(file_bytes, 1)
        img_original_rgb = cv2.cvtColor(img_original_bgr, cv2.COLOR_BGR2RGB)
        
        img_640_rgb = cv2.resize(img_original_rgb, (640, 640))
        img_640_bgr = cv2.cvtColor(img_640_rgb, cv2.COLOR_RGB2BGR)
        img_input = img_640_rgb.astype(np.float32) / 255.0
        
        st.image(img_original_rgb, caption="업로드된 분석 원본 이미지", width='stretch')
        
        if st.button("🚀 종합 정밀 진단 구동 (대량 연산 실행)", type="primary", key="btn_run_img"):
            step_progress = st.progress(0)
            step_status = st.empty()
            
            # 하드웨어 스위치 상태에 맞춰 테스트 디바이스 정의
            configs = []
            if enable_gpu:
                configs.append({"name": "PyTorch (GPU)", "model": pt_model, "device": gpu_device})
            configs.extend([
                {"name": "PyTorch (CPU)", "model": pt_model, "device": "cpu"},
                {"name": "OpenVINO (CPU)", "model": ov_model, "device": "cpu"}
            ])
            
            benchmark_results = {}
            annotated_frames = {}
            
            # --- 파트 A: 멀티 단말 벤치마킹 연산 단계 ---
            for c_idx, cfg in enumerate(configs):
                cfg_name = cfg["name"]
                model_inst = cfg["model"]
                dev = cfg["device"]
                
                step_status.text(f"⏳ [단계 {c_idx*2 + 1} / {len(configs)*2 + 6}] {cfg_name} 가속 및 {warmup_runs}회 웜업(Warmup) 처리 중...")
                step_progress.progress(int((c_idx*2 + 1) * (40 / (len(configs)*2 + 6))))
                for _ in range(warmup_runs):
                    _ = model_inst(img_640_bgr, device=dev, verbose=False)
                
                step_status.text(f"⏳ [단계 {c_idx*2 + 2} / {len(configs)*2 + 6}] {cfg_name} 연속 {benchmark_runs}회 기가플롭스 처리량 및 성능 데이터 누적 중...")
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

            # --- 파트 B: XAI 설명가능성 SHAP 연산 단계 ---
            base_step = len(configs)*2
            step_status.text(f"⏳ [단계 {base_step + 1} / {base_step + 6}] SHAP 마스킹 타겟 분할 및 가상 배경 동기화 수립 중...")
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
            
            # 🚨 SHAP 필수 하한 조건 방어 코드: 사용자 슬라이더 조작 미스로 인한 수학적 ValueError 차단
            required_min_evals = 2 * num_features + 1
            if max_evals < required_min_evals:
                max_evals = required_min_evals

            step_status.text(f"⏳ [단계 {base_step + 2} / {base_step + 6}] PyTorch 핵심 기여도 탐색을 위한 {max_evals}회 미시 연산 중...")
            step_progress.progress(65)
            explainer_pt = shap.Explainer(pt_f, shap.maskers.Independent(np.zeros((1, num_features))))
            shap_values_pt = explainer_pt(np.ones((1, num_features)), max_evals=max_evals, batch_size=batch_size)
            
            step_status.text(f"⏳ [단계 {base_step + 3} / {base_step + 6}] OpenVINO 최적화 분기 탐색을 위한 {max_evals}회 가중치 추적 연산 중...")
            step_progress.progress(85)
            explainer_ov = shap.Explainer(ov_f, shap.maskers.Independent(np.zeros((1, num_features))))
            shap_values_ov = explainer_ov(np.ones((1, num_features)), max_evals=max_evals, batch_size=batch_size)
            
            step_status.text(f"⏳ [단계 {base_step + 4} / {base_step + 6}] 샤플리 기여도 정합 대조 맵 2차원 공간 선형 보간 생성 중...")
            step_progress.progress(95)
            pt_values = shap_values_pt.values[0, :, 1].reshape(grid_size, grid_size)
            ov_values = shap_values_ov.values[0, :, 1].reshape(grid_size, grid_size)
            pt_heatmap = cv2.resize(pt_values, (640, 640), interpolation=cv2.INTER_LINEAR)
            ov_heatmap = cv2.resize(ov_values, (640, 640), interpolation=cv2.INTER_LINEAR)
            
            step_progress.progress(100)
            step_status.empty()
            step_progress.empty()
            
            st.success("🎉 정밀 종합 진단 모델 및 XAI 기여도 보고서가 출력되었습니다!")
            
            # 성능 가속화 지표 정량 데이터 산출
            pt_cpu_lat = benchmark_results["PyTorch (CPU)"]["latency"]
            ov_cpu_lat = benchmark_results["OpenVINO (CPU)"]["latency"]
            cpu_speedup_ratio = pt_cpu_lat / ov_cpu_lat
            cpu_latency_diff = pt_cpu_lat - ov_cpu_lat
            cpu_percent_improved = (cpu_latency_diff / pt_cpu_lat) * 100
            
            # 1. 딥 런타임 성능 지표 카드화
            st.markdown("### 📊 다중 디바이스 연산 정량 비교 지표 (Runtime Metrics)")
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.metric(
                    label="⚡ CPU 가속 개선 비율", 
                    value=f"{cpu_speedup_ratio:.2f} 배 향상", 
                    delta="OpenVINO 최적화 분기 고속 동작"
                )
            with col_m2:
                st.metric(
                    label="⏱️ 단일 프레임 지연 감소폭", 
                    value=f"{cpu_latency_diff:.2f} ms 단축", 
                    delta=f"기존 PyTorch CPU 대비 {cpu_percent_improved:.1f}% 개선",
                    delta_color="normal"
                )
            with col_m3:
                if enable_gpu and "PyTorch (GPU)" in benchmark_results:
                    gpu_lat = benchmark_results["PyTorch (GPU)"]["latency"]
                    gpu_speedup = pt_cpu_lat / gpu_lat
                    st.metric(
                        label="🚀 GPU 가속 대조 배율", 
                        value=f"{gpu_speedup:.1f} 배", 
                        delta=f"GPU 단일 프레임 지연: {gpu_lat:.1f} ms",
                        delta_color="normal"
                    )
                else:
                    st.metric(
                        label="🔌 GPU 성능 분석 위젯", 
                        value="비활성화", 
                        delta="사이드바 가속 모드를 활성화하십시오."
                    )
                    
            st.markdown("---")
            
            # 2. 객체 탐지 프레임 대조군
            st.markdown("### 🔲 검출 경계상자 결과 비교 (Detected Bounding Boxes)")
            col_b = st.columns(len(configs))
            for b_idx, name_b in enumerate(benchmark_results.keys()):
                with col_b[b_idx]:
                    st.image(
                        annotated_frames[name_b], 
                        caption=f"{name_b} | 지연: {benchmark_results[name_b]['latency']:.1f}ms | 신뢰도 점수: {benchmark_results[name_b]['conf']:.2f}", 
                        width='stretch'
                    )
                
            st.markdown("---")
            
            # 3. SHAP 기여도 열지도 대조군
            st.markdown("### 🔮 피쳐 가중치 기여도 분포 맵 비교 (SHAP Interpretability)")
            st.info("💡 SHAP 기여도 해석 가이드: 【빨간색 영역】은 드론 객체 탐지 확률을 높이는 데 기여한 영역(기체 외곽선, 프로펠러 등)입니다. 반대로 【파란색 영역】은 인공지능 탐지 결정에 악영향을 준 방해 및 오인지 노이즈 요소를 나타냅니다.")
            
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
            cbar.set_label("Shapley 기여치 분배 척도 (빨강: 긍정적 기여치 | 파랑: 방해 요소)", fontsize=10, fontweight='bold')
            
            col_s1, col_s2 = st.columns([4, 1])
            with col_s1:
                st.pyplot(fig_shap)
                
            st.markdown("---")
            
            # 4. 정밀 벤치마킹 데이터 차트
            st.markdown("### 📊 정밀 벤치마크 처리 성능 비교 (지연 시간 vs 실시간 처리 프레임률)")
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
    st.subheader("실시간 동영상 프레임 추적 및 데이터 합성")
    st.markdown("비디오 파일을 탑재하고 추론 하드웨어 엔드포인트를 지정하면 YOLOv11 정밀 식별 엔진이 실시간으로 표식을 렌더링하여 다운로드 파일로 재출력합니다.")
    
    uploaded_video = st.file_uploader("📂 추적 테스트용 드론 주행 영상 탐색...", type=["mp4", "avi", "mov", "mkv"], key="vid_uploader")
    
    if uploaded_video is not None:
        st.markdown("#### ⚙️ 실시간 비디오 추적 환경 설정")
        
        # GPU 가용 및 활성 상태에 매칭되는 백엔드 옵션 필터링
        engine_options = ["PyTorch (CPU)", "OpenVINO (CPU)"]
        if enable_gpu and gpu_available:
            engine_options.insert(0, "PyTorch (GPU)")
            
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            engine_option = st.radio(
                "🚀 탐지 가속화 백엔드 엔진 선택:",
                options=engine_options,
                help="GPU 자원이 가용한 상황이라면 하드웨어 가속 모드를 활성화한 뒤 PyTorch GPU 백엔드 설정을 권장하며, 일반 CPU 임베디드 장치 환경에서는 하드웨어 한계를 쥐어짜내는 OpenVINO CPU 최적화를 권장합니다."
            )
        with col_v2:
            conf_threshold = st.slider("최소 탐지 임계 확률치 (Confidence)", min_value=0.05, max_value=0.95, value=0.25, step=0.05)
            
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
            
        if st.button("🎥 비디오 실시간 탐지 개시", type="primary", key="btn_run_video"):
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_video.read())
            tfile.close()
            
            cap = cv2.VideoCapture(tfile.name)
            
            if not cap.isOpened():
                st.error("❌ 비디오 파일을 정상적으로 탑재할 수 없습니다. (H.264 코덱 형식의 MP4 확장자 사용을 권장합니다.)")
            else:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                st.info(f"📊 비디오 해상도 구조: {width}x{height} | 가변 프레임률: {fps:.1f} FPS | 전체 구성 프레임: {total_frames} 장")
                
                video_placeholder = st.empty()
                progress_bar_v = st.progress(0)
                status_text_v = st.empty()
                
                out_tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                out_tfile.close()
                
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
                    # 프레임별 메모리 버퍼 실시간 디스플레이 전송
                    video_placeholder.image(frame_rgb, caption=f"추적 및 가시화 연산 처리 중: [{frame_count}/{total_frames}]", width='stretch')
                    
                    progress_percent = int((frame_count / total_frames) * 100)
                    progress_bar_v.progress(progress_percent)
                    status_text_v.text(f"🚀 실시간 프레임 세부 활성화 성분 추출 중... 처리 현황: {frame_count}/{total_frames} ({progress_percent}%)")
                
                cap.release()
                out_writer.release()
                
                total_duration = time.time() - start_track_time
                avg_fps_run = frame_count / total_duration
                
                progress_bar_v.empty()
                status_text_v.empty()
                video_placeholder.empty()
                
                st.success(f"🎉 동적 드론 추적 및 비디오 데이터 합성이 완료되었습니다! 총 연산 소요 시간: {total_duration:.1f}초 | 하드웨어 처리율: {avg_fps_run:.1f} FPS")
                
                with open(out_tfile.name, 'rb') as f:
                    st.download_button(
                        label="📥 최종 렌더링 비디오 결과 파일 다운로드 (.mp4)",
                        data=f,
                        file_name="drone_tracked_output.mp4",
                        mime="video/mp4"
                    )