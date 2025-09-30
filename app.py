import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Настройка страницы
st.set_page_config(
    page_title="Оптимизация интервалов испытания скважин",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Заголовок приложения
st.title("🛢️ Оптимизация интервалов испытания нефтяных скважин")
st.markdown("---")

# Функции оптимизации (перенесены из Jupyter notebook)
def optimize_interval(
    df, 
    well_name,
    mu_o=0.43,  # вязкость нефти, Па·с
    mu_w=0.33,  # вязкость воды, Па·с
    P_res=30 * 1.013e5,  # пластовое давление, Па
    P_well=20 * 1.013e5, # давление на устье, Па
    r_w=0.1,             # радиус скважины, м
    r_e=500,             # радиус дренирования, м
    h=0.1,               # толщина пласта, м
    Soi=0.28,            # остаточная нефтеёмкость
    B=1.2,               # объемный коэффициент
    prise_oil=15000,     # цена нефти, руб/м3
    prise_w=800,         # утилизация воды, руб/м3
    opex_liquid=250,     # OPEX на жидкость, руб/м3
    ndpi_rate=0.01,      # НДПИ от выручки
    reculc=902007.49,      # коэффициент пересчёта
    window_sizes_min_m=4,
    window_sizes_max_m=35,
    progress_callback=None  # callback для обновления прогресс-бара
):
    """
    Поиск оптимального интервала испытания по прибыли
    """
    # фильтрация по скважине
    well = df[df.well == well_name].copy().reset_index(drop=True)

    # фазовые параметры (относительные проницаемости)
    well['k_ro'] = well['fill_cb_res'] * (1 - well['KVS_fin']) - Soi
    well['k_ro'] = np.clip(well['k_ro'], 0, 1)
    k_ro = well['k_ro']
    kv = 1 - well['fill_cb_res'] * (1 - well['KVS_fin'])
    k_rw = kv - well['KVS_fin']
    k_rw = np.clip(k_rw, 0, 1)

    well['f_o'] = (k_ro / mu_o) / (k_ro / mu_o + k_rw / mu_w)
    # Формула Дарси для радиального притока: Q = (2π*k*h*ΔP) / (μ*B*ln(re/rw))
    # Для многофазного потока используем относительные проницаемости
    # Используем абсолютную проницаемость из данных
    well['Q_total'] = (
        (2 * np.pi * well["KPR_cb"] * h * (P_res - P_well)) /
        (B * np.log(r_e / r_w))
    ) * (k_ro / mu_o + k_rw / mu_w)
    well['Q_oil'] = well['Q_total'] * well['f_o']
    well['Q_wat'] = well['Q_total'] * (1 - well['f_o'])

    # перевод метров в точки (шаг 0.1 м → умножаем на 10)
    window_sizes_min = window_sizes_min_m * 10
    window_sizes_max = window_sizes_max_m * 10

    best_window = None
    best_start = None
    best_metric = -np.inf
    best_result = {}
    
    # Подсчёт общего количества итераций для прогресс-бара
    total_windows = window_sizes_max - window_sizes_min + 1
    completed_windows = 0

    for window in range(window_sizes_min, window_sizes_max + 1):
        # Обновляем прогресс-бар
        if progress_callback is not None:
            progress = completed_windows / total_windows
            progress_callback(progress, f"Проверка окна {window/10:.1f}м ({completed_windows+1}/{total_windows})")
        for start in range(0, len(well) - window + 1):
            end = start + window
            df_window = well.iloc[start:end]

            # выручка
            revenue = prise_oil * df_window['Q_oil'].sum() / reculc
            # налог
            ndpi = ndpi_rate * revenue
            # затраты (вода + подъём жидкости)
            costs = (
                prise_w * df_window['Q_wat'].sum() / reculc +
                opex_liquid * df_window['Q_total'].sum() / reculc
            )
            # чистая прибыль
            profit = revenue - ndpi - costs

            if profit > best_metric:
                best_metric = profit
                best_window = window
                best_start = start
                Q_oil_total = df_window['Q_oil'].sum() / reculc
                Q_wat_total = df_window['Q_wat'].sum() / reculc
                water_cut = Q_wat_total / (Q_oil_total + Q_wat_total) if (Q_oil_total + Q_wat_total) > 0 else 0
                
                best_result = {
                    'revenue': revenue,
                    'ndpi': ndpi,
                    'costs': costs,
                    'profit': profit,
                    'Q_oil': Q_oil_total,
                    'Q_wat': Q_wat_total,
                    'water_cut': water_cut,
                    'depth_start': df_window.iloc[0]['DEPTH'],
                    'depth_end': df_window.iloc[-1]['DEPTH'],
                    'window_m': window / 10
                }
        
        completed_windows += 1
    
    # Завершаем прогресс-бар
    if progress_callback is not None:
        progress_callback(1.0, "Завершено!")

    return best_result

def optimize_interval_with_loss(
    df, well_name,
    mu_o=0.43, mu_w=0.33,
    P_res=30*1.013e5, P_well=20*1.013e5,
    r_w=0.1, r_e=500, h=0.1, Soi=0.28, B=1.2,
    prise_oil=15000, prise_w=800, opex_liquid=250, ndpi_rate=0.01,
    window_sizes_min_m=4, window_sizes_max_m=35,
    fact_start_depth=None, fact_end_depth=None,
    fact_q=None, fact_wc=None, fact_time=None,
    progress_callback=None  # callback для обновления прогресс-бара
):
    """
    Оптимизация интервала испытания и оценка недополученной прибыли.
    """
    well = df[df.well == well_name].reset_index(drop=True)

    # --- расчёт фазовых потоков ---
    # Относительные проницаемости (без деления на вязкость)
    well['k_ro'] = well['fill_cb_res'] * (1 - well['KVS_fin']) - Soi
    well['k_ro'] = np.clip(well['k_ro'], 0, 1)
    k_ro = well['k_ro']
    kv = 1 - well['fill_cb_res'] * (1 - well['KVS_fin'])
    k_rw = kv - well['KVS_fin']
    k_rw = np.clip(k_rw, 0, 1)
    
    well['f_o'] = (k_ro / mu_o) / (k_ro / mu_o + k_rw / mu_w)
    # Формула Дарси для радиального притока
    # Используем абсолютную проницаемость из данных
    well['Q_total'] = ((2 * np.pi * well["KPR_cb"] * h * (P_res - P_well)) / (B * np.log(r_e / r_w))) * (k_ro / mu_o + k_rw / mu_w)
    well['Q_oil'] = well['Q_total'] * well['f_o']
    well['Q_wat'] = well['Q_total'] * (1 - well['f_o'])

    # --- 1. Подгонка коэффициентов под фактические данные ---
    if fact_start_depth is not None and fact_end_depth is not None and fact_q is not None and fact_wc is not None:
        fact_window = well[(well['DEPTH'] >= fact_start_depth) & (well['DEPTH'] <= fact_end_depth)]
        if len(fact_window) == 0:
            raise ValueError("Фактический интервал не найден в данных")

        # Коэффициент пересчёта общего дебита
        model_q = fact_window['Q_oil'].sum() + fact_window['Q_wat'].sum()
        reculc = model_q / fact_q if fact_q > 0 else 1.0
        
        # Коэффициенты коррекции по фазам
        model_q_oil = fact_window['Q_oil'].sum() / reculc
        model_q_wat = fact_window['Q_wat'].sum() / reculc
        model_wc = model_q_wat / (model_q_oil + model_q_wat) if (model_q_oil + model_q_wat) > 0 else 0
        
        # Коэффициенты коррекции для нефти и воды
        # Если модель завышает обводнённость, oil_correction > 1, wat_correction < 1
        fact_q_oil_real = fact_q * (1 - fact_wc)
        fact_q_wat_real = fact_q * fact_wc
        
        oil_correction = fact_q_oil_real / model_q_oil if model_q_oil > 0 else 1.0
        wat_correction = fact_q_wat_real / model_q_wat if model_q_wat > 0 else 1.0
    else:
        reculc = 1.0
        oil_correction = 1.0
        wat_correction = 1.0

    # --- 2. Прибыль по факту ---
    if fact_q is not None and fact_wc is not None and fact_time is not None:
        fact_q_oil = fact_q * (1 - fact_wc)
        fact_q_wat = fact_q * fact_wc
        # Расчёт ИДЕНТИЧНЫЙ оптимальному: выручка - НДПИ - затраты
        fact_revenue = prise_oil * fact_q_oil * fact_time
        fact_ndpi = ndpi_rate * fact_revenue
        fact_costs = prise_w * fact_q_wat * fact_time + opex_liquid * fact_q * fact_time
        fact_profit = fact_revenue - fact_ndpi - fact_costs
    else:
        fact_profit = None

    # --- 3. Поиск оптимального интервала (БЕЗ коррекции по фазам!) ---
    window_sizes = range(window_sizes_min_m*10, window_sizes_max_m*10)
    best = {"profit": -np.inf}
    
    # Подсчёт общего количества итераций для прогресс-бара
    total_windows = len(window_sizes)
    completed_windows = 0

    for window in window_sizes:
        # Обновляем прогресс-бар
        if progress_callback is not None:
            progress = completed_windows / total_windows
            progress_callback(progress, f"Анализ потерь: окно {window/10:.1f}м ({completed_windows+1}/{total_windows})")
        for start in range(0, len(well) - window + 1):
            end = start + window
            df_window = well.iloc[start:end]

            # Модельные значения (только общая калибровка по reculc)
            Q_oil_sum = df_window['Q_oil'].sum() / reculc
            Q_wat_sum = df_window['Q_wat'].sum() / reculc
            Q_total_sum = df_window['Q_total'].sum() / reculc
            
            # Расчёт за период fact_time
            revenue = prise_oil * Q_oil_sum * fact_time
            ndpi = ndpi_rate * revenue
            costs = prise_w * Q_wat_sum * fact_time + opex_liquid * Q_total_sum * fact_time
            profit = revenue - ndpi - costs

            if profit > best["profit"]:
                best = {
                    "profit": profit,
                    "revenue": revenue,
                    "ndpi": ndpi,
                    "costs": costs,
                    "window": window,
                    "start": start,
                    "end": end,
                    "Q_oil_sum": Q_oil_sum,
                    "Q_wat_sum": Q_wat_sum,
                    "Q_total_sum": Q_total_sum,
                    "start_depth": df_window.iloc[0]['DEPTH'],
                    "end_depth": df_window.iloc[-1]['DEPTH']
                }
        
        completed_windows += 1
    
    # Завершаем прогресс-бар
    if progress_callback is not None:
        progress_callback(1.0, "Завершено!")
    
    # --- 4. Применяем коррекцию по фазам ТОЛЬКО к найденному оптимуму для сравнения ---
    best["Q_oil_corrected"] = best["Q_oil_sum"] * oil_correction
    best["Q_wat_corrected"] = best["Q_wat_sum"] * wat_correction
    best["Q_total_corrected"] = best["Q_oil_corrected"] + best["Q_wat_corrected"]
    
    # Пересчитываем экономику с корректированными дебитами
    best["revenue_corrected"] = prise_oil * best["Q_oil_corrected"] * fact_time
    best["ndpi_corrected"] = ndpi_rate * best["revenue_corrected"]
    best["costs_corrected"] = prise_w * best["Q_wat_corrected"] * fact_time + opex_liquid * best["Q_total_corrected"] * fact_time
    best["profit_corrected"] = best["revenue_corrected"] - best["ndpi_corrected"] - best["costs_corrected"]

    # --- 5. Считаем недополученную прибыль (используем скорректированные значения!) ---
    if fact_profit is not None:
        loss = best["profit_corrected"] - fact_profit
    else:
        loss = None

    return {
        "fact_profit": fact_profit,
        # Для ОТОБРАЖЕНИЯ используем скорректированные значения
        "optimal_profit": best["profit_corrected"],
        "optimal_revenue": best["revenue_corrected"],
        "optimal_ndpi": best["ndpi_corrected"],
        "optimal_costs": best["costs_corrected"],
        "loss": loss,
        "optimal_interval": (best["start_depth"], best["end_depth"]),
        "optimal_window_m": best["window"]/10,
        "optimal_Q_oil": best["Q_oil_corrected"],
        "optimal_Q_wat": best["Q_wat_corrected"],
        "reculc": reculc,
        "oil_correction": oil_correction,
        "wat_correction": wat_correction
    }

# Боковая панель с настройками
st.sidebar.header("⚙️ Параметры анализа")

# Загрузка данных
st.sidebar.subheader("📁 Загрузка данных")
uploaded_file = st.sidebar.file_uploader(
    "Загрузите Excel файл с данными скважин",
    type=['xlsx', 'xls'],
    help="Файл должен содержать колонки: well, DEPTH, fill_cb_res, KVS_fin, KPR_cb"
)

# Параметры скважины
st.sidebar.subheader("🛢️ Параметры скважины")
# well_name будет определен после загрузки данных

# Физические параметры
st.sidebar.subheader("🔬 Физические параметры")
mu_o = st.sidebar.slider("Вязкость нефти (Па·с)", 0.1, 1.0, 0.43, 0.01)
mu_w = st.sidebar.slider("Вязкость воды (Па·с)", 0.1, 1.0, 0.33, 0.01)
P_res = st.sidebar.slider("Пластовое давление (атм)", 10, 50, 30, 1) * 1.013e5
P_well = st.sidebar.slider("Давление на устье (атм)", 5, 30, 20, 1) * 1.013e5
r_w = st.sidebar.slider("Радиус скважины (м)", 0.05, 0.2, 0.1, 0.01)
r_e = st.sidebar.slider("Радиус дренирования (м)", 100, 1000, 500, 10)
h = st.sidebar.slider("Толщина пласта (м)", 0.05, 0.5, 0.1, 0.01)
Soi = st.sidebar.slider("Остаточная нефтеёмкость", 0.1, 0.5, 0.28, 0.01)
B = st.sidebar.slider("Объемный коэффициент", 1.0, 2.0, 1.2, 0.1)

# Экономические параметры
st.sidebar.subheader("💰 Экономические параметры")
prise_oil = st.sidebar.number_input("Цена нефти (руб/м³)", 5000, 50000, 15000, 500)
prise_w = st.sidebar.number_input("Утилизация воды (руб/м³)", 100, 2000, 800, 50)
opex_liquid = st.sidebar.number_input("OPEX на жидкость (руб/м³)", 50, 1000, 250, 25)
ndpi_rate = st.sidebar.slider("НДПИ от выручки", 0.0, 0.1, 0.01, 0.001)
reculc = st.sidebar.number_input("Коэффициент пересчёта", 100000.0, 5000000.0, 902007.49, 1000.0, format="%.2f")

# Параметры окна
st.sidebar.subheader("📏 Параметры окна")
window_min = st.sidebar.slider("Минимальная длина окна (м)", 1, 20, 4, 1)
window_max = st.sidebar.slider("Максимальная длина окна (м)", 10, 50, 35, 1)

# Дополнительные параметры для второй функции
st.sidebar.subheader("📊 Фактические данные (для анализа потерь)")
st.sidebar.info("⚠️ Укажите РЕАЛЬНЫЕ измеренные данные испытания. Не меняйте их для корректного сравнения!")
fact_start_depth = st.sidebar.number_input("Фактическая глубина начала (м)", 0, 5000, 2828, 1)
fact_end_depth = st.sidebar.number_input("Фактическая глубина конца (м)", 0, 5000, 2848, 1)
fact_q = st.sidebar.number_input("Фактический дебит (м³/сут)", 0.0, 1000.0, 20.0, 0.1)
fact_wc = st.sidebar.slider("Фактическая обводнённость", 0.0, 1.0, 0.7, 0.01, 
                             help="ИЗМЕРЕННАЯ обводнённость из испытания. Не меняйте для корректного сравнения!")
fact_time = st.sidebar.number_input("Время работы скважины (сутки)", 1, 3650, 365, 1)

# Кнопка пересчёта
st.sidebar.markdown("---")
calculate_button = st.sidebar.button("🔄 Пересчитать", type="primary", use_container_width=True)

# Основной контент
if uploaded_file is not None:
    try:
        # Загрузка данных
        df = pd.read_excel(uploaded_file)
        
        # Проверка наличия необходимых колонок
        required_columns = ['well', 'DEPTH', 'fill_cb_res', 'KVS_fin', 'KPR_cb']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"❌ Отсутствуют необходимые колонки: {missing_columns}")
            st.info("Требуемые колонки: well, DEPTH, fill_cb_res, KVS_fin, KPR_cb")
        else:
            # Получение списка доступных скважин
            available_wells = df['well'].unique()
            
            # Выбор скважины из списка
            st.sidebar.subheader("🛢️ Выбор скважины")
            well_name = st.sidebar.selectbox(
                "Выберите скважину для анализа",
                options=available_wells,
                index=0 if len(available_wells) > 0 else None,
                help="Выберите скважину из загруженных данных"
            )
            
            if well_name:
                # Отображение информации о выбранной скважине
                st.success(f"✅ Выбрана скважина: **{well_name}**")
                
                # Показываем статистику по скважине
                well_data = df[df.well == well_name]
                st.info(f"📊 Данные скважины: {len(well_data)} точек, глубина от {well_data['DEPTH'].min():.1f} до {well_data['DEPTH'].max():.1f} м")
                
                # Инициализация session_state для хранения результатов
                if 'results' not in st.session_state:
                    st.session_state.results = None
                if 'results_loss' not in st.session_state:
                    st.session_state.results_loss = None
                
                # Выполнение расчётов при нажатии кнопки
                if calculate_button:
                    # Создаём placeholder для прогресс-бара
                    progress_placeholder = st.empty()
                    status_placeholder = st.empty()
                    
                    # Создаём прогресс-бар
                    progress_bar = progress_placeholder.progress(0)
                    
                    # Callback функция для обновления прогресс-бара
                    def update_progress(progress, status_text):
                        progress_bar.progress(progress)
                        status_placeholder.text(status_text)
                    
                    try:
                        # Оптимизация интервала
                        update_progress(0.0, "🔄 Запуск оптимизации интервала...")
                        st.session_state.results = optimize_interval(
                            df, well_name,
                            mu_o=mu_o, mu_w=mu_w,
                            P_res=P_res, P_well=P_well,
                            r_w=r_w, r_e=r_e, h=h, Soi=Soi, B=B,
                            prise_oil=prise_oil, prise_w=prise_w,
                            opex_liquid=opex_liquid, ndpi_rate=ndpi_rate,
                            reculc=reculc,
                            window_sizes_min_m=window_min,
                            window_sizes_max_m=window_max,
                            progress_callback=update_progress
                        )
                        
                        # Анализ потерь
                        update_progress(0.0, "🔄 Запуск анализа потерь...")
                        st.session_state.results_loss = optimize_interval_with_loss(
                            df, well_name,
                            mu_o=mu_o, mu_w=mu_w,
                            P_res=P_res, P_well=P_well,
                            r_w=r_w, r_e=r_e, h=h, Soi=Soi, B=B,
                            prise_oil=prise_oil, prise_w=prise_w,
                            opex_liquid=opex_liquid, ndpi_rate=ndpi_rate,
                            window_sizes_min_m=window_min, window_sizes_max_m=window_max,
                            fact_start_depth=fact_start_depth, fact_end_depth=fact_end_depth,
                            fact_q=fact_q, fact_wc=fact_wc, fact_time=fact_time,
                            progress_callback=update_progress
                        )
                        
                        # Очищаем прогресс-бар и показываем успех
                        progress_placeholder.empty()
                        status_placeholder.empty()
                        st.success("✅ Пересчёт завершён!")
                        
                    except Exception as e:
                        progress_placeholder.empty()
                        status_placeholder.empty()
                        st.error(f"❌ Ошибка при расчёте: {str(e)}")
                
                # Создание вкладок
                tab1, tab2 = st.tabs(["🔍 Оптимизация интервала", "💰 Анализ потерь"])
                
                with tab1:
                    st.header("🔍 Оптимизация интервала испытания")
                    st.markdown("Поиск оптимального интервала по максимальной прибыли")
                    
                    # Проверка наличия результатов
                    if st.session_state.results is None:
                        st.warning("⚠️ Измените параметры и нажмите кнопку **'Пересчитать'** для выполнения оптимизации")
                    else:
                        result = st.session_state.results
                        
                        # Отображение результатов
                        st.success("✅ Оптимизация завершена!")
                        
                        # Основные метрики
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric(
                                "Чистая прибыль",
                                f"{result['profit']:,.0f} руб",
                                delta=None
                            )
                        
                        with col2:
                            st.metric(
                                "Добыча нефти",
                                f"{result['Q_oil']:.2f} м³",
                                delta=None
                            )
                        
                        with col3:
                            st.metric(
                                "Добыча воды",
                                f"{result['Q_wat']:.2f} м³",
                                delta=None
                            )
                        
                        with col4:
                            st.metric(
                                "Длина интервала",
                                f"{result['window_m']:.1f} м",
                                delta=None
                            )
                        
                        # Детальная информация
                        st.subheader("📊 Детальные результаты")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**Экономические показатели:**")
                            st.write(f"• Выручка: {result['revenue']:,.2f} руб")
                            st.write(f"• НДПИ: {result['ndpi']:,.2f} руб")
                            st.write(f"• Затраты: {result['costs']:,.2f} руб")
                            st.write(f"• Чистая прибыль: {result['profit']:,.2f} руб")
                        
                        with col2:
                            st.markdown("**Технические показатели:**")
                            st.write(f"• Глубина начала: {result['depth_start']:.1f} м")
                            st.write(f"• Глубина конца: {result['depth_end']:.1f} м")
                            st.write(f"• Длина интервала: {result['window_m']:.1f} м")
                            st.write(f"• Добыча нефти: {result['Q_oil']:.2f} м³")
                            st.write(f"• Добыча воды: {result['Q_wat']:.2f} м³")
                            st.write(f"• Обводнённость: {result['water_cut']:.1%}")
                        
                        # Визуализация
                        st.subheader("📈 Визуализация данных")
                        
                        # ========================================
                        # ПОДГОТОВКА ЕДИНОГО ДАТАСЕТА ДЛЯ ВСЕХ ГРАФИКОВ
                        # ========================================
                        
                        # Создаём единый DataFrame для всех расчётов и графиков
                        plot_data = well_data.copy().reset_index(drop=True)
                        
                        # Проверяем наличие необходимых колонок и удаляем строки с NaN
                        required_cols = ['DEPTH', 'fill_cb_res', 'KVS_fin', 'KPR_cb']
                        plot_data = plot_data.dropna(subset=required_cols)
                        
                        # Расчёт относительных проницаемостей
                        plot_data['k_ro'] = plot_data['fill_cb_res'] * (1 - plot_data['KVS_fin']) - Soi
                        plot_data['k_ro'] = np.clip(plot_data['k_ro'], 0, 1)
                        
                        plot_data['kv'] = 1 - plot_data['fill_cb_res'] * (1 - plot_data['KVS_fin'])
                        plot_data['k_rw'] = plot_data['kv'] - plot_data['KVS_fin']
                        plot_data['k_rw'] = np.clip(plot_data['k_rw'], 0, 1)
                        
                        # Расчёт доли нефти с защитой от деления на ноль и переполнения
                        # Ограничиваем вязкости снизу, чтобы избежать inf
                        mu_o_safe = max(mu_o, 1e-6)
                        mu_w_safe = max(mu_w, 1e-6)
                        
                        mobility_oil = plot_data['k_ro'] / mu_o_safe
                        mobility_wat = plot_data['k_rw'] / mu_w_safe
                        
                        # Ограничиваем мобильности сверху, чтобы избежать inf
                        mobility_oil = np.clip(mobility_oil, 0, 1e6)
                        mobility_wat = np.clip(mobility_wat, 0, 1e6)
                        
                        total_mobility = mobility_oil + mobility_wat
                        
                        # Защита от деления на ноль
                        total_mobility = np.where(total_mobility == 0, 1e-10, total_mobility)
                        plot_data['f_o'] = mobility_oil / total_mobility
                        plot_data['f_w'] = 1 - plot_data['f_o']
                        
                        # Заменяем бесконечные значения на NaN
                        plot_data = plot_data.replace([np.inf, -np.inf], np.nan)
                        
                        # Проверяем на NaN после всех расчётов
                        plot_data = plot_data.dropna(subset=['f_o','fill_cb_res', 'KVS_fin', 'KPR_cb', 'f_w'])
                        
                        # Информация о данных
                        st.info(f"📊 Используется {len(plot_data)} точек данных для визуализации (диапазон глубины: {plot_data['DEPTH'].min():.1f} - {plot_data['DEPTH'].max():.1f} м)")
                        
                        # ========================================
                        # ГРАФИК РАСПРЕДЕЛЕНИЯ ПРИБЫЛИ ПО ГЛУБИНЕ
                        # ========================================
                        
                        st.subheader("💰 Распределение прибыли по глубине")
                        
                        with st.spinner("🔄 Расчёт распределения прибыли..."):
                            # Рассчитываем дебиты для plot_data (если ещё не рассчитаны)
                            if 'Q_total' not in plot_data.columns:
                                k_ro_vals = plot_data['k_ro']
                                k_rw_vals = plot_data['k_rw']
                                
                                plot_data['Q_total'] = (
                                    (2 * np.pi * plot_data["KPR_cb"] * h * (P_res - P_well)) /
                                    (B * np.log(r_e / r_w))
                                ) * (k_ro_vals / mu_o + k_rw_vals / mu_w)
                                
                                plot_data['Q_oil'] = plot_data['Q_total'] * plot_data['f_o']
                                plot_data['Q_wat'] = plot_data['Q_total'] * (1 - plot_data['f_o'])
                            
                            # ОПТИМИЗАЦИЯ: Векторизованный расчёт распределения прибыли
                            profit_results = []
                            window_sizes_min_viz = window_min * 10
                            window_sizes_max_viz = window_max * 10
                            
                            # Берём каждое 5-е окно для визуализации (меньше линий на графике)
                            for window in range(window_sizes_min_viz, window_sizes_max_viz + 1, 50):
                                # Векторизованный расчёт для ВСЕХ позиций сразу
                                rolling_oil = plot_data['Q_oil'].rolling(window=window).sum() / reculc
                                rolling_wat = plot_data['Q_wat'].rolling(window=window).sum() / reculc
                                rolling_total = plot_data['Q_total'].rolling(window=window).sum() / reculc
                                
                                # Векторизованная экономика
                                revenues = prise_oil * rolling_oil
                                ndpis = ndpi_rate * revenues
                                costs = prise_w * rolling_wat + opex_liquid * rolling_total
                                profits = revenues - ndpis - costs
                                
                                # Берём каждую 10-ю позицию для графика
                                for i in range(window-1, len(plot_data), 10):
                                    if not pd.isna(profits.iloc[i]):
                                        profit_results.append({
                                            'window_m': window / 10,
                                            'depth_start': plot_data.iloc[i - window + 1]['DEPTH'],
                                            'profit': profits.iloc[i]
                                        })
                            
                            # Создаём DataFrame
                            df_profit = pd.DataFrame(profit_results)
                            
                            # Создаём график
                            fig_profit = go.Figure()
                            
                            # Добавляем линии для каждого размера окна
                            for window_m in sorted(df_profit['window_m'].unique()):
                                df_subset = df_profit[df_profit['window_m'] == window_m]
                                fig_profit.add_trace(go.Scatter(
                                    x=df_subset['depth_start'],
                                    y=df_subset['profit'],
                                    mode='lines',
                                    name=f'{window_m:.0f} м',
                                    opacity=0.7,
                                    line=dict(width=2)
                                ))
                            
                            # Отмечаем оптимальное окно
                            fig_profit.add_trace(go.Scatter(
                                x=[result['depth_start']],
                                y=[result['profit']],
                                mode='markers',
                                marker=dict(size=15, color='red', symbol='star', line=dict(width=2, color='white')),
                                name='Оптимум',
                                showlegend=True
                            ))
                            
                            fig_profit.update_layout(
                                title='Распределение прибыли по глубине для разных размеров окон',
                                xaxis_title='Глубина начала интервала, м',
                                yaxis_title='Прибыль, руб',
                                height=600,
                                showlegend=True,
                                hovermode='x unified',
                                xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
                                yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray')
                            )
                            
                            st.plotly_chart(fig_profit, use_container_width=True)
                            
                            # Краткая статистика
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Максимальная прибыль", f"{df_profit['profit'].max():,.0f} руб")
                            with col2:
                                st.metric("Минимальная прибыль", f"{df_profit['profit'].min():,.0f} руб")
                            with col3:
                                st.metric("Средняя прибыль", f"{df_profit['profit'].mean():,.0f} руб")
                        
                        # ========================================
                        # ПОСТРОЕНИЕ ГРАФИКОВ НА ОСНОВЕ ЕДИНОГО ДАТАСЕТА
                        # ========================================
                        
                        st.subheader("📊 Анализ скважины")
                        
                        fig = make_subplots(
                            rows=1, cols=3,
                            subplot_titles=('Профиль скважины', 'Фазовые потоки', 'Абсолютная проницаемость'),
                            specs=[[{"secondary_y": True}, {"secondary_y": False}, {"secondary_y": False}]]
                        )
                        
                        # График 1: Профиль скважины
                        fig.add_trace(
                            go.Scatter(
                                x=plot_data['fill_cb_res'].values,
                                y=plot_data['DEPTH'].values,
                                mode='lines',
                                name='Заполненность',
                                line=dict(color='blue', width=2)
                            ),
                            row=1, col=1
                        )
                        
                        fig.add_trace(
                            go.Scatter(
                                x=plot_data['KVS_fin'].values,
                                y=plot_data['DEPTH'].values,
                                mode='lines',
                                name='КВС',
                                line=dict(color='red', width=2),
                                yaxis='y2'
                            ),
                            row=1, col=1, secondary_y=True
                        )
                        
                        # График 2: Фазовые потоки (заливка для нефти)
                        fig.add_trace(
                            go.Scatter(
                                x=plot_data['f_o'].values,
                                y=plot_data['DEPTH'].values,
                                #fill='tozerox',
                                fillcolor='rgba(139, 69, 19, 0.5)',
                                line=dict(color='rgba(139, 69, 19, 0.8)', width=2),
                                name='Доля нефти',
                                mode='lines'
                            ),
                            row=1, col=2
                        )
                        
                        # Заливка для воды (синим) - от f_o до 1 (накопительная)
                        ones_array = np.ones(len(plot_data))
                        fig.add_trace(
                            go.Scatter(
                                x=ones_array,
                                y=plot_data['DEPTH'].values,
                                fill='tonextx',
                                fillcolor='rgba(0, 100, 255, 0.4)',
                                line=dict(color='rgba(0, 100, 255, 0.7)', width=1),
                                name='Доля воды',
                                mode='lines'
                            ),
                            row=1, col=2
                        )
                        
                        # График 3: Абсолютная проницаемость (логарифмическая шкала)
                        fig.add_trace(
                            go.Scatter(
                                x=plot_data['KPR_cb'].values,
                                y=plot_data['DEPTH'].values,
                                mode='lines',
                                name='Проницаемость',
                                line=dict(color='purple', width=2)
                            ),
                            row=1, col=3
                        )
                        
                        # Выделение оптимального интервала с заливкой
                        optimal_start = result['depth_start']
                        optimal_end = result['depth_end']
                        
                        # Заливка для оптимального интервала на первом и третьем графиках
                        fig.add_hrect(
                            y0=optimal_start, y1=optimal_end,
                            fillcolor="yellow", opacity=0.4,
                            layer="below", line_width=2,
                            line_color="orange",
                            annotation_text="Оптимальный интервал",
                            annotation_position="top left",
                            row=1, col=1
                        )
                        
                        fig.add_hrect(
                            y0=optimal_start, y1=optimal_end,
                            fillcolor="yellow", opacity=0.4,
                            layer="below", line_width=2,
                            line_color="orange",
                            annotation_text="Оптимальный интервал",
                            annotation_position="top left",
                            row=1, col=3
                        )
                        
                        fig.update_layout(
                            height=800,
                            title_text="Анализ скважины",
                            showlegend=True
                        )
                        
                        # Обратная шкала глубины (глубина увеличивается вниз)
                        fig.update_xaxes(title_text="Заполненность", row=1, col=1)
                        fig.update_yaxes(
                            title_text="Глубина (м)", 
                            row=1, col=1, 
                            autorange="reversed",
                            dtick=10,  # деления каждые 10 метров
                            showgrid=True,
                            gridwidth=1,
                            gridcolor="lightgray"
                        )
                        fig.update_xaxes(title_text="Фазовые потоки", row=1, col=2)
                        fig.update_yaxes(
                            title_text="Глубина (м)", 
                            row=1, col=2, 
                            autorange="reversed",
                            dtick=10,  # деления каждые 10 метров
                            showgrid=True,
                            gridwidth=1,
                            gridcolor="lightgray"
                        )
                        # Настройки для третьего графика (проницаемость в логарифмическом масштабе)
                        fig.update_xaxes(
                            title_text="Проницаемость, м²", 
                            row=1, col=3,
                            type="log",  # логарифмическая шкала
                            showgrid=True,
                            gridwidth=1,
                            gridcolor="lightgray"
                        )
                        fig.update_yaxes(
                            title_text="Глубина (м)", 
                            row=1, col=3, 
                            autorange="reversed",
                            dtick=10,  # деления каждые 10 метров
                            showgrid=True,
                            gridwidth=1,
                            gridcolor="lightgray"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                
                with tab2:
                    st.header("💰 Анализ потерь и недополученной прибыли")
                    st.markdown("Сравнение фактических результатов с оптимальными")
                    
                    # Проверка наличия результатов
                    if st.session_state.results_loss is None:
                        st.warning("⚠️ Измените параметры и нажмите кнопку **'Пересчитать'** для выполнения анализа потерь")
                    else:
                        result_loss = st.session_state.results_loss
                        
                        # Отображение результатов
                        st.success("✅ Анализ потерь завершен!")
                        
                        # Основные метрики
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric(
                                "Фактическая прибыль",
                                f"{result_loss['fact_profit']:,.0f} руб" if result_loss['fact_profit'] else "N/A",
                                delta=None
                            )
                        
                        with col2:
                            st.metric(
                                "Оптимальная прибыль",
                                f"{result_loss['optimal_profit']:,.0f} руб",
                                delta=None
                            )
                        
                        with col3:
                            st.metric(
                                "Недополученная прибыль",
                                f"{result_loss['loss']:,.0f} руб" if result_loss['loss'] else "N/A",
                                delta=f"-{result_loss['loss']:,.0f} руб" if result_loss['loss'] and result_loss['loss'] > 0 else None
                            )
                        
                        with col4:
                            st.metric(
                                "Коэффициент пересчёта",
                                f"{result_loss['reculc']:.3f}",
                                delta=None
                            )
                        
                        # Детальная информация
                        st.subheader("📊 Детальный анализ")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**Фактические данные:**")
                            st.write(f"• Интервал: {fact_start_depth}–{fact_end_depth} м")
                            st.write(f"• Дебит: {fact_q} м³/сут")
                            st.write(f"• Обводнённость: {fact_wc:.1%}")
                            st.write(f"• Время работы: {fact_time} суток")
                            st.write(f"• Прибыль: {result_loss['fact_profit']:,.0f} руб" if result_loss['fact_profit'] else "• Прибыль: N/A")
                        
                        with col2:
                            st.markdown("**Оптимальные данные (скорректированные):**")
                            st.write(f"• Интервал: {result_loss['optimal_interval'][0]:.1f}–{result_loss['optimal_interval'][1]:.1f} м")
                            st.write(f"• Длина окна: {result_loss['optimal_window_m']:.1f} м")
                            st.write(f"• Добыча нефти: {result_loss['optimal_Q_oil']:.2f} м³/сут")
                            st.write(f"• Добыча воды: {result_loss['optimal_Q_wat']:.2f} м³/сут")
                            st.write(f"• Обводнённость: {result_loss['optimal_Q_wat']/(result_loss['optimal_Q_oil']+result_loss['optimal_Q_wat'])*100:.1f}%")
                            st.write(f"• Прибыль: {result_loss['optimal_profit']:,.0f} руб за {fact_time} сут")
                            st.write(f"• Потери: {result_loss['loss']:,.0f} руб" if result_loss['loss'] else "• Потери: N/A")
                            
                            st.markdown("**Коэффициенты коррекции модели:**")
                            st.write(f"• Коррекция нефти: {result_loss['oil_correction']:.3f}")
                            st.write(f"• Коррекция воды: {result_loss['wat_correction']:.3f}")
                            
                            if abs(result_loss['oil_correction'] - 1) > 0.1 or abs(result_loss['wat_correction'] - 1) > 0.1:
                                st.warning("⚠️ Модель скорректирована для соответствия фактическим данным по обводнённости")
                        
                        # Визуализация сравнения
                        st.subheader("📈 Сравнительный анализ")
                        
                        # Рассчитываем показатели для фактического интервала
                        fact_q_oil = fact_q * (1 - fact_wc)
                        fact_q_wat = fact_q * fact_wc
                        fact_revenue = prise_oil * fact_q_oil * fact_time
                        fact_ndpi = ndpi_rate * fact_revenue
                        fact_costs = prise_w * fact_q_wat * fact_time + opex_liquid * fact_q * fact_time
                        fact_profit_calc = fact_revenue - fact_ndpi - fact_costs
                        
                        # Барчарт сравнения (данные из result_loss уже за год)
                        metrics = ['Прибыль', 'Выручка', 'НДПИ', 'Затраты']
                        fact_values = [fact_profit_calc, fact_revenue, fact_ndpi, fact_costs]
                        opt_values = [result_loss['optimal_profit'], 
                                     result_loss['optimal_revenue'],
                                     result_loss['optimal_ndpi'],
                                     result_loss['optimal_costs']]
                        
                        fig_comp = go.Figure(data=[
                            go.Bar(name='Фактический', x=metrics, y=fact_values, 
                                  marker_color='rgba(255, 0, 0, 0.7)',
                                  text=[f'{v/1000:.0f}k' for v in fact_values],
                                  textposition='outside'),
                            go.Bar(name='Оптимальный', x=metrics, y=opt_values, 
                                  marker_color='rgba(0, 200, 0, 0.7)',
                                  text=[f'{v/1000:.0f}k' for v in opt_values],
                                  textposition='outside')
                        ])
                        
                        fig_comp.update_layout(
                            title='Сравнение экономических показателей (за год)',
                            yaxis_title='Руб',
                            barmode='group',
                            height=600,
                            showlegend=True,
                            yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray')
                        )
                        
                        st.plotly_chart(fig_comp, use_container_width=True)
                        
                        # Информация о потенциальном приросте
                        if fact_profit_calc > 0:
                            potential_gain = result_loss['optimal_profit'] - fact_profit_calc
                            gain_percentage = (potential_gain / fact_profit_calc) * 100
                            
                            if potential_gain > 0:
                                st.success(f"💰 **Потенциальный прирост прибыли:** {potential_gain:,.0f} руб за {fact_time} сут (+{gain_percentage:.1f}%)")
                            else:
                                st.info(f"ℹ️ Фактический интервал близок к оптимальному")
                        
                        # Добавляем детальное сравнение
                        st.markdown("### 📊 Детальное сравнение показателей")
                        comparison_df = pd.DataFrame({
                            'Показатель': metrics,
                            'Фактический': [f'{v:,.0f}' for v in fact_values],
                            'Оптимальный': [f'{v:,.0f}' for v in opt_values],
                            'Разница': [f'{opt_values[i] - fact_values[i]:,.0f}' for i in range(len(metrics))],
                            'Прирост, %': [f'{(opt_values[i] / fact_values[i] - 1) * 100:.1f}%' if fact_values[i] != 0 else 'N/A' for i in range(len(metrics))]
                        })
                        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                        
                        # Экономический анализ
                        if result_loss['fact_profit'] and result_loss['loss']:
                            st.subheader("💰 Экономический анализ потерь")
                            
                            loss_percentage = (result_loss['loss'] / result_loss['fact_profit']) * 100 if result_loss['fact_profit'] > 0 else 0
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric(
                                    "Процент потерь",
                                    f"{loss_percentage:.1f}%",
                                    delta=f"-{loss_percentage:.1f}%"
                                )
                            
                            with col2:
                                st.metric(
                                    "Потенциал роста",
                                    f"{((result_loss['optimal_profit'] / result_loss['fact_profit']) - 1) * 100:.1f}%",
                                    delta=f"+{((result_loss['optimal_profit'] / result_loss['fact_profit']) - 1) * 100:.1f}%"
                                )
                            
                            with col3:
                                st.metric(
                                    "Эффективность",
                                    f"{(result_loss['fact_profit'] / result_loss['optimal_profit']) * 100:.1f}%",
                                    delta=f"-{(1 - result_loss['fact_profit'] / result_loss['optimal_profit']) * 100:.1f}%"
                                )
                
    except Exception as e:
        st.error(f"❌ Ошибка при загрузке файла: {str(e)}")

else:
    # Инструкции для пользователя
    st.info("👆 Пожалуйста, загрузите Excel файл с данными скважин в боковой панели")
    
    # Показываем информацию о том, что нужно загрузить файл для выбора скважины
    st.sidebar.info("📁 Загрузите файл для выбора скважины")
    
    st.markdown("""
    ### 📋 Требования к данным
    
    Файл должен содержать следующие колонки:
    - `well` - название скважины
    - `DEPTH` - глубина (м)
    - `fill_cb_res` - заполненность коллектора
    - `KVS_fin` - коэффициент вытеснения нефти водой
    - `KPR_cb` - абсолютная проницаемость (м²)
    
    ### 🔧 Возможности приложения
    
    - **Оптимизация интервалов** - поиск наиболее прибыльного интервала испытания
    - **💰 Анализ потерь** - сравнение фактических и оптимальных результатов
    - **Интерактивные параметры** - настройка физических и экономических параметров
    - **Визуализация** - графики профиля скважины и фазовых потоков
    - **Экономический анализ** - расчет прибыли, затрат и налогов
    
    ### 🛢️ Выбор скважины
    
    После загрузки файла в боковой панели появится выпадающий список со всеми доступными скважинами из ваших данных.
    """)

# Футер
st.markdown("---")
st.markdown("🛢️ **Приложение для оптимизации интервалов испытания нефтяных скважин**")
